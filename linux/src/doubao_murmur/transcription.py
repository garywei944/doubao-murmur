"""State machine orchestrator for the recording lifecycle.

Mirrors TranscriptionManager.swift.

Key design decisions:
- GLib.idle_add() marshals callbacks from the asyncio thread to GTK main thread
- GLib.timeout_add() replaces DispatchQueue.main.asyncAfter for delayed execution
- State machine exactly mirrors macOS: idle -> starting -> recording -> stopping -> idle
- Every recording owns a generation-scoped ASR client so stale callbacks are ignored
"""

from __future__ import annotations

import logging

from gi.repository import GLib

from doubao_murmur.app_state import AppState, LoginStatus, RecordingState
from doubao_murmur.asr_client import ASRClient
from doubao_murmur.audio_capture import AudioCapture
from doubao_murmur.config import (
    AUTH_EXPIRY_DELAY,
    FINAL_RESULT_QUIET_PERIOD,
    START_SAFETY_TIMEOUT,
    STOP_NO_RESULT_TIMEOUT,
    STOP_SAFETY_TIMEOUT,
)
from doubao_murmur.params_store import ASRParams, ParamsStore

logger = logging.getLogger(__name__)


def _is_http_auth_rejection(error) -> bool:
    """True if the WebSocket handshake was rejected with HTTP 401/403.

    Covers both websockets>=13 (InvalidStatus with .response.status_code)
    and older releases (InvalidStatusCode with .status_code). Anything
    else -- timeouts, DNS failures, connection resets -- is a network
    problem, not an auth problem.
    """
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    return status in (200, 401, 403)


class TranscriptionManager:
    """Orchestrates the recording lifecycle."""

    def __init__(self, app_state: AppState) -> None:
        self.app_state = app_state
        self.asr_client = ASRClient()
        self.audio_capture = AudioCapture()

        self.using_cached_params = False
        self.awaiting_final_result = False
        self._session_generation = 0
        self._active_generation: int | None = None
        self.start_timer_id: int | None = None
        self.safety_timer_id: int | None = None
        # Pending completion, rescheduled on each late result so only the last wins.
        self.quiet_timer_id: int | None = None

        # Callbacks set by app.py
        self.on_auth_expired = None  # () -> None
        self.on_show_login = None  # () -> None
        self.on_params_needed = None  # (callback: (ASRParams|None)->None) -> None
        self.on_overlay_show = None  # () -> None
        self.on_overlay_hide = None  # () -> None
        self.on_overlay_update = None  # (text: str) -> None
        self.on_paste = None  # (text: str) -> None
        self.on_cancel_enabled_changed = None  # (enabled: bool) -> None

    def _wire_asr_callbacks(
        self, client: ASRClient, generation: int
    ) -> None:
        """Wire one client's callbacks to its recording generation."""
        client.on_open = lambda: GLib.idle_add(
            self._on_asr_open, generation
        )
        client.on_result = lambda text: GLib.idle_add(
            self._on_asr_result, generation, text
        )
        client.on_finish = lambda: GLib.idle_add(
            self._on_asr_finish, generation
        )
        client.on_error = lambda err: GLib.idle_add(
            self._on_asr_error, generation, err
        )
        client.on_auth_error = lambda: GLib.idle_add(
            self._on_auth_error, generation
        )

    # --- Toggle ---

    def handle_toggle(self) -> None:
        """Called on GTK main thread from hotkey manager."""
        state = self.app_state.recording_state
        if state == RecordingState.IDLE:
            self._start_recording()
        elif state in (RecordingState.STARTING, RecordingState.RECORDING):
            self._stop_recording()
        # STOPPING: ignore

    def _start_recording(self) -> None:
        if self.app_state.login_status != LoginStatus.LOGGED_IN:
            logger.warning("Not logged in, showing login window")
            if self.on_show_login:
                self.on_show_login()
            return

        logger.info("Starting recording...")
        self._session_generation += 1
        generation = self._session_generation
        self._active_generation = generation

        # A client belongs to exactly one recording generation. Replacing it
        # prevents callbacks from an old socket from being rebound to a new run.
        self.asr_client.disconnect()
        self.asr_client = ASRClient()
        self._wire_asr_callbacks(self.asr_client, generation)

        self._set_state(RecordingState.STARTING)
        self.app_state.transcription_text = ""
        self.app_state.error_message = None
        if self.on_overlay_show:
            self.on_overlay_show()

        self.start_timer_id = GLib.timeout_add(
            int(START_SAFETY_TIMEOUT * 1000),
            self._startup_timeout,
            generation,
        )

        # Start audio immediately (buffered in ASR client until WS connects)
        try:
            self.audio_capture.start(on_audio_data=self.asr_client.send_audio)
        except Exception as e:
            logger.error("Audio capture failed: %s", e)
            self.app_state.error_message = "麦克风启动失败"
            GLib.timeout_add(
                int(AUTH_EXPIRY_DELAY * 1000),
                self._reset_to_idle,
                generation,
            )
            return

        # Try cached params first, fall back to WebView extraction
        cached = ParamsStore.load()
        if cached:
            logger.info("Using cached ASR params")
            self.using_cached_params = True
            self.asr_client.connect(cached)
        elif self.on_params_needed:
            self.using_cached_params = False
            self.on_params_needed(
                lambda params: self._on_params_extracted(generation, params)
            )
        else:
            self.app_state.error_message = "无法获取连接参数，请重新登录"
            GLib.timeout_add(
                int(AUTH_EXPIRY_DELAY * 1000),
                self._reset_to_idle,
                generation,
            )

    def _stop_recording(self) -> None:
        logger.info("Stopping recording...")
        was_starting = self.app_state.recording_state == RecordingState.STARTING
        self._cancel_start_timer()
        self._set_state(RecordingState.STOPPING)
        self.audio_capture.stop()
        # Flushes trailing silence so the server finalises the last word.
        self.asr_client.finish_sending()
        self.awaiting_final_result = True

        # A stop pressed during a slow handshake must leave enough time for the
        # queued speech and trailing silence to connect and flush. Normal
        # recordings retain the short final-result deadline; early stops keep
        # the startup allowance as a fallback until a result/finish callback
        # completes them. A recording that never produced a result (slow
        # connect, large backlog still being transcribed) gets a longer
        # allowance: the quiet-period timer still finishes it promptly once
        # results arrive, so this only extends the empty-transcript case.
        if was_starting:
            timeout = START_SAFETY_TIMEOUT
        elif not self.app_state.transcription_text:
            timeout = STOP_NO_RESULT_TIMEOUT
        else:
            timeout = STOP_SAFETY_TIMEOUT
        self._arm_safety_timeout(timeout, self._active_generation)

    def _arm_safety_timeout(
        self, timeout: float, generation: int | None
    ) -> None:
        self._cancel_safety_timer()
        self.safety_timer_id = GLib.timeout_add(
            int(timeout * 1000), self._safety_timeout, generation
        )

    def _schedule_final_completion(self, generation: int) -> None:
        """Wait for the result stream to go quiet before accepting the transcript.

        The server streams corrections right up to the end: after the audio stops
        it replays the pending partial results first, and only then sends the one
        carrying the final word. Completing on the first result to arrive after
        stopping truncates the tail -- the "last two characters are missing"
        symptom -- so each late result pushes the deadline back instead.
        """
        if not self._is_active_generation(generation):
            return
        self._cancel_quiet_timer()
        self.quiet_timer_id = GLib.timeout_add(
            int(FINAL_RESULT_QUIET_PERIOD * 1000),
            self._on_stream_quiet,
            generation,
        )

    def _on_stream_quiet(self, generation: int) -> bool:
        if not self._is_active_generation(generation):
            return GLib.SOURCE_REMOVE
        self.quiet_timer_id = None
        if (
            self.awaiting_final_result
            and self.app_state.recording_state == RecordingState.STOPPING
        ):
            logger.info("Result stream quiet, completing")
            self.awaiting_final_result = False
            self._complete_transcription()
        return GLib.SOURCE_REMOVE

    def _safety_timeout(self, generation: int | None) -> bool:
        if generation != self._active_generation:
            return GLib.SOURCE_REMOVE
        self.safety_timer_id = None
        if self.app_state.recording_state == RecordingState.STOPPING:
            logger.info("Safety timeout, completing with current text")
            self.awaiting_final_result = False
            self._complete_transcription()
        return GLib.SOURCE_REMOVE

    def _startup_timeout(self, generation: int) -> bool:
        if not self._is_active_generation(generation):
            return GLib.SOURCE_REMOVE
        self.start_timer_id = None
        if self.app_state.recording_state == RecordingState.STARTING:
            logger.warning("ASR startup timed out")
            self.app_state.error_message = "连接豆包超时"
            self.audio_capture.stop()
            self.asr_client.disconnect()
            GLib.timeout_add(
                int(AUTH_EXPIRY_DELAY * 1000),
                self._reset_to_idle,
                generation,
            )
        return GLib.SOURCE_REMOVE

    def _cancel_start_timer(self) -> None:
        if self.start_timer_id is not None:
            GLib.source_remove(self.start_timer_id)
            self.start_timer_id = None

    def _cancel_safety_timer(self) -> None:
        if self.safety_timer_id is not None:
            GLib.source_remove(self.safety_timer_id)
            self.safety_timer_id = None

    def _cancel_quiet_timer(self) -> None:
        if self.quiet_timer_id is not None:
            GLib.source_remove(self.quiet_timer_id)
            self.quiet_timer_id = None

    def _is_active_generation(self, generation: int) -> bool:
        return generation == self._active_generation

    # --- ASR callbacks (on GTK main thread via GLib.idle_add) ---

    def _on_asr_open(self, generation: int) -> bool:
        if not self._is_active_generation(generation):
            return GLib.SOURCE_REMOVE
        self._cancel_start_timer()
        if self.app_state.recording_state == RecordingState.STARTING:
            self._set_state(RecordingState.RECORDING)
        # If the user stopped during startup, keep the original 7-second
        # allowance. The socket can become publishable just before
        # finish_sending() schedules trailing silence, so on_open alone does not
        # prove that the final audio writes have completed. Results still finish
        # promptly via the quiet-period/finish callbacks; this timer is fallback.
        return GLib.SOURCE_REMOVE

    def _on_asr_result(self, generation: int, text: str) -> bool:
        if not self._is_active_generation(generation):
            return GLib.SOURCE_REMOVE
        self._cancel_start_timer()
        self.app_state.transcription_text = text
        if self.on_overlay_update:
            self.on_overlay_update(text)
        if self.app_state.recording_state == RecordingState.STARTING:
            self._set_state(RecordingState.RECORDING)
        if self.awaiting_final_result:
            self._schedule_final_completion(generation)
        return GLib.SOURCE_REMOVE

    def _on_asr_finish(self, generation: int) -> bool:
        if not self._is_active_generation(generation):
            return GLib.SOURCE_REMOVE
        self.awaiting_final_result = False
        if self.app_state.recording_state in (
            RecordingState.STOPPING,
            RecordingState.RECORDING,
        ):
            self._complete_transcription()
        return GLib.SOURCE_REMOVE

    def _on_asr_error(self, generation: int, error) -> bool:
        if not self._is_active_generation(generation):
            return GLib.SOURCE_REMOVE
        self._cancel_start_timer()
        if self.app_state.recording_state == RecordingState.IDLE:
            return GLib.SOURCE_REMOVE
        logger.error("ASR error: %s", error)
        # Only treat the error as an auth problem when the server actively
        # rejected the handshake (HTTP 401/403). Network failures (timeouts,
        # DNS, resets) used to be lumped in here, which cleared perfectly
        # valid cached credentials and forced a re-login after every
        # transient connectivity hiccup. Real in-band auth errors arrive
        # via _on_auth_error instead.
        if self.using_cached_params and _is_http_auth_rejection(error):
            self._handle_auth_failure()
            return GLib.SOURCE_REMOVE
        self.app_state.error_message = "连接出错"
        GLib.timeout_add(
            int(AUTH_EXPIRY_DELAY * 1000),
            self._reset_to_idle,
            generation,
        )
        return GLib.SOURCE_REMOVE

    def _on_auth_error(self, generation: int) -> bool:
        if not self._is_active_generation(generation):
            return GLib.SOURCE_REMOVE
        self._handle_auth_failure()
        return GLib.SOURCE_REMOVE

    # --- Completion & Reset ---

    def _cancel_stop_timers(self) -> None:
        """Drop any pending stop-phase timers so they cannot fire after reset."""
        for attr in ("quiet_timer_id", "safety_timer_id"):
            timer_id = getattr(self, attr)
            if timer_id is not None:
                GLib.source_remove(timer_id)
                setattr(self, attr, None)

    def _complete_transcription(self) -> None:
        text = self.app_state.transcription_text.strip()
        logger.info("Completing transcription: '%s'", text[:50])
        if text and self.on_paste:
            self.on_paste(text)
        self._reset_to_idle()

    def _reset_to_idle(self, generation: int | None = None) -> bool:
        if generation is not None and generation != self._active_generation:
            return GLib.SOURCE_REMOVE
        self._cancel_start_timer()
        self.awaiting_final_result = False
        self._cancel_stop_timers()
        self.audio_capture.stop()
        self.asr_client.disconnect()
        self._set_state(RecordingState.IDLE)
        self.app_state.error_message = None
        if self.on_overlay_hide:
            self.on_overlay_hide()
        self.using_cached_params = False
        finished_generation = self._active_generation
        self._active_generation = None
        # Clear text after short delay
        GLib.timeout_add(
            200, self._clear_text_after_reset, finished_generation
        )
        return GLib.SOURCE_REMOVE

    def _clear_text_after_reset(self, generation: int | None) -> bool:
        if (
            self._active_generation is None
            and generation == self._session_generation
        ):
            self.app_state.transcription_text = ""
        return GLib.SOURCE_REMOVE

    def handle_cancel(self) -> None:
        if self.app_state.recording_state == RecordingState.IDLE:
            return
        logger.info("Cancelling transcription")
        self.awaiting_final_result = False
        self.audio_capture.stop()
        self.asr_client.disconnect()
        self._reset_to_idle()

    def _handle_auth_failure(self) -> None:
        logger.warning("Auth failure, clearing cached params")
        ParamsStore.clear()
        self.using_cached_params = False
        self.audio_capture.stop()
        self.asr_client.disconnect()
        self._reset_to_idle()
        self.app_state.login_status = LoginStatus.NOT_LOGGED_IN
        if self.on_auth_expired:
            self.on_auth_expired()

    def _set_state(self, new_state: RecordingState) -> None:
        self.app_state.recording_state = new_state
        if self.on_cancel_enabled_changed:
            self.on_cancel_enabled_changed(new_state != RecordingState.IDLE)

    def _on_params_extracted(
        self, generation: int, params: ASRParams | None
    ) -> None:
        """Called when WebView param extraction completes."""
        state = self.app_state.recording_state
        can_connect = (
            state == RecordingState.STARTING
            and self.start_timer_id is not None
        ) or (
            state == RecordingState.STOPPING and self.awaiting_final_result
        )
        if not self._is_active_generation(generation) or not can_connect:
            return
        if params:
            if ParamsStore.save(params):
                self.asr_client.connect(params)
                return
        self.app_state.error_message = "无法获取连接参数，请重新登录"
        GLib.timeout_add(
            int(AUTH_EXPIRY_DELAY * 1000),
            self._reset_to_idle,
            generation,
        )
