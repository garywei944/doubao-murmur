"""Regression tests for the recording lifecycle."""

from doubao_murmur.app_state import AppState, LoginStatus, RecordingState
from doubao_murmur.params_store import ASRParams
from doubao_murmur.transcription import (
    TranscriptionManager,
    _is_http_auth_rejection,
)


class FakeAudioCapture:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self, on_audio_data):
        self.started += 1

    def stop(self):
        self.stopped += 1


class FakeASRClient:
    instances = []

    def __init__(self):
        self.connected = 0
        self.disconnected = 0
        self.on_open = None
        self.on_result = None
        self.on_finish = None
        self.on_error = None
        self.on_auth_error = None
        self.__class__.instances.append(self)

    def connect(self, params):
        self.connected += 1

    def send_audio(self, data):
        pass

    def finish_sending(self):
        pass

    def disconnect(self):
        self.disconnected += 1


def make_manager(monkeypatch, cached_params):
    timers = []
    removed_sources = []
    FakeASRClient.instances = []

    def capture_timer(interval_ms, callback, *args):
        timers.append((interval_ms, callback, args))
        return len(timers)

    monkeypatch.setattr("doubao_murmur.transcription.ASRClient", FakeASRClient)
    monkeypatch.setattr("doubao_murmur.transcription.GLib.timeout_add", capture_timer)
    monkeypatch.setattr(
        "doubao_murmur.transcription.GLib.idle_add",
        lambda callback, *args: callback(*args),
    )
    monkeypatch.setattr(
        "doubao_murmur.transcription.GLib.source_remove",
        lambda source_id: removed_sources.append(source_id),
    )
    monkeypatch.setattr(
        "doubao_murmur.transcription.ParamsStore.load",
        lambda: cached_params,
    )
    monkeypatch.setattr("doubao_murmur.transcription.ParamsStore.save", lambda params: True)
    monkeypatch.setattr("doubao_murmur.transcription.ParamsStore.clear", lambda: None)

    state = AppState()
    state.login_status = LoginStatus.LOGGED_IN
    manager = TranscriptionManager(state)
    audio_capture = FakeAudioCapture()
    manager.audio_capture = audio_capture  # type: ignore[assignment]
    return manager, state, audio_capture, timers, removed_sources


def fire_timer(timers, index):
    _, callback, args = timers[index]
    callback(*args)


def test_http_200_handshake_response_is_auth_rejection():
    class Response:
        status_code = 200

    class InvalidStatus(Exception):
        response = Response()

    assert _is_http_auth_rejection(InvalidStatus())


def test_starting_timeout_reports_error_cleans_up_and_returns_idle(monkeypatch):
    params = ASRParams(cookies={}, device_id="d", web_id="w")
    manager, state, audio, timers, _ = make_manager(monkeypatch, params)

    manager.handle_toggle()
    client = manager.asr_client

    assert state.recording_state == RecordingState.STARTING
    assert audio.started == 1
    assert client.connected == 1

    fire_timer(timers, 0)

    assert state.error_message == "连接豆包超时"
    assert audio.stopped == 1
    assert client.disconnected == 1

    fire_timer(timers, 1)

    assert state.recording_state == RecordingState.IDLE
    assert state.error_message is None


def test_timed_out_connection_cannot_affect_next_recording(monkeypatch):
    params = ASRParams(cookies={}, device_id="d", web_id="w")
    manager, state, _, timers, _ = make_manager(monkeypatch, params)

    manager.handle_toggle()
    old_client = manager.asr_client
    fire_timer(timers, 0)
    old_reset = timers[1]
    fire_timer(timers, 1)
    old_clear_text = timers[2]

    manager.handle_toggle()
    new_client = manager.asr_client

    assert new_client is not old_client
    assert state.recording_state == RecordingState.STARTING

    assert old_client.on_open is not None
    assert old_client.on_result is not None
    assert old_client.on_finish is not None
    assert old_client.on_error is not None
    assert old_client.on_auth_error is not None
    old_client.on_open()
    old_client.on_result("迟到的旧识别结果")
    old_client.on_finish()
    old_client.on_error(RuntimeError("迟到的旧错误"))
    old_client.on_auth_error()

    assert state.recording_state == RecordingState.STARTING
    assert state.transcription_text == ""
    assert state.login_status == LoginStatus.LOGGED_IN

    assert new_client.on_open is not None
    assert new_client.on_result is not None
    new_client.on_open()
    assert state.recording_state == RecordingState.RECORDING
    new_client.on_result("新的识别结果")

    _, old_reset_callback, old_reset_args = old_reset
    old_reset_callback(*old_reset_args)
    _, old_clear_callback, old_clear_args = old_clear_text
    old_clear_callback(*old_clear_args)

    assert state.recording_state == RecordingState.RECORDING
    assert state.transcription_text == "新的识别结果"


def test_late_parameter_extraction_does_not_connect_after_timeout(monkeypatch):
    manager, state, _, timers, _ = make_manager(monkeypatch, None)
    callbacks = []
    manager.on_params_needed = lambda callback: callbacks.append(callback)

    manager.handle_toggle()
    client = manager.asr_client
    assert len(callbacks) == 1

    fire_timer(timers, 0)
    fire_timer(timers, 1)
    callbacks[0](ASRParams(cookies={}, device_id="d", web_id="w"))

    assert state.recording_state == RecordingState.IDLE
    assert client.connected == 0


def test_stop_waits_for_result_stream_to_go_quiet(monkeypatch):
    params = ASRParams(cookies={}, device_id="d", web_id="w")
    manager, state, _, timers, _ = make_manager(monkeypatch, params)
    pasted = []
    manager.on_paste = pasted.append

    manager.handle_toggle()
    client = manager.asr_client
    assert client.on_open is not None
    assert client.on_result is not None
    client.on_open()

    manager.handle_toggle()
    assert state.recording_state == RecordingState.STOPPING

    client.on_result("一二三四五六七八九十")
    assert pasted == []
    assert state.recording_state == RecordingState.STOPPING

    fire_timer(timers, 2)

    assert pasted == ["一二三四五六七八九十"]
    assert state.recording_state == RecordingState.IDLE


def test_stop_while_connecting_preserves_start_deadline_then_rearms(monkeypatch):
    params = ASRParams(cookies={}, device_id="d", web_id="w")
    manager, state, _, timers, removed = make_manager(monkeypatch, params)

    manager.handle_toggle()
    client = manager.asr_client
    manager.handle_toggle()

    assert state.recording_state == RecordingState.STOPPING
    assert timers[1][0] == 7000

    assert client.on_open is not None
    client.on_open()

    assert timers[2][0] == 1500
    assert 2 in removed


def test_stale_quiet_timer_cannot_complete_new_recording(monkeypatch):
    params = ASRParams(cookies={}, device_id="d", web_id="w")
    manager, state, _, timers, _ = make_manager(monkeypatch, params)
    pasted = []
    manager.on_paste = pasted.append

    manager.handle_toggle()
    old_client = manager.asr_client
    assert old_client.on_open is not None
    assert old_client.on_result is not None
    old_client.on_open()
    manager.handle_toggle()
    old_client.on_result("旧结果")
    old_quiet_timer = timers[2]

    manager.handle_cancel()
    manager.handle_toggle()
    new_client = manager.asr_client
    assert new_client.on_open is not None
    new_client.on_open()

    _, callback, args = old_quiet_timer
    callback(*args)

    assert pasted == []
    assert state.recording_state == RecordingState.RECORDING
