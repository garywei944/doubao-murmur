"""Tests for ASR client URL building and message parsing."""

import json
import builtins
import asyncio
import gc
import threading

import pytest

from doubao_murmur.asr_client import (
    ASRClient,
    _load_websockets,
    _is_connection_closed_error,
    _websocket_header_kwargs,
)
from doubao_murmur.params_store import ASRParams


@pytest.fixture
def client():
    return ASRClient()


@pytest.fixture
def sample_params():
    return ASRParams(
        cookies={"sessionid": "abc", "sid_tt": "xyz"},
        device_id="device_12345",
        web_id="web_67890",
    )


class TestURLBuilding:
    def test_url_contains_base(self, client, sample_params):
        url = client._build_url(sample_params)
        assert url.startswith("wss://ws-samantha.doubao.com/samantha/audio/asr?")

    def test_url_contains_device_id(self, client, sample_params):
        url = client._build_url(sample_params)
        assert "device_id=device_12345" in url

    def test_url_contains_web_id(self, client, sample_params):
        url = client._build_url(sample_params)
        assert "web_id=web_67890" in url

    def test_url_contains_tea_uuid(self, client, sample_params):
        url = client._build_url(sample_params)
        assert "tea_uuid=web_67890" in url

    def test_url_contains_fixed_params(self, client, sample_params):
        url = client._build_url(sample_params)
        assert "version_code=20800" in url
        assert "language=zh" in url
        assert "aid=497858" in url
        assert "format=pcm" in url

    def test_url_contains_unique_tab_id(self, client, sample_params):
        url1 = client._build_url(sample_params)
        url2 = client._build_url(sample_params)
        # web_tab_id should be different each time (UUID)
        assert url1 != url2


class TestMessageParsing:
    def test_result_event(self, client):
        received = []
        client.on_result = lambda text: received.append(text)

        msg = json.dumps({
            "code": 0,
            "event": "result",
            "result": {"Text": "你好世界"},
        })
        client._handle_message(msg)
        assert received == ["你好世界"]

    def test_finish_event(self, client):
        finished = []
        client.on_finish = lambda: finished.append(True)

        msg = json.dumps({"code": 0, "event": "finish"})
        client._handle_message(msg)
        assert finished == [True]

    def test_auth_error_code(self, client):
        auth_errors = []
        client.on_auth_error = lambda: auth_errors.append(True)

        msg = json.dumps({
            "code": 709599054,
            "event": "",
            "message": "Invalid",
        })
        client._handle_message(msg)
        assert auth_errors == [True]

    def test_auth_error_keyword_cookie(self, client):
        auth_errors = []
        client.on_auth_error = lambda: auth_errors.append(True)

        msg = json.dumps({
            "code": 12345,
            "event": "",
            "message": "Cookie expired",
        })
        client._handle_message(msg)
        assert auth_errors == [True]

    def test_auth_error_keyword_session(self, client):
        auth_errors = []
        client.on_auth_error = lambda: auth_errors.append(True)

        msg = json.dumps({
            "code": 99,
            "event": "",
            "message": "session unauthorized",
        })
        client._handle_message(msg)
        assert auth_errors == [True]

    def test_non_auth_error_ignored(self, client):
        errors = []
        client.on_error = lambda e: errors.append(e)

        msg = json.dumps({
            "code": 1,
            "event": "",
            "message": "some other error",
        })
        client._handle_message(msg)
        assert errors == []  # non-auth errors with code != 0 are silently ignored

    def test_empty_text_ignored(self, client):
        received = []
        client.on_result = lambda text: received.append(text)

        msg = json.dumps({
            "code": 0,
            "event": "result",
            "result": {"Text": ""},
        })
        client._handle_message(msg)
        assert received == []

    def test_binary_message(self, client):
        received = []
        client.on_result = lambda text: received.append(text)

        msg = json.dumps({
            "code": 0,
            "event": "result",
            "result": {"Text": "test"},
        }).encode("utf-8")
        client._handle_message(msg)
        assert received == ["test"]

    def test_invalid_json_ignored(self, client):
        received = []
        client.on_result = lambda text: received.append(text)

        client._handle_message("not json at all")
        assert received == []


class TestCookieHeader:
    def test_cookie_header_format(self):
        params = ASRParams(
            cookies={"sessionid": "abc", "sid_tt": "xyz"},
            device_id="d",
            web_id="w",
        )
        header = params.cookie_header
        # Should contain both cookies separated by "; "
        assert "sessionid=abc" in header
        assert "sid_tt=xyz" in header


class TestDependencyHandling:
    def test_header_kwargs_new_websockets(self):
        def connect(*, additional_headers=None):
            return additional_headers

        assert _websocket_header_kwargs(connect, {"Cookie": "x"}) == {
            "additional_headers": {"Cookie": "x"}
        }

    def test_header_kwargs_old_websockets(self):
        def connect(*, extra_headers=None):
            return extra_headers

        assert _websocket_header_kwargs(connect, {"Cookie": "x"}) == {
            "extra_headers": {"Cookie": "x"}
        }

    def test_load_websockets_error_is_actionable(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "websockets":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(RuntimeError, match="websockets"):
            _load_websockets()

    def test_connection_closed_error_detection(self):
        class ConnectionClosedOK(Exception):
            pass

        assert _is_connection_closed_error(ConnectionClosedOK())
        assert not _is_connection_closed_error(RuntimeError())


class TestConnectionLifecycle:
    def test_connect_disables_periodic_protocol_keepalive(self, monkeypatch):
        connect_kwargs = {}

        class FakeWebSocket:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def close(self):
                return None

        class FakeWebsockets:
            @staticmethod
            async def connect(url, **kwargs):
                connect_kwargs.update(kwargs)
                return FakeWebSocket()

        monkeypatch.setattr(
            "doubao_murmur.asr_client._load_websockets",
            lambda: FakeWebsockets,
        )
        client = ASRClient()
        client.connect(ASRParams(cookies={}, device_id="d", web_id="w"))

        assert client._thread is not None
        client._thread.join(timeout=2)

        assert not client._thread.is_alive()
        assert connect_kwargs["ping_interval"] is None

    def test_connection_closed_while_flushing_buffer_reports_error(
        self, monkeypatch
    ):
        errors = []

        class ConnectionClosedDuringFlush(Exception):
            pass

        class FakeWebSocket:
            async def send(self, data):
                raise ConnectionClosedDuringFlush("closed while flushing")

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def close(self):
                return None

        class FakeWebsockets:
            @staticmethod
            async def connect(url, **kwargs):
                return FakeWebSocket()

        monkeypatch.setattr(
            "doubao_murmur.asr_client._load_websockets",
            lambda: FakeWebsockets,
        )
        client = ASRClient()
        client.on_error = errors.append
        client.send_audio(b"buffered-before-open")
        client.connect(ASRParams(cookies={}, device_id="d", web_id="w"))

        assert client._thread is not None
        client._thread.join(timeout=2)

        assert not client._thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], ConnectionClosedDuringFlush)

    def test_live_audio_cannot_overtake_buffered_audio_during_open(
        self, monkeypatch
    ):
        buffered_send_started = threading.Event()
        allow_buffered_send = threading.Event()
        all_audio_sent = threading.Event()
        sent = []
        buffered = b"buffered-before-open"
        live = b"live-during-open"

        class FakeWebSocket:
            async def send(self, data):
                if data == buffered:
                    buffered_send_started.set()
                    while not allow_buffered_send.is_set():
                        await asyncio.sleep(0.001)
                sent.append(data)
                if len(sent) == 2:
                    all_audio_sent.set()

            def __aiter__(self):
                return self

            async def __anext__(self):
                while not all_audio_sent.is_set():
                    await asyncio.sleep(0.001)
                raise StopAsyncIteration

            async def close(self):
                return None

        class FakeWebsockets:
            @staticmethod
            async def connect(url, **kwargs):
                return FakeWebSocket()

        monkeypatch.setattr(
            "doubao_murmur.asr_client._load_websockets",
            lambda: FakeWebsockets,
        )
        client = ASRClient()
        client.send_audio(buffered)
        client.connect(ASRParams(cookies={}, device_id="d", web_id="w"))

        try:
            assert buffered_send_started.wait(timeout=1)
            client.send_audio(live)
            assert client._loop is not None

            async def event_loop_barrier():
                await asyncio.sleep(0)

            asyncio.run_coroutine_threadsafe(
                event_loop_barrier(), client._loop
            ).result(timeout=1)
            allow_buffered_send.set()

            assert client._thread is not None
            client._thread.join(timeout=2)
            assert not client._thread.is_alive()
            assert sent == [buffered, live]
        finally:
            allow_buffered_send.set()
            if client._thread is not None and client._thread.is_alive():
                client.disconnect()
                client._thread.join(timeout=1)

    def test_early_stop_open_waits_until_buffered_audio_is_sent(
        self, monkeypatch
    ):
        send_started = threading.Event()
        allow_send = threading.Event()
        send_finished = threading.Event()
        events = []
        speech = b"speech-buffered-before-early-stop"

        class FakeWebSocket:
            async def send(self, data):
                send_started.set()
                while not allow_send.is_set():
                    await asyncio.sleep(0.001)
                events.append(("sent", data))
                send_finished.set()

            def __aiter__(self):
                return self

            async def __anext__(self):
                while not send_finished.is_set():
                    await asyncio.sleep(0.001)
                raise StopAsyncIteration

            async def close(self):
                return None

        class FakeWebsockets:
            @staticmethod
            async def connect(url, **kwargs):
                return FakeWebSocket()

        monkeypatch.setattr(
            "doubao_murmur.asr_client._load_websockets",
            lambda: FakeWebsockets,
        )
        client = ASRClient()
        client.on_open = lambda: events.append(("open", None))
        client.send_audio(speech)
        client.finish_sending(trailing_silence_ms=0)
        client.connect(ASRParams(cookies={}, device_id="d", web_id="w"))

        try:
            assert send_started.wait(timeout=1)
            allow_send.set()
            assert client._thread is not None
            client._thread.join(timeout=2)

            assert not client._thread.is_alive()
            assert events == [("sent", speech), ("open", None)]
        finally:
            allow_send.set()
            if client._thread is not None and client._thread.is_alive():
                client.disconnect()
                client._thread.join(timeout=1)

    def test_slow_connect_flushes_early_stop_audio_in_order(self, monkeypatch):
        connect_started = threading.Event()
        allow_connect = threading.Event()
        sent = []
        results = []
        finishes = []

        class FakeWebSocket:
            def __init__(self):
                self.message_index = 0

            async def send(self, data):
                sent.append(data)

            def __aiter__(self):
                return self

            async def __anext__(self):
                for _ in range(100):
                    if len(sent) >= 3:
                        break
                    await asyncio.sleep(0.001)
                if self.message_index == 0:
                    self.message_index += 1
                    return json.dumps(
                        {"code": 0, "event": "result", "result": {"Text": "完整"}}
                    )
                if self.message_index == 1:
                    self.message_index += 1
                    return json.dumps({"code": 0, "event": "finish"})
                raise StopAsyncIteration

            async def close(self):
                return None

        class FakeWebsockets:
            @staticmethod
            async def connect(url, **kwargs):
                connect_started.set()
                while not allow_connect.is_set():
                    await asyncio.sleep(0.001)
                return FakeWebSocket()

        monkeypatch.setattr(
            "doubao_murmur.asr_client._load_websockets",
            lambda: FakeWebsockets,
        )
        client = ASRClient()
        client.on_result = results.append
        client.on_finish = lambda: finishes.append(True)
        speech = b"speech-buffered-during-handshake"

        client.connect(ASRParams(cookies={}, device_id="d", web_id="w"))
        assert connect_started.wait(timeout=1)
        client.send_audio(speech)
        client.finish_sending(trailing_silence_ms=200)
        allow_connect.set()

        assert client._thread is not None
        client._thread.join(timeout=2)

        assert not client._thread.is_alive()
        assert sent == [speech, bytes(3200), bytes(3200)]
        assert results == ["完整"]
        assert finishes == [True]

    def test_finish_before_connect_preserves_audio_and_queues_silence(
        self, monkeypatch
    ):
        sent = []

        class FakeWebSocket:
            async def send(self, data):
                sent.append(data)
        client = ASRClient()
        speech = b"captured-before-websocket-opened"
        client.send_audio(speech)

        client.finish_sending(trailing_silence_ms=200)

        assert client._pending_audio == [speech, bytes(3200), bytes(3200)]

        client._ws = FakeWebSocket()
        client._connected = True
        asyncio.run(client._flush_audio_buffer(client._ws))

        assert sent == [speech, bytes(3200), bytes(3200)]
        assert client._pending_audio == []
        assert client.is_connected is False

    def test_finish_sending_flushes_trailing_silence(self, monkeypatch):
        sent = []

        class FakeWebSocket:
            async def send(self, data):
                sent.append(data)

        class FakeLoop:
            @staticmethod
            def is_running():
                return True

        def run_now(coro, loop):
            asyncio.run(coro)

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", run_now)
        client = ASRClient()
        client._ws = FakeWebSocket()
        client._loop = FakeLoop()  # type: ignore[assignment]
        client._connected = True

        client.finish_sending(trailing_silence_ms=200)

        assert client.is_connected is False
        assert sent == [bytes(3200), bytes(3200)]

    def test_connect_releases_event_loop_and_executor(self, monkeypatch):
        async def resolve_once(self, params):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: None)

        monkeypatch.setattr(ASRClient, "_connect_and_listen", resolve_once)
        client = ASRClient()

        client.connect(ASRParams(cookies={}, device_id="d", web_id="w"))
        assert client._thread is not None
        assert client._loop is not None
        client._thread.join(timeout=2)

        assert not client._thread.is_alive()
        assert client._loop.is_closed()

    def test_disconnect_cancels_connection_before_websocket_opens(
        self, monkeypatch
    ):
        started = threading.Event()

        async def blocked_connect(self, params):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(ASRClient, "_connect_and_listen", blocked_connect)
        client = ASRClient()
        client.connect(ASRParams(cookies={}, device_id="d", web_id="w"))
        assert started.wait(timeout=1)
        assert client._thread is not None
        assert client._loop is not None

        try:
            client.disconnect()
            client._thread.join(timeout=1)
            assert not client._thread.is_alive()
            assert client._loop.is_closed()
        finally:
            if client._thread.is_alive():
                client._loop.call_soon_threadsafe(
                    lambda: [task.cancel() for task in asyncio.all_tasks()]
                )
                client._thread.join(timeout=1)

    def test_disconnect_drains_pending_send_tasks_before_loop_close(
        self, monkeypatch
    ):
        opened = threading.Event()
        send_started = threading.Event()
        loop_errors = []

        class FakeWebSocket:
            async def send(self, data):
                send_started.set()
                await asyncio.Event().wait()

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()

            async def close(self):
                return None

        class FakeWebsockets:
            @staticmethod
            async def connect(url, **kwargs):
                loop = asyncio.get_running_loop()
                loop.set_exception_handler(
                    lambda _loop, context: loop_errors.append(context)
                )
                return FakeWebSocket()

        monkeypatch.setattr(
            "doubao_murmur.asr_client._load_websockets",
            lambda: FakeWebsockets,
        )
        client = ASRClient()
        client.on_open = lambda: opened.set()  # type: ignore[assignment]
        client.connect(ASRParams(cookies={}, device_id="d", web_id="w"))

        assert opened.wait(timeout=1)
        client.send_audio(b"in-flight-live-audio")
        assert send_started.wait(timeout=1)
        assert client._thread is not None

        client.disconnect()
        client._thread.join(timeout=2)
        gc.collect()

        assert not client._thread.is_alive()
        assert client._loop is not None
        assert client._loop.is_closed()
        assert [
            context
            for context in loop_errors
            if context.get("message") == "Task was destroyed but it is pending!"
        ] == []
