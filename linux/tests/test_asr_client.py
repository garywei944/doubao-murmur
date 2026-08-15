"""Tests for ASR client URL building and message parsing."""

import json
import builtins
import asyncio
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
