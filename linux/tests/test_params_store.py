"""Tests for ParamsStore."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Patch config before importing params_store
import doubao_murmur.config as config


@pytest.fixture(autouse=True)
def tmp_config_dir(tmp_path, monkeypatch):
    """Redirect config dir to a temp directory for each test."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


def test_save_and_load():
    from doubao_murmur.params_store import ASRParams, ParamsStore

    params = ASRParams(
        cookies={"sessionid": "abc123", "sid_tt": "xyz789"},
        device_id="dev_001",
        web_id="web_002",
    )
    ParamsStore.save(params)

    loaded = ParamsStore.load()
    assert loaded is not None
    assert loaded.cookies == params.cookies
    assert loaded.device_id == "dev_001"
    assert loaded.web_id == "web_002"


def test_cookie_header():
    from doubao_murmur.params_store import ASRParams

    params = ASRParams(
        cookies={"a": "1", "b": "2"},
        device_id="d",
        web_id="w",
    )
    header = params.cookie_header
    assert "a=1" in header
    assert "b=2" in header
    assert "; " in header


def test_load_nonexistent():
    from doubao_murmur.params_store import ParamsStore

    assert ParamsStore.load() is None


def test_clear():
    from doubao_murmur.params_store import ASRParams, ParamsStore

    params = ASRParams(cookies={"a": "1"}, device_id="d", web_id="w")
    ParamsStore.save(params)
    assert ParamsStore.has_saved()

    ParamsStore.clear()
    assert not ParamsStore.has_saved()
    assert ParamsStore.load() is None


def test_has_saved():
    from doubao_murmur.params_store import ParamsStore

    assert not ParamsStore.has_saved()

    from doubao_murmur.params_store import ASRParams

    ParamsStore.save(ASRParams(cookies={"a": "1"}, device_id="d", web_id="w"))
    assert ParamsStore.has_saved()


def test_save_rejects_empty_cookies():
    from doubao_murmur.params_store import ASRParams, ParamsStore

    params = ASRParams(cookies={}, device_id="device", web_id="web")

    assert ParamsStore.save(params) is False
    assert not config.get_params_path().exists()


def test_load_and_has_saved_reject_invalid_cached_params():
    from doubao_murmur.params_store import ParamsStore

    path = config.get_params_path()
    path.write_text(
        '{"cookies": {}, "device_id": "device", "web_id": "web"}',
        encoding="utf-8",
    )

    assert ParamsStore.load() is None
    assert ParamsStore.has_saved() is False


def test_save_uses_owner_only_permissions():
    from doubao_murmur.params_store import ASRParams, ParamsStore

    params = ASRParams(
        cookies={"sessionid": "secret"},
        device_id="device",
        web_id="web",
    )

    assert ParamsStore.save(params) is True
    assert config.get_params_path().stat().st_mode & 0o777 == 0o600
