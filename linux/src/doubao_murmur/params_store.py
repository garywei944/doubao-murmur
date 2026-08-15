"""Persist ASR params (cookies, device_id, web_id) to a JSON file.

Mirrors ASRParamsStore.swift.
Location: $XDG_CONFIG_HOME/doubao-murmur/asr_params.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

from doubao_murmur.config import get_params_path

logger = logging.getLogger(__name__)


@dataclass
class ASRParams:
    """Parameters needed to establish a WSS ASR connection."""

    cookies: dict[str, str]
    device_id: str
    web_id: str

    @property
    def cookie_header(self) -> str:
        """Build the Cookie header string for HTTP/WSS requests."""
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    @property
    def is_valid(self) -> bool:
        """Whether the cached values are sufficient for an ASR handshake."""
        return bool(
            any(name and value for name, value in self.cookies.items())
            and self.device_id.strip()
            and self.web_id.strip()
        )


class ParamsStore:
    """Persist ASR params to JSON file."""

    @staticmethod
    def save(params: ASRParams) -> bool:
        if not params.is_valid:
            logger.warning("Refusing to save incomplete ASR params")
            return False
        path = get_params_path()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            data = json.dumps(asdict(params), ensure_ascii=False, indent=2)
            fd = os.open(
                tmp_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                0o600,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(data)
            os.replace(tmp_path, path)
            path.chmod(0o600)
            logger.info("Saved ASR params to %s", path)
            return True
        except Exception as e:
            logger.error("Failed to save params: %s", e)
            tmp_path.unlink(missing_ok=True)
            return False

    @staticmethod
    def load() -> ASRParams | None:
        path = get_params_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            params = ASRParams(**data)
            if not params.is_valid:
                logger.warning("Ignoring incomplete cached ASR params")
                return None
            return params
        except Exception as e:
            logger.error("Failed to load params: %s", e)
            return None

    @staticmethod
    def clear() -> None:
        try:
            get_params_path().unlink(missing_ok=True)
            logger.info("Cleared saved params")
        except Exception as e:
            logger.error("Failed to clear params: %s", e)

    @staticmethod
    def has_saved() -> bool:
        return ParamsStore.load() is not None
