# -*- coding: utf-8 -*-
"""Load approved API settings from the user's central local environment file."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

DEFAULT_CENTRAL_ENV_PATH = Path.home() / "Desktop" / "api" / ".env"
MAX_CENTRAL_ENV_BYTES = 64 * 1024

ALLOWED_CENTRAL_ENV_NAMES = frozenset(
    {
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_CHAT_COMPLETIONS_PATH",
        "AGENT_WORKBENCH_LLM_MODEL",
        "AGENT_WORKBENCH_LLM_PROVIDER",
        "LAOZHANG_API_KEY",
        "LAOZHANG_BASE_URL",
        "LAOZHANG_MODEL",
    }
)

_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def resolve_central_env_path(path: Optional[Union[Path, str]] = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    configured = os.environ.get("AGENT_API_ENV_FILE")
    return Path(configured).expanduser() if configured else DEFAULT_CENTRAL_ENV_PATH


def _unquote_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_central_agent_env(
    path: Optional[Union[Path, str]] = None,
    *,
    environ: Optional[MutableMapping[str, str]] = None,
    override: bool = False,
) -> Dict[str, Any]:
    """Load allowlisted values without ever returning or logging their contents."""
    target_env = environ if environ is not None else os.environ
    disabled = (
        str(target_env.get("AGENT_WORKBENCH_DISABLE_CENTRAL_ENV") or "").strip().lower()
        in _TRUE_VALUES
    )
    if disabled:
        return {
            "enabled": False,
            "loaded_names": [],
            "skipped_names": [],
            "error_type": "central_env_disabled",
        }

    env_path = resolve_central_env_path(path)
    try:
        if not env_path.is_file():
            return {
                "enabled": True,
                "loaded_names": [],
                "skipped_names": [],
                "error_type": "central_env_missing",
            }
        if env_path.stat().st_size > MAX_CENTRAL_ENV_BYTES:
            return {
                "enabled": True,
                "loaded_names": [],
                "skipped_names": [],
                "error_type": "central_env_too_large",
            }
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return {
            "enabled": True,
            "loaded_names": [],
            "skipped_names": [],
            "error_type": "central_env_unavailable",
        }

    loaded = []
    skipped = []
    invalid_line_seen = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not _ENV_NAME_RE.fullmatch(name):
            invalid_line_seen = True
            continue
        if name not in ALLOWED_CENTRAL_ENV_NAMES:
            skipped.append(name)
            continue
        value = _unquote_env_value(raw_value)
        if not value:
            skipped.append(name)
            continue
        if not override and name in target_env:
            skipped.append(name)
            continue
        target_env[name] = value
        loaded.append(name)

    return {
        "enabled": True,
        "loaded_names": sorted(set(loaded)),
        "skipped_names": sorted(set(skipped)),
        "error_type": "central_env_invalid_line" if invalid_line_seen else None,
    }
