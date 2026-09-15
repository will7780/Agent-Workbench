# -*- coding: utf-8 -*-
"""统一执行模式定义与解析。"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union


class ExecutionMode(str, Enum):
    """Agent 执行模式。"""

    DRY_RUN = "dry_run"
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    LIVE = "live"

    @classmethod
    def from_cli(cls, value: str) -> "ExecutionMode":
        mapping = {
            "dry-run": cls.DRY_RUN,
            "dry_run": cls.DRY_RUN,
            "read-only": cls.READ_ONLY,
            "read_only": cls.READ_ONLY,
            "local-write": cls.LOCAL_WRITE,
            "local_write": cls.LOCAL_WRITE,
            "live": cls.LIVE,
        }
        key = (value or "").strip().lower()
        if key not in mapping:
            raise ValueError(f"unsupported execution mode: {value}")
        return mapping[key]

    @classmethod
    def coerce(cls, value: Union[str, "ExecutionMode", None]) -> "ExecutionMode":
        if value is None:
            return cls.DRY_RUN
        if isinstance(value, ExecutionMode):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                return cls.from_cli(value)
        raise TypeError(f"invalid execution mode type: {type(value)}")


def resolve_pipeline_execution_mode(
    *,
    execution_mode: Union[str, ExecutionMode, None] = None,
    dry_run: bool = True,
) -> ExecutionMode:
    """
    解析 pipeline 执行模式。

    - 显式 execution_mode 优先
    - 否则 dry_run=True -> dry_run
    - dry_run=False 且未指定 mode 时，兼容旧逻辑视为 read_only
    """
    if execution_mode is not None:
        return ExecutionMode.coerce(execution_mode)
    if dry_run:
        return ExecutionMode.DRY_RUN
    return ExecutionMode.READ_ONLY
