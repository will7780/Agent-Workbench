# -*- coding: utf-8 -*-
"""LangGraph human_confirm 门控（Phase 3）：local_write/live 与高风险 capability 需人工确认。"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .execution_mode import ExecutionMode
from .permissions import PermissionLevel, parse_risk_level, requires_confirmation_for_level
from .registry import ModuleCapabilityRegistry
from .schemas import Observation

VALID_EXECUTION_CONFIRMATION_SOURCES = frozenset(
    {
        "cli_explicit",
        "ui_explicit",
        "eval_explicit",
        "human_confirm_token",
    }
)


def execution_confirmation_granted(
    user_confirmed: bool,
    confirmation_source: Optional[str],
    execution_mode: ExecutionMode,
) -> bool:
    """local_write/live 除 user_confirmed 外还必须携带显式确认来源。"""
    if execution_mode in (ExecutionMode.LOCAL_WRITE, ExecutionMode.LIVE):
        return bool(user_confirmed) and confirmation_source in VALID_EXECUTION_CONFIRMATION_SOURCES
    return bool(user_confirmed)


def human_confirm_required(
    *,
    execution_mode: ExecutionMode,
    user_confirmed: bool,
    confirmation_source: Optional[str] = None,
    module: Optional[str],
    action: Optional[str],
    registry: ModuleCapabilityRegistry,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    返回 (needs_confirm, error_type, error_message)。
    needs_confirm=True 表示必须人工确认但未确认，应阻断执行。
    """
    if execution_confirmation_granted(user_confirmed, confirmation_source, execution_mode):
        return False, None, None

    if execution_mode in (ExecutionMode.LOCAL_WRITE, ExecutionMode.LIVE):
        if confirmation_source and confirmation_source not in VALID_EXECUTION_CONFIRMATION_SOURCES:
            return True, "human_confirm_required", (
                f"execution_mode={execution_mode.value} requires a valid confirmation_source"
            )
        return True, "human_confirm_required", (
            f"execution_mode={execution_mode.value} requires user confirmation before tool execution"
        )

    if module and action:
        risk = registry.get_risk_level(module, action)
        level = risk if isinstance(risk, PermissionLevel) else parse_risk_level(str(risk))
        if requires_confirmation_for_level(level):
            return True, "human_confirm_required", (
                f"capability {module}.{action} risk={risk} requires user confirmation"
            )

    return False, None, None


def human_confirm_blocked_observation(step_id: str, confirm_dict: Dict[str, Any]) -> Observation:
    error_type = confirm_dict.get("error_type") or "human_confirm_required"
    suggested_next_action = "replan" if error_type == "human_confirmation_rejected" else "require_confirmation"

    return Observation(
        step_id=step_id,
        status="failed",
        summary=confirm_dict.get("error_message") or "human confirmation required",
        error={
            "type": error_type,
            "message": confirm_dict.get("error_message") or "",
        },
        suggested_next_action=suggested_next_action,
    )
