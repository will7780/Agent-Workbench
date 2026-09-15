# -*- coding: utf-8 -*-
"""工具与步骤权限等级（L0-L5）。"""

from __future__ import annotations

from enum import IntEnum
from typing import Set


class PermissionLevel(IntEnum):
    L0_REASONING = 0
    L1_LOCAL_READ = 1
    L2_LOCAL_DRAFT_WRITE = 2
    L3_EXTERNAL_READ = 3
    L4_SIDE_EFFECT = 4
    L5_HIGH_RISK_SIDE_EFFECT = 5


PLAN_AGENT_MAX_LEVEL = PermissionLevel.L1_LOCAL_READ

# PlanAgent 可见工具组
PLAN_AGENT_TOOL_GROUPS: Set[str] = {"planning_tools", "context_tools"}

# 执行阶段可见工具组
EXECUTOR_TOOL_GROUPS: Set[str] = {
    "validation_tools",
    "execution_adapters",
    "observation_tools",
    "report_tools",
}


def parse_risk_level(level: str) -> PermissionLevel:
    """将 risk_level 字符串解析为 PermissionLevel。"""
    normalized = (level or "low").strip().upper()
    if normalized.startswith("L") and len(normalized) == 2 and normalized[1].isdigit():
        return PermissionLevel(int(normalized[1]))
    mapping = {
        "LOW": PermissionLevel.L1_LOCAL_READ,
        "MEDIUM": PermissionLevel.L2_LOCAL_DRAFT_WRITE,
        "HIGH": PermissionLevel.L4_SIDE_EFFECT,
        "CRITICAL": PermissionLevel.L5_HIGH_RISK_SIDE_EFFECT,
    }
    return mapping.get(normalized, PermissionLevel.L2_LOCAL_DRAFT_WRITE)


def requires_confirmation_for_level(level: PermissionLevel) -> bool:
    return level >= PermissionLevel.L4_SIDE_EFFECT


def requires_high_risk_flag(level: PermissionLevel) -> bool:
    return level >= PermissionLevel.L5_HIGH_RISK_SIDE_EFFECT


def permission_to_step_risk(level: PermissionLevel) -> str:
    return f"L{int(level)}"


def step_risk_to_permission(step_risk: str) -> PermissionLevel:
    return parse_risk_level(step_risk)


READ_ONLY_MAX_LEVEL = PermissionLevel.L3_EXTERNAL_READ
LOCAL_WRITE_MIN_LEVEL = PermissionLevel.L2_LOCAL_DRAFT_WRITE
LOCAL_WRITE_MAX_LEVEL = PermissionLevel.L3_EXTERNAL_READ


def allows_read_only_execution(level: PermissionLevel) -> bool:
    return level <= READ_ONLY_MAX_LEVEL


def allows_local_write_execution(level: PermissionLevel) -> bool:
    return LOCAL_WRITE_MIN_LEVEL <= level <= LOCAL_WRITE_MAX_LEVEL


def allows_live_execution(level: PermissionLevel) -> bool:
    return level >= PermissionLevel.L4_SIDE_EFFECT


def plan_agent_may_use_tool(tool_group: str, tool_level: PermissionLevel) -> bool:
    if tool_group not in PLAN_AGENT_TOOL_GROUPS:
        return False
    return tool_level <= PLAN_AGENT_MAX_LEVEL
