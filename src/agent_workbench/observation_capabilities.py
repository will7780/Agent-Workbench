# -*- coding: utf-8 -*-
"""观察 / 检查类 capability 目录（Phase 2）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .registry import ModuleCapabilityRegistry, get_registry
from .tool_schema import capability_id, build_openai_tool_schema

# 与 registry capability_kind 对齐的已知观察类能力（兜底）
_KNOWN_OBSERVATION_IDS = frozenset(
    {
        "product_comparison.compare_prices",
        "product_price_audit.inspect_price_audit_inputs",
        "product_upload.inspect_upload_inputs",
    }
)


def is_observation_capability(
    module: str,
    action: str,
    registry: Optional[ModuleCapabilityRegistry] = None,
) -> bool:
    reg = registry or get_registry()
    cid = capability_id(module, action)
    if cid in _KNOWN_OBSERVATION_IDS:
        return True
    action_def = reg.get_action(module, action) or {}
    return str(action_def.get("capability_kind") or "").lower() == "observation"


def list_observation_capabilities(
    registry: Optional[ModuleCapabilityRegistry] = None,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """返回 (module, action, action_def) 列表，仅 dry_run 可用观察类能力。"""
    reg = registry or get_registry()
    items: List[Tuple[str, str, Dict[str, Any]]] = []
    for module, action, action_def in reg.iter_actions():
        if not is_observation_capability(module, action, reg):
            continue
        if reg.get_execution_mode_status(module, action, "dry_run") != "available":
            continue
        items.append((module, action, action_def))
    return items


def list_observation_capability_ids(
    registry: Optional[ModuleCapabilityRegistry] = None,
) -> List[str]:
    return [capability_id(m, a) for m, a, _ in list_observation_capabilities(registry)]


def build_observation_tool_schemas(
    registry: Optional[ModuleCapabilityRegistry] = None,
) -> List[Dict[str, Any]]:
    reg = registry or get_registry()
    schemas: List[Dict[str, Any]] = []
    for module, action, action_def in list_observation_capabilities(reg):
        mod_def = reg.get_module(module) or {}
        display = str(mod_def.get("display_name") or module)
        schemas.append(build_openai_tool_schema(module, action, action_def, display))
    return schemas


def observation_system_hint(registry: Optional[ModuleCapabilityRegistry] = None) -> str:
    ids = list_observation_capability_ids(registry)
    if not ids:
        return ""
    joined = ", ".join(ids)
    return (
        "Observation tools (inspect state before side-effect actions): "
        f"{joined}. Prefer these when the user asks to check, inspect, or compare before acting."
    )
