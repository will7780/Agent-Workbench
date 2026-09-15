# -*- coding: utf-8 -*-
"""Tool Guard — 校验 LLM tool_call 是否允许执行。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .execution_mode import ExecutionMode
from .registry import ModuleCapabilityRegistry, get_registry
from .tool_schema import tool_name_to_capability
from .tools.validation_tools import check_required_params


@dataclass
class ToolGuardResult:
    allowed: bool
    module: Optional[str] = None
    action: Optional[str] = None
    tool_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    missing_params: List[Dict[str, str]] = field(default_factory=list)
    requires_human_confirm: bool = False
    requested_execution_mode: Optional[str] = None
    mode_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "module": self.module,
            "action": self.action,
            "tool_name": self.tool_name,
            "params": self.params,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "missing_params": self.missing_params,
            "requires_human_confirm": self.requires_human_confirm,
            "requested_execution_mode": self.requested_execution_mode,
            "mode_status": self.mode_status,
        }


def _mode_registry_key(execution_mode: ExecutionMode) -> str:
    return execution_mode.value


def _validate_and_normalize_params(
    module: str,
    action: str,
    params: Dict[str, Any],
    registry: ModuleCapabilityRegistry,
) -> Optional[str]:
    """Run only the semantic rules explicitly installed by the host."""
    validator = getattr(registry, "parameter_validator", None)
    if validator is not None:
        try:
            return validator(module, action, params)
        except Exception:
            return "parameter_policy_unavailable"
    return None


def guard_tool_call(
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    *,
    registry: Optional[ModuleCapabilityRegistry] = None,
    execution_mode: ExecutionMode = ExecutionMode.DRY_RUN,
    module_config: Optional[Dict[str, Any]] = None,
) -> ToolGuardResult:
    reg = registry or get_registry()
    mode_key = _mode_registry_key(execution_mode)
    requires_confirm = execution_mode in (ExecutionMode.LOCAL_WRITE, ExecutionMode.LIVE)

    parsed = tool_name_to_capability(tool_name)
    if parsed is None:
        return ToolGuardResult(
            allowed=False,
            tool_name=tool_name,
            error_type="unknown_tool",
            error_message=f"tool name {tool_name} does not map to registry capability",
            requires_human_confirm=requires_confirm,
            requested_execution_mode=mode_key,
        )

    module, action = parsed
    ok, err = reg.validate_module_action(module, action)
    if not ok:
        return ToolGuardResult(
            allowed=False,
            module=module,
            action=action,
            tool_name=tool_name,
            error_type="unknown_capability",
            error_message=err or f"unknown capability {module}.{action}",
            requires_human_confirm=requires_confirm,
            requested_execution_mode=mode_key,
        )

    mode_status = reg.get_execution_mode_status(module, action, mode_key)
    if mode_status != "available":
        return ToolGuardResult(
            allowed=False,
            module=module,
            action=action,
            tool_name=tool_name,
            error_type="execution_mode_blocked",
            error_message=f"{mode_key}_status={mode_status} for {module}.{action}",
            requires_human_confirm=requires_confirm,
            requested_execution_mode=mode_key,
            mode_status=mode_status,
        )

    parameter_contract = reg.get_parameter_contract(module, action)
    contract_properties = set(
        str(key) for key in (parameter_contract.get("properties") or {}).keys()
    )

    merged = dict(arguments or {})
    module_cfg = (module_config or {}).get(module)
    if isinstance(module_cfg, dict):
        for key, value in module_cfg.items():
            if contract_properties and str(key) not in contract_properties:
                continue
            if merged.get(key) is None or (
                isinstance(merged.get(key), str) and not str(merged.get(key)).strip()
            ):
                merged[key] = value

    missing = check_required_params(reg, module, action, merged)
    if missing:
        return ToolGuardResult(
            allowed=False,
            module=module,
            action=action,
            tool_name=tool_name,
            params=merged,
            error_type="missing_params",
            error_message=f"missing required params for {module}.{action}",
            missing_params=missing,
            requires_human_confirm=requires_confirm,
            requested_execution_mode=mode_key,
            mode_status=mode_status,
        )

    invalid_message = _validate_and_normalize_params(module, action, merged, reg)
    if invalid_message:
        return ToolGuardResult(
            allowed=False,
            module=module,
            action=action,
            tool_name=tool_name,
            params=merged,
            error_type="invalid_params",
            error_message=invalid_message,
            requires_human_confirm=requires_confirm,
            requested_execution_mode=mode_key,
            mode_status=mode_status,
        )

    return ToolGuardResult(
        allowed=True,
        module=module,
        action=action,
        tool_name=tool_name,
        params=merged,
        requires_human_confirm=requires_confirm,
        requested_execution_mode=mode_key,
        mode_status=mode_status,
    )
