# -*- coding: utf-8 -*-
"""Registry capability -> OpenAI-compatible tool schema 转换层。"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .execution_mode import ExecutionMode
from .registry import ModuleCapabilityRegistry, get_registry

TOOL_NAME_SEP = "__"


def capability_id(module: str, action: str) -> str:
    return f"{module}.{action}"


def capability_to_tool_name(module: str, action: str) -> str:
    """OpenAI function name（仅字母数字下划线）。"""
    return f"{module}{TOOL_NAME_SEP}{action}"


def tool_name_to_capability(tool_name: str) -> Optional[Tuple[str, str]]:
    if TOOL_NAME_SEP not in tool_name:
        return None
    module, action = tool_name.split(TOOL_NAME_SEP, 1)
    if not module or not action:
        return None
    return module, action


def _param_json_schema(
    param: str,
    *,
    module: Optional[str] = None,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    return {"description": f"Parameter {param}"}


def build_openai_tool_schema(
    module: str,
    action: str,
    action_def: Dict[str, Any],
    module_display_name: str,
    runtime_param_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    registry_required = list(action_def.get("required_params") or [])
    runtime_available = set(runtime_param_names or [])
    required = [param for param in registry_required if param not in runtime_available]
    optional = list(action_def.get("optional_params") or [])
    contract = action_def.get("parameter_contract")
    if isinstance(contract, dict):
        parameters = copy.deepcopy(contract)
        parameters.pop("$schema", None)
        parameters["type"] = "object"
        parameters["required"] = [
            str(param)
            for param in (parameters.get("required") or registry_required)
            if str(param) not in runtime_available
        ]
        parameters.setdefault("properties", {})
        for param, schema in parameters["properties"].items():
            if isinstance(schema, dict) and not schema.get("description"):
                inferred = _param_json_schema(str(param), module=module, action=action)
                schema["description"] = inferred.get("description") or f"Parameter {param}"
        parameters.setdefault("additionalProperties", False)
    else:
        properties: Dict[str, Any] = {}
        for param in registry_required + optional:
            properties[param] = _param_json_schema(param, module=module, action=action)
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    display = str(action_def.get("display_name") or action)
    notes = action_def.get("notes")
    description = f"{module_display_name}: {display} ({capability_id(module, action)})"
    if notes:
        description = f"{description}. {notes}"
    supplied = [param for param in registry_required if param in runtime_available]
    if supplied:
        description = (
            f"{description}. Runtime configuration supplies: {', '.join(supplied)}; "
            "omit these unless the user explicitly overrides them"
        )

    return {
        "type": "function",
        "function": {
            "name": capability_to_tool_name(module, action),
            "description": description[:1024],
            "parameters": parameters,
        },
    }


def build_tool_schemas_from_registry(
    registry: Optional[ModuleCapabilityRegistry] = None,
    *,
    execution_mode: ExecutionMode = ExecutionMode.DRY_RUN,
    module_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    reg = registry or get_registry()
    tools: List[Dict[str, Any]] = []
    mode_key = execution_mode.value.replace("-", "_")
    for module, action, action_def in reg.iter_actions():
        status = reg.get_execution_mode_status(module, action, mode_key)
        if status != "available":
            continue
        mod_def = reg.get_module(module) or {}
        display = str(mod_def.get("display_name") or module)
        runtime_values = (module_config or {}).get(module)
        runtime_names = [
            str(key)
            for key, value in (runtime_values.items() if isinstance(runtime_values, dict) else [])
            if value not in (None, "", [], {})
        ]
        tools.append(
            build_openai_tool_schema(
                module,
                action,
                action_def,
                display,
                runtime_param_names=runtime_names,
            )
        )
    return tools


def list_dry_run_tool_names(registry: Optional[ModuleCapabilityRegistry] = None) -> List[str]:
    reg = registry or get_registry()
    names: List[str] = []
    for module, action, _ in reg.iter_actions():
        if reg.get_execution_mode_status(module, action, "dry_run") == "available":
            names.append(capability_to_tool_name(module, action))
    return names
