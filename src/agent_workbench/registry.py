# -*- coding: utf-8 -*-
"""Module Capability Registry — 模块能力注册表。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .permissions import PermissionLevel, parse_risk_level


class ModuleCapabilityRegistry:
    """加载并查询 module_capabilities.json。"""

    def __init__(self, capabilities_path: Optional[Path] = None, *, payload=None,
                 executor=None, parameter_validator=None, artifact_validator=None,
                 system_prompt=None, answer_verifier=None, config_context_builder=None):
        if capabilities_path is not None and payload is not None:
            raise ValueError("Supply a path or payload, not both")
        if capabilities_path is not None:
            with open(capabilities_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        if not isinstance(payload, dict) or not isinstance(payload.get("modules"), dict):
            raise ValueError("An explicit tool catalog is required")
        self._modules: Dict[str, Dict[str, Any]] = copy.deepcopy(payload["modules"])
        self.version = payload.get("version", "unknown")
        self.executor = executor
        self.parameter_validator = parameter_validator
        self.artifact_validator = artifact_validator
        self.system_prompt = system_prompt
        self.answer_verifier = answer_verifier
        self.config_context_builder = config_context_builder

    def snapshot(self):
        return {"version": self.version, "modules": copy.deepcopy(self._modules)}

    def list_modules(self) -> List[str]:
        return sorted(self._modules.keys())

    def get_module(self, module: str) -> Optional[Dict[str, Any]]:
        return self._modules.get(module)

    def get_action(self, module: str, action: str) -> Optional[Dict[str, Any]]:
        mod = self.get_module(module)
        if not mod:
            return None
        return (mod.get("actions") or {}).get(action)

    def module_exists(self, module: str) -> bool:
        return module in self._modules

    def action_exists(self, module: str, action: str) -> bool:
        return self.get_action(module, action) is not None

    def get_risk_level(self, module: str, action: str) -> PermissionLevel:
        action_def = self.get_action(module, action)
        if not action_def:
            return PermissionLevel.L5_HIGH_RISK_SIDE_EFFECT
        return parse_risk_level(action_def.get("risk_level", "L2"))

    def get_required_params(self, module: str, action: str) -> List[str]:
        action_def = self.get_action(module, action)
        if not action_def:
            return []
        return list(action_def.get("required_params") or [])

    def get_parameter_contract(self, module: str, action: str) -> Dict[str, Any]:
        '''Return the canonical JSON Schema contract for one capability.'''
        action_def = self.get_action(module, action)
        if not action_def:
            return {}
        contract = action_def.get("parameter_contract")
        if isinstance(contract, dict):
            return dict(contract)

        required = list(action_def.get("required_params") or [])
        optional = list(action_def.get("optional_params") or [])
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                name: {"type": ["string", "number", "integer", "boolean", "array", "object", "null"]}
                for name in dict.fromkeys(required + optional)
            },
            "required": required,
            "additionalProperties": False,
        }

    def get_execution_adapter(self, module: str, action: str) -> Optional[str]:
        action_def = self.get_action(module, action)
        if not action_def:
            return None
        return action_def.get("execution_adapter")

    def get_workflow_file(self, module: str, action: str) -> Optional[str]:
        action_def = self.get_action(module, action)
        if not action_def:
            return None
        return action_def.get("workflow_file")

    def get_entrypoint_type(self, module: str, action: str) -> Optional[str]:
        action_def = self.get_action(module, action)
        if not action_def:
            return None
        return action_def.get("existing_entrypoint_type")

    def get_live_adapter_status(self, module: str, action: str) -> str:
        action_def = self.get_action(module, action)
        if not action_def:
            return "unknown"
        return str(action_def.get("live_adapter_status", "not_started"))

    def get_execution_mode_status(self, module: str, action: str, mode: str) -> str:
        """查询 dry_run / read_only / local_write / live 能力状态。"""
        action_def = self.get_action(module, action)
        if not action_def:
            return "unknown"
        key = f"{mode}_status"
        if key in action_def:
            return str(action_def[key])
        return self._infer_mode_status_from_legacy(action_def, mode)

    def get_execution_mode_statuses(self, module: str, action: str) -> Dict[str, str]:
        return {
            "dry_run": self.get_execution_mode_status(module, action, "dry_run"),
            "read_only": self.get_execution_mode_status(module, action, "read_only"),
            "local_write": self.get_execution_mode_status(module, action, "local_write"),
            "live": self.get_execution_mode_status(module, action, "live"),
        }

    @staticmethod
    def _infer_mode_status_from_legacy(action_def: Dict[str, Any], mode: str) -> str:
        legacy = str(action_def.get("live_adapter_status", "not_started"))
        if mode == "dry_run":
            return "available"
        if mode == "read_only":
            return "available" if legacy == "read_only_ready" else "unavailable"
        if mode == "local_write":
            return "not_started"
        if mode == "live":
            return "available" if legacy == "live_ready" else "not_started"
        return "unknown"

    def module_supports_store_modes(self, module: str) -> bool:
        mod = self.get_module(module)
        if not mod:
            return False
        return bool(mod.get("supports_store_modes"))

    def iter_actions(self) -> List[tuple[str, str, Dict[str, Any]]]:
        items: List[tuple[str, str, Dict[str, Any]]] = []
        for module, mod_def in self._modules.items():
            for action, action_def in (mod_def.get("actions") or {}).items():
                items.append((module, action, action_def))
        return items

    def validate_module_action(self, module: str, action: str) -> Tuple[bool, Optional[str]]:
        if not self.module_exists(module):
            return False, f"unknown module: {module}"
        if not self.action_exists(module, action):
            return False, f"unknown action: {module}.{action}"
        return True, None


def clear_registry_cache() -> None:
    """Compatibility hook: public registries are explicitly scoped, never global."""


def get_registry() -> ModuleCapabilityRegistry:
    raise RuntimeError("ToolRegistry must be supplied by the host")


ToolRegistry = ModuleCapabilityRegistry
