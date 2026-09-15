"""Bounded, trusted artifact policies and the injectable content-validator contract.

Rules are plugin-owned JSON, not executable expressions. Registry requirements
and host requirements are additive; conflicting rule definitions fail closed.
No policy is obtained from tool arguments, environment, or a host singleton.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Protocol


class ArtifactValidator(Protocol):
    def validate(self, snapshot: dict, policy: dict) -> dict:
        """Validate every snapshot row, returning complete/valid/errors/row_count.

        Include checked_rules covering policy.required_rules when present.
        snapshot.rows contains deterministic row_id values and parsed content;
        snapshot.manifest describes the actual selected bytes. The engine, not
        the validator, owns identity, sampling, revision, and runtime bindings.
        """
        ...


_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}")
_SCOPES = {"input_contract", "explicit_file_set", "directory_root"}
_POLICY_KEYS = {"required", "rule_version", "scope", "columns", "sheet",
                "required_rules", "rules"}


def _json_value(value, depth=0):
    if depth > 8:
        raise ValueError("artifact_policy_invalid")
    if value is None or type(value) in (bool, int, float, str):
        return
    if isinstance(value, list) and len(value) <= 256:
        for item in value:
            _json_value(item, depth + 1)
        return
    if isinstance(value, dict) and len(value) <= 256 and all(isinstance(k, str) for k in value):
        for item in value.values():
            _json_value(item, depth + 1)
        return
    raise ValueError("artifact_policy_invalid")


def validate_artifact_policy(value):
    """Validate one generic policy; return a detached copy or raise ValueError."""
    if not isinstance(value, dict) or set(value) - _POLICY_KEYS:
        raise ValueError("artifact_policy_invalid")
    _json_value(value)
    try:
        if len(json.dumps(value, allow_nan=False)) > 32768:
            raise ValueError("artifact_policy_invalid")
    except (TypeError, ValueError):
        raise ValueError("artifact_policy_invalid") from None
    if "required" in value and type(value["required"]) is not bool:
        raise ValueError("artifact_policy_invalid")
    for name in ("rule_version",):
        if name in value and (not isinstance(value[name], str) or not _NAME.fullmatch(value[name])):
            raise ValueError("artifact_policy_invalid")
    if "scope" in value and (not isinstance(value["scope"], str) or value["scope"] not in _SCOPES):
        raise ValueError("artifact_policy_invalid")
    for key in ("columns", "required_rules"):
        if key not in value:
            continue
        items = value[key]
        if (not isinstance(items, list) or len(items) > 256
                or any(not isinstance(s, str) or not s.strip() or len(s) > 80 for s in items)
                or len(set(items)) != len(items)):
            raise ValueError("artifact_policy_invalid")
        if key == "required_rules" and any(not _NAME.fullmatch(s) for s in items):
            raise ValueError("artifact_policy_invalid")
    if "sheet" in value and (not isinstance(value["sheet"], str) or not value["sheet"]
                             or len(value["sheet"]) > 80):
        raise ValueError("artifact_policy_invalid")
    if "rules" in value and (not isinstance(value["rules"], dict)
                            or any(not _NAME.fullmatch(k) for k in value["rules"])):
        raise ValueError("artifact_policy_invalid")
    return copy.deepcopy(value)


def validate_artifact_policies(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 32:
        raise ValueError("artifact_policy_invalid")
    result = {}
    for capability, policy in value.items():
        if not isinstance(capability, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", capability):
            raise ValueError("artifact_policy_invalid")
        result[capability] = validate_artifact_policy(policy)
    return result


def merge_artifact_policies(*policies):
    """Union required flags/rule IDs; never silently replace a rule definition."""
    merged = {"required": False}
    for value in policies:
        policy = validate_artifact_policy(value)
        merged["required"] = merged["required"] or policy.get("required", False)
        for key, item in policy.items():
            if key == "required":
                continue
            if key in {"required_rules", "columns"}:
                merged[key] = sorted(set(merged.get(key, [])) | set(item))
            elif key == "rules":
                rules = merged.setdefault("rules", {})
                for name, rule in item.items():
                    if name in rules and rules[name] != rule:
                        raise ValueError("artifact_policy_conflict")
                    rules[name] = copy.deepcopy(rule)
            elif key == "scope" and "directory_root" in (item, merged.get(key)):
                merged[key] = "directory_root"
            elif key == "sheet" and key in merged and merged[key] != item:
                raise ValueError("artifact_policy_conflict")
            else:
                merged[key] = copy.deepcopy(item)
    # Declaring required rules cannot leave the gate disabled.
    merged["required"] = merged["required"] or bool(merged.get("required_rules"))
    return validate_artifact_policy(merged)


# Compatibility name for existing profile imports; the schema is now generic.
validate_company_artifact_policies = validate_artifact_policies
