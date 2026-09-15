# -*- coding: utf-8 -*-
"""Shared deterministic and semantic verification for resolved business-tool parameters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from jsonschema import Draft202012Validator

from .eval_judge import EvalJudge, EvalJudgeResult
from .execution_mode import ExecutionMode
from .redaction import redact_recursive
from .registry import ModuleCapabilityRegistry, get_registry
from .schemas import Observation

PARAMETER_INTENT_THRESHOLD = 0.8
_EMPTY_VALUES = (None, "", [], {})
_SOURCE_ALIASES = {
    "demo_fixture": "fixture",
    "fixture": "fixture",
    "legacy_gui_user_prefs": "legacy_gui_config",
    "legacy_gui_config": "legacy_gui_config",
    "model_argument": "model_argument",
    "module_config": "module_config",
}


def _nonempty(value: Any) -> bool:
    return value not in _EMPTY_VALUES


def _canonical_source(value: Any) -> str:
    raw = str(value or "").strip()
    return _SOURCE_ALIASES.get(raw, raw or "module_config")


def parameter_snapshot_hash(params: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(params),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "params_" + hashlib.sha256(encoded).hexdigest()


def resolve_parameter_sources(
    *,
    module: str,
    arguments: Mapping[str, Any],
    resolved_params: Mapping[str, Any],
    module_config: Optional[Mapping[str, Any]] = None,
    parameter_sources: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    configured = (module_config or {}).get(module)
    configured = configured if isinstance(configured, Mapping) else {}
    supplied = parameter_sources or {}
    result: Dict[str, str] = {}
    for key, value in resolved_params.items():
        if not _nonempty(value):
            continue
        explicit = supplied.get(f"{module}.{key}") or supplied.get(str(key))
        argument_present = key in arguments and _nonempty(arguments.get(key))
        configured_present = key in configured and _nonempty(configured.get(key))
        if argument_present and configured_present and arguments.get(key) != configured.get(key):
            result[str(key)] = "model_argument"
            continue
        if explicit:
            result[str(key)] = _canonical_source(explicit)
        elif argument_present:
            result[str(key)] = "model_argument"
        elif configured_present:
            result[str(key)] = "module_config"
        else:
            result[str(key)] = "unexplained"
    return result


def _source_conflicts(
    *,
    module: str,
    arguments: Mapping[str, Any],
    configured: Mapping[str, Any],
    parameter_sources: Optional[Mapping[str, str]],
) -> List[Dict[str, Any]]:
    conflicts: List[Dict[str, Any]] = []
    for key, model_value in arguments.items():
        if not _nonempty(model_value) or not _nonempty(configured.get(key)):
            continue
        if model_value != configured.get(key):
            conflicts.append(
                {
                    "parameter": str(key),
                    "winner": "model_argument",
                    "overridden_source": _canonical_source(
                        (parameter_sources or {}).get(f"{module}.{key}")
                        or (parameter_sources or {}).get(str(key))
                        or "module_config"
                    ),
                }
            )
    return conflicts


def _schema_issue_code(validator_name: str) -> str:
    return {
        "required": "parameter_required",
        "additionalProperties": "unknown_parameter",
        "type": "parameter_type_invalid",
        "enum": "parameter_enum_invalid",
        "minimum": "parameter_below_minimum",
        "maximum": "parameter_above_maximum",
        "exclusiveMinimum": "parameter_below_minimum",
        "exclusiveMaximum": "parameter_above_maximum",
        "minLength": "parameter_format_invalid",
        "maxLength": "parameter_format_invalid",
        "pattern": "parameter_format_invalid",
        "minItems": "parameter_format_invalid",
        "maxItems": "parameter_format_invalid",
    }.get(str(validator_name), "parameter_schema_invalid")


def _validate_schema(contract: Mapping[str, Any], params: Mapping[str, Any]) -> List[Dict[str, Any]]:
    try:
        validator = Draft202012Validator(dict(contract))
        raw_errors = sorted(validator.iter_errors(dict(params)), key=lambda item: list(item.path))
    except Exception:
        return [{"issue_code": "parameter_contract_invalid", "path": "$", "validator": "schema"}]
    errors: List[Dict[str, Any]] = []
    for error in raw_errors:
        path = ".".join(str(item) for item in error.path)
        errors.append(
            {
                "issue_code": _schema_issue_code(str(error.validator)),
                "path": path or "$",
                "validator": str(error.validator),
            }
        )
    return errors


@dataclass
class ParameterVerificationResult:
    tool_call_id: str
    step_id: str
    tool_name: str
    capability_id: str
    applicable: bool
    schema_pass: bool
    allowed: bool
    requires_confirmation: bool
    semantic_check_required: bool
    intent_decision: Optional[str]
    intent_score: Optional[float]
    method: str
    parameter_snapshot_hash: str
    parameter_summary: Dict[str, Any] = field(default_factory=dict)
    parameter_sources: Dict[str, str] = field(default_factory=dict)
    source_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    source_conflict_count: int = 0
    schema_errors: List[Dict[str, Any]] = field(default_factory=list)
    issue_codes: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    judge_provider: Optional[str] = None
    judge_model: Optional[str] = None
    judge_error_type: Optional[str] = None
    confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        safe, _ = redact_recursive(
            {
                "type": "parameter_verification_result",
                "tool_call_id": self.tool_call_id,
                "step_id": self.step_id,
                "tool_name": self.tool_name,
                "capability_id": self.capability_id,
                "applicable": self.applicable,
                "schema_pass": self.schema_pass,
                "allowed": self.allowed,
                "requires_confirmation": self.requires_confirmation,
                "semantic_check_required": self.semantic_check_required,
                "intent_decision": self.intent_decision,
                "intent_score": self.intent_score,
                "method": self.method,
                "parameter_snapshot_hash": self.parameter_snapshot_hash,
                "parameter_summary": self.parameter_summary,
                "parameter_sources": self.parameter_sources,
                "source_conflicts": self.source_conflicts,
                "source_conflict_count": self.source_conflict_count,
                "schema_errors": self.schema_errors,
                "issue_codes": self.issue_codes,
                "evidence_refs": self.evidence_refs,
                "judge_provider": self.judge_provider,
                "judge_model": self.judge_model,
                "judge_error_type": self.judge_error_type,
                "confirmed": self.confirmed,
            }
        )
        return safe if isinstance(safe, dict) else {}


def verify_tool_call_parameters(
    *,
    tool_call_id: str,
    step_id: str,
    tool_name: str,
    module: str,
    action: str,
    arguments: Optional[Mapping[str, Any]],
    resolved_params: Optional[Mapping[str, Any]],
    user_request: str,
    execution_mode: ExecutionMode,
    registry: Optional[ModuleCapabilityRegistry] = None,
    module_config: Optional[Mapping[str, Any]] = None,
    parameter_sources: Optional[Mapping[str, str]] = None,
    recent_dialogue: Optional[Sequence[Mapping[str, Any]]] = None,
    semantic_mode: str = "auto",
    judge: Optional[EvalJudge] = None,
) -> ParameterVerificationResult:
    reg = registry or get_registry()
    capability = f"{module}.{action}"
    raw_args = dict(arguments or {})
    params = dict(resolved_params or {})
    safe_params, _ = redact_recursive(params)
    safe_params = safe_params if isinstance(safe_params, dict) else {}
    sources = resolve_parameter_sources(
        module=module,
        arguments=raw_args,
        resolved_params=params,
        module_config=module_config,
        parameter_sources=parameter_sources,
    )
    configured = (module_config or {}).get(module)
    configured = configured if isinstance(configured, Mapping) else {}
    conflicts = _source_conflicts(
        module=module,
        arguments=raw_args,
        configured=configured,
        parameter_sources=parameter_sources,
    )
    contract = reg.get_parameter_contract(module, action)
    schema_errors = _validate_schema(contract, params)
    schema_pass = not schema_errors
    issue_codes = [str(item.get("issue_code")) for item in schema_errors]
    snapshot_hash = parameter_snapshot_hash(params)
    evidence_refs = [f"contract:{capability}"]
    for item in conflicts:
        evidence_refs.append(f"source_conflict:{capability}.{item.get('parameter')}")

    semantic_setting = str(semantic_mode or "auto").strip().lower()
    semantic_required = semantic_setting == "always" or (
        semantic_setting == "auto"
        and (
            execution_mode in (ExecutionMode.LOCAL_WRITE, ExecutionMode.LIVE)
            or bool(conflicts)
            or any(source == "unexplained" for source in sources.values())
        )
    )
    common = {
        "tool_call_id": tool_call_id,
        "step_id": step_id,
        "tool_name": tool_name,
        "capability_id": capability,
        "applicable": True,
        "parameter_snapshot_hash": snapshot_hash,
        "parameter_summary": safe_params,
        "parameter_sources": sources,
        "source_conflicts": conflicts,
        "source_conflict_count": len(conflicts),
        "evidence_refs": evidence_refs,
    }
    if not schema_pass:
        return ParameterVerificationResult(
            **common,
            schema_pass=False,
            allowed=False,
            requires_confirmation=False,
            semantic_check_required=False,
            intent_decision=None,
            intent_score=None,
            method="json_schema",
            schema_errors=schema_errors,
            issue_codes=issue_codes,
        )

    if not semantic_required or semantic_setting == "never":
        return ParameterVerificationResult(
            **common,
            schema_pass=True,
            allowed=True,
            requires_confirmation=False,
            semantic_check_required=False,
            intent_decision=None,
            intent_score=None,
            method="json_schema",
        )

    if judge is None:
        judge_result = EvalJudgeResult(
            decision="uncertain",
            issue_codes=["parameter_judge_unavailable"],
            error_type="parameter_judge_unavailable",
        )
    else:
        try:
            judge_result = judge.judge_parameter_intent(
                {
                    "user_request": user_request,
                    "recent_dialogue": list(recent_dialogue or [])[-8:],
                    "capability_id": capability,
                    "parameters": safe_params,
                    "parameter_sources": sources,
                    "source_conflicts": conflicts,
                    "contract": contract,
                }
            )
        except Exception:
            judge_result = EvalJudgeResult(
                decision="uncertain",
                issue_codes=["parameter_judge_failed"],
                error_type="parameter_judge_failed",
            )

    score = judge_result.score
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
        score = None
    decision = judge_result.decision
    aligned = decision == "aligned" and score is not None and score >= PARAMETER_INTENT_THRESHOLD
    below_threshold = decision == "aligned" and not aligned
    uncertain = decision == "uncertain" or below_threshold
    semantic_issues = list(judge_result.issue_codes)
    if decision == "aligned" and not aligned:
        semantic_issues.append("parameter_intent_below_threshold")
    if decision == "misaligned":
        semantic_issues.append("parameter_intent_misaligned")
    return ParameterVerificationResult(
        **{
            **common,
            "evidence_refs": list(
                dict.fromkeys(evidence_refs + list(judge_result.evidence_refs))
            ),
        },
        schema_pass=True,
        allowed=aligned,
        requires_confirmation=uncertain,
        semantic_check_required=True,
        intent_decision=decision,
        intent_score=score,
        method="json_schema+judge",
        issue_codes=list(dict.fromkeys(semantic_issues)),
        judge_provider=judge_result.provider,
        judge_model=judge_result.model,
        judge_error_type=judge_result.error_type,
    )


def parameter_validation_observation(
    step_id: str,
    result: Mapping[str, Any],
) -> Observation:
    requires_confirmation = bool(result.get("requires_confirmation"))
    error_type = "parameter_confirmation_required" if requires_confirmation else "parameter_validation_failed"
    return Observation(
        step_id=step_id,
        status="failed",
        summary=error_type,
        error={
            "type": error_type,
            "issue_codes": list(result.get("issue_codes") or []),
            "schema_errors": list(result.get("schema_errors") or []),
            "parameter_snapshot_hash": result.get("parameter_snapshot_hash"),
        },
        suggested_next_action="confirm_parameters" if requires_confirmation else "replan",
    )
