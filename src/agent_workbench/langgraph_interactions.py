# -*- coding: utf-8 -*-
"""Interactive HITL：交互 schema、输入校验与进程内 runtime handle。"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .redaction import redact_recursive, redact_text

ASK_USER_TOOL_NAME = "agent__ask_user"
_QUESTION_MAX = 500
_REASON_MAX = 500
_FIELD_MAX = 20
_FIELD_NAME_MAX = 80
_ANSWER_MAX = 4000
_PARAMS_SUMMARY_MAX = 500
_PARAM_NAMES_MAX = 40
_FORBIDDEN_RESUME_KEYS = frozenset(
    {
        "arguments",
        "tool_arguments",
        "params",
        "risk_level",
        "execution_mode",
        "capability_id",
        "capability",
        "parameter_snapshot_hash",
        "parameter_snapshot_hashes",
        "parameter_summary",
        "confirmation_kind",
    }
)

_HANDLE_PENDING = "pending"
_HANDLE_RESUMING = "resuming"
_HANDLE_COMPLETED = "completed"
_HANDLE_FAILED = "failed"
_TERMINAL_STATUSES = frozenset({_HANDLE_COMPLETED, _HANDLE_FAILED})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bound(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit]


def _redact_bound(text: Any, limit: int) -> str:
    raw = "" if text is None else str(text)
    redacted, _ = redact_text(raw)
    return _bound(redacted, limit)


def is_interaction_tool(tool_name: str) -> bool:
    return str(tool_name or "").strip() == ASK_USER_TOOL_NAME


def build_interaction_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": ASK_USER_TOOL_NAME,
                "description": (
                    "Ask the operator for required missing information. "
                    "Call this tool alone; do not mix with business tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _QUESTION_MAX,
                            "description": "Question shown to the operator",
                        },
                        "missing_fields": {
                            "type": "array",
                            "maxItems": _FIELD_MAX,
                            "items": {"type": "string"},
                            "description": "Required field names still missing",
                        },
                        "reason": {
                            "type": "string",
                            "maxLength": _REASON_MAX,
                            "description": "Why the information is required",
                        },
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _stable_interaction_id(run_id: str, interaction_type: str, tool_call_ids: List[str]) -> str:
    raw = json.dumps([run_id, interaction_type, tool_call_ids], sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"int_{digest[:24]}"


def _payload_arguments(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    raw = tool_call.get("arguments")
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    nested = function.get("arguments")
    if isinstance(nested, dict):
        return dict(nested)
    if isinstance(nested, str):
        try:
            parsed = json.loads(nested)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def validate_clarification_tool_call(tool_call: Dict[str, Any]) -> Optional[str]:
    """Validate untrusted model output before creating a user-facing interrupt."""
    if not isinstance(tool_call, dict):
        return "interaction_request_invalid"
    args = _payload_arguments(tool_call)
    if set(args) - {"question", "missing_fields", "reason"}:
        return "interaction_request_invalid"
    question = args.get("question")
    if not isinstance(question, str) or not question.strip() or len(question) > _QUESTION_MAX:
        return "interaction_request_invalid"
    reason = args.get("reason")
    if reason is not None and (not isinstance(reason, str) or len(reason) > _REASON_MAX):
        return "interaction_request_invalid"
    missing_fields = args.get("missing_fields", [])
    if missing_fields is None:
        missing_fields = []
    if not isinstance(missing_fields, list) or len(missing_fields) > _FIELD_MAX:
        return "interaction_request_invalid"
    for item in missing_fields:
        if not isinstance(item, str) or not item.strip() or len(item) > _FIELD_NAME_MAX:
            return "interaction_request_invalid"
    return None


def build_clarification_interaction(run_id: str, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    error_type = validate_clarification_tool_call(tool_call)
    if error_type:
        raise ValueError(error_type)
    args = _payload_arguments(tool_call)
    tool_call_id = str(tool_call.get("tool_call_id") or tool_call.get("id") or "")
    missing = []
    for item in list(args.get("missing_fields") or [])[:_FIELD_MAX]:
        text = _redact_bound(item, _FIELD_NAME_MAX)
        if text:
            missing.append(text)
    payload = {
        "interaction_id": _stable_interaction_id(str(run_id or ""), "clarification", [
            str(tool_call.get("step_id") or ""), tool_call_id,
            hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest(),
        ]),
        "type": "clarification",
        "status": "pending",
        "question": _redact_bound(args.get("question"), _QUESTION_MAX),
        "missing_fields": missing,
        "reason": _redact_bound(args.get("reason") or "缺少执行所需参数", _REASON_MAX),
        "actions": [],
        "created_at": _utc_now(),
    }
    redacted, _ = redact_recursive(payload)
    return redacted if isinstance(redacted, dict) else payload


def _params_summary(arguments: Any) -> str:
    redacted, _ = redact_recursive(arguments if isinstance(arguments, dict) else {})
    try:
        text = json.dumps(redacted, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        text = str(redacted)
    return _bound(text, _PARAMS_SUMMARY_MAX)


def build_confirmation_interaction(
    run_id: str,
    pending_tool_calls: List[Dict[str, Any]],
    guard_results: List[Dict[str, Any]],
    execution_mode: str,
    parameter_verification_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    effective_mode = str(execution_mode or "dry_run").strip().lower().replace("-", "_")
    if effective_mode not in {"dry_run", "read_only", "local_write", "live"}:
        effective_mode = "unknown"
    actions: List[Dict[str, Any]] = []
    tool_call_ids: List[str] = []
    confirmation_bindings: List[str] = []
    parameter_hashes: List[str] = []
    parameter_confirmation_needed = False
    risk_confirmation_needed = False
    guards = {str(item.get("tool_call_id") or ""): item for item in guard_results if isinstance(item, dict)}
    verifications = {
        str(item.get("tool_call_id") or ""): item
        for item in (parameter_verification_results or [])
        if isinstance(item, dict)
    }
    for entry in pending_tool_calls or []:
        if not isinstance(entry, dict):
            continue
        call_id = str(entry.get("tool_call_id") or entry.get("id") or "")
        guard = guards.get(call_id) or {}
        verification = verifications.get(call_id) or {}
        if guard.get("allowed") is False:
            continue
        module = str(guard.get("module") or "")
        action = str(guard.get("action") or "")
        capability = f"{module}.{action}" if module and action else str(entry.get("tool_name") or "")
        risk = str(guard.get("risk_level") or entry.get("risk_level") or "L4")
        if risk not in {"L0", "L1", "L2", "L3", "L4", "L5"}:
            risk = "L4"
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        resolved_params = (
            verification.get("parameter_summary")
            if isinstance(verification.get("parameter_summary"), dict)
            else arguments
        )
        param_names = [
            _redact_bound(key, _FIELD_NAME_MAX)
            for key in list(resolved_params.keys())[:_PARAM_NAMES_MAX]
        ]
        snapshot_hash = _redact_bound(verification.get("parameter_snapshot_hash"), 80)
        if snapshot_hash:
            parameter_hashes.append(snapshot_hash)
        parameter_needed = bool(verification.get("requires_confirmation"))
        risk_needed = bool(guard.get("requires_human_confirm")) or effective_mode in {"local_write", "live"}
        parameter_confirmation_needed = parameter_confirmation_needed or parameter_needed
        risk_confirmation_needed = risk_confirmation_needed or risk_needed
        actions.append(
            {
                "tool_call_id": call_id,
                "capability_id": capability,
                "risk_level": risk,
                "execution_mode": effective_mode,
                "param_names": param_names,
                "params_summary": _params_summary(resolved_params),
                "parameter_snapshot_hash": snapshot_hash or None,
                "parameter_confirmation_required": parameter_needed,
                "risk_confirmation_required": risk_needed,
            }
        )
        tool_call_ids.append(call_id)
        # Model call IDs may repeat. Engine steps and frozen values define this approval.
        confirmation_bindings.append(json.dumps({
            "step_id": entry.get("step_id"), "call_id": call_id,
            "capability": capability, "execution_mode": effective_mode,
            "snapshot_hash": snapshot_hash,
            "arguments_hash": hashlib.sha256(json.dumps(
                arguments, sort_keys=True, default=str).encode()).hexdigest(),
        }, sort_keys=True))
    if parameter_confirmation_needed and risk_confirmation_needed:
        confirmation_kind = "combined"
    elif parameter_confirmation_needed:
        confirmation_kind = "parameter"
    else:
        confirmation_kind = "risk"
    payload = {
        "interaction_id": _stable_interaction_id(str(run_id or ""), "confirmation", confirmation_bindings),
        "type": "confirmation",
        "confirmation_kind": confirmation_kind,
        "status": "pending",
        "question": "请确认是否继续执行以下高风险工具。",
        "missing_fields": [],
        "reason": "当前执行模式或工具风险等级需要人工确认",
        "actions": actions,
        "parameter_snapshot_hashes": list(dict.fromkeys(parameter_hashes)),
        "created_at": _utc_now(),
    }
    if confirmation_kind == "parameter":
        payload["question"] = "\u8bf7\u786e\u8ba4\u4ee5\u4e0b\u5de5\u5177\u53c2\u6570\u662f\u5426\u7b26\u5408\u4f60\u7684\u610f\u56fe\u3002"
        payload["reason"] = "\u53c2\u6570\u610f\u56fe\u6821\u9a8c\u4e0d\u786e\u5b9a\uff0c\u9700\u8981\u4f60\u786e\u8ba4\u51bb\u7ed3\u53c2\u6570"
    elif confirmation_kind == "combined":
        payload["question"] = "\u8bf7\u786e\u8ba4\u4ee5\u4e0b\u5de5\u5177\u53c2\u6570\u548c\u9ad8\u98ce\u9669\u6267\u884c\u3002"
        payload["reason"] = (
            "\u53c2\u6570\u9700\u8981\u786e\u8ba4\uff0c\u4e14\u5f53\u524d\u6267\u884c\u6a21\u5f0f\u6216\u5de5\u5177\u98ce\u9669\u7b49\u7ea7\u9700\u8981\u4eba\u5de5\u653e\u884c"
        )
    redacted, _ = redact_recursive(payload)
    return redacted if isinstance(redacted, dict) else payload


def sanitize_interaction_response(
    payload: Dict[str, Any],
    expected: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(payload, dict) or not isinstance(expected, dict):
        return None, "interaction_response_invalid"
    if any(key in payload for key in _FORBIDDEN_RESUME_KEYS):
        return None, "interaction_response_invalid"
    interaction_id = str(payload.get("interaction_id") or "").strip()
    expected_id = str(expected.get("interaction_id") or "").strip()
    response_type = str(payload.get("type") or "").strip()
    expected_type = str(expected.get("type") or "").strip()
    if not interaction_id or not expected_id or interaction_id != expected_id or response_type != expected_type:
        return None, "interaction_mismatch"
    if expected_type == "clarification":
        answer = payload.get("answer")
        if not isinstance(answer, str) or len(answer) > _ANSWER_MAX:
            return None, "interaction_response_invalid"
        cleaned = _redact_bound(answer, _ANSWER_MAX)
        if not cleaned.strip():
            return None, "interaction_response_invalid"
        sanitized = {
            "interaction_id": expected_id,
            "type": "clarification",
            "answer": cleaned,
        }
        redacted, _ = redact_recursive(sanitized)
        return (redacted if isinstance(redacted, dict) else sanitized), None
    if expected_type == "artifact_review":
        if set(payload) - {"interaction_id", "type", "decision", "comment"}:
            return None, "interaction_response_invalid"
        decision = payload.get("decision")
        if decision not in {"approve", "reject", "request_changes"}:
            return None, "interaction_response_invalid"
        if decision == "approve" and not expected.get("can_approve"):
            return None, "artifact_has_errors"
        comment = payload.get("comment", "")
        if not isinstance(comment, str) or len(comment) > _ANSWER_MAX:
            return None, "interaction_response_invalid"
        return {"interaction_id": expected_id, "type": expected_type,
                "decision": decision, "comment": _redact_bound(comment, _ANSWER_MAX)}, None
    if expected_type == "confirmation":
        if set(payload) - {"interaction_id", "type", "decision"}:
            return None, "interaction_response_invalid"
        decision = str(payload.get("decision") or "").strip().lower()
        if decision not in {"approve", "reject"}:
            return None, "interaction_response_invalid"
        return {
            "interaction_id": expected_id,
            "type": "confirmation",
            "decision": decision,
        }, None
    return None, "interaction_response_invalid"


def extract_pending_interaction(invoke_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(invoke_result, dict):
        return None
    interrupts = invoke_result.get("__interrupt__") or []
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    if isinstance(value, dict):
        redacted, _ = redact_recursive(value)
        return redacted if isinstance(redacted, dict) else value
    return None


@dataclass
class InteractionRuntimeHandle:
    run_id: str
    compiled_graph: Any
    runtime_config: Any
    invoke_config: Dict[str, Any]
    pending_interaction: Dict[str, Any]
    status: str
    created_at: str
    resume_error_type: Optional[str] = None


class InteractionRuntimeStore:
    def __init__(self, *, max_entries: int = 32):
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._handles: Dict[str, InteractionRuntimeHandle] = {}
        self._order: List[str] = []

    def create_checkpointer(self) -> Any:
        try:
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        return InMemorySaver()

    def _evict_terminal_locked(self, *, target_size: Optional[int] = None) -> None:
        limit = self._max_entries if target_size is None else max(0, int(target_size))
        if len(self._handles) <= limit:
            return
        removable = [
            run_id
            for run_id in list(self._order)
            if self._handles.get(run_id) is not None and self._handles[run_id].status in _TERMINAL_STATUSES
        ]
        while len(self._handles) > limit and removable:
            run_id = removable.pop(0)
            self._handles.pop(run_id, None)
            if run_id in self._order:
                self._order.remove(run_id)

    def register(self, handle: InteractionRuntimeHandle) -> Optional[str]:
        with self._lock:
            if not handle.run_id or handle.run_id in self._handles:
                return "interaction_run_conflict"
            self._evict_terminal_locked(target_size=self._max_entries - 1)
            if len(self._handles) >= self._max_entries:
                return "interaction_capacity_exceeded"
            self._handles[handle.run_id] = handle
            self._order.append(handle.run_id)
            return None

    def get_pending(self, run_id: str) -> Tuple[Optional[InteractionRuntimeHandle], Optional[str]]:
        with self._lock:
            handle = self._handles.get(str(run_id or ""))
            if handle is None:
                return None, "interaction_expired"
            if handle.status == _HANDLE_COMPLETED:
                return None, "interaction_already_resolved"
            if handle.status == _HANDLE_FAILED:
                return None, handle.resume_error_type or "interaction_resume_failed"
            return handle, None

    def begin_resume(
        self,
        run_id: str,
        response: Dict[str, Any],
    ) -> Tuple[Optional[InteractionRuntimeHandle], Optional[Dict[str, Any]], Optional[str]]:
        with self._lock:
            handle = self._handles.get(str(run_id or ""))
            if handle is None:
                return None, None, "interaction_expired"
            if handle.status == _HANDLE_COMPLETED:
                return None, None, "interaction_already_resolved"
            if handle.status == _HANDLE_FAILED:
                return None, None, handle.resume_error_type or "interaction_resume_failed"
            if handle.status == _HANDLE_RESUMING:
                return None, None, "interaction_resume_in_progress"
            sanitized, error_type = sanitize_interaction_response(response or {}, handle.pending_interaction or {})
            if error_type:
                return None, None, error_type
            handle.status = _HANDLE_RESUMING
            return handle, sanitized, None

    def update_pending(self, run_id: str, pending: Dict[str, Any]) -> None:
        with self._lock:
            handle = self._handles.get(str(run_id or ""))
            if handle is None:
                return
            handle.pending_interaction = dict(pending or {})
            handle.status = _HANDLE_PENDING
            handle.resume_error_type = None

    def complete(self, run_id: str) -> None:
        with self._lock:
            handle = self._handles.get(str(run_id or ""))
            if handle is None:
                return
            handle.status = _HANDLE_COMPLETED
            handle.pending_interaction = {}
            self._evict_terminal_locked()

    def fail_resume(self, run_id: str) -> None:
        with self._lock:
            handle = self._handles.get(str(run_id or ""))
            if handle is None:
                return
            handle.status = _HANDLE_FAILED
            handle.resume_error_type = "interaction_resume_failed"
