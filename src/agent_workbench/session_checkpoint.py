# -*- coding: utf-8 -*-
"""Phase 13.1 — 线程会话、短期记忆与文件型 Session Checkpoint。"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .context_manager import _compact_artifact_refs
from .redaction import redact_plan_dict, redact_recursive, redact_text

THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_RECENT_TURNS = 6
DEFAULT_SESSION_DIR_NAME = "agent_sessions"


def default_session_dir(base: Optional[Path] = None) -> Path:
    if base is not None:
        return Path(base) / "runs" / DEFAULT_SESSION_DIR_NAME
    root = Path(os.environ.get("AGENT_WORKBENCH_HOME") or Path.home() / ".agent-workbench").expanduser()
    return root / "sessions"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_thread_id(thread_id: str) -> Optional[str]:
    tid = (thread_id or "").strip()
    if not tid:
        return None
    if not THREAD_ID_PATTERN.match(tid):
        return "invalid_thread_id"
    return None


DISABLED_COMPANY_PROFILE_BINDING_ID = "disabled"


def opaque_company_profile_binding_id(company_skill_runtime: Optional[Dict[str, Any]]) -> str:
    """只返回不透明 binding/snapshot ID，不保存 tenant、路径或配置正文。"""
    runtime = company_skill_runtime if isinstance(company_skill_runtime, dict) else {}
    profile = runtime.get("company_profile") if isinstance(runtime.get("company_profile"), dict) else {}
    if not profile.get("enabled"):
        return DISABLED_COMPANY_PROFILE_BINDING_ID
    snapshot_id = str(runtime.get("company_profile_snapshot_id") or "").strip()
    if snapshot_id:
        return snapshot_id
    snapshot = profile.get("snapshot") if isinstance(profile.get("snapshot"), dict) else {}
    nested = str(snapshot.get("snapshot_id") or "").strip()
    return nested or "enabled"


@dataclass
class SessionRecord:
    thread_id: str
    created_at: str
    updated_at: str
    turn_count: int = 0
    recent_turns: List[Dict[str, Any]] = field(default_factory=list)
    session_summary: str = ""
    latest_checkpoint_summary: str = ""
    pending_questions: List[str] = field(default_factory=list)
    confirmed_decisions: List[str] = field(default_factory=list)
    known_parameters: Dict[str, Any] = field(default_factory=dict)
    active_plan_summary: Optional[Dict[str, Any]] = None
    latest_execution_status: Optional[str] = None
    corrections: List[str] = field(default_factory=list)
    artifact_refs: List[Dict[str, Any]] = field(default_factory=list)
    company_profile_binding_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        binding = data.get("company_profile_binding_id")
        binding_id = str(binding).strip() if binding not in (None, "") else None
        return cls(
            thread_id=str(data.get("thread_id") or ""),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            turn_count=int(data.get("turn_count") or 0),
            recent_turns=list(data.get("recent_turns") or []),
            session_summary=str(data.get("session_summary") or ""),
            latest_checkpoint_summary=str(data.get("latest_checkpoint_summary") or ""),
            pending_questions=list(data.get("pending_questions") or []),
            confirmed_decisions=list(data.get("confirmed_decisions") or []),
            known_parameters=dict(data.get("known_parameters") or {}),
            active_plan_summary=data.get("active_plan_summary"),
            latest_execution_status=data.get("latest_execution_status"),
            corrections=list(data.get("corrections") or []),
            artifact_refs=list(data.get("artifact_refs") or []),
            company_profile_binding_id=binding_id,
        )

    @classmethod
    def new_thread(cls, thread_id: str) -> "SessionRecord":
        now = _utc_now()
        return cls(thread_id=thread_id, created_at=now, updated_at=now)


class SessionCheckpointStore:
    """标准库文件型 session 存储（原子写入、按 thread 隔离）。"""

    def __init__(self, session_dir: Optional[Path] = None):
        self.session_dir = Path(session_dir or default_session_dir())
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _thread_path(self, thread_id: str) -> Optional[Path]:
        if validate_thread_id(thread_id):
            return None
        safe = thread_id.replace("/", "_")
        return self.session_dir / f"{safe}.json"

    def load(self, thread_id: str) -> Tuple[Optional[SessionRecord], Optional[str]]:
        tid_err = validate_thread_id(thread_id)
        if tid_err:
            return None, tid_err
        path = self._thread_path(thread_id)
        assert path is not None
        if not path.exists():
            return None, None
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None, "session_file_invalid_shape"
            record = SessionRecord.from_dict(data)
            if record.thread_id != thread_id:
                return None, "session_thread_id_mismatch"
            return record, None
        except json.JSONDecodeError:
            return None, "session_file_corrupt"
        except OSError as exc:
            return None, f"session_read_error:{exc}"

    def save(self, record: SessionRecord) -> Optional[str]:
        """保存 session；非法 thread_id 返回错误码，不抛异常。"""
        tid_err = validate_thread_id(record.thread_id)
        if tid_err:
            return tid_err
        path = self._thread_path(record.thread_id)
        if path is None:
            return "invalid_thread_id"
        record = _redact_session_record(record)
        record.updated_at = _utc_now()
        payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=2)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{record.thread_id}.", suffix=".tmp", dir=self.session_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
        return None

    def reset_thread(self, thread_id: str) -> Tuple[bool, Optional[str]]:
        tid_err = validate_thread_id(thread_id)
        if tid_err:
            return False, tid_err
        path = self._thread_path(thread_id)
        if path is None:
            return False, "invalid_thread_id"
        if path.exists():
            try:
                path.unlink()
            except OSError:
                return False, "session_checkpoint_reset_failed"
            return True, None
        return True, None

    def load_or_prepare(self, thread_id: str) -> Tuple[Optional[SessionRecord], Optional[str]]:
        """加载已有 session 或准备新线程；损坏/非法时 fail-closed 返回 (None, error)。"""
        tid_err = validate_thread_id(thread_id)
        if tid_err:
            return None, tid_err
        record, error = self.load(thread_id)
        if error:
            return None, error
        if record is None:
            return SessionRecord.new_thread(thread_id), None
        return record, None

    def get_or_create(self, thread_id: str) -> Tuple[Optional[SessionRecord], Optional[str]]:
        """兼容旧接口；损坏 session 不创建占位记录。"""
        return self.load_or_prepare(thread_id)


def _extract_known_parameters(module_config: Optional[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    known: Dict[str, Any] = {}
    if module_config:
        flat: Dict[str, Any] = {}
        for module, cfg in module_config.items():
            if isinstance(cfg, dict):
                for key, value in cfg.items():
                    if value is None or (isinstance(value, str) and not value.strip()):
                        continue
                    flat[f"{module}.{key}"] = value
        known, _ = redact_recursive(flat)
    intent = state.get("intent")
    if intent:
        known["intent"] = intent
    final_source = state.get("final_plan_source")
    if final_source:
        known["final_plan_source"] = final_source
    return known


def _build_session_summary(record: SessionRecord, *, user_request: str, state: Dict[str, Any], report: Dict[str, Any]) -> str:
    """确定性更新 session_summary；当前事实与本请求优先于历史。"""
    safe_request, _ = redact_text(user_request)
    lines: List[str] = []
    lines.append(f"Latest request: {safe_request[:240]}")
    intent = state.get("intent")
    if intent:
        lines.append(f"Intent: {intent}")
    status = report.get("status") or state.get("execution_status")
    if status:
        lines.append(f"Execution status: {status}")
    decision = state.get("llm_review_decision")
    if decision:
        lines.append(f"LLM review: {decision}")
    plan = state.get("plan") or {}
    steps = plan.get("steps") or []
    if steps:
        caps = [f"{s.get('module')}.{s.get('action')}" for s in steps]
        lines.append(f"Active plan steps: {', '.join(caps)}")
    if record.confirmed_decisions:
        lines.append("Confirmed: " + "; ".join(record.confirmed_decisions[-3:]))
    if state.get("llm_review_decision") == "ask_user":
        summary = state.get("llm_review_summary") or report.get("running_summary")
        if summary:
            safe_summary, _ = redact_text(str(summary))
            lines.append(f"Pending clarification: {safe_summary[:200]}")
    text, _ = redact_text("\n".join(lines))
    return text[:2000]


def _redact_session_record(record: SessionRecord) -> SessionRecord:
    """持久化前对 session 字段脱敏。"""
    summary, _ = redact_text(record.session_summary)
    checkpoint, _ = redact_text(record.latest_checkpoint_summary)
    turns, _ = redact_recursive(record.recent_turns)
    pending, _ = redact_recursive(record.pending_questions)
    confirmed, _ = redact_recursive(record.confirmed_decisions)
    known, _ = redact_recursive(record.known_parameters)
    corrections, _ = redact_recursive(record.corrections)
    artifacts, _ = redact_recursive(record.artifact_refs)
    active = record.active_plan_summary
    if active:
        active, _ = redact_recursive(active)
    record.session_summary = summary
    record.latest_checkpoint_summary = checkpoint
    record.recent_turns = turns
    record.pending_questions = pending
    record.confirmed_decisions = confirmed
    record.known_parameters = known
    record.corrections = corrections
    record.artifact_refs = artifacts
    record.active_plan_summary = active
    return record


def session_error_next_actions(error_type: Optional[str]) -> List[str]:
    if not error_type:
        return []
    if error_type == "invalid_thread_id":
        return ["fix_thread_id"]
    if error_type == "company_profile_thread_reset_required":
        return ["reset_thread"]
    if error_type.startswith("session_read_error"):
        return ["reset_thread", "fix_session_file"]
    return ["reset_thread", "fix_session_file"]


def update_session_from_pipeline(
    record: SessionRecord,
    *,
    user_request: str,
    state: Dict[str, Any],
    report: Dict[str, Any],
    module_config: Optional[Dict[str, Any]] = None,
) -> SessionRecord:
    """从结构化 state/report 更新 session；不写入原始工具输出正文。"""
    safe_request, _ = redact_text(user_request)
    turn_index = record.turn_count + 1
    turn = {
        "turn": turn_index,
        "timestamp": _utc_now(),
        "user_request": safe_request[:500],
        "intent": state.get("intent"),
        "execution_status": report.get("status") or state.get("execution_status"),
        "llm_review_decision": state.get("llm_review_decision"),
        "final_plan_source": state.get("final_plan_source"),
        "summary": redact_text(str(report.get("running_summary") or ""))[0][:400],
    }
    record.recent_turns = (record.recent_turns + [turn])[-MAX_RECENT_TURNS:]
    record.turn_count = turn_index
    record.latest_execution_status = report.get("status") or state.get("execution_status")

    if state.get("llm_review_decision") == "ask_user":
        question = str(state.get("llm_review_summary") or report.get("running_summary") or "").strip()
        safe_q, _ = redact_text(question)
        if safe_q and safe_q not in record.pending_questions:
            record.pending_questions.append(safe_q[:300])
    else:
        record.pending_questions = []

    if state.get("user_confirmed"):
        cap = ", ".join(state.get("selected_capability_ids") or [])
        if cap:
            decision = f"user_confirmed plan: {cap}"
            if decision not in record.confirmed_decisions:
                record.confirmed_decisions.append(decision)

    known = _extract_known_parameters(module_config, state)
    record.known_parameters.update(known)

    plan = state.get("plan")
    if plan:
        redacted_plan, _ = redact_plan_dict(plan)
        record.active_plan_summary = {
            "plan_id": redacted_plan.get("plan_id") if redacted_plan else plan.get("plan_id"),
            "goal": str((redacted_plan or plan).get("goal") or "")[:200],
            "step_count": len((redacted_plan or plan).get("steps") or []),
            "capabilities": [
                f"{s.get('module')}.{s.get('action')}" for s in ((redacted_plan or plan).get("steps") or [])
            ],
        }

    record.artifact_refs = _compact_artifact_refs(list(state.get("artifacts") or []))
    record.session_summary = _build_session_summary(record, user_request=user_request, state=state, report=report)
    record.latest_checkpoint_summary = record.session_summary[:600]
    return _redact_session_record(record)
