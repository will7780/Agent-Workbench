# -*- coding: utf-8 -*-
"""Phase 13.2 — 类型化长期记忆、候选记忆与人工审批（文件型 MemoryStore）。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .redaction import contains_secret_blob, redact_recursive, redact_text

DEFAULT_MEMORY_DIR_NAME = "agent_memory"
FORGOTTEN_TITLE = "[FORGOTTEN]"

MEMORY_TYPES = frozenset(
    {
        "operator_preference",
        "business_feedback",
        "incident_case",
        "reference",
        "user_profile",
        "workflow_preference",
        "store_knowledge",
        "tool_experience",
        "incident_lesson",
    }
)
MEMORY_STATUSES = frozenset({"candidate", "approved", "rejected", "superseded", "forgotten"})
SCOPE_TYPES = frozenset({"user", "store", "workflow", "global"})
ACTIVE_DEDUP_STATUSES = frozenset({"candidate", "approved"})

MAX_TITLE_LEN = 200
MAX_SUMMARY_LEN = 2000
MAX_WHY_LEN = 1500
MAX_HOW_LEN = 1500

ALLOWED_TRANSITIONS: Dict[str, frozenset[str]] = {
    "candidate": frozenset({"approved", "rejected", "forgotten"}),
    "approved": frozenset({"superseded", "forgotten"}),
    "rejected": frozenset({"forgotten"}),
    "superseded": frozenset(),
    "forgotten": frozenset(),
}

BUSINESS_REVIEW_FEEDBACK_CODES = frozenset(
    {
        "as_expected",
        "missing_steps",
        "extra_steps",
        "wrong_tool",
        "wrong_params",
        "wrong_order",
        "other",
    }
)

MEMORY_ID_PATTERN = re.compile(r"^mem_[A-Za-z0-9._-]+$")


def default_memory_dir(base: Optional[Path] = None) -> Path:
    if base is not None:
        return Path(base) / "runs" / DEFAULT_MEMORY_DIR_NAME
    root = Path(os.environ.get("AGENT_WORKBENCH_HOME") or Path.home() / ".agent-workbench").expanduser()
    return root / "memory"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_memory_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"mem_{ts}_{secrets.token_hex(4)}"


def _normalize_fingerprint_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def compute_fingerprint(
    *,
    memory_type: str,
    scope_type: str,
    scope_key: str,
    title: str,
    summary: str,
) -> str:
    raw = "|".join(
        [
            memory_type,
            scope_type,
            _normalize_fingerprint_text(scope_key),
            _normalize_fingerprint_text(title),
            _normalize_fingerprint_text(summary),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _validate_iso_timestamp(value: Any) -> bool:
    if value is None or value == "":
        return True
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return True
    except (TypeError, ValueError):
        return False


def _parse_confidence(value: Any, default: float = 0.5) -> float:
    if value is None:
        return default
    return float(value)


def _parse_version(value: Any, default: int = 1) -> int:
    if value is None:
        return default
    return int(value)


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _sanitize_event_detail(detail: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not detail:
        return {}
    safe, _ = redact_recursive(detail)
    if not isinstance(safe, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in safe.items():
        if isinstance(value, str) and contains_secret_blob(value):
            out[key] = "[REDACTED]"
        else:
            out[key] = value
    return out


def _append_transition_event(
    record: "MemoryRecord",
    event_type: str,
    *,
    from_status: Optional[str],
    to_status: str,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    record.events.append(
        {
            "event_type": event_type,
            "timestamp": _utc_now(),
            "from_status": from_status,
            "to_status": to_status,
            "detail": _sanitize_event_detail(detail),
        }
    )


@dataclass
class MemoryRecord:
    memory_id: str
    memory_type: str
    status: str
    scope_type: str
    scope_key: str
    title: str
    summary: str
    why: str
    how_to_apply: str
    source_run_id: Optional[str] = None
    source_thread_id: Optional[str] = None
    source_turn: Optional[int] = None
    source_business_review: Optional[Dict[str, Any]] = None
    confidence: float = 0.5
    created_at: str = ""
    updated_at: str = ""
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    supersedes_memory_id: Optional[str] = None
    expires_at: Optional[str] = None
    version: int = 1
    fingerprint: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryRecord":
        """解析磁盘记录；类型错误由调用方捕获并映射为 memory_file_invalid_schema。"""
        events_raw = data.get("events")
        if events_raw is not None and not isinstance(events_raw, list):
            raise ValueError("events must be a list")
        br = data.get("source_business_review")
        if br is not None and not isinstance(br, dict):
            raise ValueError("source_business_review must be a dict")

        created_at = data.get("created_at")
        updated_at = data.get("updated_at")
        if created_at is not None and not isinstance(created_at, str):
            raise ValueError("created_at must be a string")
        if updated_at is not None and not isinstance(updated_at, str):
            raise ValueError("updated_at must be a string")

        return cls(
            memory_id=str(data.get("memory_id") or ""),
            memory_type=str(data.get("memory_type") or ""),
            status=str(data.get("status") or "candidate"),
            scope_type=str(data.get("scope_type") or "global"),
            scope_key=str(data.get("scope_key") or ""),
            title=str(data.get("title") or ""),
            summary=str(data.get("summary") or ""),
            why=str(data.get("why") or ""),
            how_to_apply=str(data.get("how_to_apply") or ""),
            source_run_id=data.get("source_run_id"),
            source_thread_id=data.get("source_thread_id"),
            source_turn=_parse_optional_int(data.get("source_turn")),
            source_business_review=br,
            confidence=_parse_confidence(data.get("confidence")),
            created_at=str(created_at or _utc_now()),
            updated_at=str(updated_at or _utc_now()),
            approved_at=data.get("approved_at"),
            approved_by=data.get("approved_by"),
            rejected_at=data.get("rejected_at"),
            rejection_reason=data.get("rejection_reason"),
            supersedes_memory_id=data.get("supersedes_memory_id"),
            expires_at=data.get("expires_at"),
            version=_parse_version(data.get("version")),
            fingerprint=str(data.get("fingerprint") or ""),
            events=list(events_raw or []),
        )


@dataclass
class MemoryListResult:
    memories: List[MemoryRecord]
    load_errors: List[Dict[str, str]]
    index_error_type: Optional[str] = None


def _validate_common_fields(data: Dict[str, Any]) -> Optional[str]:
    mtype = str(data.get("memory_type") or "")
    if mtype and mtype not in MEMORY_TYPES:
        return f"memory_invalid_type:{mtype}"
    scope = str(data.get("scope_type") or "")
    if scope and scope not in SCOPE_TYPES:
        return f"memory_invalid_scope_type:{scope}"
    status = str(data.get("status") or "")
    if status and status not in MEMORY_STATUSES:
        return f"memory_invalid_status:{status}"
    memory_id = data.get("memory_id")
    if memory_id is not None and str(memory_id) and not MEMORY_ID_PATTERN.match(str(memory_id)):
        return "memory_invalid_id"
    conf = data.get("confidence")
    if conf is not None:
        try:
            c = float(conf)
            if c < 0 or c > 1:
                return "memory_invalid_confidence"
        except (TypeError, ValueError):
            return "memory_invalid_confidence"
    turn = data.get("source_turn")
    if turn is not None:
        try:
            if int(turn) < 0:
                return "memory_invalid_source_turn"
        except (TypeError, ValueError):
            return "memory_invalid_source_turn"
    version = data.get("version")
    if version is not None:
        try:
            if int(version) < 1:
                return "memory_invalid_version"
        except (TypeError, ValueError):
            return "memory_invalid_version"
    expires_at = data.get("expires_at")
    if expires_at is not None and expires_at != "" and not _validate_iso_timestamp(expires_at):
        return "memory_invalid_expires_at"
    br = data.get("source_business_review")
    if br is not None and not isinstance(br, dict):
        return "memory_invalid_source_business_review"
    created_at = data.get("created_at")
    updated_at = data.get("updated_at")
    if created_at is not None and created_at != "" and not _validate_iso_timestamp(created_at):
        return "memory_invalid_timestamp"
    if updated_at is not None and updated_at != "" and not _validate_iso_timestamp(updated_at):
        return "memory_invalid_timestamp"
    events = data.get("events")
    if events is not None and not isinstance(events, list):
        return "memory_invalid_events"
    return None


def _validate_content_fields(data: Dict[str, Any], *, partial: bool) -> Optional[str]:
    required_text = ["memory_type", "scope_type", "title", "summary"]
    if not partial:
        required_text.extend(["why", "how_to_apply"])
    for key in required_text:
        if key not in data and not partial:
            return f"memory_missing_field:{key}"
        val = data.get(key)
        if val is None and not partial:
            return f"memory_missing_field:{key}"
        if isinstance(val, str) and not val.strip() and key in required_text:
            return f"memory_empty_field:{key}"
    scope = str(data.get("scope_type") or "")
    if scope and scope != "global":
        scope_key = str(data.get("scope_key") or "").strip()
        if not scope_key:
            return "memory_missing_scope_key"
    for field_name, limit in (
        ("title", MAX_TITLE_LEN),
        ("summary", MAX_SUMMARY_LEN),
        ("why", MAX_WHY_LEN),
        ("how_to_apply", MAX_HOW_LEN),
    ):
        val = data.get(field_name)
        if val is not None and len(str(val)) > limit:
            return f"memory_field_too_long:{field_name}"
    status = str(data.get("status") or "")
    if status in ACTIVE_DEDUP_STATUSES:
        fp = str(data.get("fingerprint") or "").strip()
        if not fp:
            return "memory_missing_fingerprint"
    return None


def _validate_forgotten_fields(data: Dict[str, Any]) -> Optional[str]:
    if str(data.get("status") or "") != "forgotten":
        return "memory_invalid_status:forgotten"
    if str(data.get("title") or "") != FORGOTTEN_TITLE:
        return "memory_invalid_forgotten_title"
    for key in ("summary", "why", "how_to_apply", "scope_key", "fingerprint"):
        if str(data.get(key) or "").strip():
            return f"memory_forgotten_field_not_empty:{key}"
    for key in (
        "source_run_id",
        "source_thread_id",
        "source_turn",
        "source_business_review",
        "approved_by",
        "rejection_reason",
        "supersedes_memory_id",
    ):
        if data.get(key) not in (None, "", []):
            return f"memory_forgotten_source_not_cleared:{key}"
    if not str(data.get("memory_id") or "").strip():
        return "memory_missing_field:memory_id"
    if not str(data.get("created_at") or "").strip():
        return "memory_missing_field:created_at"
    if not str(data.get("updated_at") or "").strip():
        return "memory_missing_field:updated_at"
    try:
        if int(data.get("version") or 0) < 1:
            return "memory_invalid_version"
    except (TypeError, ValueError):
        return "memory_invalid_version"
    events = data.get("events") or []
    if not isinstance(events, list):
        return "memory_invalid_events"
    for event in events:
        if not isinstance(event, dict):
            return "memory_invalid_events"
        detail = event.get("detail") or {}
        if detail:
            return "memory_forgotten_event_detail_not_empty"
    return None


def validate_memory_fields(data: Dict[str, Any], *, partial: bool = False) -> Optional[str]:
    status = str(data.get("status") or "candidate")
    err = _validate_common_fields(data)
    if err:
        return err
    if status == "forgotten":
        return _validate_forgotten_fields(data)
    return _validate_content_fields(data, partial=partial)


def validate_memory_record(record: MemoryRecord) -> Optional[str]:
    return validate_memory_fields(record.to_dict(), partial=False)


def check_sensitive_memory_content(data: Dict[str, Any]) -> Optional[str]:
    """检测敏感内容；返回 memory_sensitive_content 或 None。"""
    text_fields = (
        "title",
        "summary",
        "why",
        "how_to_apply",
        "scope_key",
        "approved_by",
        "rejection_reason",
        "source_run_id",
        "source_thread_id",
    )
    for key in text_fields:
        val = data.get(key)
        if val and contains_secret_blob(str(val)):
            return "memory_sensitive_content"
    br = data.get("source_business_review")
    if isinstance(br, dict):
        for v in br.values():
            if v is not None and contains_secret_blob(str(v)):
                return "memory_sensitive_content"
    detail = data.get("detail")
    if isinstance(detail, dict):
        for v in detail.values():
            if v is not None and contains_secret_blob(str(v)):
                return "memory_sensitive_content"
    return None


def sanitize_memory_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    out: Dict[str, Any] = {}
    redacted_any = False
    for key, value in data.items():
        if isinstance(value, str):
            safe, was = redact_text(value)
            out[key] = safe[
                : {
                    "title": MAX_TITLE_LEN,
                    "summary": MAX_SUMMARY_LEN,
                    "why": MAX_WHY_LEN,
                    "how_to_apply": MAX_HOW_LEN,
                    "scope_key": 128,
                }.get(key, len(safe))
            ]
            redacted_any = redacted_any or was
        elif isinstance(value, dict):
            safe, was = redact_recursive(value)
            out[key] = safe
            redacted_any = redacted_any or was
        else:
            out[key] = value
    return out, redacted_any


def _tombstone_events(prior: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """仅保留最小安全审计结构，剥离业务 detail。"""
    kept: List[Dict[str, Any]] = []
    for event in prior[-2:]:
        if not isinstance(event, dict):
            continue
        kept.append(
            {
                "event_type": str(event.get("event_type") or "unknown"),
                "timestamp": str(event.get("timestamp") or _utc_now()),
                "from_status": event.get("from_status"),
                "to_status": event.get("to_status"),
                "detail": {},
            }
        )
    return kept


def _apply_forget_tombstone(record: MemoryRecord) -> None:
    """清除可召回正文，仅保留最小 tombstone。"""
    record.title = FORGOTTEN_TITLE
    record.summary = ""
    record.why = ""
    record.how_to_apply = ""
    record.source_business_review = None
    record.source_run_id = None
    record.source_thread_id = None
    record.source_turn = None
    record.scope_key = ""
    record.fingerprint = ""
    record.confidence = 0.0
    record.approved_at = None
    record.approved_by = None
    record.rejected_at = None
    record.rejection_reason = None
    record.supersedes_memory_id = None
    record.expires_at = None
    record.events = _tombstone_events(record.events)


class MemoryStore:
    """标准库文件型长期记忆存储（每条 memory 独立 JSON，原子写入，RLock）。"""

    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = Path(memory_dir or default_memory_dir())
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.memory_dir / "_index.json"
        self._lock = threading.RLock()
        self._inject_fail_on: Optional[str] = None

    def _memory_path(self, memory_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", memory_id)
        return self.memory_dir / f"{safe}.json"

    def _load_index_unlocked(self) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        if not self._index_path.exists():
            return {}, None
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("fingerprints"), dict):
                return None, "memory_index_corrupt"
            return {str(k): str(v) for k, v in data["fingerprints"].items()}, None
        except json.JSONDecodeError:
            return None, "memory_index_corrupt"
        except OSError as exc:
            return None, f"memory_read_error:{exc}"

    def _ensure_writable_index(self) -> Optional[str]:
        _, err = self._load_index_unlocked()
        return err

    def _inject_fail(self, stage: str) -> Optional[str]:
        if self._inject_fail_on == stage:
            return "memory_write_error:injected"
        if self._inject_fail_on == "record" and stage in (
            "save_record",
            "supersede_save_candidate",
            "supersede_save_approved",
            "supersede_save_old",
        ):
            return "memory_write_error:injected"
        if self._inject_fail_on == "index" and stage in (
            "save_index",
            "supersede_index_add",
            "supersede_index_final",
        ):
            return "memory_write_error:injected"
        return None

    def _save_index_unlocked(self, fingerprints: Dict[str, str], *, stage: str = "save_index") -> Optional[str]:
        fail = self._inject_fail(stage)
        if fail:
            return fail
        payload = json.dumps({"fingerprints": fingerprints}, ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(prefix=".mem_index.", suffix=".tmp", dir=self.memory_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._index_path)
        except OSError as exc:
            return f"memory_write_error:{exc}"
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return None

    def _atomic_write(self, path: Path, payload: str) -> None:
        fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=self.memory_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _read_file_bytes(self, memory_id: str) -> Optional[bytes]:
        path = self._memory_path(memory_id)
        if not path.exists():
            return None
        return path.read_bytes()

    def _write_file_bytes(self, memory_id: str, content: Optional[bytes]) -> None:
        path = self._memory_path(memory_id)
        if content is None:
            if path.exists():
                path.unlink()
            return
        path.write_bytes(content)

    def _delete_record_file(self, memory_id: str) -> None:
        path = self._memory_path(memory_id)
        if path.exists():
            path.unlink()

    def _load_record_file(self, path: Path) -> Tuple[Optional[MemoryRecord], Optional[str], Optional[str]]:
        memory_id = path.stem
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None, memory_id, "memory_file_invalid_shape"
            try:
                record = MemoryRecord.from_dict(data)
            except (TypeError, ValueError):
                return None, memory_id, "memory_file_invalid_schema"
            schema_err = validate_memory_record(record)
            if schema_err:
                return None, memory_id, "memory_file_invalid_schema"
            if record.memory_id != memory_id:
                return None, memory_id, "memory_id_mismatch"
            return record, memory_id, None
        except json.JSONDecodeError:
            return None, memory_id, "memory_file_corrupt"
        except OSError as exc:
            return None, memory_id, f"memory_read_error:{exc}"

    def _save_record_unlocked(self, record: MemoryRecord, *, stage: str = "save_record") -> Optional[str]:
        fail = self._inject_fail(stage)
        if fail:
            return fail
        path = self._memory_path(record.memory_id)
        payload = json.dumps(record.to_dict(), ensure_ascii=False, indent=2)
        try:
            self._atomic_write(path, payload)
        except OSError as exc:
            return f"memory_write_error:{exc}"
        return None

    def _scan_active_fingerprint_owner(
        self,
        fingerprint: str,
        *,
        exclude_id: Optional[str] = None,
    ) -> Optional[str]:
        if not fingerprint:
            return None
        for path in sorted(self.memory_dir.glob("mem_*.json")):
            record, memory_id, err = self._load_record_file(path)
            if err or record is None:
                continue
            if record.fingerprint != fingerprint:
                continue
            if record.status not in ACTIVE_DEDUP_STATUSES:
                continue
            if exclude_id and record.memory_id == exclude_id:
                continue
            return record.memory_id
        return None

    def _count_approved_on_disk(self) -> int:
        count = 0
        for path in self.memory_dir.glob("mem_*.json"):
            record, _, err = self._load_record_file(path)
            if err or record is None:
                continue
            if record.status == "approved":
                count += 1
        return count

    def rebuild_index(self) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        """从通过校验的记录文件重建索引；冲突或写入失败时不覆盖损坏索引。"""
        with self._lock:
            new_index: Dict[str, str] = {}
            for path in sorted(self.memory_dir.glob("mem_*.json")):
                record, _, err = self._load_record_file(path)
                if err or record is None:
                    continue
                if record.status not in ACTIVE_DEDUP_STATUSES:
                    continue
                if not record.fingerprint:
                    continue
                if record.fingerprint in new_index:
                    return None, "memory_duplicate_fingerprint"
                new_index[record.fingerprint] = record.memory_id
            save_err = self._save_index_unlocked(new_index)
            if save_err:
                return None, save_err
            return new_index, None

    def get(self, memory_id: str) -> Tuple[Optional[MemoryRecord], Optional[str]]:
        with self._lock:
            path = self._memory_path(memory_id)
            if not path.exists():
                return None, "memory_not_found"
            record, _, err = self._load_record_file(path)
            if err:
                return None, err
            return record, None

    def list(
        self,
        *,
        status: Optional[str] = None,
        memory_type: Optional[str] = None,
        scope_type: Optional[str] = None,
        scope_key: Optional[str] = None,
    ) -> MemoryListResult:
        with self._lock:
            _, index_err = self._load_index_unlocked()
            records: List[MemoryRecord] = []
            load_errors: List[Dict[str, str]] = []
            for path in sorted(self.memory_dir.glob("mem_*.json")):
                record, memory_id, err = self._load_record_file(path)
                if err:
                    load_errors.append({"memory_id": memory_id or path.stem, "error_type": err})
                    continue
                assert record is not None
                if status and record.status != status:
                    continue
                if memory_type and record.memory_type != memory_type:
                    continue
                if scope_type and record.scope_type != scope_type:
                    continue
                if scope_key and record.scope_key != scope_key:
                    continue
                records.append(record)
            records.sort(key=lambda r: r.updated_at, reverse=True)
            return MemoryListResult(memories=records, load_errors=load_errors, index_error_type=index_err)

    def create_candidate(self, fields: Dict[str, Any]) -> Tuple[Optional[MemoryRecord], Optional[str]]:
        with self._lock:
            index_err = self._ensure_writable_index()
            if index_err:
                return None, index_err
            err = validate_memory_fields(fields)
            if err:
                return None, err
            sensitive = check_sensitive_memory_content(fields)
            if sensitive:
                return None, sensitive
            safe_fields, _ = sanitize_memory_payload(fields)
            memory_type = str(safe_fields["memory_type"])
            scope_type = str(safe_fields.get("scope_type") or "global")
            scope_key = str(safe_fields.get("scope_key") or "")
            title = str(safe_fields["title"])
            summary = str(safe_fields["summary"])
            fingerprint = compute_fingerprint(
                memory_type=memory_type,
                scope_type=scope_type,
                scope_key=scope_key,
                title=title,
                summary=summary,
            )
            if self._scan_active_fingerprint_owner(fingerprint):
                return None, "memory_duplicate_fingerprint"
            now = _utc_now()
            record = MemoryRecord(
                memory_id=_new_memory_id(),
                memory_type=memory_type,
                status="candidate",
                scope_type=scope_type,
                scope_key=scope_key,
                title=title,
                summary=summary,
                why=str(safe_fields.get("why") or ""),
                how_to_apply=str(safe_fields.get("how_to_apply") or ""),
                source_run_id=safe_fields.get("source_run_id"),
                source_thread_id=safe_fields.get("source_thread_id"),
                source_turn=safe_fields.get("source_turn"),
                source_business_review=safe_fields.get("source_business_review"),
                confidence=float(safe_fields.get("confidence") or 0.5),
                created_at=now,
                updated_at=now,
                expires_at=safe_fields.get("expires_at"),
                version=1,
                fingerprint=fingerprint,
            )
            _append_transition_event(
                record,
                "candidate_created",
                from_status=None,
                to_status="candidate",
                detail={"memory_type": memory_type, "scope_type": scope_type},
            )
            save_err = self._save_record_unlocked(record)
            if save_err:
                return None, save_err
            fingerprints, _ = self._load_index_unlocked()
            assert fingerprints is not None
            fingerprints = dict(fingerprints)
            fingerprints[fingerprint] = record.memory_id
            index_save_err = self._save_index_unlocked(fingerprints)
            if index_save_err:
                self._delete_record_file(record.memory_id)
                return None, index_save_err
            return record, None

    def update_candidate(self, memory_id: str, fields: Dict[str, Any]) -> Tuple[Optional[MemoryRecord], Optional[str]]:
        with self._lock:
            index_err = self._ensure_writable_index()
            if index_err:
                return None, index_err
            record, err = self.get(memory_id)
            if err:
                return None, err
            assert record is not None
            if record.status != "candidate":
                return None, "memory_invalid_transition"
            merged = {**record.to_dict(), **fields}
            merged.pop("memory_id", None)
            merged.pop("status", None)
            merged.pop("events", None)
            err = validate_memory_fields(merged, partial=True)
            if err:
                return None, err
            sensitive = check_sensitive_memory_content(merged)
            if sensitive:
                return None, sensitive
            safe_fields, _ = sanitize_memory_payload(merged)
            old_fp = record.fingerprint
            for key in (
                "memory_type",
                "scope_type",
                "scope_key",
                "title",
                "summary",
                "why",
                "how_to_apply",
                "confidence",
                "expires_at",
                "source_run_id",
                "source_thread_id",
                "source_turn",
                "source_business_review",
            ):
                if key in safe_fields:
                    setattr(record, key, safe_fields[key])
            new_fp = compute_fingerprint(
                memory_type=record.memory_type,
                scope_type=record.scope_type,
                scope_key=record.scope_key,
                title=record.title,
                summary=record.summary,
            )
            fingerprints, _ = self._load_index_unlocked()
            assert fingerprints is not None
            fingerprints = dict(fingerprints)
            if new_fp != old_fp:
                if self._scan_active_fingerprint_owner(new_fp, exclude_id=memory_id):
                    return None, "memory_duplicate_fingerprint"
                if old_fp in fingerprints:
                    del fingerprints[old_fp]
                fingerprints[new_fp] = record.memory_id
                record.fingerprint = new_fp
            record.version += 1
            record.updated_at = _utc_now()
            from_status = record.status
            _append_transition_event(
                record,
                "candidate_updated",
                from_status=from_status,
                to_status=from_status,
            )
            save_err = self._save_record_unlocked(record)
            if save_err:
                return None, save_err
            if new_fp != old_fp:
                index_save_err = self._save_index_unlocked(fingerprints)
                if index_save_err:
                    return None, index_save_err
            return record, None

    def _transition(
        self,
        memory_id: str,
        *,
        target_status: str,
        event_type: str,
        mutator: Callable[[MemoryRecord], None],
        detail_factory: Optional[Callable[[MemoryRecord], Dict[str, Any]]] = None,
        post_save: Optional[Callable[[MemoryRecord], Optional[str]]] = None,
    ) -> Tuple[Optional[MemoryRecord], Optional[str]]:
        with self._lock:
            index_err = self._ensure_writable_index()
            if index_err:
                return None, index_err
            record, err = self.get(memory_id)
            if err:
                return None, err
            assert record is not None
            from_status = record.status
            allowed = ALLOWED_TRANSITIONS.get(from_status, frozenset())
            if target_status not in allowed:
                return None, "memory_invalid_transition"
            mutator(record)
            detail = detail_factory(record) if detail_factory else {}
            sensitive = check_sensitive_memory_content({"detail": detail, **detail})
            if sensitive:
                return None, sensitive
            _append_transition_event(
                record,
                event_type,
                from_status=from_status,
                to_status=target_status,
                detail=detail,
            )
            record.status = target_status
            record.updated_at = _utc_now()
            if target_status == "forgotten":
                schema_err = validate_memory_record(record)
                if schema_err:
                    return None, schema_err
            save_err = self._save_record_unlocked(record)
            if save_err:
                return None, save_err
            if post_save:
                post_err = post_save(record)
                if post_err:
                    return None, post_err
            return record, None

    def approve(self, memory_id: str, *, approved_by: str) -> Tuple[Optional[MemoryRecord], Optional[str]]:
        if contains_secret_blob(approved_by):
            return None, "memory_sensitive_content"

        def _mut(rec: MemoryRecord) -> None:
            rec.approved_at = _utc_now()
            rec.approved_by = approved_by[:128]

        return self._transition(
            memory_id,
            target_status="approved",
            event_type="approved",
            mutator=_mut,
            detail_factory=lambda r: {"approved_by": r.approved_by},
        )

    def reject(self, memory_id: str, *, rejection_reason: str) -> Tuple[Optional[MemoryRecord], Optional[str]]:
        if contains_secret_blob(rejection_reason):
            return None, "memory_sensitive_content"

        def _mut(rec: MemoryRecord) -> None:
            rec.rejected_at = _utc_now()
            rec.rejection_reason = rejection_reason[:500]

        return self._transition(
            memory_id,
            target_status="rejected",
            event_type="rejected",
            mutator=_mut,
            detail_factory=lambda r: {"reason": r.rejection_reason},
        )

    def forget(self, memory_id: str) -> Tuple[Optional[MemoryRecord], Optional[str]]:
        old_fingerprint_holder: Dict[str, Optional[str]] = {"fp": None}

        def _mut(rec: MemoryRecord) -> None:
            old_fingerprint_holder["fp"] = rec.fingerprint
            _apply_forget_tombstone(rec)

        def _post(rec: MemoryRecord) -> Optional[str]:
            old_fp = old_fingerprint_holder["fp"]
            if not old_fp:
                return None
            fingerprints, idx_err = self._load_index_unlocked()
            if idx_err or fingerprints is None:
                return idx_err
            if old_fp in fingerprints and fingerprints[old_fp] == memory_id:
                updated = dict(fingerprints)
                del updated[old_fp]
                return self._save_index_unlocked(updated)
            return None

        return self._transition(
            memory_id,
            target_status="forgotten",
            event_type="forgotten",
            mutator=_mut,
            detail_factory=lambda _r: {},
            post_save=_post,
        )

    def _rollback_supersede(
        self,
        *,
        old_id: str,
        old_bytes: Optional[bytes],
        new_id: Optional[str],
        index_snapshot: Dict[str, str],
    ) -> None:
        self._write_file_bytes(old_id, old_bytes)
        if new_id:
            self._delete_record_file(new_id)
        self._save_index_unlocked(dict(index_snapshot))

    def supersede(
        self,
        memory_id: str,
        new_fields: Dict[str, Any],
    ) -> Tuple[Optional[MemoryRecord], Optional[str]]:
        with self._lock:
            index_err = self._ensure_writable_index()
            if index_err:
                return None, index_err
            old, err = self.get(memory_id)
            if err:
                return None, err
            assert old is not None
            if old.status != "approved":
                return None, "memory_invalid_transition"

            merged = {
                "memory_type": old.memory_type,
                "scope_type": old.scope_type,
                "scope_key": old.scope_key,
                "title": new_fields.get("title") or old.title,
                "summary": new_fields.get("summary") or old.summary,
                "why": new_fields.get("why") or old.why,
                "how_to_apply": new_fields.get("how_to_apply") or old.how_to_apply,
                "confidence": new_fields.get("confidence", old.confidence),
                "expires_at": new_fields.get("expires_at", old.expires_at),
                "source_run_id": new_fields.get("source_run_id", old.source_run_id),
                "source_thread_id": new_fields.get("source_thread_id", old.source_thread_id),
                "source_turn": new_fields.get("source_turn", old.source_turn),
                "source_business_review": new_fields.get("source_business_review", old.source_business_review),
            }
            sensitive = check_sensitive_memory_content(merged)
            if sensitive:
                return None, sensitive
            err = validate_memory_fields(merged)
            if err:
                return None, err
            safe_fields, _ = sanitize_memory_payload(merged)
            fingerprint = compute_fingerprint(
                memory_type=str(safe_fields["memory_type"]),
                scope_type=str(safe_fields["scope_type"]),
                scope_key=str(safe_fields.get("scope_key") or ""),
                title=str(safe_fields["title"]),
                summary=str(safe_fields["summary"]),
            )
            if self._scan_active_fingerprint_owner(fingerprint, exclude_id=old.memory_id):
                return None, "memory_duplicate_fingerprint"

            fingerprints_before, _ = self._load_index_unlocked()
            index_snapshot = dict(fingerprints_before or {})
            old_bytes = self._read_file_bytes(old.memory_id)

            now = _utc_now()
            new_record = MemoryRecord(
                memory_id=_new_memory_id(),
                memory_type=str(safe_fields["memory_type"]),
                status="candidate",
                scope_type=str(safe_fields["scope_type"]),
                scope_key=str(safe_fields.get("scope_key") or ""),
                title=str(safe_fields["title"]),
                summary=str(safe_fields["summary"]),
                why=str(safe_fields.get("why") or ""),
                how_to_apply=str(safe_fields.get("how_to_apply") or ""),
                source_run_id=safe_fields.get("source_run_id"),
                source_thread_id=safe_fields.get("source_thread_id"),
                source_turn=safe_fields.get("source_turn"),
                source_business_review=safe_fields.get("source_business_review"),
                confidence=float(safe_fields.get("confidence") or old.confidence),
                created_at=now,
                updated_at=now,
                expires_at=safe_fields.get("expires_at"),
                version=old.version + 1,
                fingerprint=fingerprint,
                supersedes_memory_id=old.memory_id,
            )
            _append_transition_event(
                new_record,
                "candidate_created",
                from_status=None,
                to_status="candidate",
                detail={"supersedes": old.memory_id},
            )

            # 1) 新 candidate 文件
            save_new_err = self._save_record_unlocked(new_record, stage="supersede_save_candidate")
            if save_new_err:
                return None, save_new_err

            # 2) index 添加新 fingerprint（暂保留旧项）
            fingerprints = dict(index_snapshot)
            fingerprints[fingerprint] = new_record.memory_id
            index_err2 = self._save_index_unlocked(fingerprints, stage="supersede_index_add")
            if index_err2:
                self._rollback_supersede(
                    old_id=old.memory_id,
                    old_bytes=old_bytes,
                    new_id=new_record.memory_id,
                    index_snapshot=index_snapshot,
                )
                return None, index_err2

            # 3) 新记录 approved 落盘
            new_record.approved_at = _utc_now()
            new_record.approved_by = "supersede"
            from_candidate = new_record.status
            _append_transition_event(
                new_record,
                "approved",
                from_status=from_candidate,
                to_status="approved",
                detail={"approved_by": "supersede"},
            )
            new_record.status = "approved"
            new_record.updated_at = _utc_now()
            save_approved_err = self._save_record_unlocked(new_record, stage="supersede_save_approved")
            if save_approved_err:
                self._rollback_supersede(
                    old_id=old.memory_id,
                    old_bytes=old_bytes,
                    new_id=new_record.memory_id,
                    index_snapshot=index_snapshot,
                )
                return None, save_approved_err

            # 4) 旧记录 superseded 落盘
            old_from = old.status
            _append_transition_event(
                old,
                "superseded",
                from_status=old_from,
                to_status="superseded",
                detail={"replaced_by": new_record.memory_id},
            )
            old.status = "superseded"
            old.updated_at = _utc_now()
            save_old_err = self._save_record_unlocked(old, stage="supersede_save_old")
            if save_old_err:
                self._rollback_supersede(
                    old_id=old.memory_id,
                    old_bytes=old_bytes,
                    new_id=new_record.memory_id,
                    index_snapshot=index_snapshot,
                )
                return None, save_old_err

            # 5) index 移除旧 fingerprint
            fingerprints_final = dict(fingerprints)
            if old.fingerprint in fingerprints_final and fingerprints_final[old.fingerprint] == old.memory_id:
                del fingerprints_final[old.fingerprint]
            index_err3 = self._save_index_unlocked(fingerprints_final, stage="supersede_index_final")
            if index_err3:
                self._rollback_supersede(
                    old_id=old.memory_id,
                    old_bytes=old_bytes,
                    new_id=new_record.memory_id,
                    index_snapshot=index_snapshot,
                )
                return None, index_err3

            return new_record, None


def build_candidate_from_business_review(
    run: Dict[str, Any],
    *,
    memory_type: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    why: Optional[str] = None,
    how_to_apply: Optional[str] = None,
    scope_type: str = "workflow",
    scope_key: str = "",
) -> Dict[str, Any]:
    """从 Run + Business Review 构建候选字段（不写入存储）。"""
    review = run.get("business_review") or {}
    reason = str(review.get("reason_code") or "other")
    if memory_type is None:
        memory_type = "incident_case" if review.get("status") == "FAIL" and reason != "as_expected" else "business_feedback"
    run_id = str(run.get("run_id") or "")
    thread_id = run.get("thread_id")
    session = run.get("session_snapshot") or {}
    turn = session.get("turn_count")
    plan = run.get("plan") or {}
    goal = str(plan.get("goal") or run.get("user_request") or "")[:120]
    auto_title = title or f"{memory_type}: {reason} — {goal[:80]}"
    auto_summary = summary or str(review.get("comment") or run.get("message") or goal)[:500]
    auto_why = why or f"Business review {review.get('status')} ({reason}) on run {run_id}"
    auto_how = how_to_apply or "Apply during similar workflow planning after Phase 13.3 recall is enabled."
    return {
        "memory_type": memory_type,
        "scope_type": scope_type,
        "scope_key": scope_key or str(plan.get("plan_type") or "general"),
        "title": auto_title,
        "summary": auto_summary,
        "why": auto_why,
        "how_to_apply": auto_how,
        "source_run_id": run_id,
        "source_thread_id": thread_id,
        "source_turn": turn,
        "source_business_review": {
            "status": review.get("status"),
            "reason_code": reason,
            "comment": str(review.get("comment") or "")[:300],
        },
        "confidence": 0.6 if review.get("status") == "PASS" else 0.4,
    }


def feedback_to_candidate_fields(
    *,
    reason_code: str,
    comment: str = "",
    scope_type: str = "workflow",
    scope_key: str = "",
    source_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """将人工反馈码转为 business_feedback 候选字段。"""
    code = reason_code if reason_code in BUSINESS_REVIEW_FEEDBACK_CODES else "other"
    labels = {
        "as_expected": "符合预期的业务经验",
        "missing_steps": "漏步骤修正",
        "extra_steps": "多余步骤修正",
        "wrong_tool": "工具选择修正",
        "wrong_params": "参数修正",
        "wrong_order": "顺序修正",
        "other": "其他业务反馈",
    }
    return {
        "memory_type": "business_feedback",
        "scope_type": scope_type,
        "scope_key": scope_key,
        "title": labels.get(code, "业务反馈"),
        "summary": comment or labels.get(code, ""),
        "why": f"人工验收反馈：{code}",
        "how_to_apply": "在同类请求规划时参考此反馈（Phase 13.3 召回后生效）",
        "source_run_id": source_run_id,
        "confidence": 0.55,
    }
