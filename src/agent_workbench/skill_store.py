# -*- coding: utf-8 -*-
"""Skill 不可变版本存储。只存哈希、生命周期与受限摘要，不存正文/路径/明文身份。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

from .redaction import contains_secret_blob

DEFAULT_GOVERNANCE_DB_RELATIVE = Path("runs") / "agent_skills" / "governance.db"
LIFECYCLE_STATES = (
    "draft",
    "security_reviewed",
    "eval_passed",
    "active",
    "deprecated",
    "archived",
)
FORWARD_LIFECYCLE = {
    "draft": frozenset({"security_reviewed"}),
    "security_reviewed": frozenset({"eval_passed"}),
    "eval_passed": frozenset(),
    "active": frozenset({"deprecated"}),
    "deprecated": frozenset({"archived"}),
    "archived": frozenset(),
}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_CHECKSUM_RE = re.compile(r"^[a-f0-9]{64}$")
_STABLE_ID_RE = re.compile(r"^skill_[a-f0-9]{24}$")
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "tenant_id",
        "source_id",
        "actor_id",
        "author_id",
        "approver_id",
        "body",
        "path",
        "file_path",
        "resource_path",
        "skill_root",
        "profiles_root",
        "db_path",
        "password",
        "token",
        "api_key",
        "secret",
        "credential",
    }
)


class SkillGovernanceStoreError(Exception):
    error_type = "skill_governance_unavailable"


def default_skill_governance_db(base: Optional[Path] = None) -> Path:
    if base is not None:
        return Path(base) / DEFAULT_GOVERNANCE_DB_RELATIVE
    root = Path(os.environ.get("AGENT_WORKBENCH_HOME") or Path.home() / ".agent-workbench").expanduser()
    return root / "skills" / "governance.db"


def hash_governance_identity(kind: str, value: str) -> str:
    seed = f"{kind}:{value or ''}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"{kind}_{digest}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _contains_forbidden_payload(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key or "").strip().lower()
            if lowered in _FORBIDDEN_PAYLOAD_KEYS or contains_secret_blob(str(key)):
                return True
            if _contains_forbidden_payload(value):
                return True
        return False
    if isinstance(payload, (list, tuple)):
        return any(_contains_forbidden_payload(item) for item in payload)
    text = str(payload or "")
    return bool(text) and contains_secret_blob(text)


def validate_skill_identity_fields(
    *,
    skill_id: str,
    skill_name: str,
    version: str,
    checksum: str,
) -> Optional[str]:
    if not _STABLE_ID_RE.match(str(skill_id or "").strip()):
        return "skill_not_found"
    if not _NAME_RE.match(str(skill_name or "").strip()):
        return "skill_not_found"
    if not _VERSION_RE.match(str(version or "").strip()):
        return "skill_version_mismatch"
    if not _CHECKSUM_RE.match(str(checksum or "").strip()):
        return "skill_checksum_mismatch"
    blob = " ".join([skill_id, skill_name, version, checksum])
    if contains_secret_blob(blob):
        return "skill_secret_detected"
    return None


class SkillVersionStore(Protocol):
    def register_version(self, record: Dict[str, Any]) -> Dict[str, Any]: ...

    def get_version(self, skill_id: str, version: str) -> Optional[Dict[str, Any]]: ...

    def list_versions(self, skill_name: str, tenant_id: str) -> List[Dict[str, Any]]: ...

    def set_lifecycle(
        self,
        skill_id: str,
        version: str,
        status: str,
        actor_id: str,
    ) -> Dict[str, Any]: ...

    def resolve_active(self, skill_name: str, tenant_id: str) -> Optional[Dict[str, Any]]: ...

    def resolve_last_known_good(self, skill_name: str, tenant_id: str) -> Optional[Dict[str, Any]]: ...

    def create_improvement_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]: ...

    def list_improvement_candidates(
        self,
        tenant_id: str,
        skill_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]: ...


class SqliteSkillVersionStore:
    """SQLite 治理库。测试必须注入 tmp 路径，禁止写入 Skill 正文或明文身份。"""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self._db_path = Path(db_path) if db_path is not None else default_skill_governance_db()
        self._lock = threading.RLock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS skill_versions (
                        skill_id TEXT NOT NULL,
                        version TEXT NOT NULL,
                        skill_name TEXT NOT NULL,
                        tenant_hash TEXT NOT NULL,
                        source_hash TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        lifecycle TEXT NOT NULL,
                        is_last_known_good INTEGER NOT NULL DEFAULT 0,
                        author_hash TEXT NOT NULL,
                        eval_summary_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (skill_id, version),
                        UNIQUE (tenant_hash, skill_name, version)
                    );
                    CREATE TABLE IF NOT EXISTS skill_audit_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        skill_id TEXT,
                        version TEXT,
                        skill_name TEXT,
                        tenant_hash TEXT,
                        actor_hash TEXT,
                        action TEXT NOT NULL,
                        from_lifecycle TEXT,
                        to_lifecycle TEXT,
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS skill_improvement_candidates (
                        candidate_id TEXT PRIMARY KEY,
                        tenant_hash TEXT NOT NULL,
                        skill_name TEXT,
                        skill_id TEXT,
                        version TEXT,
                        checksum TEXT,
                        reason TEXT,
                        trigger_reasons_json TEXT NOT NULL DEFAULT '[]',
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.commit()
            except sqlite3.Error as exc:
                raise SkillGovernanceStoreError(str(exc)) from exc
            finally:
                conn.close()

    def _public_version(self, row: sqlite3.Row) -> Dict[str, Any]:
        eval_summary = {}
        raw = row["eval_summary_json"] if "eval_summary_json" in row.keys() else "{}"
        try:
            parsed = json.loads(raw or "{}")
            if isinstance(parsed, dict):
                eval_summary = parsed
        except json.JSONDecodeError:
            eval_summary = {}
        checksum = str(row["checksum"] or "")
        return {
            "skill_id": row["skill_id"],
            "skill_name": row["skill_name"],
            "version": row["version"],
            "checksum": checksum,
            "checksum_digest": checksum[:12],
            "lifecycle": row["lifecycle"],
            "is_last_known_good": bool(row["is_last_known_good"]),
            "tenant_hash": row["tenant_hash"],
            "source_hash": row["source_hash"],
            "author_hash": row["author_hash"],
            "eval_summary": eval_summary,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "error_type": None,
        }

    def _write_audit(
        self,
        conn: sqlite3.Connection,
        *,
        skill_id: Optional[str],
        version: Optional[str],
        skill_name: Optional[str],
        tenant_hash: Optional[str],
        actor_hash: Optional[str],
        action: str,
        from_lifecycle: Optional[str],
        to_lifecycle: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        safe_payload = payload or {}
        if _contains_forbidden_payload(safe_payload):
            safe_payload = {"redacted": True}
        conn.execute(
            """
            INSERT INTO skill_audit_events (
                skill_id, version, skill_name, tenant_hash, actor_hash,
                action, from_lifecycle, to_lifecycle, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill_id,
                version,
                skill_name,
                tenant_hash,
                actor_hash,
                action,
                from_lifecycle,
                to_lifecycle,
                _canonical_json(safe_payload),
                _utc_now(),
            ),
        )

    def register_version(self, record: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(record or {})
        skill_id = str(payload.get("skill_id") or "").strip()
        skill_name = str(payload.get("skill_name") or payload.get("name") or "").strip()
        version = str(payload.get("version") or "").strip()
        checksum = str(payload.get("checksum") or "").strip()
        identity_error = validate_skill_identity_fields(
            skill_id=skill_id,
            skill_name=skill_name,
            version=version,
            checksum=checksum,
        )
        if identity_error:
            return {"record": None, "error_type": identity_error}
        if _contains_forbidden_payload({k: v for k, v in payload.items() if k not in {"tenant_id", "source_id", "author_id"}}):
            return {"record": None, "error_type": "skill_secret_detected"}
        tenant_hash = hash_governance_identity("tenant", str(payload.get("tenant_id") or ""))
        source_hash = hash_governance_identity("source", str(payload.get("source_id") or ""))
        author_hash = hash_governance_identity("actor", str(payload.get("author_id") or ""))
        requested_lifecycle = str(payload.get("lifecycle") or "draft").strip() or "draft"
        if requested_lifecycle != "draft":
            return {"record": None, "error_type": "skill_lifecycle_transition_denied"}
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """
                    SELECT * FROM skill_versions
                    WHERE tenant_hash = ? AND skill_name = ? AND version = ?
                    """,
                    (tenant_hash, skill_name, version),
                ).fetchone()
                if existing is not None:
                    if str(existing["checksum"]) != checksum:
                        conn.rollback()
                        return {"record": None, "error_type": "skill_version_immutable_conflict"}
                    conn.rollback()
                    return {"record": self._public_version(existing), "error_type": None}
                conn.execute(
                    """
                    INSERT INTO skill_versions (
                        skill_id, version, skill_name, tenant_hash, source_hash,
                        checksum, lifecycle, is_last_known_good, author_hash,
                        eval_summary_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, '{}', ?, ?)
                    """,
                    (
                        skill_id,
                        version,
                        skill_name,
                        tenant_hash,
                        source_hash,
                        checksum,
                        "draft",
                        author_hash,
                        now,
                        now,
                    ),
                )
                self._write_audit(
                    conn,
                    skill_id=skill_id,
                    version=version,
                    skill_name=skill_name,
                    tenant_hash=tenant_hash,
                    actor_hash=author_hash,
                    action="register_version",
                    from_lifecycle=None,
                    to_lifecycle="draft",
                )
                conn.commit()
                row = conn.execute(
                    "SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?",
                    (skill_id, version),
                ).fetchone()
                return {"record": self._public_version(row), "error_type": None}
            except sqlite3.Error as exc:
                conn.rollback()
                raise SkillGovernanceStoreError(str(exc)) from exc
            finally:
                conn.close()

    def get_version(self, skill_id: str, version: str) -> Optional[Dict[str, Any]]:
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        "SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?",
                        (str(skill_id or "").strip(), str(version or "").strip()),
                    ).fetchone()
                finally:
                    conn.close()
            if row is None:
                return None
            return self._public_version(row)
        except sqlite3.Error as exc:
            raise SkillGovernanceStoreError(str(exc)) from exc

    def list_versions(self, skill_name: str, tenant_id: str) -> List[Dict[str, Any]]:
        tenant_hash = hash_governance_identity("tenant", str(tenant_id or ""))
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        """
                        SELECT * FROM skill_versions
                        WHERE tenant_hash = ? AND skill_name = ?
                        ORDER BY created_at ASC
                        """,
                        (tenant_hash, str(skill_name or "").strip()),
                    ).fetchall()
                finally:
                    conn.close()
            return [self._public_version(row) for row in rows]
        except sqlite3.Error as exc:
            raise SkillGovernanceStoreError(str(exc)) from exc

    def set_lifecycle(
        self,
        skill_id: str,
        version: str,
        status: str,
        actor_id: str,
    ) -> Dict[str, Any]:
        target = str(status or "").strip()
        actor_hash = hash_governance_identity("actor", str(actor_id or ""))
        if not str(actor_id or "").strip():
            return {"record": None, "error_type": "skill_self_approval_denied"}
        if target not in LIFECYCLE_STATES:
            return {"record": None, "error_type": "skill_lifecycle_transition_denied"}
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?",
                    (str(skill_id or "").strip(), str(version or "").strip()),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return {"record": None, "error_type": "skill_not_found"}
                current = str(row["lifecycle"])
                allowed = FORWARD_LIFECYCLE.get(current, frozenset())
                if target not in allowed:
                    conn.rollback()
                    return {"record": None, "error_type": "skill_lifecycle_transition_denied"}
                now = _utc_now()
                conn.execute(
                    """
                    UPDATE skill_versions
                    SET lifecycle = ?, updated_at = ?
                    WHERE skill_id = ? AND version = ?
                    """,
                    (target, now, row["skill_id"], row["version"]),
                )
                self._write_audit(
                    conn,
                    skill_id=row["skill_id"],
                    version=row["version"],
                    skill_name=row["skill_name"],
                    tenant_hash=row["tenant_hash"],
                    actor_hash=actor_hash,
                    action="set_lifecycle",
                    from_lifecycle=current,
                    to_lifecycle=target,
                )
                conn.commit()
                updated = conn.execute(
                    "SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?",
                    (row["skill_id"], row["version"]),
                ).fetchone()
                return {"record": self._public_version(updated), "error_type": None}
            except sqlite3.Error as exc:
                conn.rollback()
                raise SkillGovernanceStoreError(str(exc)) from exc
            finally:
                conn.close()

    def resolve_active(self, skill_name: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        tenant_hash = hash_governance_identity("tenant", str(tenant_id or ""))
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        """
                        SELECT * FROM skill_versions
                        WHERE tenant_hash = ? AND skill_name = ? AND lifecycle = 'active'
                        LIMIT 1
                        """,
                        (tenant_hash, str(skill_name or "").strip()),
                    ).fetchone()
                finally:
                    conn.close()
            return self._public_version(row) if row is not None else None
        except sqlite3.Error as exc:
            raise SkillGovernanceStoreError(str(exc)) from exc

    def resolve_last_known_good(self, skill_name: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        tenant_hash = hash_governance_identity("tenant", str(tenant_id or ""))
        try:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute(
                        """
                        SELECT * FROM skill_versions
                        WHERE tenant_hash = ? AND skill_name = ? AND is_last_known_good = 1
                        LIMIT 1
                        """,
                        (tenant_hash, str(skill_name or "").strip()),
                    ).fetchone()
                finally:
                    conn.close()
            return self._public_version(row) if row is not None else None
        except sqlite3.Error as exc:
            raise SkillGovernanceStoreError(str(exc)) from exc

    def create_improvement_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(candidate or {})
        tenant_id = str(payload.pop("tenant_id", "") or "")
        tenant_hash = str(payload.get("tenant_hash") or hash_governance_identity("tenant", tenant_id))
        skill_name = str(payload.get("skill_name") or "").strip() or None
        skill_id = str(payload.get("skill_id") or "").strip() or None
        version = str(payload.get("version") or "").strip() or None
        checksum = str(payload.get("checksum") or "").strip() or None
        reason = str(payload.get("reason") or "").strip() or None
        trigger_reasons = payload.get("trigger_reasons") or []
        if not isinstance(trigger_reasons, list):
            trigger_reasons = [str(trigger_reasons)]
        stored_payload = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "tenant_hash",
                "skill_name",
                "skill_id",
                "version",
                "checksum",
                "reason",
                "trigger_reasons",
                "candidate_id",
            }
        }
        if _contains_forbidden_payload(stored_payload) or (
            checksum and contains_secret_blob(checksum)
        ):
            return {"candidate": None, "error_type": "skill_secret_detected"}
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if not candidate_id:
            digest = hashlib.sha256(
                f"{tenant_hash}:{skill_id}:{version}:{reason}:{_utc_now()}".encode("utf-8")
            ).hexdigest()[:20]
            candidate_id = f"skcand_{digest}"
        now = _utc_now()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO skill_improvement_candidates (
                        candidate_id, tenant_hash, skill_name, skill_id, version,
                        checksum, reason, trigger_reasons_json, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        tenant_hash,
                        skill_name,
                        skill_id,
                        version,
                        checksum,
                        reason,
                        _canonical_json(trigger_reasons),
                        _canonical_json(stored_payload),
                        now,
                    ),
                )
                self._write_audit(
                    conn,
                    skill_id=skill_id,
                    version=version,
                    skill_name=skill_name,
                    tenant_hash=tenant_hash,
                    actor_hash=None,
                    action="create_improvement_candidate",
                    from_lifecycle=None,
                    to_lifecycle=None,
                    payload={"candidate_id": candidate_id, "reason": reason},
                )
                conn.commit()
                return {
                    "candidate": self._public_candidate(
                        {
                            "candidate_id": candidate_id,
                            "tenant_hash": tenant_hash,
                            "skill_name": skill_name,
                            "skill_id": skill_id,
                            "version": version,
                            "checksum": checksum,
                            "reason": reason,
                            "trigger_reasons_json": _canonical_json(trigger_reasons),
                            "payload_json": _canonical_json(stored_payload),
                            "created_at": now,
                        }
                    ),
                    "candidate_id": candidate_id,
                    "error_type": None,
                }
            except sqlite3.Error as exc:
                conn.rollback()
                raise SkillGovernanceStoreError(str(exc)) from exc
            finally:
                conn.close()

    def _public_candidate(self, row: Any) -> Dict[str, Any]:
        if isinstance(row, sqlite3.Row):
            mapping = {key: row[key] for key in row.keys()}
        else:
            mapping = dict(row)
        triggers = []
        try:
            parsed = json.loads(mapping.get("trigger_reasons_json") or "[]")
            if isinstance(parsed, list):
                triggers = parsed
        except json.JSONDecodeError:
            triggers = []
        extra = {}
        try:
            parsed_payload = json.loads(mapping.get("payload_json") or "{}")
            if isinstance(parsed_payload, dict):
                extra = parsed_payload
        except json.JSONDecodeError:
            extra = {}
        checksum = str(mapping.get("checksum") or "")
        return {
            "candidate_id": mapping.get("candidate_id"),
            "tenant_hash": mapping.get("tenant_hash"),
            "skill_name": mapping.get("skill_name"),
            "skill_id": mapping.get("skill_id"),
            "version": mapping.get("version"),
            "checksum": checksum,
            "checksum_digest": checksum[:12] if checksum else None,
            "reason": mapping.get("reason"),
            "trigger_reasons": triggers,
            "payload": extra,
            "created_at": mapping.get("created_at"),
            "error_type": None,
        }

    def list_improvement_candidates(
        self,
        tenant_id: str,
        skill_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        tenant_hash = hash_governance_identity("tenant", str(tenant_id or ""))
        try:
            with self._lock:
                conn = self._connect()
                try:
                    if skill_name:
                        rows = conn.execute(
                            """
                            SELECT * FROM skill_improvement_candidates
                            WHERE tenant_hash = ? AND skill_name = ?
                            ORDER BY created_at DESC
                            """,
                            (tenant_hash, str(skill_name).strip()),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """
                            SELECT * FROM skill_improvement_candidates
                            WHERE tenant_hash = ?
                            ORDER BY created_at DESC
                            """,
                            (tenant_hash,),
                        ).fetchall()
                finally:
                    conn.close()
            return [self._public_candidate(row) for row in rows]
        except sqlite3.Error as exc:
            raise SkillGovernanceStoreError(str(exc)) from exc

    def list_audit_events(
        self,
        *,
        skill_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        tenant_hash = hash_governance_identity("tenant", str(tenant_id or "")) if tenant_id else None
        clauses = []
        params: List[Any] = []
        if tenant_hash:
            clauses.append("tenant_hash = ?")
            params.append(tenant_hash)
        if skill_name:
            clauses.append("skill_name = ?")
            params.append(str(skill_name).strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with self._lock:
                conn = self._connect()
                try:
                    rows = conn.execute(
                        f"""
                        SELECT event_id, skill_id, version, skill_name, tenant_hash,
                               actor_hash, action, from_lifecycle, to_lifecycle,
                               payload_json, created_at
                        FROM skill_audit_events
                        {where}
                        ORDER BY event_id DESC
                        LIMIT ?
                        """,
                        (*params, max(1, int(limit))),
                    ).fetchall()
                finally:
                    conn.close()
            events = []
            for row in rows:
                payload = {}
                try:
                    parsed = json.loads(row["payload_json"] or "{}")
                    if isinstance(parsed, dict):
                        payload = parsed
                except json.JSONDecodeError:
                    payload = {}
                events.append(
                    {
                        "event_id": row["event_id"],
                        "skill_id": row["skill_id"],
                        "version": row["version"],
                        "skill_name": row["skill_name"],
                        "tenant_hash": row["tenant_hash"],
                        "actor_hash": row["actor_hash"],
                        "action": row["action"],
                        "from_lifecycle": row["from_lifecycle"],
                        "to_lifecycle": row["to_lifecycle"],
                        "payload": payload,
                        "created_at": row["created_at"],
                    }
                )
            return events
        except sqlite3.Error as exc:
            raise SkillGovernanceStoreError(str(exc)) from exc

    def promote_version_atomic(
        self,
        *,
        skill_id: str,
        version: str,
        actor_id: str,
        eval_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        actor_hash = hash_governance_identity("actor", str(actor_id or ""))
        summary = _restricted_eval_summary(eval_summary)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                target = conn.execute(
                    "SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?",
                    (str(skill_id or "").strip(), str(version or "").strip()),
                ).fetchone()
                if target is None:
                    conn.rollback()
                    return {"record": None, "error_type": "skill_not_found"}
                now = _utc_now()
                previous = conn.execute(
                    """
                    SELECT * FROM skill_versions
                    WHERE tenant_hash = ? AND skill_name = ? AND lifecycle = 'active'
                    """,
                    (target["tenant_hash"], target["skill_name"]),
                ).fetchone()
                if previous is not None and (
                    previous["skill_id"] != target["skill_id"] or previous["version"] != target["version"]
                ):
                    conn.execute(
                        """
                        UPDATE skill_versions
                        SET is_last_known_good = 0, updated_at = ?
                        WHERE tenant_hash = ? AND skill_name = ? AND is_last_known_good = 1
                        """,
                        (now, target["tenant_hash"], target["skill_name"]),
                    )
                    conn.execute(
                        """
                        UPDATE skill_versions
                        SET lifecycle = 'deprecated', is_last_known_good = 1, updated_at = ?
                        WHERE skill_id = ? AND version = ?
                        """,
                        (now, previous["skill_id"], previous["version"]),
                    )
                    self._write_audit(
                        conn,
                        skill_id=previous["skill_id"],
                        version=previous["version"],
                        skill_name=previous["skill_name"],
                        tenant_hash=previous["tenant_hash"],
                        actor_hash=actor_hash,
                        action="promote_demote_previous_active",
                        from_lifecycle="active",
                        to_lifecycle="deprecated",
                        payload={"last_known_good": True},
                    )
                conn.execute(
                    """
                    UPDATE skill_versions
                    SET lifecycle = 'active', eval_summary_json = ?, updated_at = ?
                    WHERE skill_id = ? AND version = ?
                    """,
                    (_canonical_json(summary), now, target["skill_id"], target["version"]),
                )
                self._write_audit(
                    conn,
                    skill_id=target["skill_id"],
                    version=target["version"],
                    skill_name=target["skill_name"],
                    tenant_hash=target["tenant_hash"],
                    actor_hash=actor_hash,
                    action="promote_skill_version",
                    from_lifecycle=target["lifecycle"],
                    to_lifecycle="active",
                    payload={"eval_summary": summary},
                )
                conn.commit()
                updated = conn.execute(
                    "SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?",
                    (target["skill_id"], target["version"]),
                ).fetchone()
                lkg = conn.execute(
                    """
                    SELECT * FROM skill_versions
                    WHERE tenant_hash = ? AND skill_name = ? AND is_last_known_good = 1
                    LIMIT 1
                    """,
                    (target["tenant_hash"], target["skill_name"]),
                ).fetchone()
                return {
                    "record": self._public_version(updated),
                    "last_known_good": self._public_version(lkg) if lkg is not None else None,
                    "error_type": None,
                }
            except sqlite3.Error as exc:
                conn.rollback()
                raise SkillGovernanceStoreError(str(exc)) from exc
            finally:
                conn.close()

    def rollback_version_atomic(
        self,
        *,
        skill_name: str,
        tenant_id: str,
        actor_id: str,
    ) -> Dict[str, Any]:
        actor_hash = hash_governance_identity("actor", str(actor_id or ""))
        tenant_hash = hash_governance_identity("tenant", str(tenant_id or ""))
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                lkg = conn.execute(
                    """
                    SELECT * FROM skill_versions
                    WHERE tenant_hash = ? AND skill_name = ? AND is_last_known_good = 1
                    LIMIT 1
                    """,
                    (tenant_hash, str(skill_name or "").strip()),
                ).fetchone()
                if lkg is None:
                    conn.rollback()
                    return {"record": None, "error_type": "skill_no_last_known_good"}
                now = _utc_now()
                current = conn.execute(
                    """
                    SELECT * FROM skill_versions
                    WHERE tenant_hash = ? AND skill_name = ? AND lifecycle = 'active'
                    LIMIT 1
                    """,
                    (tenant_hash, str(skill_name or "").strip()),
                ).fetchone()
                if current is not None:
                    conn.execute(
                        """
                        UPDATE skill_versions
                        SET lifecycle = 'deprecated', is_last_known_good = 0, updated_at = ?
                        WHERE skill_id = ? AND version = ?
                        """,
                        (now, current["skill_id"], current["version"]),
                    )
                    self._write_audit(
                        conn,
                        skill_id=current["skill_id"],
                        version=current["version"],
                        skill_name=current["skill_name"],
                        tenant_hash=tenant_hash,
                        actor_hash=actor_hash,
                        action="rollback_demote_active",
                        from_lifecycle="active",
                        to_lifecycle="deprecated",
                    )
                conn.execute(
                    """
                    UPDATE skill_versions
                    SET lifecycle = 'active', updated_at = ?
                    WHERE skill_id = ? AND version = ?
                    """,
                    (now, lkg["skill_id"], lkg["version"]),
                )
                self._write_audit(
                    conn,
                    skill_id=lkg["skill_id"],
                    version=lkg["version"],
                    skill_name=lkg["skill_name"],
                    tenant_hash=tenant_hash,
                    actor_hash=actor_hash,
                    action="rollback_skill_version",
                    from_lifecycle=lkg["lifecycle"],
                    to_lifecycle="active",
                    payload={"restored_last_known_good": True},
                )
                conn.commit()
                restored = conn.execute(
                    "SELECT * FROM skill_versions WHERE skill_id = ? AND version = ?",
                    (lkg["skill_id"], lkg["version"]),
                ).fetchone()
                return {"record": self._public_version(restored), "error_type": None}
            except sqlite3.Error as exc:
                conn.rollback()
                raise SkillGovernanceStoreError(str(exc)) from exc
            finally:
                conn.close()


def _restricted_eval_summary(eval_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    allowed = (
        "skill_trigger_precision",
        "skill_trigger_recall",
        "skill_top_k_accuracy",
        "skill_false_activation_rate",
        "skill_coexistence_conflict_rate",
        "skill_instruction_follow_rate",
        "skill_tool_alignment",
        "cross_company_leak_rate",
        "policy_violation_rate",
        "skill_context_chars",
        "outcome_score_delta",
        "case_count",
        "promotion_gate_passed",
    )
    source = eval_summary if isinstance(eval_summary, dict) else {}
    out: Dict[str, Any] = {}
    for key in allowed:
        if key in source:
            out[key] = source[key]
    return out
