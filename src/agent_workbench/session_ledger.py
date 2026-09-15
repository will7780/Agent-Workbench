# -*- coding: utf-8 -*-
"""Phase 1 — Session Ledger：本地全量会话历史库（SQLite + FTS），fail-open 写入。"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .memory_store import default_memory_dir
from .redaction import contains_secret_blob, redact_recursive, redact_text

DEFAULT_LEDGER_DB_NAME = "state.db"
DEFAULT_SEARCH_TOP_K = 5
MAX_SNIPPET_CHARS = 240
MAX_SUMMARY_CHARS = 2000
MAX_ARGS_CHARS = 800

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def default_ledger_db_path(base: Optional[Path] = None) -> Path:
    return default_memory_dir(base) / DEFAULT_LEDGER_DB_NAME


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{secrets.token_hex(4)}"


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_json(data: Any, *, max_chars: int = MAX_ARGS_CHARS) -> str:
    try:
        raw = json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        raw = str(data)
    return _truncate(raw, max_chars)


def _redact_and_store_text(text: str, *, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    cleaned, _ = redact_text(text or "")
    if contains_secret_blob(cleaned):
        cleaned = "[REDACTED]"
    return _truncate(cleaned, max_chars)


def _redact_and_store_obj(data: Any) -> Any:
    cleaned, _ = redact_recursive(data)
    if isinstance(cleaned, str):
        return _redact_and_store_text(cleaned)
    return cleaned


def _fts_query(text: str) -> str:
    tokens = _TOKEN_RE.findall(text or "")
    if not tokens:
        return ""
    parts = []
    for token in tokens[:12]:
        safe = token.replace('"', "")
        if len(safe) >= 1:
            parts.append(f'"{safe}"')
    return " OR ".join(parts)


def _opaque_binding_id(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _row_field(row: Any, key: str) -> Optional[str]:
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return None
    return _opaque_binding_id(value if value is not None else None)


def _session_isolation_sql(
    *,
    thread_id: Optional[str],
    company_profile_binding_id: Optional[str],
    table_alias: str,
) -> Tuple[str, List[Any]]:
    prefix = f"{table_alias}." if table_alias else ""
    clauses: List[str] = []
    params: List[Any] = []
    requested_thread = _opaque_binding_id(thread_id)
    requested_binding = _opaque_binding_id(company_profile_binding_id)
    if requested_thread:
        clauses.append(f"{prefix}thread_id = ?")
        params.append(requested_thread)
    if requested_binding:
        clauses.append(f"{prefix}company_profile_binding_id = ?")
        params.append(requested_binding)
    if not clauses:
        return "", params
    return " AND " + " AND ".join(clauses), params


def _row_matches_isolation(
    row: Any,
    *,
    thread_id: Optional[str],
    company_profile_binding_id: Optional[str],
) -> bool:
    requested_thread = _opaque_binding_id(thread_id)
    requested_binding = _opaque_binding_id(company_profile_binding_id)
    row_thread = _row_field(row, "thread_id")
    row_binding = _row_field(row, "company_profile_binding_id")
    if requested_thread and row_thread != requested_thread:
        return False
    if requested_binding and row_binding != requested_binding:
        return False
    return True


@dataclass
class SessionSearchHit:
    run_id: str
    thread_id: Optional[str]
    match_reason: str
    snippet: str
    score: float
    user_request_preview: str
    created_at: Optional[str] = None
    included_in_llm: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "match_reason": self.match_reason,
            "snippet": self.snippet,
            "score": self.score,
            "user_request_preview": self.user_request_preview,
            "created_at": self.created_at,
            "included_in_llm": self.included_in_llm,
        }


@dataclass
class SessionSearchResult:
    enabled: bool = True
    used: bool = False
    query: str = ""
    retrieved_sessions: List[Dict[str, Any]] = field(default_factory=list)
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "used": self.used,
            "query": self.query,
            "retrieved_sessions": list(self.retrieved_sessions),
            "error_type": self.error_type,
        }


class SessionLedger:
    """SQLite session ledger；初始化或写入失败时 fail-open（不抛异常阻断主流程）。"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or default_ledger_db_path())
        self._lock = threading.RLock()
        self._healthy = True
        self._error: Optional[str] = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
        except Exception as exc:
            self._healthy = False
            self._error = f"session_ledger_init_failed:{exc}"

    @property
    def healthy(self) -> bool:
        return self._healthy

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        run_id TEXT PRIMARY KEY,
                        thread_id TEXT,
                        turn_index INTEGER,
                        agent_mode TEXT,
                        execution_mode TEXT,
                        status TEXT,
                        user_request TEXT,
                        final_response TEXT,
                        verification_summary TEXT,
                        created_at TEXT,
                        company_profile_binding_id TEXT
                    );
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        role TEXT,
                        content TEXT,
                        FOREIGN KEY(run_id) REFERENCES sessions(run_id)
                    );
                    CREATE TABLE IF NOT EXISTS tool_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        round_idx INTEGER,
                        tool_name TEXT,
                        capability_id TEXT,
                        arguments_summary TEXT,
                        guard_allowed INTEGER,
                        guard_error_type TEXT,
                        adapter_status TEXT,
                        FOREIGN KEY(run_id) REFERENCES sessions(run_id)
                    );
                    CREATE TABLE IF NOT EXISTS observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        step_id TEXT,
                        status TEXT,
                        summary TEXT,
                        artifact_refs_json TEXT,
                        error_type TEXT,
                        FOREIGN KEY(run_id) REFERENCES sessions(run_id)
                    );
                    CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
                        run_id UNINDEXED,
                        thread_id UNINDEXED,
                        search_text,
                        tokenize='trigram'
                    );
                    """
                )
                self._ensure_company_profile_binding_column(conn)
                conn.commit()

    def _ensure_company_profile_binding_column(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(sessions)").fetchall()
        names = set()
        for row in rows:
            try:
                names.add(str(row["name"]))
            except (KeyError, IndexError, TypeError):
                names.add(str(row[1]))
        if "company_profile_binding_id" not in names:
            conn.execute("ALTER TABLE sessions ADD COLUMN company_profile_binding_id TEXT")

    def _next_turn_index(self, conn: sqlite3.Connection, thread_id: Optional[str]) -> int:
        if not thread_id:
            return 0
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return int(row["c"] if row else 0)

    def reset_thread(self, thread_id: Optional[str]) -> Tuple[bool, Optional[str]]:
        """事务删除该 Thread 的 sessions/messages/tool_calls/observations/FTS。失败结构化返回。"""
        tid = str(thread_id or "").strip()
        if not tid:
            return False, "invalid_thread_id"
        if not self._healthy:
            return False, "session_ledger_reset_failed"
        try:
            with self._lock:
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    run_ids = [
                        str(row["run_id"])
                        for row in conn.execute(
                            "SELECT run_id FROM sessions WHERE thread_id = ?",
                            (tid,),
                        ).fetchall()
                    ]
                    if run_ids:
                        placeholders = ",".join("?" * len(run_ids))
                        conn.execute(
                            f"DELETE FROM messages WHERE run_id IN ({placeholders})",
                            run_ids,
                        )
                        conn.execute(
                            f"DELETE FROM tool_calls WHERE run_id IN ({placeholders})",
                            run_ids,
                        )
                        conn.execute(
                            f"DELETE FROM observations WHERE run_id IN ({placeholders})",
                            run_ids,
                        )
                        conn.execute(
                            f"DELETE FROM session_fts WHERE run_id IN ({placeholders})",
                            run_ids,
                        )
                    conn.execute("DELETE FROM sessions WHERE thread_id = ?", (tid,))
                    conn.execute("DELETE FROM session_fts WHERE thread_id = ?", (tid,))
                    conn.commit()
            return True, None
        except Exception:
            return False, "session_ledger_reset_failed"

    def record_run(
        self,
        *,
        run_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        agent_mode: str = "langgraph",
        execution_mode: str = "dry_run",
        status: str = "completed",
        user_request: str = "",
        final_response: Optional[str] = None,
        verification: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
        observations: Optional[Sequence[Dict[str, Any]]] = None,
        artifacts: Optional[Sequence[Dict[str, Any]]] = None,
        company_profile_binding_id: Optional[str] = None,
    ) -> Optional[str]:
        if not self._healthy:
            return None
        rid = run_id or _new_run_id()
        binding_id = _opaque_binding_id(company_profile_binding_id)
        safe_user = _redact_and_store_text(user_request)
        safe_final = _redact_and_store_text(final_response or "")
        verification_summary = _build_verification_summary(verification)
        search_parts = [safe_user, safe_final, verification_summary]

        try:
            with self._lock:
                with self._connect() as conn:
                    turn_index = self._next_turn_index(conn, thread_id)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO sessions (
                            run_id, thread_id, turn_index, agent_mode, execution_mode,
                            status, user_request, final_response, verification_summary, created_at,
                            company_profile_binding_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rid,
                            thread_id,
                            turn_index,
                            agent_mode,
                            execution_mode,
                            status,
                            safe_user,
                            safe_final,
                            verification_summary,
                            _utc_now(),
                            binding_id,
                        ),
                    )
                    conn.execute("DELETE FROM messages WHERE run_id = ?", (rid,))
                    conn.execute("DELETE FROM tool_calls WHERE run_id = ?", (rid,))
                    conn.execute("DELETE FROM observations WHERE run_id = ?", (rid,))
                    conn.execute("DELETE FROM session_fts WHERE run_id = ?", (rid,))

                    if safe_user:
                        conn.execute(
                            "INSERT INTO messages (run_id, role, content) VALUES (?, ?, ?)",
                            (rid, "user", safe_user),
                        )
                        search_parts.append(safe_user)
                    if safe_final:
                        conn.execute(
                            "INSERT INTO messages (run_id, role, content) VALUES (?, ?, ?)",
                            (rid, "assistant", safe_final),
                        )

                    for tc in tool_calls or []:
                        guard = tc.get("guard") or {}
                        args = _redact_and_store_obj(tc.get("arguments") or {})
                        args_summary = _safe_json(args)
                        cap = tc.get("capability_id") or ""
                        tool_name = str(tc.get("tool_name") or "")
                        adapter_status = str(
                            tc.get("observation_status")
                            or tc.get("adapter_status")
                            or ""
                        )
                        guard_allowed = 1 if guard.get("allowed") else 0
                        guard_error = guard.get("error_type")
                        conn.execute(
                            """
                            INSERT INTO tool_calls (
                                run_id, round_idx, tool_name, capability_id,
                                arguments_summary, guard_allowed, guard_error_type, adapter_status
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rid,
                                int(tc.get("round") or 0),
                                tool_name,
                                cap,
                                args_summary,
                                guard_allowed,
                                guard_error,
                                adapter_status,
                            ),
                        )
                        tool_line = f"{tool_name} {cap} {args_summary} {adapter_status}"
                        search_parts.append(_truncate(tool_line, 400))

                    for obs in observations or []:
                        step_id = str(obs.get("step_id") or "")
                        obs_status = str(obs.get("status") or "")
                        summary = _redact_and_store_text(str(obs.get("summary") or ""))
                        err = obs.get("error") or {}
                        error_type = err.get("type") if isinstance(err, dict) else None
                        artifact_refs = _extract_artifact_refs(obs, artifacts or [])
                        conn.execute(
                            """
                            INSERT INTO observations (
                                run_id, step_id, status, summary, artifact_refs_json, error_type
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                rid,
                                step_id,
                                obs_status,
                                summary,
                                json.dumps(artifact_refs, ensure_ascii=False),
                                error_type,
                            ),
                        )
                        if summary:
                            search_parts.append(summary)

                    search_text = _redact_and_store_text("\n".join(p for p in search_parts if p), max_chars=4000)
                    conn.execute(
                        "INSERT INTO session_fts (run_id, thread_id, search_text) VALUES (?, ?, ?)",
                        (rid, thread_id or "", search_text),
                    )
                    conn.commit()
            return rid
        except Exception:
            return None

    def search(
        self,
        query: str,
        *,
        thread_id: Optional[str] = None,
        company_profile_binding_id: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_TOP_K,
        exclude_run_id: Optional[str] = None,
    ) -> List[SessionSearchHit]:
        if not self._healthy:
            return []
        hits = self._search_fts(
            query,
            thread_id=thread_id,
            company_profile_binding_id=company_profile_binding_id,
            limit=limit,
            exclude_run_id=exclude_run_id,
        )
        if not hits:
            hits = self._search_like(
                query,
                thread_id=thread_id,
                company_profile_binding_id=company_profile_binding_id,
                limit=limit,
                exclude_run_id=exclude_run_id,
            )
        return hits

    def _search_fts(
        self,
        query: str,
        *,
        thread_id: Optional[str] = None,
        company_profile_binding_id: Optional[str] = None,
        limit: int,
        exclude_run_id: Optional[str] = None,
    ) -> List[SessionSearchHit]:
        fts_q = _fts_query(query)
        if not fts_q:
            return []
        isolation_sql, isolation_params = _session_isolation_sql(
            thread_id=thread_id,
            company_profile_binding_id=company_profile_binding_id,
            table_alias="s",
        )
        try:
            with self._lock:
                with self._connect() as conn:
                    rows = conn.execute(
                        f"""
                        SELECT f.run_id, s.thread_id, f.search_text, s.user_request, s.created_at,
                               s.company_profile_binding_id, bm25(session_fts) AS rank
                        FROM session_fts f
                        LEFT JOIN sessions s ON s.run_id = f.run_id
                        WHERE session_fts MATCH ?{isolation_sql}
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_q, *isolation_params, max(1, min(limit, 20))),
                    ).fetchall()
            return self._rows_to_hits(
                rows,
                query,
                thread_id=thread_id,
                company_profile_binding_id=company_profile_binding_id,
                exclude_run_id=exclude_run_id,
                limit=limit,
                match_reason="fts_keyword",
            )
        except Exception:
            return []

    def _search_like(
        self,
        query: str,
        *,
        thread_id: Optional[str] = None,
        company_profile_binding_id: Optional[str] = None,
        limit: int,
        exclude_run_id: Optional[str] = None,
    ) -> List[SessionSearchHit]:
        tokens = [t for t in _TOKEN_RE.findall(query or "") if len(t) >= 2]
        if not tokens:
            tokens = [query.strip()] if query and query.strip() else []
        if not tokens:
            return []
        isolation_sql, isolation_params = _session_isolation_sql(
            thread_id=thread_id,
            company_profile_binding_id=company_profile_binding_id,
            table_alias="",
        )
        try:
            with self._lock:
                with self._connect() as conn:
                    rows = conn.execute(
                        f"""
                        SELECT run_id, thread_id, user_request, final_response, created_at,
                               company_profile_binding_id
                        FROM sessions
                        WHERE 1=1{isolation_sql}
                        ORDER BY created_at DESC
                        LIMIT 200
                        """,
                        tuple(isolation_params),
                    ).fetchall()
            scored: List[Tuple[int, Any]] = []
            for row in rows:
                rid = row["run_id"]
                if exclude_run_id and rid == exclude_run_id:
                    continue
                if not _row_matches_isolation(
                    row,
                    thread_id=thread_id,
                    company_profile_binding_id=company_profile_binding_id,
                ):
                    continue
                text = f"{row['user_request'] or ''} {row['final_response'] or ''}"
                score = sum(1 for t in tokens if t in text)
                if score > 0:
                    scored.append((score, row))
            scored.sort(key=lambda x: (-x[0], x[1]["created_at"] or ""))
            hits: List[SessionSearchHit] = []
            for score, row in scored[:limit]:
                snippet = _make_snippet(row["user_request"] or row["final_response"] or "", query)
                hits.append(
                    SessionSearchHit(
                        run_id=row["run_id"],
                        thread_id=row["thread_id"] or None,
                        match_reason="keyword_overlap",
                        snippet=snippet,
                        score=float(score),
                        user_request_preview=_truncate(row["user_request"] or "", 120),
                        created_at=row["created_at"],
                        included_in_llm=False,
                    )
                )
            return hits
        except Exception:
            return []

    def _rows_to_hits(
        self,
        rows: Sequence[Any],
        query: str,
        *,
        thread_id: Optional[str],
        company_profile_binding_id: Optional[str] = None,
        exclude_run_id: Optional[str],
        limit: int,
        match_reason: str,
    ) -> List[SessionSearchHit]:
        hits: List[SessionSearchHit] = []
        for row in rows:
            rid = row["run_id"]
            if exclude_run_id and rid == exclude_run_id:
                continue
            if not _row_matches_isolation(
                row,
                thread_id=thread_id,
                company_profile_binding_id=company_profile_binding_id,
            ):
                continue
            snippet = _make_snippet(row["search_text"] or row["user_request"] or "", query)
            preview = _truncate(row["user_request"] or "", 120)
            hits.append(
                SessionSearchHit(
                    run_id=rid,
                    thread_id=row["thread_id"] or None,
                    match_reason=match_reason,
                    snippet=snippet,
                    score=float(-(row["rank"] or 0)),
                    user_request_preview=preview,
                    created_at=row["created_at"],
                    included_in_llm=False,
                )
            )
        return hits[:limit]


class SessionSearchRetriever:
    """按用户请求关键词 / FTS 检索历史会话片段（fail-open）。"""

    def __init__(self, ledger: Optional[SessionLedger] = None):
        self._ledger_store = ledger

    def get_ledger(self) -> SessionLedger:
        if self._ledger_store is None:
            self._ledger_store = SessionLedger()
        return self._ledger_store

    def search(
        self,
        user_request: str,
        *,
        thread_id: Optional[str] = None,
        company_profile_binding_id: Optional[str] = None,
        limit: int = DEFAULT_SEARCH_TOP_K,
        exclude_run_id: Optional[str] = None,
    ) -> SessionSearchResult:
        if self._ledger_store is None:
            return SessionSearchResult(enabled=False, used=False, query=user_request)
        ledger = self.get_ledger()
        if not ledger.healthy:
            return SessionSearchResult(
                enabled=True,
                used=False,
                query=user_request,
                error_type=ledger._error or "session_ledger_unavailable",
            )
        try:
            hits = ledger.search(
                user_request,
                thread_id=thread_id,
                company_profile_binding_id=company_profile_binding_id,
                limit=limit,
                exclude_run_id=exclude_run_id,
            )
            retrieved = [h.to_dict() for h in hits]
            return SessionSearchResult(
                enabled=True,
                used=bool(retrieved),
                query=user_request,
                retrieved_sessions=retrieved,
            )
        except Exception as exc:
            return SessionSearchResult(
                enabled=True,
                used=False,
                query=user_request,
                error_type=f"session_search_failed:{exc}",
            )


def _build_verification_summary(verification: Optional[Dict[str, Any]]) -> str:
    if not verification:
        return ""
    parts: List[str] = []
    for key in ("status", "summary", "reason", "message"):
        val = verification.get(key)
        if val:
            parts.append(f"{key}={val}")
    issues = verification.get("issues") or []
    if issues:
        parts.append(f"issues={len(issues)}")
    return _redact_and_store_text("; ".join(parts))


def _extract_artifact_refs(
    observation: Dict[str, Any],
    artifacts: Sequence[Dict[str, Any]],
) -> List[str]:
    refs: List[str] = []
    raw_ref = observation.get("raw_output_ref")
    if raw_ref:
        refs.append(str(raw_ref))
    obs_compact = observation.get("observation") or {}
    if isinstance(obs_compact, dict):
        for art in obs_compact.get("artifacts") or []:
            if isinstance(art, dict) and art.get("path"):
                refs.append(str(art["path"]))
    step_id = observation.get("step_id")
    for art in artifacts:
        if not isinstance(art, dict):
            continue
        if step_id and art.get("step_id") == step_id and art.get("path"):
            refs.append(str(art["path"]))
    deduped: List[str] = []
    seen = set()
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped[:20]


def _make_snippet(text: str, query: str) -> str:
    cleaned = _redact_and_store_text(text, max_chars=MAX_SNIPPET_CHARS * 2)
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for token in _TOKEN_RE.findall(query):
        pos = lowered.find(token.lower())
        if pos >= 0:
            start = max(0, pos - 60)
            end = min(len(cleaned), pos + len(token) + 120)
            snippet = cleaned[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(cleaned):
                snippet = snippet + "..."
            return _truncate(snippet, MAX_SNIPPET_CHARS)
    return _truncate(cleaned, MAX_SNIPPET_CHARS)


def record_langgraph_run_to_ledger(
    *,
    state: Dict[str, Any],
    report: Dict[str, Any],
    run_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    company_profile_binding_id: Optional[str] = None,
    ledger: Optional[SessionLedger] = None,
    enabled: Optional[bool] = None,
) -> Optional[str]:
    """Run 结束后写入 Session Ledger（fail-open，不抛异常）。"""
    if enabled is False or ledger is None:
        return None
    try:
        store = ledger or SessionLedger()
        rid = run_id or state.get("run_id") or report.get("run_id")
        binding_id = _opaque_binding_id(
            company_profile_binding_id
            or state.get("company_profile_binding_id")
            or report.get("company_profile_binding_id")
        )
        return store.record_run(
            run_id=rid,
            thread_id=thread_id or state.get("thread_id") or report.get("thread_id"),
            company_profile_binding_id=binding_id,
            agent_mode=str(state.get("agent_mode") or report.get("agent_mode") or "langgraph"),
            execution_mode=str(
                report.get("execution_mode")
                or state.get("execution_mode")
                or state.get("requested_execution_mode")
                or "dry_run"
            ),
            status=str(report.get("status") or state.get("execution_status") or "completed"),
            user_request=str(state.get("user_request") or report.get("goal") or ""),
            final_response=state.get("final_response") or report.get("final_response"),
            verification=state.get("verification") or report.get("verification"),
            tool_calls=report.get("tool_calls") or state.get("tool_calls") or [],
            observations=report.get("observations") or state.get("observations") or [],
            artifacts=report.get("artifacts") or state.get("artifacts") or [],
        )
    except Exception:
        return None


def retrieve_session_search(
    user_request: str,
    *,
    thread_id: Optional[str] = None,
    company_profile_binding_id: Optional[str] = None,
    exclude_run_id: Optional[str] = None,
    retriever: Optional[SessionSearchRetriever] = None,
    enabled: Optional[bool] = None,
) -> SessionSearchResult:
    """retrieve_context_node 使用的 fail-open 检索入口。"""
    if enabled is False or retriever is None:
        return SessionSearchResult(enabled=False, used=False, query=user_request)
    try:
        r = retriever or SessionSearchRetriever()
        return r.search(
            user_request,
            thread_id=thread_id,
            company_profile_binding_id=company_profile_binding_id,
            exclude_run_id=exclude_run_id,
        )
    except Exception as exc:
        return SessionSearchResult(
            enabled=True,
            used=False,
            query=user_request,
            error_type=f"session_search_failed:{exc}",
        )
