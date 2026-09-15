# -*- coding: utf-8 -*-
"""LocalKnowledgeProvider：配置 root 内 UTF-8 文档 + SQLite FTS5 / 关键词回退。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .knowledge_provider import (
    PERMISSION_MODE_TENANT_ONLY,
    KnowledgeDocument,
    KnowledgeFetchRequest,
    KnowledgeProviderHealth,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeSearchResult,
    KnowledgeTimeout,
    _safe_text,
    ensure_knowledge_deadline,
    format_citation,
    knowledge_deadline_exceeded,
)
from .redaction import contains_secret_blob, redact_text

SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}
MAX_CHUNK_CHARS = 800
MAX_FILE_CHARS = 200_000
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TENANT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_QUERY_TOKENS = 24


def default_knowledge_index_path(tenant_id: str, *, base: Optional[Path] = None) -> Path:
    if base is not None:
        root = Path(base) / "runs" / "agent_knowledge"
    else:
        root = Path(os.environ.get("AGENT_WORKBENCH_HOME") or Path.home() / ".agent-workbench").expanduser() / "knowledge"
    return root / sanitize_tenant_id(tenant_id) / "knowledge.db"


def sanitize_tenant_id(tenant_id: Optional[str]) -> str:
    raw = str(tenant_id or "default")
    cleaned = _TENANT_RE.sub("_", raw).strip("._-")
    if cleaned == raw and len(cleaned) <= 64:
        return cleaned
    # Lossy path normalization must not merge distinct tenant namespaces.
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{cleaned[:47] or 'tenant'}-{digest}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    """英文按词；连续中文再拆 2/3-gram，避免整句查询无法命中文档片段。"""
    tokens: List[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        cleaned = token.strip().lower()
        if not cleaned or cleaned in seen or len(tokens) >= _MAX_QUERY_TOKENS:
            return
        seen.add(cleaned)
        tokens.append(cleaned)

    for raw in _TOKEN_RE.findall(text or ""):
        add(raw)
        if _CJK_RE.fullmatch(raw) and len(raw) >= 2:
            for n in (2, 3):
                if len(raw) < n:
                    continue
                for idx in range(len(raw) - n + 1):
                    add(raw[idx : idx + n])
    return tokens


_FTS_SPECIAL_RE = re.compile(r'["*():^]')


def _fts_query(text: str, *, trigram: bool = False) -> str:
    if trigram:
        compact = _FTS_SPECIAL_RE.sub(" ", text or "")
        compact = re.sub(r"\s+", " ", compact).strip()
        parts: List[str] = []
        if len(re.sub(r"\s+", "", compact)) >= 3:
            parts.append(compact)
        for token in _tokenize(text):
            safe = _FTS_SPECIAL_RE.sub("", token)
            if len(safe) >= 3:
                parts.append(safe)
        uniq: List[str] = []
        seen: set[str] = set()
        for part in parts:
            if part not in seen:
                seen.add(part)
                uniq.append(part)
        return " OR ".join(uniq)
    tokens = _tokenize(text)
    if not tokens:
        return ""
    parts = []
    for token in tokens:
        safe = token.replace('"', "")
        if safe:
            parts.append(f'"{safe}"')
    return " OR ".join(parts)


def _file_sha256(path: Path, *, deadline: Optional[float] = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            ensure_knowledge_deadline(deadline)
            digest.update(chunk)
    return digest.hexdigest()


def opaque_document_id(tenant_id: str, relative_path: str) -> str:
    rel = str(relative_path or "").replace("\\", "/").lstrip("/")
    digest = hashlib.sha256(f"{sanitize_tenant_id(tenant_id)}\n{rel}".encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


def _path_fingerprint(relative_path: str) -> str:
    rel = str(relative_path or "").replace("\\", "/").lstrip("/")
    return hashlib.sha256(rel.encode("utf-8")).hexdigest()[:16]


def _looks_like_filesystem_path(document_id: str) -> bool:
    value = str(document_id or "")
    if not value:
        return False
    normalized = value.replace("\\", "/")
    if ".." in normalized.split("/"):
        return True
    if "/" in normalized:
        return True
    if len(value) >= 2 and value[1] == ":":
        return True
    return False


def _opaque_source_uri(provider_id: str, document_id: str) -> str:
    return f"kb://{provider_id}/{document_id}"


DEFAULT_SQLITE_TIMEOUT_S = 5.0
_SQLITE_LOCK_MARKERS = (
    "database is locked",
    "database table is locked",
    "interrupted",
)


def sqlite_wait_timeout(deadline: Optional[float]) -> Tuple[float, int]:
    """按 deadline 剩余时间计算 sqlite3.connect timeout 与 PRAGMA busy_timeout。"""
    ensure_knowledge_deadline(deadline)
    if deadline is None:
        timeout_ms = int(DEFAULT_SQLITE_TIMEOUT_S * 1000)
        return DEFAULT_SQLITE_TIMEOUT_S, timeout_ms
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise KnowledgeTimeout()
    timeout_ms = max(1, int(remaining * 1000))
    return remaining, timeout_ms


def is_sqlite_lock_or_interrupt(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _SQLITE_LOCK_MARKERS)


def reraise_sqlite_error(exc: sqlite3.OperationalError, deadline: Optional[float]) -> None:
    """deadline 已到或锁/中断耗尽剩余等待时映射为 knowledge_timeout。"""
    if deadline is not None and (knowledge_deadline_exceeded(deadline) or is_sqlite_lock_or_interrupt(exc)):
        raise KnowledgeTimeout() from exc
    raise


def _is_within_root(root: Path, candidate: Path) -> bool:
    try:
        root_resolved = root.resolve()
        cand_resolved = candidate.resolve()
        return cand_resolved == root_resolved or root_resolved in cand_resolved.parents
    except OSError:
        return False


def _chunk_text(text: str, *, title: str) -> List[Dict[str, str]]:
    sections: List[Tuple[str, str]] = []
    headings: List[str] = [title] if title else []
    buf: List[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            heading_path = " > ".join([h for h in headings if h]) or title
            sections.append((heading_path, body))
        buf.clear()

    for line in (text or "").splitlines():
        match = _HEADING_RE.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            heading = match.group(2).strip()
            headings = headings[:level] + [heading]
            if not headings or headings[0] != title:
                headings = [title] + [h for h in headings if h != title]
            continue
        buf.append(line)
    flush()
    if not sections and (text or "").strip():
        sections = [(title, text.strip())]

    chunks: List[Dict[str, str]] = []
    for heading_path, body in sections:
        remaining = body
        part_idx = 0
        while remaining:
            piece = remaining[:MAX_CHUNK_CHARS]
            if len(remaining) > MAX_CHUNK_CHARS:
                cut = piece.rfind("\n")
                if cut >= MAX_CHUNK_CHARS // 2:
                    piece = piece[:cut]
            cleaned, _ = redact_text(piece)
            if contains_secret_blob(cleaned):
                cleaned = "[REDACTED]"
            heading_safe, _ = redact_text(heading_path if part_idx == 0 else f"{heading_path} (cont.)")
            if contains_secret_blob(heading_safe):
                heading_safe = "document"
            if cleaned.strip():
                chunks.append(
                    {
                        "heading_path": heading_safe,
                        "text": cleaned.strip(),
                    }
                )
                part_idx += 1
            remaining = remaining[len(piece) :].lstrip()
    return chunks


def _read_utf8_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if len(raw) > MAX_FILE_CHARS:
        raw = raw[:MAX_FILE_CHARS]
    cleaned, _ = redact_text(raw)
    return cleaned


def _document_display_title(path: Path, raw: str) -> Tuple[str, str]:
    """标题来自文档内容，不用文件名，避免路径泄漏。"""
    body = raw
    title = "document"
    if path.suffix.lower() == ".json":
        title, body = _json_to_text(raw, "document")
    else:
        for line in (raw or "").splitlines():
            match = _HEADING_RE.match(line.strip())
            if match:
                title = match.group(2).strip() or "document"
                break
    safe_title, _ = redact_text(title)
    if not safe_title.strip() or contains_secret_blob(safe_title):
        safe_title = "document"
    return _safe_text(safe_title, 200), body


def _json_to_text(raw: str, fallback_title: str) -> Tuple[str, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return fallback_title, raw
    title = fallback_title
    if isinstance(data, dict):
        title = str(data.get("title") or fallback_title)
        if "content" in data:
            content = data.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
            return title, text
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return title, text
    return title, json.dumps(data, ensure_ascii=False, indent=2)


class LocalKnowledgeProvider:
    provider_id = "local"

    def __init__(
        self,
        *,
        root: str | Path,
        tenant_id: str = "default",
        index_path: Optional[str | Path] = None,
        index_dir: Optional[str | Path] = None,
        force_keyword: bool = False,
    ) -> None:
        self.tenant_id = sanitize_tenant_id(tenant_id)
        self.root = Path(root).expanduser().resolve()
        if index_path:
            self.index_path = Path(index_path)
        elif index_dir:
            self.index_path = Path(index_dir) / self.tenant_id / "knowledge.db"
        else:
            self.index_path = default_knowledge_index_path(self.tenant_id)
        self._lock = threading.RLock()
        self._force_keyword = force_keyword
        self._fts_available = not force_keyword
        self._fts_tokenize = "unicode61"
        self._schema_ready = False
        self._init_error: Optional[str] = None

    def _connect(self, *, deadline: Optional[float] = None) -> sqlite3.Connection:
        ensure_knowledge_deadline(deadline)
        timeout_s, timeout_ms = sqlite_wait_timeout(deadline)
        try:
            conn = sqlite3.connect(str(self.index_path), timeout=timeout_s)
        except sqlite3.OperationalError as exc:
            reraise_sqlite_error(exc, deadline)
            raise
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout = {timeout_ms}")
        except sqlite3.OperationalError as exc:
            conn.close()
            reraise_sqlite_error(exc, deadline)
            raise
        if deadline is not None:
            def _progress() -> int:
                return 1 if knowledge_deadline_exceeded(deadline) else 0

            conn.set_progress_handler(_progress, 64)
        ensure_knowledge_deadline(deadline)
        return conn

    def _init_schema(self, *, deadline: Optional[float] = None) -> None:
        if self._schema_ready:
            return
        ensure_knowledge_deadline(deadline)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._schema_ready:
                return
            with self._connect(deadline=deadline) as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        title TEXT,
                        content_hash TEXT,
                        mtime REAL,
                        updated_at TEXT,
                        version TEXT
                    );
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL,
                        tenant_id TEXT NOT NULL,
                        chunk_index INTEGER,
                        heading_path TEXT,
                        text TEXT,
                        FOREIGN KEY(document_id) REFERENCES documents(document_id)
                    );
                    """
                )
                if self._force_keyword:
                    self._fts_available = False
                elif self._fts_available:
                    self._fts_available = self._ensure_fts(conn, deadline=deadline)
                conn.commit()
            self._schema_ready = True

    def _ensure_fts(self, conn: sqlite3.Connection, *, deadline: Optional[float] = None) -> bool:
        for tokenize in ("trigram", None):
            token_sql = ", tokenize = 'trigram'" if tokenize else ""
            try:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        document_id UNINDEXED,
                        tenant_id UNINDEXED,
                        title,
                        heading_path,
                        text{token_sql}
                    )
                    """
                )
                conn.execute("SELECT * FROM chunks_fts LIMIT 0")
                self._fts_tokenize = tokenize or "unicode61"
                return True
            except sqlite3.OperationalError as exc:
                if deadline is not None and (
                    knowledge_deadline_exceeded(deadline) or is_sqlite_lock_or_interrupt(exc)
                ):
                    raise KnowledgeTimeout() from exc
                continue
        return False

    def _iter_source_files(self, *, deadline: Optional[float] = None) -> Iterable[Path]:
        if not self.root.exists() or not self.root.is_dir():
            return []
        files: List[Path] = []
        for path in self.root.rglob("*"):
            ensure_knowledge_deadline(deadline)
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if not _is_within_root(self.root, path):
                continue
            files.append(path)
        return files

    def _sync_index(self, *, deadline: Optional[float] = None) -> None:
        if self._init_error:
            return
        try:
            self._init_schema(deadline=deadline)
            with self._lock:
                with self._connect(deadline=deadline) as conn:
                    existing = {
                        row["document_id"]: row
                        for row in conn.execute(
                            "SELECT document_id, content_hash, mtime FROM documents WHERE tenant_id = ?",
                            (self.tenant_id,),
                        )
                    }
                    seen: set[str] = set()
                    for path in self._iter_source_files(deadline=deadline):
                        ensure_knowledge_deadline(deadline)
                        rel = path.relative_to(self.root).as_posix()
                        document_id = opaque_document_id(self.tenant_id, rel)
                        seen.add(document_id)
                        stat = path.stat()
                        content_hash = _file_sha256(path, deadline=deadline)
                        prev = existing.get(document_id)
                        if (
                            prev
                            and prev["content_hash"] == content_hash
                            and abs(float(prev["mtime"] or 0) - stat.st_mtime) < 1e-6
                        ):
                            continue
                        self._index_file(
                            conn,
                            path,
                            document_id,
                            rel,
                            content_hash,
                            stat.st_mtime,
                            deadline=deadline,
                        )
                    stale = [doc_id for doc_id in existing if doc_id not in seen]
                    for doc_id in stale:
                        ensure_knowledge_deadline(deadline)
                        self._delete_document(conn, doc_id)
                    conn.commit()
        except sqlite3.OperationalError as exc:
            reraise_sqlite_error(exc, deadline)

    def _delete_document(self, conn: sqlite3.Connection, document_id: str) -> None:
        conn.execute(
            "DELETE FROM document_chunks WHERE tenant_id = ? AND document_id = ?",
            (self.tenant_id, document_id),
        )
        conn.execute(
            "DELETE FROM documents WHERE tenant_id = ? AND document_id = ?",
            (self.tenant_id, document_id),
        )
        if self._fts_available:
            try:
                conn.execute(
                    "DELETE FROM chunks_fts WHERE tenant_id = ? AND document_id = ?",
                    (self.tenant_id, document_id),
                )
            except sqlite3.OperationalError:
                pass

    def _index_file(
        self,
        conn: sqlite3.Connection,
        path: Path,
        document_id: str,
        relative_path: str,
        content_hash: str,
        mtime: float,
        deadline: Optional[float] = None,
    ) -> None:
        ensure_knowledge_deadline(deadline)
        raw = _read_utf8_file(path)
        title, raw = _document_display_title(path, raw)
        chunks = _chunk_text(raw, title=title)
        self._delete_document(conn, document_id)
        now = _utc_now()
        conn.execute(
            """
            INSERT OR REPLACE INTO documents (
                document_id, tenant_id, relative_path, title, content_hash, mtime, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                self.tenant_id,
                _path_fingerprint(relative_path),
                title,
                content_hash,
                mtime,
                now,
                content_hash[:12],
            ),
        )
        for idx, chunk in enumerate(chunks):
            ensure_knowledge_deadline(deadline)
            chunk_id = f"c{idx:03d}"
            conn.execute(
                """
                INSERT INTO document_chunks (
                    chunk_id, document_id, tenant_id, chunk_index, heading_path, text
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{document_id}#{chunk_id}",
                    document_id,
                    self.tenant_id,
                    idx,
                    chunk["heading_path"],
                    chunk["text"],
                ),
            )
            if self._fts_available:
                try:
                    conn.execute(
                        """
                        INSERT INTO chunks_fts (chunk_id, document_id, tenant_id, title, heading_path, text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"{document_id}#{chunk_id}",
                            document_id,
                            self.tenant_id,
                            title,
                            chunk["heading_path"],
                            chunk["text"],
                        ),
                    )
                except sqlite3.OperationalError as exc:
                    if deadline is not None and (
                        knowledge_deadline_exceeded(deadline) or is_sqlite_lock_or_interrupt(exc)
                    ):
                        raise KnowledgeTimeout() from exc
                    self._fts_available = False

    def _keyword_search(
        self,
        conn: sqlite3.Connection,
        request: KnowledgeSearchRequest,
        *,
        deadline: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        tokens = _tokenize(request.query)
        raw = (request.query or "").strip().lower()
        needles = list(tokens)
        if raw and raw not in needles:
            needles.append(raw)
        try:
            rows = conn.execute(
                """
                SELECT c.chunk_id, c.document_id, c.heading_path, c.text, c.chunk_index,
                       d.title, d.relative_path, d.updated_at, d.version
                FROM document_chunks c
                JOIN documents d ON d.document_id = c.document_id AND d.tenant_id = c.tenant_id
                WHERE c.tenant_id = ?
                """,
                (self.tenant_id,),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            reraise_sqlite_error(exc, deadline)
            raise
        scored: List[Tuple[float, sqlite3.Row]] = []
        for row in rows:
            ensure_knowledge_deadline(deadline)
            blob = " ".join(
                [
                    str(row["title"] or ""),
                    str(row["heading_path"] or ""),
                    str(row["text"] or ""),
                ]
            ).lower()
            if needles:
                overlap = sum(1 for token in needles if token in blob)
                if overlap <= 0:
                    continue
                score = overlap / max(len(needles), 1)
            else:
                score = 0.0
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for score, row in scored[: request.top_k] if score > 0]

    def _fts_search(
        self,
        conn: sqlite3.Connection,
        request: KnowledgeSearchRequest,
        *,
        deadline: Optional[float] = None,
    ) -> List[sqlite3.Row]:
        ensure_knowledge_deadline(deadline)
        query = _fts_query(request.query, trigram=self._fts_tokenize == "trigram")
        if not query:
            return []
        try:
            return conn.execute(
                """
                SELECT c.chunk_id, c.document_id, c.heading_path, c.text, c.chunk_index,
                       d.title, d.relative_path, d.updated_at, d.version,
                       bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN document_chunks c ON c.chunk_id = chunks_fts.chunk_id AND c.tenant_id = chunks_fts.tenant_id
                JOIN documents d ON d.document_id = c.document_id AND d.tenant_id = c.tenant_id
                WHERE chunks_fts.tenant_id = ? AND chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (self.tenant_id, query, request.top_k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if deadline is not None and (
                knowledge_deadline_exceeded(deadline) or is_sqlite_lock_or_interrupt(exc)
            ):
                raise KnowledgeTimeout() from exc
            self._fts_available = False
            return self._keyword_search(conn, request, deadline=deadline)

    def _rows_to_hits(
        self,
        rows: Sequence[sqlite3.Row],
        request: KnowledgeSearchRequest,
        *,
        backend: str = "keyword",
    ) -> List[KnowledgeSearchResult]:
        hits: List[KnowledgeSearchResult] = []
        for idx, row in enumerate(rows):
            document_id = str(row["document_id"])
            raw_chunk_id = str(row["chunk_id"])
            chunk_id = raw_chunk_id.split("#")[-1] if "#" in raw_chunk_id else f"c{int(row['chunk_index']):03d}"
            snippet, _ = redact_text(str(row["text"] or ""))
            if contains_secret_blob(snippet):
                continue
            title = _safe_text(row["title"] or "document", 200)
            heading = _safe_text(row["heading_path"] or "", 200)
            score = 1.0 / (idx + 1)
            if "rank" in row.keys() and row["rank"] is not None:
                try:
                    score = max(0.0, 1.0 / (1.0 + abs(float(row["rank"]))))
                except (TypeError, ValueError):
                    pass
            hits.append(
                KnowledgeSearchResult(
                    provider_id=self.provider_id,
                    tenant_id=self.tenant_id,
                    source_type="local_file",
                    source_id=document_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    title=title,
                    snippet=snippet,
                    score=float(score),
                    source_uri=_opaque_source_uri(self.provider_id, document_id),
                    updated_at=row["updated_at"],
                    version=row["version"],
                    sensitivity_level="internal",
                    metadata={
                        "heading_path": heading,
                        "search_backend": backend,
                    },
                    citation=format_citation(self.provider_id, document_id, chunk_id),
                    retrieval_reason="fts_match" if backend == "fts" else "keyword_overlap",
                )
            )
        return hits

    def search(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        tenant = sanitize_tenant_id(request.tenant_id)
        if tenant != self.tenant_id:
            return KnowledgeSearchResponse(
                enabled=True,
                used=False,
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                query=request.query,
                error_type="tenant_mismatch",
                omission_reason="tenant_mismatch",
            )
        if self._init_error:
            return KnowledgeSearchResponse(
                enabled=True,
                used=False,
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                query=request.query,
                error_type=self._init_error,
                omission_reason=self._init_error,
                degraded=True,
            )
        try:
            deadline = request.resolved_deadline()
            started = time.perf_counter()
            ensure_knowledge_deadline(deadline)
            self._sync_index(deadline=deadline)
            with self._lock:
                with self._connect(deadline=deadline) as conn:
                    used_fts = False
                    if self._fts_available:
                        rows = self._fts_search(conn, request, deadline=deadline)
                        used_fts = bool(rows)
                        if not rows:
                            rows = self._keyword_search(conn, request, deadline=deadline)
                    else:
                        rows = self._keyword_search(conn, request, deadline=deadline)
            hits = self._rows_to_hits(rows, request, backend="fts" if used_fts else "keyword")
            latency_ms = (time.perf_counter() - started) * 1000
            return KnowledgeSearchResponse(
                enabled=True,
                used=bool(hits),
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                query=request.query,
                hits=hits,
                latency_ms=latency_ms,
                omission_reason=None if hits else "empty_result",
                degraded=not self._fts_available,
                retrieval_mode="bm25" if used_fts else "keyword",
                bm25_hit_count=len(hits) if used_fts else 0,
                vector_hit_count=0,
                fused_hit_count=len(hits),
                reranked_count=0,
                acl_filtered_count=0,
                stale_filtered_count=0,
                degraded_reason=None if self._fts_available else "fts_unavailable",
                provider_latency_ms=latency_ms,
            )
        except KnowledgeTimeout:
            return KnowledgeSearchResponse(
                enabled=True,
                used=False,
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                query=request.query,
                error_type="knowledge_timeout",
                omission_reason="knowledge_timeout",
                degraded=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except sqlite3.OperationalError as exc:
            deadline = request.resolved_deadline()
            if deadline is not None and (
                knowledge_deadline_exceeded(deadline) or is_sqlite_lock_or_interrupt(exc)
            ):
                return KnowledgeSearchResponse(
                    enabled=True,
                    used=False,
                    provider=self.provider_id,
                    tenant_id=self.tenant_id,
                    permission_mode=PERMISSION_MODE_TENANT_ONLY,
                    query=request.query,
                    error_type="knowledge_timeout",
                    omission_reason="knowledge_timeout",
                    degraded=True,
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            return KnowledgeSearchResponse(
                enabled=True,
                used=False,
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                query=request.query,
                error_type=f"knowledge_search_failed:{type(exc).__name__}",
                omission_reason="provider_error",
                degraded=True,
            )
        except Exception as exc:
            return KnowledgeSearchResponse(
                enabled=True,
                used=False,
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                query=request.query,
                error_type=f"knowledge_search_failed:{type(exc).__name__}",
                omission_reason="provider_error",
                degraded=True,
            )

    def fetch(self, request: KnowledgeFetchRequest) -> KnowledgeDocument:
        tenant = sanitize_tenant_id(request.tenant_id)
        if tenant != self.tenant_id:
            return KnowledgeDocument(
                provider_id=self.provider_id,
                tenant_id=self.tenant_id,
                document_id=request.document_id,
                title="",
                source_uri=None,
                text="",
                error_type="tenant_mismatch",
            )
        document_id = str(request.document_id or "")
        if _looks_like_filesystem_path(document_id):
            return KnowledgeDocument(
                provider_id=self.provider_id,
                tenant_id=self.tenant_id,
                document_id=document_id,
                title="",
                source_uri=None,
                text="",
                error_type="path_outside_root",
            )
        try:
            self._sync_index()
            with self._lock:
                with self._connect() as conn:
                    doc = conn.execute(
                        "SELECT * FROM documents WHERE tenant_id = ? AND document_id = ?",
                        (self.tenant_id, document_id),
                    ).fetchone()
                    if not doc:
                        return KnowledgeDocument(
                            provider_id=self.provider_id,
                            tenant_id=self.tenant_id,
                            document_id=document_id,
                            title="",
                            source_uri=None,
                            text="",
                            error_type="document_not_found",
                        )
                    chunk_rows = conn.execute(
                        """
                        SELECT chunk_id, chunk_index, heading_path, text
                        FROM document_chunks
                        WHERE tenant_id = ? AND document_id = ?
                        ORDER BY chunk_index
                        """,
                        (self.tenant_id, document_id),
                    ).fetchall()
            chunks = [
                {
                    "chunk_id": str(row["chunk_id"]).split("#")[-1],
                    "heading_path": _safe_text(row["heading_path"] or "", 200),
                    "text": row["text"],
                }
                for row in chunk_rows
            ]
            if request.chunk_id:
                chunks = [c for c in chunks if c["chunk_id"] == request.chunk_id]
            text = "\n\n".join(c["text"] for c in chunks)
            return KnowledgeDocument(
                provider_id=self.provider_id,
                tenant_id=self.tenant_id,
                document_id=document_id,
                title=_safe_text(doc["title"] or "document", 200),
                source_uri=_opaque_source_uri(self.provider_id, document_id),
                text=text,
                chunks=chunks,
                updated_at=doc["updated_at"],
                version=doc["version"],
            )
        except Exception as exc:
            return KnowledgeDocument(
                provider_id=self.provider_id,
                tenant_id=self.tenant_id,
                document_id=document_id,
                title="",
                source_uri=None,
                text="",
                error_type=f"knowledge_fetch_failed:{type(exc).__name__}",
            )

    def health(self) -> KnowledgeProviderHealth:
        details = {
            "root_exists": self.root.exists(),
            "fts_available": self._fts_available,
            "index_path": str(self.index_path),
            "tenant_id": self.tenant_id,
        }
        if self._init_error:
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=False,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                error_type=self._init_error,
                details=details,
            )
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=True,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                details=details,
            )
        except Exception as exc:
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=False,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                error_type=f"knowledge_health_failed:{type(exc).__name__}",
                details=details,
            )
