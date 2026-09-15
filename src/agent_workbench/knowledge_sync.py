# -*- coding: utf-8 -*-
"""企业知识源同步：Connector 读文档，Indexer 切块/脱敏/ACL/增量更新。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from .knowledge_acl import KnowledgeACL, parse_knowledge_acl
from .knowledge_provider import KnowledgeTimeout, ensure_knowledge_deadline, knowledge_deadline_exceeded
from .local_knowledge_provider import sanitize_tenant_id
from .redaction import redact_recursive, redact_text

SYNC_STATUS_OK = "ok"
SYNC_STATUS_PARTIAL = "partial"
SYNC_STATUS_FAILED = "failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_err(message: Any, limit: int = 240) -> str:
    cleaned, _ = redact_text(str(message or ""))
    return cleaned[:limit]


@dataclass
class KnowledgeSourceDocument:
    source_id: str
    tenant_id: str
    document_id: str
    title: str
    text: str
    content_hash: str
    version: str
    updated_at: str
    source_uri: Optional[str] = None
    source_type: str = "connector"
    acl: KnowledgeACL = field(default_factory=KnowledgeACL)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["acl"] = self.acl.to_dict() if isinstance(self.acl, KnowledgeACL) else parse_knowledge_acl(self.acl).to_dict()
        text, _ = redact_text(self.text or "")
        payload["text"] = text[:400]
        payload["title"], _ = redact_text(self.title or "")
        meta, _ = redact_recursive(self.metadata or {})
        payload["metadata"] = meta
        return payload


@dataclass
class KnowledgeSourceCheckpoint:
    source_id: str
    tenant_id: str
    sync_cursor: Optional[str] = None
    document_ids: List[str] = field(default_factory=list)
    hashes: Dict[str, str] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "tenant_id": sanitize_tenant_id(self.tenant_id),
            "sync_cursor": self.sync_cursor,
            "document_ids": list(self.document_ids),
            "hashes": dict(self.hashes),
            "versions": dict(self.versions),
        }


@dataclass
class IndexedChunk:
    chunk_id: str
    text: str
    heading_path: str = ""
    embedding: Optional[List[float]] = None


@dataclass
class IndexedDocument:
    source_id: str
    tenant_id: str
    document_id: str
    title: str
    content_hash: str
    version: str
    updated_at: str
    source_uri: Optional[str] = None
    source_type: str = "connector"
    acl: KnowledgeACL = field(default_factory=KnowledgeACL)
    chunks: List[IndexedChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeSyncJob:
    source_id: str
    tenant_id: str
    status: str
    created: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    sync_cursor: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    store_backend: str = "memory"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "tenant_id": sanitize_tenant_id(self.tenant_id),
            "status": self.status,
            "created": int(self.created),
            "updated": int(self.updated),
            "deleted": int(self.deleted),
            "unchanged": int(self.unchanged),
            "sync_cursor": self.sync_cursor,
            "error_type": self.error_type,
            "latency_ms": round(float(self.latency_ms or 0.0), 2),
            "errors": list(self.errors or []),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "store_backend": self.store_backend,
        }


@dataclass
class KnowledgeSyncResult:
    source_id: str
    tenant_id: str
    created: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    sync_cursor: Optional[str] = None
    error_type: Optional[str] = None
    latency_ms: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    status: str = SYNC_STATUS_OK
    job: Optional[Dict[str, Any]] = None
    store_backend: str = "memory"
    mode: str = "incremental"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "tenant_id": sanitize_tenant_id(self.tenant_id),
            "created": int(self.created),
            "updated": int(self.updated),
            "deleted": int(self.deleted),
            "unchanged": int(self.unchanged),
            "sync_cursor": self.sync_cursor,
            "error_type": self.error_type,
            "latency_ms": round(float(self.latency_ms or 0.0), 2),
            "errors": list(self.errors or []),
            "status": self.status,
            "job": dict(self.job or {}),
            "store_backend": self.store_backend,
            "mode": self.mode,
        }


class KnowledgeSourceConnector(Protocol):
    source_id: str

    def list_documents(
        self,
        checkpoint: Optional[KnowledgeSourceCheckpoint] = None,
        *,
        tenant_id: str,
        deadline: Optional[float] = None,
    ) -> Iterable[KnowledgeSourceDocument]: ...

    def health(self) -> Dict[str, Any]: ...


class KnowledgeDocumentStore(Protocol):
    backend: str

    def get_document(
        self, tenant_id: str, source_id: str, document_id: str, *, deadline: Optional[float] = None
    ) -> Optional[IndexedDocument]: ...

    def upsert_document(self, document: IndexedDocument, *, deadline: Optional[float] = None) -> None: ...

    def delete_document(
        self, tenant_id: str, source_id: str, document_id: str, *, deadline: Optional[float] = None
    ) -> None: ...

    def list_document_ids(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> List[str]: ...

    def load_checkpoint(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> Optional[KnowledgeSourceCheckpoint]: ...

    def save_checkpoint(self, checkpoint: KnowledgeSourceCheckpoint, *, deadline: Optional[float] = None) -> None: ...

    def record_sync_job(self, job: KnowledgeSyncJob, *, deadline: Optional[float] = None) -> None: ...

    def latest_sync_job(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> Optional[KnowledgeSyncJob]: ...


class InMemoryKnowledgeStore:
    """测试与验收用内存索引，不访问网络。"""

    backend = "memory"

    def __init__(self) -> None:
        self.documents: Dict[tuple, IndexedDocument] = {}
        self.checkpoints: Dict[tuple, KnowledgeSourceCheckpoint] = {}
        self.sync_jobs: List[KnowledgeSyncJob] = []
        self.retrieval_logs: List[Dict[str, Any]] = []

    def _key(self, tenant_id: str, source_id: str, document_id: str) -> tuple:
        return (sanitize_tenant_id(tenant_id), str(source_id), str(document_id))

    def get_document(
        self, tenant_id: str, source_id: str, document_id: str, *, deadline: Optional[float] = None
    ) -> Optional[IndexedDocument]:
        ensure_knowledge_deadline(deadline)
        return self.documents.get(self._key(tenant_id, source_id, document_id))

    def upsert_document(self, document: IndexedDocument, *, deadline: Optional[float] = None) -> None:
        ensure_knowledge_deadline(deadline)
        self.documents[self._key(document.tenant_id, document.source_id, document.document_id)] = document

    def delete_document(
        self, tenant_id: str, source_id: str, document_id: str, *, deadline: Optional[float] = None
    ) -> None:
        ensure_knowledge_deadline(deadline)
        self.documents.pop(self._key(tenant_id, source_id, document_id), None)

    def list_document_ids(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> List[str]:
        ensure_knowledge_deadline(deadline)
        tenant = sanitize_tenant_id(tenant_id)
        return [
            doc.document_id
            for (t, src, _doc_id), doc in self.documents.items()
            if t == tenant and src == source_id
        ]

    def load_checkpoint(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> Optional[KnowledgeSourceCheckpoint]:
        ensure_knowledge_deadline(deadline)
        return self.checkpoints.get((sanitize_tenant_id(tenant_id), str(source_id)))

    def save_checkpoint(self, checkpoint: KnowledgeSourceCheckpoint, *, deadline: Optional[float] = None) -> None:
        ensure_knowledge_deadline(deadline)
        self.checkpoints[(sanitize_tenant_id(checkpoint.tenant_id), str(checkpoint.source_id))] = checkpoint

    def record_sync_job(self, job: KnowledgeSyncJob, *, deadline: Optional[float] = None) -> None:
        ensure_knowledge_deadline(deadline)
        self.sync_jobs.append(job)

    def latest_sync_job(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> Optional[KnowledgeSyncJob]:
        ensure_knowledge_deadline(deadline)
        tenant = sanitize_tenant_id(tenant_id)
        for job in reversed(self.sync_jobs):
            if sanitize_tenant_id(job.tenant_id) == tenant and job.source_id == source_id:
                return job
        return None

    def record_retrieval(self, payload: Dict[str, Any]) -> None:
        safe, _ = redact_recursive(payload or {})
        self.retrieval_logs.append(safe)


class KnowledgeIndexer:
    """切块、脱敏、权限映射、增量写入与删除失效文档。"""

    def __init__(self, store: KnowledgeDocumentStore, *, embedding: Any = None) -> None:
        self.store = store
        self.embedding = embedding

    def index_document(self, document: KnowledgeSourceDocument, *, deadline: Optional[float] = None) -> IndexedDocument:
        from .local_knowledge_provider import _chunk_text

        ensure_knowledge_deadline(deadline)
        title, _ = redact_text(document.title or "document")
        chunks_raw = _chunk_text(document.text or "", title=title)
        indexed_chunks: List[IndexedChunk] = []
        texts = [str(item.get("text") or "") for item in chunks_raw]
        vectors: List[Optional[List[float]]] = [None] * len(texts)
        if self.embedding is not None and texts:
            ensure_knowledge_deadline(deadline)
            try:
                vectors = list(self.embedding.embed(texts, deadline=deadline))
                ensure_knowledge_deadline(deadline)
            except KnowledgeTimeout:
                raise
            except Exception:
                if knowledge_deadline_exceeded(deadline):
                    raise KnowledgeTimeout()
                vectors = [None] * len(texts)
        for idx, item in enumerate(chunks_raw):
            indexed_chunks.append(
                IndexedChunk(
                    chunk_id=f"c{idx:03d}",
                    text=str(item.get("text") or ""),
                    heading_path=str(item.get("heading_path") or title),
                    embedding=vectors[idx] if idx < len(vectors) else None,
                )
            )
        indexed = IndexedDocument(
            source_id=document.source_id,
            tenant_id=sanitize_tenant_id(document.tenant_id),
            document_id=document.document_id,
            title=title,
            content_hash=document.content_hash,
            version=document.version,
            updated_at=document.updated_at,
            source_uri=document.source_uri,
            source_type=document.source_type,
            acl=parse_knowledge_acl(document.acl),
            chunks=indexed_chunks,
            metadata=dict(document.metadata or {}),
        )
        self.store.upsert_document(indexed, deadline=deadline)
        ensure_knowledge_deadline(deadline)
        return indexed

    def delete_document(
        self,
        tenant_id: str,
        source_id: str,
        document_id: str,
        *,
        deadline: Optional[float] = None,
    ) -> None:
        ensure_knowledge_deadline(deadline)
        self.store.delete_document(tenant_id, source_id, document_id, deadline=deadline)
        ensure_knowledge_deadline(deadline)


class KnowledgeSyncService:
    def __init__(
        self,
        connector: KnowledgeSourceConnector,
        store: Optional[KnowledgeDocumentStore] = None,
        *,
        indexer: Optional[KnowledgeIndexer] = None,
        embedding: Any = None,
    ) -> None:
        self.connector = connector
        self.store = store or InMemoryKnowledgeStore()
        self.indexer = indexer or KnowledgeIndexer(self.store, embedding=embedding)

    def _store_backend(self) -> str:
        return str(getattr(self.store, "backend", None) or "memory")

    def _persist_checkpoint(
        self,
        *,
        source_id: str,
        tenant: str,
        hashes: Dict[str, str],
        versions: Dict[str, str],
        document_ids: List[str],
        sync_cursor: Optional[str],
        deadline: Optional[float],
    ) -> None:
        self.store.save_checkpoint(
            KnowledgeSourceCheckpoint(
                source_id=source_id,
                tenant_id=tenant,
                sync_cursor=sync_cursor,
                document_ids=sorted(document_ids),
                hashes=dict(hashes),
                versions=dict(versions),
            ),
            deadline=deadline,
        )

    def sync(
        self,
        *,
        tenant_id: str,
        checkpoint: Optional[KnowledgeSourceCheckpoint] = None,
        deadline: Optional[float] = None,
        mode: str = "incremental",
    ) -> KnowledgeSyncResult:
        started = time.perf_counter()
        started_at = _utc_now()
        tenant = sanitize_tenant_id(tenant_id)
        source_id = str(getattr(self.connector, "source_id", "") or "source")
        store_backend = self._store_backend()
        sync_mode = "full" if str(mode or "incremental").strip().lower() == "full" else "incremental"
        force = sync_mode == "full"
        errors: List[Dict[str, Any]] = []
        created = updated = deleted = unchanged = 0
        sync_cursor = None
        error_type = None
        status = SYNC_STATUS_OK
        previous = None
        hashes: Dict[str, str] = {}
        versions: Dict[str, str] = {}
        document_ids: List[str] = []
        try:
            ensure_knowledge_deadline(deadline)
            previous = checkpoint or self.store.load_checkpoint(tenant, source_id, deadline=deadline)
            hashes = dict(previous.hashes) if previous else {}
            versions = dict(previous.versions) if previous else {}
            document_ids = list(previous.document_ids) if previous else []
            previous_cursor = previous.sync_cursor if previous else None
            id_set = set(document_ids)
            current_docs: Dict[str, KnowledgeSourceDocument] = {}
            for doc in self.connector.list_documents(previous, tenant_id=tenant, deadline=deadline):
                ensure_knowledge_deadline(deadline)
                if sanitize_tenant_id(doc.tenant_id) != tenant:
                    errors.append(
                        {
                            "document_id": doc.document_id,
                            "error_type": "tenant_mismatch",
                        }
                    )
                    continue
                current_docs[doc.document_id] = doc
            for document_id, doc in current_docs.items():
                ensure_knowledge_deadline(deadline)
                old_hash = hashes.get(document_id)
                try:
                    if old_hash == doc.content_hash and not force:
                        unchanged += 1
                        continue
                    self.indexer.index_document(doc, deadline=deadline)
                    hashes[document_id] = doc.content_hash
                    versions[document_id] = doc.version
                    id_set.add(document_id)
                    if old_hash is None:
                        created += 1
                    else:
                        updated += 1
                except KnowledgeTimeout:
                    raise
                except Exception as exc:
                    errors.append(
                        {
                            "document_id": document_id,
                            "error_type": f"index_failed:{type(exc).__name__}",
                            "message": _safe_err(exc),
                        }
                    )
            stale_ids = [doc_id for doc_id in list(hashes.keys()) if doc_id not in current_docs]
            for document_id in stale_ids:
                ensure_knowledge_deadline(deadline)
                try:
                    self.indexer.delete_document(tenant, source_id, document_id, deadline=deadline)
                    hashes.pop(document_id, None)
                    versions.pop(document_id, None)
                    id_set.discard(document_id)
                    deleted += 1
                except KnowledgeTimeout:
                    raise
                except Exception as exc:
                    errors.append(
                        {
                            "document_id": document_id,
                            "error_type": f"delete_failed:{type(exc).__name__}",
                            "message": _safe_err(exc),
                        }
                    )
            document_ids = sorted(id_set)
            if errors:
                status = SYNC_STATUS_PARTIAL
                error_type = "sync_partial"
                sync_cursor = previous_cursor
            else:
                successful_times = [doc.updated_at for doc in current_docs.values() if doc.updated_at and doc.document_id in hashes]
                sync_cursor = max(successful_times) if successful_times else (previous_cursor or _utc_now())
            self._persist_checkpoint(
                source_id=source_id,
                tenant=tenant,
                hashes=hashes,
                versions=versions,
                document_ids=document_ids,
                sync_cursor=sync_cursor,
                deadline=deadline,
            )
        except KnowledgeTimeout:
            status = SYNC_STATUS_FAILED
            error_type = "knowledge_timeout"
            sync_cursor = previous.sync_cursor if previous else None
            try:
                self._persist_checkpoint(
                    source_id=source_id,
                    tenant=tenant,
                    hashes=hashes,
                    versions=versions,
                    document_ids=sorted(set(document_ids) | set(hashes.keys())),
                    sync_cursor=sync_cursor,
                    deadline=None,
                )
            except Exception:
                pass
        except Exception as exc:
            status = SYNC_STATUS_FAILED
            error_type = f"sync_failed:{type(exc).__name__}"
            errors.append({"error_type": error_type, "message": _safe_err(exc)})
            sync_cursor = previous.sync_cursor if previous else None
            try:
                self._persist_checkpoint(
                    source_id=source_id,
                    tenant=tenant,
                    hashes=hashes,
                    versions=versions,
                    document_ids=sorted(set(document_ids) | set(hashes.keys())),
                    sync_cursor=sync_cursor,
                    deadline=None,
                )
            except Exception:
                pass

        latency_ms = (time.perf_counter() - started) * 1000
        job = KnowledgeSyncJob(
            source_id=source_id,
            tenant_id=tenant,
            status=status,
            created=created,
            updated=updated,
            deleted=deleted,
            unchanged=unchanged,
            sync_cursor=sync_cursor,
            error_type=error_type,
            latency_ms=latency_ms,
            errors=errors,
            started_at=started_at,
            finished_at=_utc_now(),
            store_backend=store_backend,
        )
        self.store.record_sync_job(job)
        return KnowledgeSyncResult(
            source_id=source_id,
            tenant_id=tenant,
            created=created,
            updated=updated,
            deleted=deleted,
            unchanged=unchanged,
            sync_cursor=sync_cursor,
            error_type=error_type,
            latency_ms=latency_ms,
            errors=errors,
            status=status,
            job=job.to_dict(),
            store_backend=store_backend,
            mode=sync_mode,
        )
