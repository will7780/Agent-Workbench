# -*- coding: utf-8 -*-
"""OpenSearchKnowledgeStore：生产写入、checkpoint 与 sync job 持久化。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .knowledge_acl import parse_knowledge_acl
from .knowledge_opensearch import (
    OpenSearchHttpResponse,
    OpenSearchTransport,
    RequestsOpenSearchTransport,
    remaining_timeout_s,
    resolve_opensearch_auth,
)
from .knowledge_provider import KnowledgeTimeout, ensure_knowledge_deadline
from .knowledge_sync import (
    IndexedChunk,
    IndexedDocument,
    KnowledgeSourceCheckpoint,
    KnowledgeSyncJob,
)
from .local_knowledge_provider import sanitize_tenant_id
from .redaction import redact_recursive


RECORD_CHUNK = "chunk"
RECORD_DOCUMENT = "document"
RECORD_CHECKPOINT = "checkpoint"
RECORD_SYNC_JOB = "sync_job"
RECORD_SYNC_LATEST = "sync_job_latest"


def opensearch_document_id(tenant_id: str, source_id: str, document_id: str) -> str:
    return f"{sanitize_tenant_id(tenant_id)}::{source_id}::{document_id}"


def opensearch_chunk_id(tenant_id: str, source_id: str, document_id: str, chunk_id: str) -> str:
    return f"{opensearch_document_id(tenant_id, source_id, document_id)}::{chunk_id}"


def opensearch_checkpoint_id(tenant_id: str, source_id: str) -> str:
    return f"{sanitize_tenant_id(tenant_id)}::{source_id}::__checkpoint__"


def opensearch_sync_latest_id(tenant_id: str, source_id: str) -> str:
    return f"{sanitize_tenant_id(tenant_id)}::{source_id}::__sync_job_latest__"


class OpenSearchKnowledgeStore:
    backend = "opensearch"

    def __init__(
        self,
        *,
        endpoint: str,
        index: str,
        auth_mode: str = "none",
        credential_env: Optional[str] = None,
        username_env: Optional[str] = None,
        transport: Optional[OpenSearchTransport] = None,
        source_id: str = "opensearch",
    ) -> None:
        self.endpoint = str(endpoint).strip()
        self.index = str(index).strip()
        self.auth_mode = str(auth_mode or "none").strip().lower() or "none"
        self.credential_env = str(credential_env).strip() if credential_env else None
        self.username_env = str(username_env).strip() if username_env else None
        self.transport = transport or RequestsOpenSearchTransport()
        self.source_id = source_id

    def _auth(self):
        return resolve_opensearch_auth(self.auth_mode, self.credential_env, self.username_env)

    def _raise_http(self, http: OpenSearchHttpResponse) -> None:
        if http.error_type == "knowledge_timeout":
            raise KnowledgeTimeout()
        if http.error_type and http.error_type != "not_found":
            raise RuntimeError(http.error_type)

    def _bulk(self, operations: List[Dict[str, Any]], *, deadline: Optional[float]) -> None:
        ensure_knowledge_deadline(deadline)
        headers, auth = self._auth()
        http = self.transport.bulk(
            endpoint=self.endpoint,
            index=self.index,
            operations=operations,
            timeout_s=remaining_timeout_s(deadline),
            headers=headers,
            auth=auth,
        )
        ensure_knowledge_deadline(deadline)
        self._raise_http(http)
        payload = http.payload if isinstance(http.payload, dict) else {}
        if payload.get("errors"):
            raise RuntimeError("opensearch_bulk_errors")

    def _get(self, document_id: str, *, tenant_id: str, deadline: Optional[float]) -> Optional[Dict[str, Any]]:
        ensure_knowledge_deadline(deadline)
        headers, auth = self._auth()
        http = self.transport.get_document(
            endpoint=self.endpoint,
            index=self.index,
            document_id=document_id,
            timeout_s=remaining_timeout_s(deadline),
            headers=headers,
            auth=auth,
            routing=sanitize_tenant_id(tenant_id),
        )
        ensure_knowledge_deadline(deadline)
        if http.error_type == "not_found" or http.status_code == 404:
            return None
        self._raise_http(http)
        payload = http.payload if isinstance(http.payload, dict) else {}
        source = payload.get("_source") if isinstance(payload.get("_source"), dict) else payload
        if not isinstance(source, dict) or not source:
            return None
        return source

    def _search(self, body: Dict[str, Any], *, deadline: Optional[float]) -> Dict[str, Any]:
        ensure_knowledge_deadline(deadline)
        headers, auth = self._auth()
        http = self.transport.search(
            endpoint=self.endpoint,
            index=self.index,
            body=body,
            timeout_s=remaining_timeout_s(deadline),
            headers=headers,
            auth=auth,
        )
        ensure_knowledge_deadline(deadline)
        self._raise_http(http)
        return http.payload if isinstance(http.payload, dict) else {}

    def _document_source(self, document: IndexedDocument) -> Dict[str, Any]:
        tenant = sanitize_tenant_id(document.tenant_id)
        acl = document.acl.to_dict() if hasattr(document.acl, "to_dict") else parse_knowledge_acl(document.acl).to_dict()
        chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "heading_path": chunk.heading_path,
                "embedding": chunk.embedding,
            }
            for chunk in document.chunks
        ]
        return {
            "record_type": RECORD_DOCUMENT,
            "tenant_id": tenant,
            "source_id": document.source_id,
            "source_type": document.source_type,
            "document_id": document.document_id,
            "title": document.title,
            "content_hash": document.content_hash,
            "version": document.version,
            "updated_at": document.updated_at,
            "source_uri": document.source_uri,
            "acl": acl,
            "chunks": chunks,
            "metadata": dict(document.metadata or {}),
        }

    def _chunk_source(self, document: IndexedDocument, chunk: IndexedChunk) -> Dict[str, Any]:
        tenant = sanitize_tenant_id(document.tenant_id)
        acl = document.acl.to_dict() if hasattr(document.acl, "to_dict") else parse_knowledge_acl(document.acl).to_dict()
        return {
            "record_type": RECORD_CHUNK,
            "tenant_id": tenant,
            "source_id": document.source_id,
            "source_type": document.source_type,
            "document_id": document.document_id,
            "chunk_id": chunk.chunk_id,
            "title": document.title,
            "text": chunk.text,
            "heading_path": chunk.heading_path,
            "acl": acl,
            "content_hash": document.content_hash,
            "version": document.version,
            "updated_at": document.updated_at,
            "source_uri": document.source_uri,
            "embedding": chunk.embedding,
        }

    def _from_source(self, source: Dict[str, Any]) -> IndexedDocument:
        chunks = []
        for item in source.get("chunks") or []:
            if not isinstance(item, dict):
                continue
            chunks.append(
                IndexedChunk(
                    chunk_id=str(item.get("chunk_id") or "c000"),
                    text=str(item.get("text") or ""),
                    heading_path=str(item.get("heading_path") or ""),
                    embedding=item.get("embedding") if isinstance(item.get("embedding"), list) else None,
                )
            )
        return IndexedDocument(
            source_id=str(source.get("source_id") or self.source_id),
            tenant_id=sanitize_tenant_id(str(source.get("tenant_id") or "")),
            document_id=str(source.get("document_id") or ""),
            title=str(source.get("title") or "document"),
            content_hash=str(source.get("content_hash") or ""),
            version=str(source.get("version") or ""),
            updated_at=str(source.get("updated_at") or ""),
            source_uri=source.get("source_uri"),
            source_type=str(source.get("source_type") or "opensearch"),
            acl=parse_knowledge_acl(source.get("acl")),
            chunks=chunks,
            metadata=dict(source.get("metadata") or {}),
        )

    def get_document(
        self, tenant_id: str, source_id: str, document_id: str, *, deadline: Optional[float] = None
    ) -> Optional[IndexedDocument]:
        tenant = sanitize_tenant_id(tenant_id)
        source = self._get(opensearch_document_id(tenant, source_id, document_id), tenant_id=tenant, deadline=deadline)
        if source is None or source.get("record_type") not in (RECORD_DOCUMENT, None):
            return None
        if sanitize_tenant_id(str(source.get("tenant_id") or "")) != tenant:
            return None
        return self._from_source(source)

    def upsert_document(self, document: IndexedDocument, *, deadline: Optional[float] = None) -> None:
        tenant = sanitize_tenant_id(document.tenant_id)
        new_chunk_ids = {str(chunk.chunk_id) for chunk in document.chunks}
        old_chunk_ids: set[str] = set()
        existing = self.get_document(tenant, document.source_id, document.document_id, deadline=deadline)
        if existing is not None:
            old_chunk_ids = {str(chunk.chunk_id) for chunk in existing.chunks}
        stale_chunk_ids = old_chunk_ids - new_chunk_ids
        operations: List[Dict[str, Any]] = []
        meta_id = opensearch_document_id(tenant, document.source_id, document.document_id)
        operations.append({"index": {"_index": self.index, "_id": meta_id, "routing": tenant}})
        operations.append(self._document_source(document))
        for chunk in document.chunks:
            chunk_id = opensearch_chunk_id(tenant, document.source_id, document.document_id, chunk.chunk_id)
            operations.append({"index": {"_index": self.index, "_id": chunk_id, "routing": tenant}})
            operations.append(self._chunk_source(document, chunk))
        for stale_id in sorted(stale_chunk_ids):
            operations.append(
                {
                    "delete": {
                        "_index": self.index,
                        "_id": opensearch_chunk_id(tenant, document.source_id, document.document_id, stale_id),
                        "routing": tenant,
                    }
                }
            )
        self._bulk(operations, deadline=deadline)

    def delete_document(
        self, tenant_id: str, source_id: str, document_id: str, *, deadline: Optional[float] = None
    ) -> None:
        tenant = sanitize_tenant_id(tenant_id)
        existing = self.get_document(tenant, source_id, document_id, deadline=deadline)
        ids = [opensearch_document_id(tenant, source_id, document_id)]
        if existing:
            for chunk in existing.chunks:
                ids.append(opensearch_chunk_id(tenant, source_id, document_id, chunk.chunk_id))
        operations: List[Dict[str, Any]] = []
        for doc_id in ids:
            operations.append({"delete": {"_index": self.index, "_id": doc_id, "routing": tenant}})
        if operations:
            self._bulk(operations, deadline=deadline)

    def list_document_ids(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> List[str]:
        tenant = sanitize_tenant_id(tenant_id)
        payload = self._search(
            {
                "size": 1000,
                "_source": ["document_id", "tenant_id"],
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"tenant_id": tenant}},
                            {"term": {"source_id": source_id}},
                            {"term": {"record_type": RECORD_DOCUMENT}},
                        ]
                    }
                },
            },
            deadline=deadline,
        )
        ids: List[str] = []
        for hit in ((payload.get("hits") or {}).get("hits") or []):
            source = hit.get("_source") if isinstance(hit, dict) else {}
            if isinstance(source, dict) and source.get("document_id"):
                ids.append(str(source["document_id"]))
        return ids

    def load_checkpoint(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> Optional[KnowledgeSourceCheckpoint]:
        tenant = sanitize_tenant_id(tenant_id)
        source = self._get(opensearch_checkpoint_id(tenant, source_id), tenant_id=tenant, deadline=deadline)
        if not source or source.get("record_type") != RECORD_CHECKPOINT:
            return None
        return KnowledgeSourceCheckpoint(
            source_id=str(source.get("source_id") or source_id),
            tenant_id=tenant,
            sync_cursor=source.get("sync_cursor"),
            document_ids=list(source.get("document_ids") or []),
            hashes=dict(source.get("hashes") or {}),
            versions=dict(source.get("versions") or {}),
        )

    def save_checkpoint(self, checkpoint: KnowledgeSourceCheckpoint, *, deadline: Optional[float] = None) -> None:
        tenant = sanitize_tenant_id(checkpoint.tenant_id)
        cid = opensearch_checkpoint_id(tenant, checkpoint.source_id)
        body = {
            "record_type": RECORD_CHECKPOINT,
            "tenant_id": tenant,
            "source_id": checkpoint.source_id,
            "sync_cursor": checkpoint.sync_cursor,
            "document_ids": list(checkpoint.document_ids),
            "hashes": dict(checkpoint.hashes),
            "versions": dict(checkpoint.versions),
        }
        self._bulk(
            [
                {"index": {"_index": self.index, "_id": cid, "routing": tenant}},
                body,
            ],
            deadline=deadline,
        )

    def record_sync_job(self, job: KnowledgeSyncJob, *, deadline: Optional[float] = None) -> None:
        tenant = sanitize_tenant_id(job.tenant_id)
        latest_id = opensearch_sync_latest_id(tenant, job.source_id)
        job_id = f"{latest_id}::{job.started_at or job.finished_at or 'job'}"
        payload, _ = redact_recursive(job.to_dict())
        payload["record_type"] = RECORD_SYNC_JOB
        payload["tenant_id"] = tenant
        payload["store_backend"] = self.backend
        latest = dict(payload)
        latest["record_type"] = RECORD_SYNC_LATEST
        self._bulk(
            [
                {"index": {"_index": self.index, "_id": job_id, "routing": tenant}},
                payload,
                {"index": {"_index": self.index, "_id": latest_id, "routing": tenant}},
                latest,
            ],
            deadline=deadline,
        )

    def latest_sync_job(
        self, tenant_id: str, source_id: str, *, deadline: Optional[float] = None
    ) -> Optional[KnowledgeSyncJob]:
        tenant = sanitize_tenant_id(tenant_id)
        source = self._get(opensearch_sync_latest_id(tenant, source_id), tenant_id=tenant, deadline=deadline)
        if not source:
            return None
        return KnowledgeSyncJob(
            source_id=str(source.get("source_id") or source_id),
            tenant_id=tenant,
            status=str(source.get("status") or ""),
            created=int(source.get("created") or 0),
            updated=int(source.get("updated") or 0),
            deleted=int(source.get("deleted") or 0),
            unchanged=int(source.get("unchanged") or 0),
            sync_cursor=source.get("sync_cursor"),
            error_type=source.get("error_type"),
            latency_ms=float(source.get("latency_ms") or 0.0),
            errors=list(source.get("errors") or []),
            started_at=source.get("started_at"),
            finished_at=source.get("finished_at"),
            store_backend=str(source.get("store_backend") or self.backend),
        )
