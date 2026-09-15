# -*- coding: utf-8 -*-
"""OpenSearchKnowledgeProvider：可注入 transport，服务端 tenant/ACL filter，BM25 + 可选向量。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from .knowledge_acl import (
    PERMISSION_MODE_ACL,
    KnowledgeIdentity,
    build_identity_summary,
    check_knowledge_acl,
    identity_from_search_request,
)
from .knowledge_embedding import EmbeddingUnavailable, cosine_similarity
from .knowledge_fusion import reciprocal_rank_fusion
from .knowledge_provider import (
    DEFAULT_TOP_K,
    KnowledgeDocument,
    KnowledgeFetchRequest,
    KnowledgeProviderConfigError,
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
from .local_knowledge_provider import sanitize_tenant_id
from .redaction import redact_text

PROVIDER_ID = "opensearch"


@dataclass
class OpenSearchHttpResponse:
    status_code: int
    payload: Any = None
    error_type: Optional[str] = None


class OpenSearchTransport(Protocol):
    def search(
        self,
        *,
        endpoint: str,
        index: str,
        body: Dict[str, Any],
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
    ) -> OpenSearchHttpResponse: ...

    def get_document(
        self,
        *,
        endpoint: str,
        index: str,
        document_id: str,
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
        routing: Optional[str] = None,
    ) -> OpenSearchHttpResponse: ...

    def bulk(
        self,
        *,
        endpoint: str,
        index: str,
        operations: List[Dict[str, Any]],
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
    ) -> OpenSearchHttpResponse: ...

    def delete_document(
        self,
        *,
        endpoint: str,
        index: str,
        document_id: str,
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
        routing: Optional[str] = None,
    ) -> OpenSearchHttpResponse: ...


class RequestsOpenSearchTransport:
    """复用项目已有 requests，不引入重量级 OpenSearch 客户端。"""

    def search(
        self,
        *,
        endpoint: str,
        index: str,
        body: Dict[str, Any],
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
    ) -> OpenSearchHttpResponse:
        return self._request(
            "POST",
            f"{endpoint.rstrip('/')}/{index}/_search",
            json_body=body,
            timeout_s=timeout_s,
            headers=headers,
            auth=auth,
        )

    def get_document(
        self,
        *,
        endpoint: str,
        index: str,
        document_id: str,
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
        routing: Optional[str] = None,
    ) -> OpenSearchHttpResponse:
        url = f"{endpoint.rstrip('/')}/{index}/_doc/{document_id}"
        if routing:
            url = f"{url}?routing={routing}"
        return self._request(
            "GET",
            url,
            json_body=None,
            timeout_s=timeout_s,
            headers=headers,
            auth=auth,
        )

    def bulk(
        self,
        *,
        endpoint: str,
        index: str,
        operations: List[Dict[str, Any]],
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
    ) -> OpenSearchHttpResponse:
        import requests

        lines = [json.dumps(item, ensure_ascii=False) for item in operations]
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        bulk_headers = dict(headers or {})
        bulk_headers["Content-Type"] = "application/x-ndjson"
        try:
            response = requests.post(
                f"{endpoint.rstrip('/')}/_bulk",
                params={"index": index},
                data=payload,
                headers=bulk_headers,
                auth=auth,
                timeout=max(0.001, float(timeout_s)),
            )
        except requests.Timeout:
            return OpenSearchHttpResponse(status_code=0, error_type="knowledge_timeout")
        except requests.RequestException:
            return OpenSearchHttpResponse(status_code=0, error_type="provider_error")
        return self._from_http(response.status_code, response.content)

    def delete_document(
        self,
        *,
        endpoint: str,
        index: str,
        document_id: str,
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
        routing: Optional[str] = None,
    ) -> OpenSearchHttpResponse:
        url = f"{endpoint.rstrip('/')}/{index}/_doc/{document_id}"
        if routing:
            url = f"{url}?routing={routing}"
        return self._request(
            "DELETE",
            url,
            json_body=None,
            timeout_s=timeout_s,
            headers=headers,
            auth=auth,
        )

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Optional[Dict[str, Any]],
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]],
    ) -> OpenSearchHttpResponse:
        import requests

        try:
            response = requests.request(
                method,
                url,
                json=json_body,
                headers=headers,
                auth=auth,
                timeout=max(0.001, float(timeout_s)),
            )
        except requests.Timeout:
            return OpenSearchHttpResponse(status_code=0, error_type="knowledge_timeout")
        except requests.RequestException:
            return OpenSearchHttpResponse(status_code=0, error_type="provider_error")
        return self._from_http(response.status_code, response.content)

    @staticmethod
    def _from_http(status_code: int, content: bytes) -> OpenSearchHttpResponse:
        if status_code == 404:
            return OpenSearchHttpResponse(status_code=status_code, payload={}, error_type="not_found")
        if status_code == 429:
            return OpenSearchHttpResponse(status_code=status_code, error_type="knowledge_rate_limited")
        if status_code >= 500:
            return OpenSearchHttpResponse(status_code=status_code, error_type="knowledge_backend_error")
        if status_code >= 400:
            return OpenSearchHttpResponse(status_code=status_code, error_type="provider_error")
        try:
            payload = json.loads(content.decode("utf-8")) if content else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return OpenSearchHttpResponse(status_code=status_code, error_type="knowledge_invalid_response")
        if not isinstance(payload, dict):
            return OpenSearchHttpResponse(status_code=status_code, error_type="knowledge_invalid_response")
        return OpenSearchHttpResponse(status_code=status_code, payload=payload)


def remaining_timeout_s(deadline: Optional[float], *, default: float = 10.0) -> float:
    if deadline is None:
        return default
    return max(0.001, deadline - time.monotonic())


def resolve_opensearch_auth(
    auth_mode: str,
    credential_env: Optional[str],
    username_env: Optional[str],
) -> Tuple[Dict[str, str], Optional[Tuple[str, str]]]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    mode = str(auth_mode or "none").strip().lower() or "none"
    if mode in ("none", "", "disabled"):
        return headers, None
    credential = os.environ.get(credential_env, "") if credential_env else ""
    username = os.environ.get(username_env, "") if username_env else ""
    if mode == "basic":
        if not username or not credential:
            return headers, None
        return headers, (username, credential)
    if mode == "bearer" and credential:
        headers["Authorization"] = f"Bearer {credential}"
    return headers, None


def build_opensearch_acl_filter(identity: KnowledgeIdentity) -> Dict[str, Any]:
    ident = identity.normalized()
    user = ident.user_id or ""
    groups = list(ident.groups)
    roles = list(ident.roles)
    deny_clauses: List[Dict[str, Any]] = []
    if user:
        deny_clauses.append({"term": {"acl.denied_users": user}})
    if groups:
        deny_clauses.append({"terms": {"acl.denied_groups": groups}})
    if roles:
        deny_clauses.append({"terms": {"acl.denied_roles": roles}})

    allow_should: List[Dict[str, Any]] = [{"term": {"acl.visibility": "tenant"}}]
    if user:
        allow_should.append(
            {
                "bool": {
                    "must": [
                        {"term": {"acl.visibility": "private"}},
                        {"term": {"acl.allowed_users": user}},
                    ]
                }
            }
        )
        restricted_should: List[Dict[str, Any]] = [{"term": {"acl.allowed_users": user}}]
        if groups:
            restricted_should.append({"terms": {"acl.allowed_groups": groups}})
        if roles:
            restricted_should.append({"terms": {"acl.allowed_roles": roles}})
        allow_should.append(
            {
                "bool": {
                    "must": [
                        {"term": {"acl.visibility": "restricted"}},
                        {"bool": {"should": restricted_should, "minimum_should_match": 1}},
                    ]
                }
            }
        )
    elif groups or roles:
        restricted_should = []
        if groups:
            restricted_should.append({"terms": {"acl.allowed_groups": groups}})
        if roles:
            restricted_should.append({"terms": {"acl.allowed_roles": roles}})
        if restricted_should:
            allow_should.append(
                {
                    "bool": {
                        "must": [
                            {"term": {"acl.visibility": "restricted"}},
                            {"bool": {"should": restricted_should, "minimum_should_match": 1}},
                        ]
                    }
                }
            )

    filters: List[Dict[str, Any]] = [{"term": {"tenant_id": ident.tenant_id}}]
    if deny_clauses:
        filters.append({"bool": {"must_not": deny_clauses}})
    filters.append({"bool": {"should": allow_should, "minimum_should_match": 1}})
    return {"bool": {"filter": filters}}


def build_bm25_query(query: str, identity: KnowledgeIdentity, *, size: int) -> Dict[str, Any]:
    acl_filter = build_opensearch_acl_filter(identity)
    return {
        "size": size,
        "query": {
            "bool": {
                "must": [{"match": {"text": query}}],
                "filter": acl_filter.get("bool", {}).get("filter") or [],
            }
        },
    }


def build_vector_query(
    vector: Sequence[float],
    identity: KnowledgeIdentity,
    *,
    size: int,
    vector_field: str,
) -> Dict[str, Any]:
    acl_filter = build_opensearch_acl_filter(identity)
    return {
        "size": size,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            vector_field: {
                                "vector": [float(v) for v in vector],
                                "k": size,
                            }
                        }
                    }
                ],
                "filter": acl_filter.get("bool", {}).get("filter") or [],
            }
        },
    }


class OpenSearchKnowledgeProvider:
    provider_id = PROVIDER_ID

    def __init__(
        self,
        *,
        endpoint: str,
        index: str,
        tenant_id: str = "default",
        auth_mode: str = "none",
        credential_env: Optional[str] = None,
        username_env: Optional[str] = None,
        enable_vector: bool = False,
        vector_field: str = "embedding",
        enable_rerank: bool = False,
        k_bm25: int = 20,
        k_vector: int = 20,
        transport: Optional[OpenSearchTransport] = None,
        embedding: Any = None,
        sync_store: Any = None,
        source_id: str = "opensearch",
    ) -> None:
        if not str(endpoint or "").strip() or not str(index or "").strip():
            raise KnowledgeProviderConfigError("opensearch_config_missing")
        self.endpoint = str(endpoint).strip()
        self.index = str(index).strip()
        self.tenant_id = sanitize_tenant_id(tenant_id)
        self.auth_mode = str(auth_mode or "none").strip().lower() or "none"
        self.credential_env = str(credential_env).strip() if credential_env else None
        self.username_env = str(username_env).strip() if username_env else None
        self.enable_vector = bool(enable_vector)
        self.vector_field = str(vector_field or "embedding")
        self.enable_rerank = bool(enable_rerank)
        self.k_bm25 = max(1, int(k_bm25 or 20))
        self.k_vector = max(1, int(k_vector or 20))
        self.transport = transport or RequestsOpenSearchTransport()
        self.embedding = embedding
        self.sync_store = sync_store
        self.source_id = source_id
        self.last_search_bodies: List[Dict[str, Any]] = []

    def _auth_headers(self) -> Tuple[Dict[str, str], Optional[Tuple[str, str]]]:
        return resolve_opensearch_auth(self.auth_mode, self.credential_env, self.username_env)

    def _sync_status_error_type(self, exc: BaseException) -> str:
        if isinstance(exc, KnowledgeTimeout):
            return "knowledge_timeout"
        message = str(exc).strip()
        if message and len(message) < 80 and all(ch.isalnum() or ch in "._:" for ch in message):
            return message
        return f"sync_status_unavailable:{type(exc).__name__}"

    def _sync_status(self, *, deadline: Optional[float] = None) -> Dict[str, Any]:
        backend = getattr(self.sync_store, "backend", None) or "opensearch"
        payload: Dict[str, Any] = {"store_backend": backend}
        if self.sync_store is None:
            return payload
        try:
            ensure_knowledge_deadline(deadline)
            job = self.sync_store.latest_sync_job(self.tenant_id, self.source_id, deadline=deadline)
            ensure_knowledge_deadline(deadline)
        except Exception as exc:
            payload["status_unavailable"] = True
            payload["error_type"] = self._sync_status_error_type(exc)
            return payload
        if job is None:
            return payload
        data = job.to_dict() if hasattr(job, "to_dict") else dict(job)
        data.setdefault("store_backend", backend)
        payload.update(data)
        return payload

    def _sync_version(self, status: Optional[Dict[str, Any]] = None) -> Optional[str]:
        payload = status if status is not None else self._sync_status()
        return payload.get("sync_cursor") if payload else None

    def _fail_open(
        self,
        request: KnowledgeSearchRequest,
        *,
        error_type: str,
        started: float,
        identity: Optional[KnowledgeIdentity] = None,
        degraded_reason: Optional[str] = None,
        deadline: Optional[float] = None,
        sync_status: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeSearchResponse:
        ident = identity or identity_from_search_request(request)
        status = sync_status if sync_status is not None else self._sync_status(deadline=deadline)
        return KnowledgeSearchResponse(
            enabled=True,
            used=False,
            provider=self.provider_id,
            tenant_id=self.tenant_id,
            permission_mode=PERMISSION_MODE_ACL,
            query=request.query,
            error_type=error_type,
            omission_reason=error_type,
            degraded=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            retrieval_mode="bm25",
            degraded_reason=degraded_reason or error_type,
            provider_latency_ms=(time.perf_counter() - started) * 1000,
            identity_summary=build_identity_summary(ident),
            sync_status=status,
            sync_version=self._sync_version(status),
        )

    def _http_search(self, body: Dict[str, Any], *, deadline: Optional[float]) -> OpenSearchHttpResponse:
        ensure_knowledge_deadline(deadline)
        headers, auth = self._auth_headers()
        timeout_s = remaining_timeout_s(deadline)
        self.last_search_bodies.append(body)
        return self.transport.search(
            endpoint=self.endpoint,
            index=self.index,
            body=body,
            timeout_s=timeout_s,
            headers=headers,
            auth=auth,
        )

    def _parse_hits(
        self,
        payload: Any,
        *,
        identity: KnowledgeIdentity,
        retrieval_reason: str,
        max_age_seconds: Optional[float] = None,
        now_ts: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], int, Dict[str, int], int]:
        acl_filtered = 0
        stale_filtered = 0
        denial_reasons: Dict[str, int] = {}
        hits: List[Dict[str, Any]] = []
        raw_hits = ((payload or {}).get("hits") or {}).get("hits") or []
        if not isinstance(raw_hits, list):
            return [], 0, {}, 0
        for item in raw_hits:
            if not isinstance(item, dict):
                continue
            source = item.get("_source") if isinstance(item.get("_source"), dict) else {}
            doc_tenant = str(source.get("tenant_id") or "")
            raw_acl = source.get("acl") if "acl" in source else None
            decision = check_knowledge_acl(raw_acl, identity, doc_tenant)
            if not decision.allowed:
                acl_filtered += 1
                denial_reasons[decision.reason] = denial_reasons.get(decision.reason, 0) + 1
                continue
            updated_at = str(source.get("updated_at") or "")
            if max_age_seconds and updated_at:
                try:
                    from datetime import datetime

                    parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    age = (time.time() if now_ts is None else now_ts) - parsed.timestamp()
                    if age > max_age_seconds:
                        stale_filtered += 1
                        continue
                except (ValueError, OSError, TypeError):
                    pass
            snippet, _ = redact_text(str(source.get("text") or source.get("snippet") or ""))
            hit = {
                "document_id": str(source.get("document_id") or item.get("_id") or ""),
                "chunk_id": str(source.get("chunk_id") or "c000"),
                "title": _safe_text(source.get("title") or "document", 200),
                "snippet": _safe_text(snippet, 400),
                "score": float(item.get("_score") or source.get("score") or 0.0),
                "source_id": str(source.get("source_id") or self.source_id),
                "source_type": str(source.get("source_type") or "opensearch"),
                "source_uri": _safe_text(source.get("source_uri") or "", 240),
                "updated_at": updated_at or None,
                "version": source.get("version"),
                "embedding": source.get("embedding") if isinstance(source.get("embedding"), list) else None,
                "retrieval_reason": retrieval_reason,
                "tenant_id": sanitize_tenant_id(doc_tenant or identity.tenant_id),
            }
            hits.append(hit)
        return hits, acl_filtered, denial_reasons, stale_filtered

    def _to_results(self, hits: List[Dict[str, Any]], *, top_k: int) -> List[KnowledgeSearchResult]:
        results: List[KnowledgeSearchResult] = []
        for hit in hits[:top_k]:
            document_id = str(hit.get("document_id") or "")
            chunk_id = str(hit.get("chunk_id") or "c000")
            results.append(
                KnowledgeSearchResult(
                    provider_id=self.provider_id,
                    tenant_id=str(hit.get("tenant_id") or self.tenant_id),
                    source_type=str(hit.get("source_type") or "opensearch"),
                    source_id=str(hit.get("source_id") or self.source_id),
                    document_id=document_id,
                    chunk_id=chunk_id,
                    title=str(hit.get("title") or "document"),
                    snippet=str(hit.get("snippet") or ""),
                    score=float(hit.get("fusion_score") or hit.get("score") or 0.0),
                    source_uri=hit.get("source_uri") or f"kb://{self.provider_id}/{document_id}",
                    updated_at=hit.get("updated_at"),
                    version=str(hit.get("version") or "") or None,
                    citation=format_citation(self.provider_id, document_id, chunk_id),
                    retrieval_reason=str(hit.get("retrieval_reason") or "bm25"),
                    metadata={"channels": list(hit.get("channels") or [])},
                )
            )
        return results

    def search(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        started = time.perf_counter()
        identity = identity_from_search_request(request)
        if sanitize_tenant_id(request.tenant_id) != self.tenant_id:
            return KnowledgeSearchResponse(
                enabled=True,
                used=False,
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_ACL,
                query=request.query,
                error_type="tenant_mismatch",
                omission_reason="tenant_mismatch",
                retrieval_mode="bm25",
                identity_summary=build_identity_summary(identity),
                sync_status=self._sync_status(deadline=None),
            )
        deadline = request.resolved_deadline()
        top_k = max(1, int(request.top_k or DEFAULT_TOP_K))
        filters = request.filters if isinstance(request.filters, dict) else {}
        max_age = filters.get("max_age_seconds")
        try:
            max_age_seconds = float(max_age) if max_age is not None else None
        except (TypeError, ValueError):
            max_age_seconds = None
        try:
            ensure_knowledge_deadline(deadline)
            bm25_body = build_bm25_query(request.query, identity, size=max(self.k_bm25, top_k))
            bm25_http = self._http_search(bm25_body, deadline=deadline)
            if bm25_http.error_type:
                return self._fail_open(
                    request,
                    error_type=bm25_http.error_type,
                    started=started,
                    identity=identity,
                    deadline=deadline,
                )
            bm25_hits, acl_filtered, denial_reasons, stale_filtered = self._parse_hits(
                bm25_http.payload,
                identity=identity,
                retrieval_reason="bm25",
                max_age_seconds=max_age_seconds,
            )
            vector_hits: List[Dict[str, Any]] = []
            degraded_reason = None
            retrieval_mode = "bm25"
            if self.enable_vector:
                try:
                    if self.embedding is None:
                        raise EmbeddingUnavailable("vector_unavailable")
                    ensure_knowledge_deadline(deadline)
                    query_vectors = self.embedding.embed([request.query], deadline=deadline)
                    ensure_knowledge_deadline(deadline)
                    if not query_vectors:
                        raise EmbeddingUnavailable("vector_unavailable")
                    vector_body = build_vector_query(
                        query_vectors[0],
                        identity,
                        size=max(self.k_vector, top_k),
                        vector_field=self.vector_field,
                    )
                    vector_http = self._http_search(vector_body, deadline=deadline)
                    ensure_knowledge_deadline(deadline)
                    if vector_http.error_type:
                        if vector_http.error_type == "knowledge_timeout" or knowledge_deadline_exceeded(deadline):
                            raise KnowledgeTimeout()
                        degraded_reason = "vector_unavailable"
                    else:
                        extra_hits, extra_acl, extra_denial, extra_stale = self._parse_hits(
                            vector_http.payload,
                            identity=identity,
                            retrieval_reason="vector",
                            max_age_seconds=max_age_seconds,
                        )
                        vector_hits = extra_hits
                        acl_filtered += extra_acl
                        stale_filtered += extra_stale
                        for reason, count in extra_denial.items():
                            denial_reasons[reason] = denial_reasons.get(reason, 0) + count
                        retrieval_mode = "hybrid"
                except KnowledgeTimeout:
                    raise
                except EmbeddingUnavailable:
                    if knowledge_deadline_exceeded(deadline):
                        raise KnowledgeTimeout()
                    degraded_reason = "vector_unavailable"
                    retrieval_mode = "bm25"
                except Exception:
                    if knowledge_deadline_exceeded(deadline):
                        raise KnowledgeTimeout()
                    degraded_reason = "vector_unavailable"
                    retrieval_mode = "bm25"

            fused = reciprocal_rank_fusion(bm25_hits, vector_hits)
            reranked_count = 0
            if self.enable_rerank and fused and self.embedding is not None and degraded_reason != "vector_unavailable":
                try:
                    query_vec = self.embedding.embed([request.query], deadline=deadline)[0]
                    ensure_knowledge_deadline(deadline)
                    for hit in fused:
                        embedding = hit.get("embedding")
                        if isinstance(embedding, list) and embedding:
                            hit["rerank_score"] = cosine_similarity(query_vec, embedding)
                        else:
                            hit["rerank_score"] = float(hit.get("fusion_score") or 0.0)
                    fused.sort(
                        key=lambda hit: (
                            -float(hit.get("rerank_score") or 0.0),
                            str(hit.get("document_id") or ""),
                            str(hit.get("chunk_id") or ""),
                        )
                    )
                    ensure_knowledge_deadline(deadline)
                    reranked_count = len(fused)
                    for hit in fused:
                        reason = str(hit.get("retrieval_reason") or "")
                        if "rerank" not in reason:
                            hit["retrieval_reason"] = f"{reason}+rerank" if reason else "rerank"
                except KnowledgeTimeout:
                    raise
                except Exception:
                    if knowledge_deadline_exceeded(deadline):
                        raise KnowledgeTimeout()
                    if degraded_reason is None:
                        degraded_reason = "rerank_unavailable"

            results = self._to_results(fused, top_k=top_k)
            omission = None
            if not results:
                if acl_filtered and not bm25_hits and not vector_hits:
                    omission = "permission_denied"
                elif stale_filtered and not fused:
                    omission = "stale_document"
                else:
                    omission = "empty_result"
            latency = (time.perf_counter() - started) * 1000
            sync_status = self._sync_status(deadline=deadline)
            return KnowledgeSearchResponse(
                enabled=True,
                used=bool(results),
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_ACL,
                query=request.query,
                hits=results,
                latency_ms=latency,
                omission_reason=omission,
                degraded=bool(degraded_reason),
                retrieval_mode=retrieval_mode,
                bm25_hit_count=len(bm25_hits),
                vector_hit_count=len(vector_hits),
                fused_hit_count=len(fused),
                reranked_count=reranked_count,
                acl_filtered_count=acl_filtered,
                stale_filtered_count=stale_filtered,
                degraded_reason=degraded_reason,
                sync_version=self._sync_version(sync_status),
                provider_latency_ms=latency,
                identity_summary=build_identity_summary(identity),
                acl_denial_reasons=denial_reasons,
                sync_status=sync_status,
            )
        except KnowledgeTimeout:
            return self._fail_open(
                request,
                error_type="knowledge_timeout",
                started=started,
                identity=identity,
                deadline=deadline,
            )
        except Exception as exc:
            return self._fail_open(
                request,
                error_type=f"knowledge_search_failed:{type(exc).__name__}",
                started=started,
                identity=identity,
                deadline=deadline,
            )

    def fetch(self, request: KnowledgeFetchRequest) -> KnowledgeDocument:
        identity = KnowledgeIdentity(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            groups=tuple(request.groups or []),
            roles=tuple(request.roles or []),
        )
        if sanitize_tenant_id(request.tenant_id) != self.tenant_id:
            return KnowledgeDocument(
                provider_id=self.provider_id,
                tenant_id=self.tenant_id,
                document_id=request.document_id,
                title="",
                source_uri=None,
                text="",
                error_type="tenant_mismatch",
            )
        headers, auth = self._auth_headers()
        http = self.transport.get_document(
            endpoint=self.endpoint,
            index=self.index,
            document_id=str(request.document_id),
            timeout_s=5.0,
            headers=headers,
            auth=auth,
            routing=self.tenant_id,
        )
        if http.error_type or not isinstance(http.payload, dict):
            return KnowledgeDocument(
                provider_id=self.provider_id,
                tenant_id=self.tenant_id,
                document_id=request.document_id,
                title="",
                source_uri=None,
                text="",
                error_type=http.error_type or "provider_error",
            )
        source = http.payload.get("_source") if isinstance(http.payload.get("_source"), dict) else http.payload
        decision = check_knowledge_acl(source.get("acl") if isinstance(source, dict) else None, identity, str((source or {}).get("tenant_id") or ""))
        if not decision.allowed:
            return KnowledgeDocument(
                provider_id=self.provider_id,
                tenant_id=self.tenant_id,
                document_id=request.document_id,
                title="",
                source_uri=None,
                text="",
                error_type="permission_denied",
            )
        text, _ = redact_text(str(source.get("text") or ""))
        return KnowledgeDocument(
            provider_id=self.provider_id,
            tenant_id=self.tenant_id,
            document_id=str(source.get("document_id") or request.document_id),
            title=_safe_text(source.get("title") or "document", 200),
            source_uri=_safe_text(source.get("source_uri") or "", 240) or None,
            text=text,
            updated_at=source.get("updated_at"),
            version=str(source.get("version") or "") or None,
        )

    def health(self) -> KnowledgeProviderHealth:
        return KnowledgeProviderHealth(
            provider_id=self.provider_id,
            healthy=True,
            permission_mode=PERMISSION_MODE_ACL,
            details={
                "endpoint": self.endpoint,
                "index": self.index,
                "auth_mode": self.auth_mode,
                "credential_env": self.credential_env,
                "username_env": self.username_env,
                "enable_vector": self.enable_vector,
            },
        )
