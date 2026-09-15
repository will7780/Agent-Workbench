# -*- coding: utf-8 -*-
"""HttpRagProvider：版本化 JSON 契约，可注入 transport，测试不访问真实网络。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple
from urllib.parse import urljoin

from .knowledge_acl import (
    PERMISSION_MODE_ACL,
    KnowledgeIdentity,
    build_identity_summary,
    check_knowledge_acl,
    identity_from_search_request,
)
from .knowledge_opensearch import remaining_timeout_s, resolve_opensearch_auth
from .knowledge_provider import (
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TOP_K,
    KnowledgeDocument,
    KnowledgeFetchRequest,
    KnowledgeProviderConfigError,
    KnowledgeProviderHealth,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeTimeout,
    _safe_text,
    ensure_knowledge_deadline,
    knowledge_deadline,
)
from .knowledge_remote import (
    PROTOCOL_VERSION,
    classify_http_status,
    classify_transport_exception,
    http_provider_instance_id,
    opaque_remote_document_id,
    process_remote_id_store,
    protocol_compatible,
    resolve_http_error,
    sanitize_remote_error_type,
    sanitize_remote_hits,
    sanitize_remote_title,
    sanitize_updated_at,
    sanitize_version,
    strip_untrusted_hit,
)
from .local_knowledge_provider import sanitize_tenant_id
from .redaction import redact_text

PROVIDER_ID = "http_rag"


@dataclass
class HttpRagHttpResponse:
    status_code: int
    payload: Any = None
    error_type: Optional[str] = None


class HttpRagTransport(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        json_body: Optional[Dict[str, Any]],
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
    ) -> HttpRagHttpResponse: ...


class RequestsHttpRagTransport:
    def request(
        self,
        *,
        method: str,
        url: str,
        json_body: Optional[Dict[str, Any]],
        timeout_s: float,
        headers: Dict[str, str],
        auth: Optional[Tuple[str, str]] = None,
    ) -> HttpRagHttpResponse:
        import requests

        try:
            response = requests.request(
                method.upper(),
                url,
                json=json_body,
                headers=headers,
                auth=auth,
                timeout=max(0.001, float(timeout_s)),
            )
        except requests.Timeout:
            return HttpRagHttpResponse(status_code=0, error_type="knowledge_timeout")
        except requests.RequestException:
            return HttpRagHttpResponse(status_code=0, error_type="knowledge_disconnected")
        error_type = classify_http_status(response.status_code)
        try:
            payload = response.json() if response.content else {}
        except ValueError:
            return HttpRagHttpResponse(
                status_code=response.status_code,
                error_type="knowledge_invalid_response",
            )
        if error_type:
            return HttpRagHttpResponse(status_code=response.status_code, payload=payload, error_type=error_type)
        if not isinstance(payload, dict):
            return HttpRagHttpResponse(
                status_code=response.status_code,
                error_type="knowledge_invalid_response",
            )
        return HttpRagHttpResponse(status_code=response.status_code, payload=payload)


class HttpRagProvider:
    provider_id = PROVIDER_ID

    def __init__(
        self,
        *,
        endpoint: str,
        tenant_id: str = "default",
        search_path: str = "/v1/knowledge/search",
        fetch_path: str = "/v1/knowledge/fetch",
        health_path: str = "/v1/knowledge/health",
        auth_mode: str = "none",
        credential_env: Optional[str] = None,
        username_env: Optional[str] = None,
        protocol_version: str = PROTOCOL_VERSION,
        transport: Optional[HttpRagTransport] = None,
        id_store: Optional[Any] = None,
    ) -> None:
        if not str(endpoint or "").strip():
            raise KnowledgeProviderConfigError("http_rag_config_missing")
        self.endpoint = str(endpoint).strip().rstrip("/")
        self.tenant_id = sanitize_tenant_id(tenant_id)
        self.search_path = str(search_path or "/v1/knowledge/search")
        self.fetch_path = str(fetch_path or "/v1/knowledge/fetch")
        self.health_path = str(health_path or "/v1/knowledge/health")
        self.auth_mode = str(auth_mode or "none").strip().lower() or "none"
        self.credential_env = str(credential_env).strip() if credential_env else None
        self.username_env = str(username_env).strip() if username_env else None
        self.protocol_version = str(protocol_version or PROTOCOL_VERSION)
        self.transport = transport or RequestsHttpRagTransport()
        self.id_store = id_store or process_remote_id_store()
        self.provider_instance_id = http_provider_instance_id(self.endpoint, self.search_path, self.fetch_path)

    def _url(self, path: str) -> str:
        return urljoin(self.endpoint + "/", path.lstrip("/"))

    def _auth(self) -> Tuple[Dict[str, str], Optional[Tuple[str, str]]]:
        return resolve_opensearch_auth(self.auth_mode, self.credential_env, self.username_env)

    def _call(
        self,
        path: str,
        body: Dict[str, Any],
        *,
        deadline: Optional[float],
        method: str = "POST",
    ) -> HttpRagHttpResponse:
        ensure_knowledge_deadline(deadline)
        headers, auth = self._auth()
        try:
            http = self.transport.request(
                method=method,
                url=self._url(path),
                json_body=body,
                timeout_s=remaining_timeout_s(deadline),
                headers=headers,
                auth=auth,
            )
        except KnowledgeTimeout:
            raise
        except Exception as exc:
            return HttpRagHttpResponse(status_code=0, error_type=classify_transport_exception(exc))
        ensure_knowledge_deadline(deadline)
        resolved = resolve_http_error(
            status_code=http.status_code,
            error_type=http.error_type,
            payload=http.payload,
        )
        if resolved:
            return HttpRagHttpResponse(status_code=http.status_code, error_type=resolved)
        return http

    def _fail_open(
        self,
        request: KnowledgeSearchRequest,
        *,
        error_type: str,
        started: float,
        identity: KnowledgeIdentity,
        acl_filtered: int = 0,
        denial_reasons: Optional[Dict[str, int]] = None,
        omission: Optional[str] = None,
    ) -> KnowledgeSearchResponse:
        safe_error = sanitize_remote_error_type(error_type) or "knowledge_backend_error"
        security = safe_error in ("permission_denied", "authentication_denied", "tenant_mismatch")
        resolved_omission = omission or (safe_error if security else None) or safe_error
        return KnowledgeSearchResponse(
            enabled=True,
            used=False,
            provider=self.provider_id,
            tenant_id=self.tenant_id,
            permission_mode=PERMISSION_MODE_ACL,
            query=request.query,
            error_type=None if resolved_omission == "empty_result" else safe_error,
            omission_reason=resolved_omission,
            degraded=not security and resolved_omission != "empty_result",
            latency_ms=(time.perf_counter() - started) * 1000,
            retrieval_mode="http_rag",
            provider_latency_ms=(time.perf_counter() - started) * 1000,
            identity_summary=build_identity_summary(identity),
            acl_filtered_count=acl_filtered,
            acl_denial_reasons=dict(denial_reasons or {}),
            protocol_version=self.protocol_version,
        )

    def _search_body(self, request: KnowledgeSearchRequest, identity: KnowledgeIdentity) -> Dict[str, Any]:
        remaining_ms = int(remaining_timeout_s(request.resolved_deadline()) * 1000)
        return {
            "protocol_version": self.protocol_version,
            "query": request.query,
            "tenant_id": self.tenant_id,
            "identity": build_identity_summary(identity),
            "groups": list(request.groups or []),
            "roles": list(request.roles or []),
            "filters": dict(request.filters or {}),
            "top_k": max(1, int(request.top_k or DEFAULT_TOP_K)),
            "request_id": request.request_id,
            "timeout_ms": remaining_ms,
        }

    def search(self, request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        started = time.perf_counter()
        identity = identity_from_search_request(request)
        if sanitize_tenant_id(request.tenant_id) != self.tenant_id:
            return self._fail_open(
                request,
                error_type="tenant_mismatch",
                started=started,
                identity=identity,
                omission="tenant_mismatch",
            )
        deadline = request.resolved_deadline()
        try:
            http = self._call(self.search_path, self._search_body(request, identity), deadline=deadline)
            remote_error = resolve_http_error(
                status_code=http.status_code,
                error_type=http.error_type,
                payload=http.payload,
            )
            if remote_error:
                return self._fail_open(request, error_type=remote_error, started=started, identity=identity)
            payload = http.payload if isinstance(http.payload, dict) else None
            if payload is None:
                return self._fail_open(
                    request, error_type="knowledge_invalid_response", started=started, identity=identity
                )
            if not protocol_compatible(payload):
                return self._fail_open(
                    request, error_type="knowledge_protocol_incompatible", started=started, identity=identity
                )
            hits, acl_filtered, denial_reasons, id_map = sanitize_remote_hits(
                payload.get("hits"),
                provider_id=self.provider_id,
                identity=identity,
                instance_id=self.provider_instance_id,
            )
            self.id_store.put_many(
                self.tenant_id,
                self.provider_id,
                id_map,
                instance_id=self.provider_instance_id,
            )
            latency = (time.perf_counter() - started) * 1000
            omission = None
            if not hits:
                if acl_filtered:
                    omission = "permission_denied"
                else:
                    omission = "empty_result"
            return KnowledgeSearchResponse(
                enabled=True,
                used=bool(hits),
                provider=self.provider_id,
                tenant_id=self.tenant_id,
                permission_mode=PERMISSION_MODE_ACL,
                query=request.query,
                hits=hits,
                latency_ms=latency,
                omission_reason=omission,
                retrieval_mode="http_rag",
                acl_filtered_count=acl_filtered,
                provider_latency_ms=latency,
                identity_summary=build_identity_summary(identity),
                acl_denial_reasons=denial_reasons,
                protocol_version=self.protocol_version,
            )
        except KnowledgeTimeout:
            return self._fail_open(request, error_type="knowledge_timeout", started=started, identity=identity)
        except Exception:
            return self._fail_open(
                request, error_type="knowledge_search_failed:Exception", started=started, identity=identity
            )

    def fetch(self, request: KnowledgeFetchRequest) -> KnowledgeDocument:
        identity = KnowledgeIdentity(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            groups=tuple(request.groups or []),
            roles=tuple(request.roles or []),
        )
        empty = KnowledgeDocument(
            provider_id=self.provider_id,
            tenant_id=self.tenant_id,
            document_id=request.document_id,
            title="",
            source_uri=None,
            text="",
        )
        if sanitize_tenant_id(request.tenant_id) != self.tenant_id:
            empty.error_type = "tenant_mismatch"
            return empty
        remote_id = self.id_store.get(
            self.tenant_id,
            self.provider_id,
            str(request.document_id),
            instance_id=self.provider_instance_id,
        )
        if not remote_id:
            empty.error_type = "knowledge_invalid_response"
            return empty
        deadline = request.resolved_deadline() or knowledge_deadline(DEFAULT_TIMEOUT_MS)
        try:
            http = self._call(
                self.fetch_path,
                {
                    "protocol_version": self.protocol_version,
                    "tenant_id": self.tenant_id,
                    "document_id": remote_id,
                    "chunk_id": request.chunk_id,
                    "identity": build_identity_summary(identity),
                    "timeout_ms": int(remaining_timeout_s(deadline) * 1000),
                    "request_id": None,
                },
                deadline=deadline,
            )
        except KnowledgeTimeout:
            empty.error_type = "knowledge_timeout"
            return empty
        except Exception as exc:
            empty.error_type = classify_transport_exception(exc)
            return empty
        remote_error = resolve_http_error(
            status_code=http.status_code,
            error_type=http.error_type,
            payload=http.payload,
        )
        if remote_error or not isinstance(http.payload, dict):
            empty.error_type = sanitize_remote_error_type(remote_error) or "knowledge_invalid_response"
            return empty
        if not protocol_compatible(http.payload):
            empty.error_type = "knowledge_protocol_incompatible"
            return empty
        source = strip_untrusted_hit(http.payload.get("document") or http.payload)
        decision = check_knowledge_acl(source.get("acl") if "acl" in source else None, identity, str(source.get("tenant_id") or ""))
        if not decision.allowed:
            empty.error_type = "permission_denied"
            return empty
        opaque_id = opaque_remote_document_id(
            self.tenant_id,
            self.provider_id,
            str(source.get("document_id") or remote_id),
            instance_id=self.provider_instance_id,
        )
        text, _ = redact_text(str(source.get("text") or ""))
        return KnowledgeDocument(
            provider_id=self.provider_id,
            tenant_id=self.tenant_id,
            document_id=opaque_id,
            title=sanitize_remote_title(source.get("title") or "document"),
            source_uri=f"kb://{self.provider_id}/{opaque_id}",
            text=text,
            updated_at=sanitize_updated_at(source.get("updated_at")),
            version=sanitize_version(source.get("version")),
        )

    def health(self) -> KnowledgeProviderHealth:
        details = {"protocol_version": self.protocol_version, "contract_version": PROTOCOL_VERSION}
        deadline = knowledge_deadline(DEFAULT_TIMEOUT_MS)
        try:
            http = self._call(self.health_path, {"protocol_version": self.protocol_version}, deadline=deadline, method="POST")
        except KnowledgeTimeout:
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=False,
                permission_mode=PERMISSION_MODE_ACL,
                error_type="knowledge_timeout",
                details=details,
            )
        except Exception as exc:
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=False,
                permission_mode=PERMISSION_MODE_ACL,
                error_type=classify_transport_exception(exc),
                details=details,
            )
        remote_error = resolve_http_error(
            status_code=http.status_code,
            error_type=http.error_type,
            payload=http.payload,
        )
        if remote_error:
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=False,
                permission_mode=PERMISSION_MODE_ACL,
                error_type=sanitize_remote_error_type(remote_error) or "knowledge_backend_error",
                details=details,
            )
        payload = http.payload if isinstance(http.payload, dict) else {}
        if not protocol_compatible(payload):
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=False,
                permission_mode=PERMISSION_MODE_ACL,
                error_type="knowledge_protocol_incompatible",
                details=details,
            )
        return KnowledgeProviderHealth(
            provider_id=self.provider_id,
            healthy=bool(payload.get("healthy", True)),
            permission_mode=PERMISSION_MODE_ACL,
            details=details,
        )
