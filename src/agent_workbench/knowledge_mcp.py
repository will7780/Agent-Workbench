# -*- coding: utf-8 -*-
"""MCPKnowledgeProvider：可注入 MCP client Protocol，不绑定具体 SDK。"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Protocol, Tuple

from .knowledge_acl import (
    PERMISSION_MODE_ACL,
    KnowledgeIdentity,
    build_identity_summary,
    check_knowledge_acl,
    identity_from_search_request,
)
from .knowledge_opensearch import remaining_timeout_s
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
    ensure_knowledge_deadline,
    knowledge_deadline,
)
from .knowledge_remote import (
    PROTOCOL_VERSION,
    classify_remote_error,
    classify_transport_exception,
    mcp_provider_instance_id,
    opaque_remote_document_id,
    process_remote_id_store,
    protocol_compatible,
    sanitize_remote_error_type,
    sanitize_remote_hits,
    sanitize_remote_title,
    sanitize_updated_at,
    sanitize_version,
    strip_untrusted_hit,
)
from .local_knowledge_provider import sanitize_tenant_id
from .redaction import redact_text

PROVIDER_ID = "mcp"


class KnowledgeMcpClient(Protocol):
    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        timeout_s: float,
    ) -> Dict[str, Any]: ...


class MCPKnowledgeProvider:
    provider_id = PROVIDER_ID

    def __init__(
        self,
        *,
        server_id: str,
        tenant_id: str = "default",
        search_tool: str = "search",
        fetch_tool: str = "fetch",
        health_tool: str = "health",
        credential_env: Optional[str] = None,
        protocol_version: str = PROTOCOL_VERSION,
        client: Optional[KnowledgeMcpClient] = None,
        transport_config: Optional[Dict[str, Any]] = None,
        id_store: Optional[Any] = None,
    ) -> None:
        if not str(server_id or "").strip():
            raise KnowledgeProviderConfigError("mcp_config_missing")
        if client is None:
            raise KnowledgeProviderConfigError("mcp_client_missing")
        self.server_id = str(server_id).strip()
        self.tenant_id = sanitize_tenant_id(tenant_id)
        self.search_tool = str(search_tool or "search")
        self.fetch_tool = str(fetch_tool or "fetch")
        self.health_tool = str(health_tool or "health")
        self.credential_env = str(credential_env).strip() if credential_env else None
        self.protocol_version = str(protocol_version or PROTOCOL_VERSION)
        self.client = client
        self.transport_config = dict(transport_config or {})
        self.id_store = id_store or process_remote_id_store()
        self.provider_instance_id = mcp_provider_instance_id(self.server_id, self.search_tool, self.fetch_tool)

    def _call(self, tool: str, arguments: Dict[str, Any], *, deadline: Optional[float]) -> Dict[str, Any]:
        ensure_knowledge_deadline(deadline)
        try:
            result = self.client.call_tool(tool, arguments, timeout_s=remaining_timeout_s(deadline))
        except KnowledgeTimeout:
            raise
        except Exception as exc:
            return {"error_type": classify_transport_exception(exc)}
        ensure_knowledge_deadline(deadline)
        return result if isinstance(result, dict) else {"error_type": "knowledge_invalid_response"}

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
            retrieval_mode="mcp",
            provider_latency_ms=(time.perf_counter() - started) * 1000,
            identity_summary=build_identity_summary(identity),
            acl_filtered_count=acl_filtered,
            acl_denial_reasons=dict(denial_reasons or {}),
            protocol_version=self.protocol_version,
        )

    def _search_arguments(self, request: KnowledgeSearchRequest, identity: KnowledgeIdentity) -> Dict[str, Any]:
        remaining_ms = int(remaining_timeout_s(request.resolved_deadline()) * 1000)
        return {
            "protocol_version": self.protocol_version,
            "server_id": self.server_id,
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
            raw = self._call(self.search_tool, self._search_arguments(request, identity), deadline=deadline)
            error_type = classify_remote_error(raw)
            if error_type:
                return self._fail_open(request, error_type=error_type, started=started, identity=identity)
            payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
            if not isinstance(payload, dict):
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
                omission = "permission_denied" if acl_filtered else "empty_result"
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
                retrieval_mode="mcp",
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
            raw = self._call(
                self.fetch_tool,
                {
                    "protocol_version": self.protocol_version,
                    "server_id": self.server_id,
                    "tenant_id": self.tenant_id,
                    "document_id": remote_id,
                    "chunk_id": request.chunk_id,
                    "identity": build_identity_summary(identity),
                    "timeout_ms": int(remaining_timeout_s(deadline) * 1000),
                },
                deadline=deadline,
            )
        except KnowledgeTimeout:
            empty.error_type = "knowledge_timeout"
            return empty
        except Exception as exc:
            empty.error_type = classify_transport_exception(exc)
            return empty
        remote_error = classify_remote_error(raw)
        if remote_error:
            empty.error_type = remote_error
            return empty
        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
        if not isinstance(payload, dict) or not protocol_compatible(payload):
            empty.error_type = "knowledge_protocol_incompatible" if isinstance(payload, dict) else "knowledge_invalid_response"
            return empty
        source = strip_untrusted_hit(payload.get("document") or payload)
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
        details = {
            "protocol_version": self.protocol_version,
            "contract_version": PROTOCOL_VERSION,
        }
        deadline = knowledge_deadline(DEFAULT_TIMEOUT_MS)
        try:
            raw = self._call(
                self.health_tool,
                {"protocol_version": self.protocol_version, "server_id": self.server_id},
                deadline=deadline,
            )
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
        remote_error = classify_remote_error(raw)
        if remote_error:
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=False,
                permission_mode=PERMISSION_MODE_ACL,
                error_type=remote_error,
                details=details,
            )
        payload = raw.get("payload") if isinstance(raw.get("payload"), dict) else raw
        if not isinstance(payload, dict) or not protocol_compatible(payload):
            return KnowledgeProviderHealth(
                provider_id=self.provider_id,
                healthy=False,
                permission_mode=PERMISSION_MODE_ACL,
                error_type="knowledge_protocol_incompatible" if isinstance(payload, dict) else "knowledge_invalid_response",
                details=details,
            )
        return KnowledgeProviderHealth(
            provider_id=self.provider_id,
            healthy=bool(payload.get("healthy", True)),
            permission_mode=PERMISSION_MODE_ACL,
            details=details,
        )
