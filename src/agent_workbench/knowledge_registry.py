# -*- coding: utf-8 -*-
"""Knowledge Provider Registry：provider_id -> factory，未知实现返回结构化 unavailable。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_provider import (
    DEFAULT_TIMEOUT_MS,
    PERMISSION_MODE_DISABLED,
    PERMISSION_MODE_TENANT_ONLY,
    PERMISSION_MODE_UNAVAILABLE,
    KnowledgeProvider,
    KnowledgeProviderConfigError,
    KnowledgeProviderFactory,
    KnowledgeSearchRequest,
    KnowledgeTimeout,
    empty_knowledge_search,
    ensure_phase2_search_fields,
    knowledge_deadline,
    knowledge_deadline_exceeded,
    normalize_knowledge_config,
)
from .knowledge_acl import KnowledgeIdentity, build_identity_summary
from .redaction import redact_text

_PROVIDER_FACTORIES: Dict[str, KnowledgeProviderFactory] = {}


def register_knowledge_provider(provider_id: str, factory: KnowledgeProviderFactory) -> None:
    key = str(provider_id or "").strip()
    if not key:
        raise ValueError("provider_id is required")
    _PROVIDER_FACTORIES[key] = factory


def unregister_knowledge_provider(provider_id: str) -> None:
    _PROVIDER_FACTORIES.pop(str(provider_id or "").strip(), None)


def registered_knowledge_provider_ids() -> Tuple[str, ...]:
    return tuple(_PROVIDER_FACTORIES.keys())


def _build_local_knowledge_provider(cfg: Dict[str, Any]) -> KnowledgeProvider:
    from .local_knowledge_provider import LocalKnowledgeProvider

    root = cfg.get("root")
    if not root:
        raise KnowledgeProviderConfigError("knowledge_root_missing")
    return LocalKnowledgeProvider(
        root=root,
        tenant_id=cfg.get("tenant_id") or "default",
        index_dir=cfg.get("index_dir"),
    )


register_knowledge_provider("local", _build_local_knowledge_provider)


def _build_opensearch_knowledge_provider(cfg: Dict[str, Any]) -> KnowledgeProvider:
    from .knowledge_opensearch import OpenSearchKnowledgeProvider, RequestsOpenSearchTransport
    from .knowledge_opensearch_store import OpenSearchKnowledgeStore

    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    os_cfg = providers.get("opensearch") if isinstance(providers.get("opensearch"), dict) else {}
    endpoint = os_cfg.get("endpoint") or cfg.get("endpoint")
    index = os_cfg.get("index") or cfg.get("index")
    if not endpoint or not index:
        raise KnowledgeProviderConfigError("opensearch_config_missing")
    transport = cfg.get("_transport") or RequestsOpenSearchTransport()
    source_id = str(os_cfg.get("source_id") or "opensearch")
    auth_mode = str(os_cfg.get("auth_mode") or cfg.get("auth_mode") or "none")
    credential_env = os_cfg.get("credential_env") or cfg.get("credential_env")
    username_env = os_cfg.get("username_env") or cfg.get("username_env")
    store = cfg.get("_sync_store") or OpenSearchKnowledgeStore(
        endpoint=str(endpoint),
        index=str(index),
        auth_mode=auth_mode,
        credential_env=credential_env,
        username_env=username_env,
        transport=transport,
        source_id=source_id,
    )
    return OpenSearchKnowledgeProvider(
        endpoint=str(endpoint),
        index=str(index),
        tenant_id=cfg.get("tenant_id") or "default",
        auth_mode=auth_mode,
        credential_env=credential_env,
        username_env=username_env,
        enable_vector=bool(os_cfg.get("enable_vector") or cfg.get("enable_vector")),
        vector_field=str(os_cfg.get("vector_field") or cfg.get("vector_field") or "embedding"),
        enable_rerank=bool(os_cfg.get("enable_rerank") or os_cfg.get("rerank") or cfg.get("enable_rerank")),
        k_bm25=int(os_cfg.get("k_bm25") or 20),
        k_vector=int(os_cfg.get("k_vector") or 20),
        transport=transport,
        embedding=cfg.get("_embedding"),
        sync_store=store,
        source_id=source_id,
    )


register_knowledge_provider("opensearch", _build_opensearch_knowledge_provider)


def _build_http_rag_knowledge_provider(cfg: Dict[str, Any]) -> KnowledgeProvider:
    from .knowledge_http_rag import HttpRagProvider, RequestsHttpRagTransport

    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    http_cfg = providers.get("http_rag") if isinstance(providers.get("http_rag"), dict) else {}
    endpoint = http_cfg.get("endpoint") or cfg.get("endpoint")
    if not endpoint:
        raise KnowledgeProviderConfigError("http_rag_config_missing")
    return HttpRagProvider(
        endpoint=str(endpoint),
        tenant_id=cfg.get("tenant_id") or "default",
        search_path=str(http_cfg.get("search_path") or "/v1/knowledge/search"),
        fetch_path=str(http_cfg.get("fetch_path") or "/v1/knowledge/fetch"),
        health_path=str(http_cfg.get("health_path") or "/v1/knowledge/health"),
        auth_mode=str(http_cfg.get("auth_mode") or cfg.get("auth_mode") or "none"),
        credential_env=http_cfg.get("credential_env") or cfg.get("credential_env"),
        username_env=http_cfg.get("username_env") or cfg.get("username_env"),
        protocol_version=str(http_cfg.get("protocol_version") or "1"),
        transport=cfg.get("_http_transport") or RequestsHttpRagTransport(),
        id_store=cfg.get("_id_map_store"),
    )


register_knowledge_provider("http_rag", _build_http_rag_knowledge_provider)


def _build_mcp_knowledge_provider(cfg: Dict[str, Any]) -> KnowledgeProvider:
    from .knowledge_mcp import MCPKnowledgeProvider

    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    mcp_cfg = providers.get("mcp") if isinstance(providers.get("mcp"), dict) else {}
    server_id = mcp_cfg.get("server_id") or cfg.get("server_id")
    client = cfg.get("_mcp_client")
    if not server_id:
        raise KnowledgeProviderConfigError("mcp_config_missing")
    return MCPKnowledgeProvider(
        server_id=str(server_id),
        tenant_id=cfg.get("tenant_id") or "default",
        search_tool=str(mcp_cfg.get("search_tool") or "search"),
        fetch_tool=str(mcp_cfg.get("fetch_tool") or "fetch"),
        health_tool=str(mcp_cfg.get("health_tool") or "health"),
        credential_env=mcp_cfg.get("credential_env") or cfg.get("credential_env"),
        protocol_version=str(mcp_cfg.get("protocol_version") or "1"),
        client=client,
        transport_config=mcp_cfg.get("transport") if isinstance(mcp_cfg.get("transport"), dict) else {},
        id_store=cfg.get("_id_map_store"),
    )


register_knowledge_provider("mcp", _build_mcp_knowledge_provider)


def build_knowledge_provider(
    knowledge_config: Optional[Dict[str, Any]],
) -> Tuple[Optional[KnowledgeProvider], Optional[str], Dict[str, Any]]:
    """
    按配置构建 provider。
    返回 (provider, error_type, normalized_config)。
    未知 provider 不在业务代码中硬编码公司分支，只返回 unavailable。
    """
    cfg = normalize_knowledge_config(knowledge_config)
    if not cfg.get("enabled"):
        return None, None, cfg
    provider_id = str(cfg.get("provider") or "local").strip() or "local"
    factory = _PROVIDER_FACTORIES.get(provider_id)
    if factory is None:
        return None, "knowledge_provider_unavailable", cfg
    try:
        provider = factory(cfg)
        return provider, None, cfg
    except KnowledgeProviderConfigError as exc:
        reason = str(exc).strip() or "knowledge_provider_init_failed"
        return None, reason, cfg
    except Exception as exc:
        return None, f"knowledge_provider_init_failed:{type(exc).__name__}", cfg


def _finalize_search_payload(
    payload: Dict[str, Any],
    *,
    request: Optional[KnowledgeSearchRequest] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    search = ensure_phase2_search_fields(payload)
    if request is not None and not search.get("identity_summary"):
        search["identity_summary"] = build_identity_summary(
            KnowledgeIdentity(
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                groups=tuple(request.groups or []),
                roles=tuple(request.roles or []),
            )
        )
    elif cfg and not search.get("identity_summary"):
        search["identity_summary"] = build_identity_summary(
            KnowledgeIdentity(
                tenant_id=str(cfg.get("tenant_id") or "default"),
                user_id=cfg.get("user_id"),
                groups=tuple(cfg.get("groups") or []),
                roles=tuple(cfg.get("roles") or []),
            )
        )
    return search


_SECURITY_DENY = frozenset(
    {
        "permission_denied",
        "authentication_denied",
        "tenant_mismatch",
        "acl_missing",
        "acl_invalid",
        "identity_ambiguous",
        "deny_user",
        "deny_group",
        "deny_role",
        "private_no_user",
        "restricted_no_allow",
    }
)
_FALLBACK_ERRORS = frozenset(
    {
        "knowledge_provider_unavailable",
        "knowledge_timeout",
        "knowledge_rate_limited",
        "knowledge_backend_error",
        "knowledge_disconnected",
        "knowledge_invalid_response",
        "knowledge_protocol_incompatible",
        "provider_error",
        "circuit_open",
        "http_rag_config_missing",
        "mcp_config_missing",
        "mcp_client_missing",
        "opensearch_config_missing",
        "knowledge_root_missing",
    }
)


def _chain_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("fallback_enabled") or cfg.get("circuit_enabled"))


def _is_security_deny(payload: Dict[str, Any]) -> bool:
    error_type = str(payload.get("error_type") or "")
    omission = str(payload.get("omission_reason") or "")
    if error_type in _SECURITY_DENY or omission in _SECURITY_DENY:
        return True
    denials = payload.get("acl_denial_reasons") if isinstance(payload.get("acl_denial_reasons"), dict) else {}
    if denials and not payload.get("hits") and omission == "permission_denied":
        return True
    return False


def _is_fallback_eligible(error_type: Optional[str]) -> bool:
    if not error_type:
        return False
    if error_type in _FALLBACK_ERRORS:
        return True
    return error_type.startswith("knowledge_provider_failed:") or error_type.startswith("knowledge_search_failed:") or error_type.startswith("knowledge_provider_init_failed:")


def _safe_attempt(
    *,
    provider_id: str,
    status: str,
    error_type: Optional[str] = None,
    latency_ms: float = 0.0,
    circuit_state: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "provider": provider_id,
        "status": status,
        "error_type": error_type,
        "latency_ms": round(float(latency_ms or 0.0), 2),
        "circuit_state": circuit_state,
        "reason": reason,
    }


def _attach_chain_fields(
    payload: Dict[str, Any],
    *,
    primary: Optional[str],
    effective: Optional[str],
    attempts: List[Dict[str, Any]],
    fallback_used: bool,
    fallback_reason: Optional[str],
    all_failed: bool,
    circuit_state: Optional[str],
) -> Dict[str, Any]:
    payload["primary_provider"] = primary
    payload["effective_provider"] = effective
    payload["provider_attempts"] = list(attempts)
    payload["fallback_used"] = bool(fallback_used)
    payload["fallback_reason"] = fallback_reason
    payload["all_providers_failed"] = bool(all_failed)
    payload["circuit_state"] = circuit_state
    if payload.get("protocol_version") is None and effective in ("http_rag", "mcp"):
        payload["protocol_version"] = "1"
        payload["contract_version"] = "1"
    return payload


def reset_knowledge_runtime_state() -> None:
    """测试清理：重置进程级熔断与远端 ID 映射，避免用例互相污染。"""
    from .knowledge_circuit import reset_circuit_store
    from .knowledge_remote import reset_remote_id_store

    reset_circuit_store()
    reset_remote_id_store()


def retrieve_knowledge_for_langgraph(
    user_request: str,
    knowledge_config: Optional[Dict[str, Any]] = None,
    *,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """retrieve_context_node 使用的企业知识检索（fail-open）。普通请求不隐式同步。"""
    started = time.perf_counter()
    deadline_started = time.monotonic()
    safe_query, _ = redact_text(user_request or "")
    cfg = normalize_knowledge_config(knowledge_config)
    from .local_knowledge_provider import sanitize_tenant_id

    tenant_id = sanitize_tenant_id(str(cfg.get("tenant_id") or "default")) if cfg.get("enabled") else None
    provider_name = cfg.get("provider")

    if not cfg.get("enabled"):
        return _finalize_search_payload(
            empty_knowledge_search(
                enabled=False,
                provider=provider_name,
                tenant_id=None,
                permission_mode=PERMISSION_MODE_DISABLED,
                query=safe_query,
                omission_reason="knowledge_disabled",
            ),
            cfg=cfg,
        )

    timeout_ms = int(cfg.get("timeout_ms") or DEFAULT_TIMEOUT_MS)
    deadline = knowledge_deadline(timeout_ms, started=deadline_started)
    request = KnowledgeSearchRequest(
        query=safe_query,
        tenant_id=str(cfg.get("tenant_id") or "default"),
        user_id=str(cfg.get("user_id") or "") or None,
        groups=list(cfg.get("groups") or []),
        roles=list(cfg.get("roles") or []),
        top_k=int(cfg.get("top_k") or 5),
        request_id=request_id or cfg.get("request_id"),
        timeout_ms=timeout_ms,
        deadline_monotonic=deadline,
    )

    if not _chain_enabled(cfg):
        provider, error_type, cfg = build_knowledge_provider(knowledge_config)
        if error_type:
            return _finalize_search_payload(
                empty_knowledge_search(
                    enabled=True,
                    provider=cfg.get("provider"),
                    tenant_id=tenant_id,
                    permission_mode=PERMISSION_MODE_UNAVAILABLE,
                    query=safe_query,
                    error_type=error_type,
                    omission_reason=error_type,
                    degraded=True,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    primary_provider=cfg.get("provider"),
                ),
                cfg=cfg,
            )
        assert provider is not None
        try:
            response = provider.search(request)
            payload = response.to_dict()
            payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            payload["provider_latency_ms"] = payload.get("provider_latency_ms") or payload["latency_ms"]
            payload["primary_provider"] = provider.provider_id
            payload["effective_provider"] = provider.provider_id if payload.get("used") else None
            return _finalize_search_payload(payload, request=request, cfg=cfg)
        except KnowledgeTimeout:
            return _finalize_search_payload(
                empty_knowledge_search(
                    enabled=True,
                    provider=provider.provider_id,
                    tenant_id=tenant_id,
                    permission_mode=PERMISSION_MODE_TENANT_ONLY,
                    query=safe_query,
                    error_type="knowledge_timeout",
                    omission_reason="knowledge_timeout",
                    degraded=True,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    primary_provider=provider.provider_id,
                ),
                request=request,
                cfg=cfg,
            )
        except Exception as exc:
            return _finalize_search_payload(
                empty_knowledge_search(
                    enabled=True,
                    provider=provider.provider_id,
                    tenant_id=tenant_id,
                    permission_mode=PERMISSION_MODE_TENANT_ONLY,
                    query=safe_query,
                    error_type=f"knowledge_provider_failed:{type(exc).__name__}",
                    omission_reason="provider_error",
                    degraded=True,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    primary_provider=provider.provider_id,
                ),
                request=request,
                cfg=cfg,
            )

    from .knowledge_circuit import KnowledgeCircuitBreaker, process_circuit_store

    primary = str(cfg.get("provider") or "local")
    chain = [primary]
    if cfg.get("fallback_enabled"):
        for item in cfg.get("fallback_providers") or []:
            name = str(item).strip()
            if name and name not in chain:
                chain.append(name)
    circuit_enabled = bool(cfg.get("circuit_enabled"))
    breaker = cfg.get("_circuit_breaker")
    if breaker is None and circuit_enabled:
        breaker = KnowledgeCircuitBreaker(
            failure_threshold=int(cfg.get("failure_threshold") or 3),
            cooldown_seconds=float(cfg.get("cooldown_seconds") or 30),
            clock=cfg.get("_circuit_clock") or time.monotonic,
            store=cfg.get("_circuit_store") or process_circuit_store(),
        )
    attempts: List[Dict[str, Any]] = []
    last_payload: Optional[Dict[str, Any]] = None
    fallback_reason = None
    last_circuit = None
    fallback_on_empty = bool(cfg.get("fallback_on_empty"))

    for index, provider_id in enumerate(chain):
        if knowledge_deadline_exceeded(deadline):
            attempts.append(
                _safe_attempt(
                    provider_id=provider_id,
                    status="skipped",
                    error_type="knowledge_timeout",
                    reason="deadline_exhausted",
                    circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
                )
            )
            fallback_reason = "deadline_exhausted"
            break
        if breaker is not None:
            allowed, circuit_state = breaker.allow(str(tenant_id or "default"), provider_id)
        else:
            allowed, circuit_state = True, None
        last_circuit = circuit_state
        if not allowed:
            attempts.append(
                _safe_attempt(
                    provider_id=provider_id,
                    status="skipped",
                    error_type="circuit_open",
                    reason="circuit_open",
                    circuit_state=circuit_state,
                )
            )
            fallback_reason = "circuit_open"
            last_payload = empty_knowledge_search(
                enabled=True,
                provider=provider_id,
                tenant_id=tenant_id,
                permission_mode=PERMISSION_MODE_UNAVAILABLE,
                query=safe_query,
                error_type="circuit_open",
                omission_reason="circuit_open",
                degraded=True,
                circuit_state=circuit_state,
            )
            continue
        cloned = dict(knowledge_config or {})
        cloned["provider"] = provider_id
        cloned["enabled"] = True
        attempt_started = time.perf_counter()
        provider, error_type, _built_cfg = build_knowledge_provider(cloned)
        if error_type or provider is None:
            latency = (time.perf_counter() - attempt_started) * 1000
            if breaker is not None:
                breaker.record_failure(str(tenant_id or "default"), provider_id)
            attempts.append(
                _safe_attempt(
                    provider_id=provider_id,
                    status="error",
                    error_type=error_type or "knowledge_provider_unavailable",
                    latency_ms=latency,
                    circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
                    reason=error_type or "unavailable",
                )
            )
            fallback_reason = error_type or "knowledge_provider_unavailable"
            last_payload = empty_knowledge_search(
                enabled=True,
                provider=provider_id,
                tenant_id=tenant_id,
                permission_mode=PERMISSION_MODE_UNAVAILABLE,
                query=safe_query,
                error_type=error_type,
                omission_reason=error_type,
                degraded=True,
            )
            continue
        try:
            response = provider.search(request)
            payload = response.to_dict()
        except KnowledgeTimeout:
            payload = empty_knowledge_search(
                enabled=True,
                provider=provider.provider_id,
                tenant_id=tenant_id,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                query=safe_query,
                error_type="knowledge_timeout",
                omission_reason="knowledge_timeout",
                degraded=True,
            )
        except Exception as exc:
            payload = empty_knowledge_search(
                enabled=True,
                provider=provider.provider_id,
                tenant_id=tenant_id,
                permission_mode=PERMISSION_MODE_TENANT_ONLY,
                query=safe_query,
                error_type=f"knowledge_provider_failed:{type(exc).__name__}",
                omission_reason="provider_error",
                degraded=True,
            )
        latency = (time.perf_counter() - attempt_started) * 1000
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        payload["provider_latency_ms"] = payload.get("provider_latency_ms") or round(latency, 2)
        last_payload = payload
        if _is_security_deny(payload):
            if breaker is not None:
                breaker.record_success(str(tenant_id or "default"), provider_id)
            attempts.append(
                _safe_attempt(
                    provider_id=provider_id,
                    status="denied",
                    error_type=payload.get("error_type") or payload.get("omission_reason"),
                    latency_ms=latency,
                    circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
                    reason="security_deny",
                )
            )
            attached = _attach_chain_fields(
                payload,
                primary=primary,
                effective=None,
                attempts=attempts,
                fallback_used=False,
                fallback_reason=None,
                all_failed=False,
                circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
            )
            return _finalize_search_payload(attached, request=request, cfg=cfg)
        if payload.get("used") and payload.get("hits"):
            if breaker is not None:
                breaker.record_success(str(tenant_id or "default"), provider_id)
            fallback_used = index > 0
            attempts.append(
                _safe_attempt(
                    provider_id=provider_id,
                    status="ok",
                    latency_ms=latency,
                    circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
                    reason="fallback_hit" if fallback_used else "primary_hit",
                )
            )
            attached = _attach_chain_fields(
                payload,
                primary=primary,
                effective=provider_id,
                attempts=attempts,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason if fallback_used else None,
                all_failed=False,
                circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
            )
            return _finalize_search_payload(attached, request=request, cfg=cfg)
        error_type = payload.get("error_type")
        empty = not payload.get("hits") and not error_type
        if empty and not fallback_on_empty:
            if breaker is not None:
                breaker.record_success(str(tenant_id or "default"), provider_id)
            attempts.append(
                _safe_attempt(
                    provider_id=provider_id,
                    status="empty",
                    latency_ms=latency,
                    circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
                    reason="empty_result",
                )
            )
            attached = _attach_chain_fields(
                payload,
                primary=primary,
                effective=None,
                attempts=attempts,
                fallback_used=False,
                fallback_reason=None,
                all_failed=False,
                circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
            )
            return _finalize_search_payload(attached, request=request, cfg=cfg)
        eligible = empty or _is_fallback_eligible(str(error_type or ""))
        if error_type and _is_fallback_eligible(str(error_type)):
            if breaker is not None:
                breaker.record_failure(str(tenant_id or "default"), provider_id)
        else:
            if breaker is not None:
                breaker.record_success(str(tenant_id or "default"), provider_id)
        attempts.append(
            _safe_attempt(
                provider_id=provider_id,
                status="empty" if empty else "error",
                error_type=error_type,
                latency_ms=latency,
                circuit_state=breaker.state(str(tenant_id or "default"), provider_id) if breaker else None,
                reason="empty_result" if empty else str(error_type or "error"),
            )
        )
        fallback_reason = "empty_result" if empty else str(error_type or fallback_reason)
        if not eligible:
            break

    all_failed = True
    if last_payload is None:
        last_payload = empty_knowledge_search(
            enabled=True,
            provider=primary,
            tenant_id=tenant_id,
            permission_mode=PERMISSION_MODE_UNAVAILABLE,
            query=safe_query,
            error_type="knowledge_provider_unavailable",
            omission_reason="knowledge_provider_unavailable",
            degraded=True,
        )
    if last_payload.get("used") and last_payload.get("hits"):
        all_failed = False
    elif last_payload.get("omission_reason") == "empty_result" and not last_payload.get("error_type"):
        all_failed = False
    if all_failed and not last_payload.get("error_type"):
        last_payload["error_type"] = fallback_reason or "knowledge_provider_unavailable"
        last_payload["degraded"] = True
        last_payload["omission_reason"] = last_payload.get("omission_reason") or last_payload["error_type"]
    last_payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    attached = _attach_chain_fields(
        last_payload,
        primary=primary,
        effective=None,
        attempts=attempts,
        fallback_used=False,
        fallback_reason=fallback_reason,
        all_failed=all_failed,
        circuit_state=last_circuit,
    )
    return _finalize_search_payload(attached, request=request, cfg=cfg)


def build_knowledge_sync_service(
    knowledge_config: Optional[Dict[str, Any]],
    *,
    transport: Any = None,
    embedding: Any = None,
    source_id: Optional[str] = None,
):
    """用 Knowledge Registry/config 组装 Connector + OpenSearchKnowledgeStore + SyncService。"""
    from .knowledge_local_connector import LocalDirectoryConnector
    from .knowledge_opensearch import RequestsOpenSearchTransport
    from .knowledge_opensearch_store import OpenSearchKnowledgeStore
    from .knowledge_sync import KnowledgeSyncService
    from .local_knowledge_provider import sanitize_tenant_id

    cfg = normalize_knowledge_config(knowledge_config)
    providers = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
    local_cfg = providers.get("local") if isinstance(providers.get("local"), dict) else {}
    os_cfg = providers.get("opensearch") if isinstance(providers.get("opensearch"), dict) else {}
    root = cfg.get("root") or local_cfg.get("root")
    endpoint = os_cfg.get("endpoint") or cfg.get("endpoint")
    index = os_cfg.get("index") or cfg.get("index")
    if not root:
        raise KnowledgeProviderConfigError("knowledge_root_missing")
    if not endpoint or not index:
        raise KnowledgeProviderConfigError("opensearch_config_missing")
    tenant_id = sanitize_tenant_id(str(cfg.get("tenant_id") or "default"))
    resolved_source = str(
        source_id or os_cfg.get("source_id") or local_cfg.get("source_id") or "opensearch"
    )
    http = transport or cfg.get("_transport") or RequestsOpenSearchTransport()
    store = OpenSearchKnowledgeStore(
        endpoint=str(endpoint),
        index=str(index),
        auth_mode=str(os_cfg.get("auth_mode") or cfg.get("auth_mode") or "none"),
        credential_env=os_cfg.get("credential_env") or cfg.get("credential_env"),
        username_env=os_cfg.get("username_env") or cfg.get("username_env"),
        transport=http,
        source_id=resolved_source,
    )
    connector = LocalDirectoryConnector(root, tenant_id=tenant_id, source_id=resolved_source)
    embedder = embedding if embedding is not None else cfg.get("_embedding")
    service = KnowledgeSyncService(connector, store, embedding=embedder)
    return service, cfg


def run_knowledge_sync(
    knowledge_config: Optional[Dict[str, Any]],
    *,
    mode: str = "incremental",
    tenant_id: Optional[str] = None,
    source_id: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    transport: Any = None,
    embedding: Any = None,
    deadline: Optional[float] = None,
):
    """独立同步入口。普通 Agent 查询不得调用此函数。"""
    from .knowledge_sync import SYNC_STATUS_FAILED, KnowledgeSyncResult
    from .local_knowledge_provider import sanitize_tenant_id

    cfg = normalize_knowledge_config(knowledge_config)
    sync_mode = "full" if str(mode or "incremental").strip().lower() == "full" else "incremental"
    tenant = sanitize_tenant_id(str(tenant_id or cfg.get("tenant_id") or "default"))
    try:
        timeout = int(timeout_ms if timeout_ms is not None else (cfg.get("timeout_ms") or DEFAULT_TIMEOUT_MS))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_MS
    sync_deadline = deadline if deadline is not None else knowledge_deadline(max(1, timeout))
    try:
        service, _cfg = build_knowledge_sync_service(
            knowledge_config,
            transport=transport,
            embedding=embedding,
            source_id=source_id,
        )
        return service.sync(tenant_id=tenant, deadline=sync_deadline, mode=sync_mode)
    except KnowledgeTimeout:
        return KnowledgeSyncResult(
            source_id=str(source_id or "opensearch"),
            tenant_id=tenant,
            status=SYNC_STATUS_FAILED,
            error_type="knowledge_timeout",
            store_backend="opensearch",
            mode=sync_mode,
        )
    except KnowledgeProviderConfigError as exc:
        reason = str(exc).strip() or "knowledge_provider_init_failed"
        return KnowledgeSyncResult(
            source_id=str(source_id or "opensearch"),
            tenant_id=tenant,
            status=SYNC_STATUS_FAILED,
            error_type=reason,
            store_backend="opensearch",
            mode=sync_mode,
        )
    except Exception as exc:
        return KnowledgeSyncResult(
            source_id=str(source_id or "opensearch"),
            tenant_id=tenant,
            status=SYNC_STATUS_FAILED,
            error_type=f"sync_failed:{type(exc).__name__}",
            store_backend="opensearch",
            mode=sync_mode,
        )
