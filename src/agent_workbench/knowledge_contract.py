# -*- coding: utf-8 -*-
"""跨 Provider 契约断言：标准字段、权限、脱敏、不泄露远端专有字段。"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .knowledge_provider import KnowledgeFetchRequest, KnowledgeSearchRequest
from .redaction import contains_secret_blob

STANDARD_SEARCH_KEYS = (
    "enabled",
    "used",
    "provider",
    "tenant_id",
    "permission_mode",
    "query",
    "hits",
    "error_type",
    "omission_reason",
)
STANDARD_HIT_KEYS = (
    "provider_id",
    "tenant_id",
    "source_type",
    "source_id",
    "document_id",
    "chunk_id",
    "title",
    "snippet",
    "citation",
)
PROHIBITED_REMOTE_KEYS = (
    "included_in_llm",
    "permission_decision",
    "tenant_authorized",
    "acl_decision",
)
PROPRIETARY_MARKERS = ("_source", "raw_hit", "vendor_payload", "mcp_meta")


def assert_standard_search_shape(payload: Dict[str, Any]) -> None:
    for key in STANDARD_SEARCH_KEYS:
        assert key in payload, f"missing search field {key}"
    assert isinstance(payload.get("hits"), list)


def assert_no_secret_leak(payload: Dict[str, Any], secret: str) -> None:
    dumped = json.dumps(payload, ensure_ascii=False)
    assert secret not in dumped
    assert not contains_secret_blob(dumped)


def assert_no_proprietary_fields(payload: Dict[str, Any]) -> None:
    dumped = json.dumps(payload, ensure_ascii=False)
    for marker in PROPRIETARY_MARKERS:
        assert marker not in dumped
    for hit in payload.get("hits") or []:
        if not isinstance(hit, dict):
            continue
        for key in PROHIBITED_REMOTE_KEYS:
            if key == "included_in_llm":
                continue
            assert key not in hit


def assert_attempts_are_safe(payload: Dict[str, Any], secret: str = "") -> None:
    dumped = json.dumps(payload.get("provider_attempts") or [], ensure_ascii=False)
    assert "password" not in dumped.lower() or "credential_env" in dumped.lower()
    if secret:
        assert secret not in dumped
    for item in payload.get("provider_attempts") or []:
        if not isinstance(item, dict):
            continue
        assert "query" not in item
        assert "snippet" not in item
        assert "text" not in item
        assert "user_id" not in item


def run_provider_contract(
    provider: Any,
    *,
    request: KnowledgeSearchRequest,
    secret: str,
    fetch_document_id: Optional[str] = None,
) -> Dict[str, Any]:
    search = provider.search(request)
    payload = search.to_dict() if hasattr(search, "to_dict") else dict(search)
    assert_standard_search_shape(payload)
    assert_no_secret_leak(payload, secret)
    assert_no_proprietary_fields(payload)
    for hit in payload.get("hits") or []:
        for key in STANDARD_HIT_KEYS:
            assert key in hit
        assert str(hit.get("citation") or "").startswith("[KB:")
        assert str(hit.get("source_uri") or "").startswith("kb://")
    if fetch_document_id is None and payload.get("hits"):
        fetch_document_id = str(payload["hits"][0].get("document_id") or "") or None
    health = provider.health()
    health_payload = health.to_dict() if hasattr(health, "to_dict") else dict(health)
    assert "provider_id" in health_payload
    assert "healthy" in health_payload
    assert_no_secret_leak(health_payload, secret)
    if fetch_document_id:
        fetched = provider.fetch(
            KnowledgeFetchRequest(
                document_id=fetch_document_id,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                groups=list(request.groups or []),
                roles=list(request.roles or []),
                timeout_ms=request.timeout_ms,
            )
        )
        fetched_payload = fetched.to_dict() if hasattr(fetched, "to_dict") else dict(fetched)
        assert_no_secret_leak(fetched_payload, secret)
        if fetched.error_type is None:
            assert str(fetched.source_uri or "").startswith("kb://")
    mismatch = KnowledgeSearchRequest(
        query=request.query,
        tenant_id="other-tenant",
        user_id=request.user_id,
        groups=list(request.groups or []),
        roles=list(request.roles or []),
        timeout_ms=request.timeout_ms,
    )
    denied = provider.search(mismatch).to_dict()
    assert denied.get("error_type") == "tenant_mismatch" or denied.get("omission_reason") == "tenant_mismatch"
    assert secret not in json.dumps(denied, ensure_ascii=False)
    assert not denied.get("hits")
    return payload


_SECURITY_CLASSES = frozenset(
    {
        "permission_denied",
        "authentication_denied",
        "tenant_mismatch",
        "acl_missing",
        "acl_invalid",
    }
)


def _permission_class(search: Dict[str, Any]) -> str:
    error = str(search.get("error_type") or "")
    omission = str(search.get("omission_reason") or "")
    if error in _SECURITY_CLASSES or omission in _SECURITY_CLASSES:
        return "denied"
    if search.get("used") and search.get("hits"):
        return "hit"
    if omission == "empty_result" or (not search.get("hits") and not error):
        return "empty"
    return "error"


def _normalized_hit_shape(search: Dict[str, Any]) -> Dict[str, Any]:
    hits = [hit for hit in (search.get("hits") or []) if isinstance(hit, dict)]
    return {
        "has_hits": bool(hits),
        "all_kb_citation": all(str(hit.get("citation") or "").startswith("[KB:") for hit in hits) if hits else True,
        "all_kb_uri": all(str(hit.get("source_uri") or "").startswith("kb://") for hit in hits) if hits else True,
        "tenant_consistent": all(hit.get("tenant_id") == search.get("tenant_id") for hit in hits) if hits else True,
        "has_snippet": all(bool(hit.get("snippet")) for hit in hits) if hits else True,
    }


def _answer_behavior(result: Dict[str, Any]) -> Dict[str, Any]:
    state = result.get("state") or {}
    report = result.get("report") or {}
    return {
        "agent_mode": state.get("agent_mode") or report.get("agent_mode"),
        "has_final_response": bool(report.get("final_response") or state.get("final_response")),
        "has_tool_loop": bool(state.get("observations") or report.get("observations") or state.get("completed_steps")),
    }


def compare_provider_runs(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """按归一化 hits、权限、citation、注入和最终回答行为计算真实 equivalence score。"""
    left_state = left.get("state") or {}
    right_state = right.get("state") or {}
    left_search = left_state.get("knowledge_search") or left.get("knowledge_search") or {}
    right_search = right_state.get("knowledge_search") or right.get("knowledge_search") or {}
    checks = {
        "permission_class": _permission_class(left_search) == _permission_class(right_search),
        "normalized_hits": _normalized_hit_shape(left_search) == _normalized_hit_shape(right_search),
        "citation_available": _normalized_hit_shape(left_search)["all_kb_citation"]
        and _normalized_hit_shape(right_search)["all_kb_citation"],
        "context_injected": bool(left_state.get("knowledge_context_injected"))
        == bool(right_state.get("knowledge_context_injected")),
        "final_answer_behavior": _answer_behavior(left) == _answer_behavior(right),
    }
    passed = sum(1 for value in checks.values() if value)
    score = round(passed / max(1, len(checks)), 4)
    return {"score": score, "checks": checks}
