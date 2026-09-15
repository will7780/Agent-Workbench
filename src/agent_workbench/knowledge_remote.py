# -*- coding: utf-8 -*-
"""远端 HTTP/MCP 知识结果的本地安全边界：不可信输入必须本地重建。"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .knowledge_acl import ACLDecision, KnowledgeIdentity, check_knowledge_acl
from .knowledge_provider import (
    MAX_SNIPPET_CHARS,
    KnowledgeSearchResult,
    _safe_text,
    format_citation,
)
from .local_knowledge_provider import opaque_document_id, sanitize_tenant_id
from .redaction import contains_secret_blob, redact_text

PROTOCOL_VERSION = "1"
_CHUNK_ID_RE = re.compile(r"^c\d{3,}$")
_ISO_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_UNSAFE_TEXT_RE = re.compile(r"(secret|password|token|bearer\s|sk-|api[_-]?key)", re.IGNORECASE)
_REMOTE_DROP_KEYS = frozenset(
    {
        "citation",
        "included_in_llm",
        "omission_reason",
        "permission",
        "allowed",
        "acl_decision",
        "permission_decision",
        "authorized",
        "tenant_authorized",
        "raw",
        "vendor",
        "proprietary",
        "vendor_payload",
        "mcp_meta",
        "source_uri",
        "error_type",
        "error",
        "message",
        "detail",
        "traceback",
    }
)
ALLOWED_REMOTE_ERROR_TYPES = frozenset(
    {
        "knowledge_timeout",
        "knowledge_rate_limited",
        "knowledge_backend_error",
        "knowledge_disconnected",
        "knowledge_invalid_response",
        "knowledge_protocol_incompatible",
        "permission_denied",
        "authentication_denied",
        "tenant_mismatch",
        "circuit_open",
        "provider_error",
    }
)
_AUTH_ALIASES = {
    "unauthorized": "authentication_denied",
    "authentication_denied": "authentication_denied",
    "401": "authentication_denied",
    "forbidden": "permission_denied",
    "permission_denied": "permission_denied",
    "403": "permission_denied",
}


class RemoteDocumentIdStore:
    """
    进程级远端 document_id 映射。
    key = (tenant_id, provider_id, provider_instance_id, opaque_id)
    LRU + TTL，线程安全。不保存 endpoint/server_id 明文。
    """

    def __init__(
        self,
        *,
        max_entries: int = 4096,
        ttl_seconds: float = 3600.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._map: "OrderedDict[Tuple[str, str, str, str], Tuple[str, float]]" = OrderedDict()
        self.max_entries = max(1, int(max_entries or 4096))
        self.ttl_seconds = max(0.0, float(ttl_seconds if ttl_seconds is not None else 3600.0))
        self.clock = clock or time.monotonic

    def _key(self, tenant_id: str, provider_id: str, instance_id: str, opaque_id: str) -> Tuple[str, str, str, str]:
        return (
            sanitize_tenant_id(tenant_id),
            str(provider_id or ""),
            str(instance_id or ""),
            str(opaque_id or ""),
        )

    def _purge_locked(self) -> None:
        now = self.clock()
        expired = [key for key, (_value, expires_at) in self._map.items() if expires_at <= now]
        for key in expired:
            self._map.pop(key, None)

    def put(
        self,
        tenant_id: str,
        provider_id: str,
        opaque_id: str,
        remote_id: str,
        *,
        instance_id: str = "",
    ) -> None:
        value = str(remote_id or "").strip()
        if not str(opaque_id or "").strip() or not value:
            return
        key = self._key(tenant_id, provider_id, instance_id, opaque_id)
        with self._lock:
            self._purge_locked()
            if key in self._map:
                self._map.pop(key, None)
            elif len(self._map) >= self.max_entries:
                self._map.popitem(last=False)
            self._map[key] = (value, self.clock() + self.ttl_seconds)

    def get(
        self,
        tenant_id: str,
        provider_id: str,
        opaque_id: str,
        *,
        instance_id: str = "",
    ) -> Optional[str]:
        key = self._key(tenant_id, provider_id, instance_id, opaque_id)
        with self._lock:
            self._purge_locked()
            item = self._map.get(key)
            if item is None:
                return None
            remote_id, _expires = item
            self._map.move_to_end(key)
            return remote_id

    def put_many(
        self,
        tenant_id: str,
        provider_id: str,
        mapping: Dict[str, str],
        *,
        instance_id: str = "",
    ) -> None:
        for opaque_id, remote_id in (mapping or {}).items():
            self.put(tenant_id, provider_id, opaque_id, remote_id, instance_id=instance_id)

    def reset(self) -> None:
        with self._lock:
            self._map.clear()


_PROCESS_ID_STORE = RemoteDocumentIdStore()


def process_remote_id_store() -> RemoteDocumentIdStore:
    return _PROCESS_ID_STORE


def reset_remote_id_store() -> None:
    _PROCESS_ID_STORE.reset()


def protocol_compatible(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    version = str(payload.get("protocol_version") or "").strip()
    return version == PROTOCOL_VERSION


def stable_chunk_id(raw: Any, index: int) -> str:
    value = str(raw or "").strip()
    if _CHUNK_ID_RE.match(value):
        return value
    return f"c{max(0, int(index)):03d}"


def provider_instance_id(*parts: str) -> str:
    seed = "\n".join(str(part or "").strip() for part in parts)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"inst_{digest}"


def http_provider_instance_id(endpoint: str, search_path: str, fetch_path: str) -> str:
    return provider_instance_id("http_rag", endpoint, search_path, fetch_path)


def mcp_provider_instance_id(server_id: str, search_tool: str, fetch_tool: str) -> str:
    return provider_instance_id("mcp", server_id, search_tool, fetch_tool)


def opaque_remote_document_id(
    tenant_id: str,
    provider_id: str,
    remote_document_id: str,
    *,
    instance_id: str = "",
) -> str:
    seed = f"{provider_id}::{instance_id}::{str(remote_document_id or '').strip()}"
    return opaque_document_id(tenant_id, seed)


def _looks_unsafe(value: Any) -> bool:
    text = str(value or "")
    if not text:
        return False
    if contains_secret_blob(text):
        return True
    return bool(_UNSAFE_TEXT_RE.search(text))


def sanitize_remote_title(raw: Any) -> str:
    text, _ = redact_text(str(raw or "document") or "document")
    if _looks_unsafe(raw) or _looks_unsafe(text):
        return "document"
    return _safe_text(text, 200) or "document"


def sanitize_updated_at(raw: Any) -> Optional[str]:
    if raw is None or isinstance(raw, (dict, list)):
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None
    text = str(raw).strip()
    if not text or not _ISO_DT_RE.match(text) or _looks_unsafe(text):
        return None
    return text[:40]


def sanitize_version(raw: Any) -> Optional[str]:
    if raw is None or isinstance(raw, (dict, list, bool)):
        return None
    text = str(raw).strip()
    if not _VERSION_RE.match(text) or _looks_unsafe(text):
        return None
    return text


def sanitize_remote_error_type(raw: Any) -> Optional[str]:
    if raw is None or isinstance(raw, (dict, list)):
        return None
    text = str(raw or "").strip()
    if not text:
        return None
    alias = _AUTH_ALIASES.get(text.lower())
    if alias:
        return alias
    if text in ALLOWED_REMOTE_ERROR_TYPES:
        return text
    return "knowledge_backend_error"


def classify_http_status(status_code: Any) -> Optional[str]:
    try:
        code = int(status_code)
    except (TypeError, ValueError):
        return None
    if code == 401:
        return "authentication_denied"
    if code == 403:
        return "permission_denied"
    if code == 429:
        return "knowledge_rate_limited"
    if code >= 500:
        return "knowledge_backend_error"
    if code >= 400:
        return "knowledge_backend_error"
    return None


def classify_remote_error(payload: Any, *, status_code: Any = None) -> Optional[str]:
    status_error = classify_http_status(status_code)
    if status_error:
        return status_error
    if not isinstance(payload, dict):
        return None
    nested_status = payload.get("status_code")
    if nested_status is None and isinstance(payload.get("payload"), dict):
        nested_status = payload["payload"].get("status_code")
    status_error = classify_http_status(nested_status)
    if status_error:
        return status_error
    raw_error = payload.get("error_type") or payload.get("error")
    if isinstance(payload.get("payload"), dict) and not raw_error:
        raw_error = payload["payload"].get("error_type") or payload["payload"].get("error")
    return sanitize_remote_error_type(raw_error)


def resolve_http_error(
    *,
    status_code: Any = None,
    error_type: Any = None,
    payload: Any = None,
) -> Optional[str]:
    """HTTP status 映射优先于 transport.error_type / payload error_type。"""
    status_error = classify_http_status(status_code)
    if status_error:
        return status_error
    if isinstance(payload, dict):
        nested_status = payload.get("status_code")
        if nested_status is None and isinstance(payload.get("payload"), dict):
            nested_status = payload["payload"].get("status_code")
        nested = classify_http_status(nested_status)
        if nested:
            return nested
    if error_type:
        sanitized = sanitize_remote_error_type(error_type)
        if sanitized:
            return sanitized
    return classify_remote_error(payload, status_code=None)


def strip_untrusted_hit(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in raw.items():
        lowered = str(key).lower()
        if lowered in _REMOTE_DROP_KEYS or any(
            part in lowered for part in ("citation", "permission", "authorized", "error", "token", "password", "secret")
        ):
            continue
        cleaned[str(key)] = value
    return cleaned


def sanitize_remote_hit(
    raw: Any,
    *,
    provider_id: str,
    identity: KnowledgeIdentity,
    index: int,
    instance_id: str = "",
) -> Tuple[Optional[KnowledgeSearchResult], Optional[str], Optional[str]]:
    """
    返回 (result, denial_reason, raw_document_id)。
    拒绝时 result 为 None，且不得携带正文。
    """
    source = strip_untrusted_hit(raw)
    if not source:
        return None, "acl_invalid", None
    ident = identity.normalized()
    doc_tenant = sanitize_tenant_id(str(source.get("tenant_id") or ""))
    raw_document_id = str(source.get("document_id") or source.get("id") or "").strip()
    if _looks_unsafe(raw_document_id):
        raw_document_id = f"hit-{index}"
    decision: ACLDecision = check_knowledge_acl(source.get("acl") if "acl" in source else None, ident, doc_tenant)
    if not decision.allowed:
        return None, decision.reason, None
    text, _ = redact_text(str(source.get("text") or source.get("snippet") or ""))
    snippet = _safe_text(text, MAX_SNIPPET_CHARS)
    opaque_id = opaque_remote_document_id(
        ident.tenant_id,
        provider_id,
        raw_document_id or f"hit-{index}",
        instance_id=instance_id,
    )
    chunk_id = stable_chunk_id(source.get("chunk_id"), index)
    title = sanitize_remote_title(source.get("title") or "document")
    try:
        score = float(source.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    result = KnowledgeSearchResult(
        provider_id=provider_id,
        tenant_id=ident.tenant_id,
        source_type=provider_id,
        source_id=provider_id,
        document_id=opaque_id,
        chunk_id=chunk_id,
        title=title,
        snippet=snippet,
        score=score,
        source_uri=f"kb://{provider_id}/{opaque_id}",
        updated_at=sanitize_updated_at(source.get("updated_at")),
        version=sanitize_version(source.get("version")),
        citation=format_citation(provider_id, opaque_id, chunk_id),
        retrieval_reason="remote_search",
        included_in_llm=False,
        metadata={},
    )
    return result, None, raw_document_id or None


def sanitize_remote_hits(
    hits: Any,
    *,
    provider_id: str,
    identity: KnowledgeIdentity,
    instance_id: str = "",
) -> Tuple[List[KnowledgeSearchResult], int, Dict[str, int], Dict[str, str]]:
    results: List[KnowledgeSearchResult] = []
    acl_filtered = 0
    denial_reasons: Dict[str, int] = {}
    id_map: Dict[str, str] = {}
    raw_hits = hits if isinstance(hits, list) else []
    for index, item in enumerate(raw_hits):
        result, reason, raw_id = sanitize_remote_hit(
            item,
            provider_id=provider_id,
            identity=identity,
            index=index,
            instance_id=instance_id,
        )
        if result is None:
            acl_filtered += 1
            key = reason or "acl_invalid"
            denial_reasons[key] = denial_reasons.get(key, 0) + 1
            continue
        results.append(result)
        if raw_id:
            id_map[result.document_id] = raw_id
    return results, acl_filtered, denial_reasons, id_map


def classify_transport_exception(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    message = str(exc or "").lower()
    if "timeout" in name or "timeout" in message:
        return "knowledge_timeout"
    if any(token in name or token in message for token in ("connect", "disconnect", "reset", "brokenpipe", "connection")):
        return "knowledge_disconnected"
    if any(token in name for token in ("json", "value", "type", "decode")):
        return "knowledge_invalid_response"
    return "knowledge_disconnected"
