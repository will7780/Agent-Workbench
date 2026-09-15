# -*- coding: utf-8 -*-
"""本地目录 Connector：复用 Phase 1 UTF-8 md/txt/json 解析，不复制切块逻辑。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .knowledge_acl import KnowledgeACL, explicit_tenant_acl, parse_knowledge_acl
from .knowledge_provider import _safe_text, ensure_knowledge_deadline
from .knowledge_sync import KnowledgeSourceDocument
from .local_knowledge_provider import (
    SUPPORTED_SUFFIXES,
    _document_display_title,
    _file_sha256,
    _is_within_root,
    _read_utf8_file,
    opaque_document_id,
    sanitize_tenant_id,
)


ACL_SUFFIX = ".acl.json"


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _load_sidecar_acl(path: Path) -> Optional[KnowledgeACL]:
    """None 表示没有 sidecar；非法 sidecar 返回空 visibility（fail-closed）。"""
    candidates = [
        path.with_name(path.stem + ACL_SUFFIX),
        path.with_name(path.name + ACL_SUFFIX),
    ]
    for candidate in candidates:
        if candidate == path:
            continue
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                return KnowledgeACL(visibility="")
            return parse_knowledge_acl(payload)
    return None


def _acl_from_json_text(raw: str) -> Optional[KnowledgeACL]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "acl" not in data:
        return None
    if not isinstance(data.get("acl"), dict):
        return KnowledgeACL(visibility="")
    return parse_knowledge_acl(data.get("acl"))


def _is_acl_sidecar(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(ACL_SUFFIX)


class LocalDirectoryConnector:
    source_id = "local_directory"
    source_type = "local_directory"

    def __init__(self, root: str | Path, *, tenant_id: str = "default", source_id: Optional[str] = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.tenant_id = sanitize_tenant_id(tenant_id)
        if source_id:
            self.source_id = str(source_id)

    def health(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "healthy": self.root.is_dir(),
            "error_type": None if self.root.is_dir() else "source_root_missing",
        }

    def list_documents(
        self,
        checkpoint=None,
        *,
        tenant_id: str,
        deadline: Optional[float] = None,
    ) -> Iterable[KnowledgeSourceDocument]:
        tenant = sanitize_tenant_id(tenant_id)
        if tenant != self.tenant_id:
            return []
        if not self.root.is_dir():
            return []
        docs: List[KnowledgeSourceDocument] = []
        for path in sorted(self.root.rglob("*")):
            ensure_knowledge_deadline(deadline)
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if _is_acl_sidecar(path):
                continue
            if not _is_within_root(self.root, path):
                continue
            rel = path.relative_to(self.root).as_posix()
            raw = _read_utf8_file(path)
            title, body = _document_display_title(path, raw)
            content_hash = _file_sha256(path, deadline=deadline)
            acl = _acl_from_json_text(raw) if path.suffix.lower() == ".json" else None
            if acl is None:
                sidecar = _load_sidecar_acl(path)
                # 无 ACL 来源时显式生成 tenant ACL；非法 sidecar 保持 fail-closed。
                acl = sidecar if sidecar is not None else explicit_tenant_acl()
            docs.append(
                KnowledgeSourceDocument(
                    source_id=self.source_id,
                    tenant_id=tenant,
                    document_id=opaque_document_id(tenant, rel),
                    title=_safe_text(title, 200),
                    text=body,
                    content_hash=content_hash,
                    version=content_hash[:12],
                    updated_at=_mtime_iso(path),
                    source_uri=f"kb://{self.source_id}/{opaque_document_id(tenant, rel)}",
                    source_type=self.source_type,
                    acl=acl,
                    metadata={"path_fingerprint": opaque_document_id(tenant, rel)},
                )
            )
        return docs
