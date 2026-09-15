# -*- coding: utf-8 -*-
"""企业知识 ACL：tenant 强制匹配、deny 优先、不明确则 fail-closed。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .local_knowledge_provider import sanitize_tenant_id

VISIBILITY_PRIVATE = "private"
VISIBILITY_TENANT = "tenant"
VISIBILITY_RESTRICTED = "restricted"
ALLOWED_VISIBILITIES = (VISIBILITY_PRIVATE, VISIBILITY_TENANT, VISIBILITY_RESTRICTED)

PERMISSION_MODE_ACL = "acl"


def _norm_id(value: Any) -> str:
    return str(value or "").strip()


def _unique_ids(values: Optional[Iterable[Any]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for raw in values or []:
        item = _norm_id(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


@dataclass
class KnowledgeACL:
    visibility: str = ""
    allowed_users: List[str] = field(default_factory=list)
    allowed_groups: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)
    denied_users: List[str] = field(default_factory=list)
    denied_groups: List[str] = field(default_factory=list)
    denied_roles: List[str] = field(default_factory=list)

    def normalized(self) -> "KnowledgeACL":
        vis = _norm_id(self.visibility).lower()
        return KnowledgeACL(
            visibility=vis,
            allowed_users=_unique_ids(self.allowed_users),
            allowed_groups=_unique_ids(self.allowed_groups),
            allowed_roles=_unique_ids(self.allowed_roles),
            denied_users=_unique_ids(self.denied_users),
            denied_groups=_unique_ids(self.denied_groups),
            denied_roles=_unique_ids(self.denied_roles),
        )

    def to_dict(self) -> Dict[str, Any]:
        acl = self.normalized()
        return {
            "visibility": acl.visibility,
            "allowed_users": list(acl.allowed_users),
            "allowed_groups": list(acl.allowed_groups),
            "allowed_roles": list(acl.allowed_roles),
            "denied_users": list(acl.denied_users),
            "denied_groups": list(acl.denied_groups),
            "denied_roles": list(acl.denied_roles),
        }


@dataclass(frozen=True)
class KnowledgeIdentity:
    tenant_id: str
    user_id: Optional[str] = None
    groups: Sequence[str] = field(default_factory=tuple)
    roles: Sequence[str] = field(default_factory=tuple)

    def normalized(self) -> "KnowledgeIdentity":
        return KnowledgeIdentity(
            tenant_id=sanitize_tenant_id(self.tenant_id),
            user_id=_norm_id(self.user_id) or None,
            groups=tuple(_unique_ids(self.groups)),
            roles=tuple(_unique_ids(self.roles)),
        )


@dataclass(frozen=True)
class ACLDecision:
    allowed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "reason": self.reason}


def explicit_tenant_acl() -> KnowledgeACL:
    """验收/本地目录在明确需要 tenant 可见时必须调用此函数，不得依赖 parse(None)。"""
    return KnowledgeACL(visibility=VISIBILITY_TENANT)


def parse_knowledge_acl(payload: Any) -> KnowledgeACL:
    """缺失或非法 ACL 不得默认成 tenant 可见。"""
    if payload is None:
        return KnowledgeACL(visibility="")
    if isinstance(payload, KnowledgeACL):
        return payload.normalized()
    if not isinstance(payload, dict):
        return KnowledgeACL(visibility="")
    raw_vis = payload.get("visibility")
    visibility = _norm_id(raw_vis).lower() if raw_vis is not None else ""
    return KnowledgeACL(
        visibility=visibility,
        allowed_users=list(payload.get("allowed_users") or []),
        allowed_groups=list(payload.get("allowed_groups") or []),
        allowed_roles=list(payload.get("allowed_roles") or []),
        denied_users=list(payload.get("denied_users") or []),
        denied_groups=list(payload.get("denied_groups") or []),
        denied_roles=list(payload.get("denied_roles") or []),
    ).normalized()


def identity_from_search_request(request: Any) -> KnowledgeIdentity:
    return KnowledgeIdentity(
        tenant_id=str(getattr(request, "tenant_id", "") or ""),
        user_id=getattr(request, "user_id", None),
        groups=tuple(getattr(request, "groups", None) or []),
        roles=tuple(getattr(request, "roles", None) or []),
    ).normalized()


def build_identity_summary(identity: KnowledgeIdentity) -> Dict[str, Any]:
    ident = identity.normalized()
    return {
        "tenant_id": ident.tenant_id,
        "has_user": bool(ident.user_id),
        "group_count": len(ident.groups),
        "role_count": len(ident.roles),
    }


def check_knowledge_acl(
    acl: Any,
    identity: KnowledgeIdentity,
    document_tenant_id: str,
) -> ACLDecision:
    """
    权限不明确时 fail-closed，且不返回文档正文。
    deny 优先于 allow；tenant_id 必须强制匹配。
    """
    ident = identity.normalized()
    doc_tenant = sanitize_tenant_id(document_tenant_id)
    if not ident.tenant_id or not doc_tenant or ident.tenant_id != doc_tenant:
        return ACLDecision(False, "tenant_mismatch")

    if acl is None:
        return ACLDecision(False, "acl_missing")
    if not isinstance(acl, (KnowledgeACL, dict)):
        return ACLDecision(False, "acl_invalid")
    if isinstance(acl, dict) and ("visibility" not in acl or not _norm_id(acl.get("visibility"))):
        return ACLDecision(False, "acl_missing")
    parsed = parse_knowledge_acl(acl)
    vis = parsed.visibility
    if vis not in ALLOWED_VISIBILITIES:
        return ACLDecision(False, "acl_missing" if not vis else "identity_ambiguous")

    user = ident.user_id or ""
    groups = set(ident.groups)
    roles = set(ident.roles)

    if user and user in parsed.denied_users:
        return ACLDecision(False, "deny_user")
    if groups & set(parsed.denied_groups):
        return ACLDecision(False, "deny_group")
    if roles & set(parsed.denied_roles):
        return ACLDecision(False, "deny_role")

    if vis == VISIBILITY_PRIVATE:
        if not user:
            return ACLDecision(False, "identity_ambiguous")
        if user in parsed.allowed_users:
            return ACLDecision(True, "allowed_user")
        return ACLDecision(False, "private_no_user")

    if vis == VISIBILITY_RESTRICTED:
        if user and user in parsed.allowed_users:
            return ACLDecision(True, "allowed_user")
        if groups & set(parsed.allowed_groups):
            return ACLDecision(True, "allowed_group")
        if roles & set(parsed.allowed_roles):
            return ACLDecision(True, "allowed_role")
        return ACLDecision(False, "restricted_no_allow")

    return ACLDecision(True, "allowed_tenant")
