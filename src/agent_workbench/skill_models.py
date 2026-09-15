# -*- coding: utf-8 -*-
"""Company Skills 领域对象。字段与 to_dict() 以主设计第 15.3 / 16.2 节为准。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_tenant_id(tenant_id: str) -> str:
    digest = hashlib.sha256(str(tenant_id or "").encode("utf-8")).hexdigest()[:16]
    return f"tenant_{digest}"


@dataclass(frozen=True)
class SkillRef:
    source_id: str
    tenant_id: str
    name: str
    version: str
    checksum: str

    def stable_id(self) -> str:
        seed = _canonical_json(
            {
                "checksum": self.checksum,
                "name": self.name,
                "source_id": self.source_id,
                "tenant_id": self.tenant_id,
                "version": self.version,
            }
        )
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
        return f"skill_{digest}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "version": self.version,
            "checksum": self.checksum,
            "stable_id": self.stable_id(),
        }


@dataclass(frozen=True)
class SkillMetadata:
    ref: SkillRef
    description: str
    compatibility: str
    owner: str
    skill_type: str
    risk_tier: str
    preferred_capabilities: Tuple[str, ...]
    required_observations: Tuple[str, ...]
    applicable_modes: Tuple[str, ...]
    conflicts_with: Tuple[str, ...]
    replaces: Tuple[str, ...]
    resources: Tuple[str, ...]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "description": self.description,
            "compatibility": self.compatibility,
            "owner": self.owner,
            "skill_type": self.skill_type,
            "risk_tier": self.risk_tier,
            "preferred_capabilities": list(self.preferred_capabilities),
            "required_observations": list(self.required_observations),
            "applicable_modes": list(self.applicable_modes),
            "conflicts_with": list(self.conflicts_with),
            "replaces": list(self.replaces),
            "resources": list(self.resources),
            "status": self.status,
        }


@dataclass(frozen=True)
class SkillDocument:
    metadata: SkillMetadata
    body: str
    loaded_at: str

    def to_dict(self, *, include_body: bool = True) -> Dict[str, Any]:
        payload = {
            "metadata": self.metadata.to_dict(),
            "loaded_at": self.loaded_at,
        }
        if include_body:
            payload["body"] = self.body
        return payload


@dataclass
class SkillCatalogResult:
    enabled: bool
    provider: Optional[str]
    tenant_id_hash: Optional[str]
    profile_snapshot_id: Optional[str]
    skills: List[SkillMetadata] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    omission_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "tenant_id_hash": self.tenant_id_hash,
            "profile_snapshot_id": self.profile_snapshot_id,
            "skills": [item.to_dict() for item in self.skills],
            "conflicts": list(self.conflicts),
            "errors": list(self.errors),
            "degraded": self.degraded,
            "omission_reason": self.omission_reason,
        }


@dataclass
class SkillLoadResult:
    loaded: bool
    skill: Optional[SkillDocument] = None
    resource_path: Optional[str] = None
    resource_checksum: Optional[str] = None
    error_type: Optional[str] = None
    omission_reason: Optional[str] = None
    redacted: bool = False

    def to_dict(self, *, include_body: bool = True) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "skill": self.skill.to_dict(include_body=include_body) if self.skill is not None else None,
            "resource_path": self.resource_path,
            "resource_checksum": self.resource_checksum,
            "error_type": self.error_type,
            "omission_reason": self.omission_reason,
            "redacted": self.redacted,
        }


@dataclass(frozen=True)
class CompanyProfileSnapshot:
    profile_id: str
    tenant_id: str
    display_name: str
    version: str
    status: str
    provider_id: str
    source_id: str
    provider_settings: Mapping[str, str]
    required_skills: Tuple[str, ...]
    available_skills: Tuple[str, ...]
    disabled_skills: Tuple[str, ...]
    pins: Dict[str, str]
    priority: Tuple[str, ...]
    policy_ref: Optional[str]
    knowledge_profile: Optional[str]
    memory_namespace: Optional[str]
    snapshot_id: str
    artifact_policies: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "tenant_id": self.tenant_id,
            "display_name": self.display_name,
            "version": self.version,
            "status": self.status,
            "provider_id": self.provider_id,
            "source_id": self.source_id,
            "provider_settings": dict(self.provider_settings),
            "required_skills": list(self.required_skills),
            "available_skills": list(self.available_skills),
            "disabled_skills": list(self.disabled_skills),
            "pins": dict(self.pins),
            "priority": list(self.priority),
            "policy_ref": self.policy_ref,
            "knowledge_profile": self.knowledge_profile,
            "memory_namespace": self.memory_namespace,
            "snapshot_id": self.snapshot_id,
            "artifact_policies": dict(self.artifact_policies),
        }


@dataclass
class CompanyProfileResolveResult:
    enabled: bool
    snapshot: Optional[CompanyProfileSnapshot] = None
    error_type: Optional[str] = None
    degraded: bool = False
    omission_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "snapshot": self.snapshot.to_dict() if self.snapshot is not None else None,
            "error_type": self.error_type,
            "degraded": self.degraded,
            "omission_reason": self.omission_reason,
        }


def empty_skill_catalog(
    *,
    reason: str = "company_profile_disabled",
    provider: Optional[str] = None,
    tenant_id_hash: Optional[str] = None,
    profile_snapshot_id: Optional[str] = None,
    degraded: bool = False,
    error_type: Optional[str] = None,
) -> SkillCatalogResult:
    errors: List[Dict[str, Any]] = []
    if error_type:
        errors.append({"error_type": error_type})
    return SkillCatalogResult(
        enabled=False,
        provider=provider,
        tenant_id_hash=tenant_id_hash,
        profile_snapshot_id=profile_snapshot_id,
        skills=[],
        conflicts=[],
        errors=errors,
        degraded=degraded,
        omission_reason=reason,
    )


def empty_company_profile_result(
    reason: str = "company_profile_disabled",
    *,
    error_type: Optional[str] = None,
    degraded: bool = False,
) -> CompanyProfileResolveResult:
    return CompanyProfileResolveResult(
        enabled=False,
        snapshot=None,
        error_type=error_type,
        degraded=degraded,
        omission_reason=reason,
    )


def empty_company_skill_pipeline_fields(
    reason: str = "company_profile_disabled",
) -> Dict[str, Any]:
    profile = empty_company_profile_result(reason)
    catalog = empty_skill_catalog(reason=reason)
    return {
        "company_profile": profile.to_dict(),
        "company_profile_snapshot_id": None,
        "skill_catalog": catalog.to_dict(),
        "skill_suggestions": [],
        "loaded_skills": [],
        "skill_resources": [],
        "skill_context_manifest": [],
        "skill_conflicts": [],
        "skill_errors": list(catalog.errors),
        "deferred_business_tool_calls": [],
        "skill_governance": {
            "enabled": False,
            "error_type": None,
            "legacy": True,
            "managed": False,
            "active": [],
            "last_known_good": [],
            "omission_reason": reason,
        },
        "skill_replay_manifest": {
            "replayed": False,
            "error_type": None,
            "profile_snapshot_id": None,
            "skills": [],
            "omission_reason": reason,
        },
    }


COMPANY_SKILL_PIPELINE_FIELD_NAMES = tuple(empty_company_skill_pipeline_fields().keys())


def select_company_skill_pipeline_fields(
    runtime: Optional[Dict[str, Any]] = None,
    *,
    reason: str = "company_profile_disabled",
) -> Dict[str, Any]:
    fields = empty_company_skill_pipeline_fields(reason)
    if not runtime:
        return fields
    for key in COMPANY_SKILL_PIPELINE_FIELD_NAMES:
        if key in runtime:
            fields[key] = runtime[key]
    return fields


def skill_ref_from_dict(payload: Optional[Dict[str, Any]]) -> Optional[SkillRef]:
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name") or "").strip()
    version = str(payload.get("version") or "").strip()
    checksum = str(payload.get("checksum") or "").strip()
    if not name or not version or not checksum:
        return None
    return SkillRef(
        source_id=str(payload.get("source_id") or "").strip(),
        tenant_id=str(payload.get("tenant_id") or "").strip(),
        name=name,
        version=version,
        checksum=checksum,
    )


def skill_metadata_from_dict(payload: Optional[Dict[str, Any]]) -> Optional[SkillMetadata]:
    if not isinstance(payload, dict):
        return None
    ref = skill_ref_from_dict(payload.get("ref") if isinstance(payload.get("ref"), dict) else None)
    if ref is None:
        return None
    return SkillMetadata(
        ref=ref,
        description=str(payload.get("description") or ""),
        compatibility=str(payload.get("compatibility") or ""),
        owner=str(payload.get("owner") or "unknown"),
        skill_type=str(payload.get("skill_type") or "operational"),
        risk_tier=str(payload.get("risk_tier") or "medium"),
        preferred_capabilities=tuple(str(item) for item in (payload.get("preferred_capabilities") or [])),
        required_observations=tuple(str(item) for item in (payload.get("required_observations") or [])),
        applicable_modes=tuple(str(item) for item in (payload.get("applicable_modes") or [])),
        conflicts_with=tuple(str(item) for item in (payload.get("conflicts_with") or [])),
        replaces=tuple(str(item) for item in (payload.get("replaces") or [])),
        resources=tuple(str(item) for item in (payload.get("resources") or [])),
        status=str(payload.get("status") or "active"),
    )


def skill_catalog_result_from_dict(payload: Optional[Dict[str, Any]]) -> SkillCatalogResult:
    data = payload if isinstance(payload, dict) else {}
    skills: List[SkillMetadata] = []
    for item in data.get("skills") or []:
        meta = skill_metadata_from_dict(item if isinstance(item, dict) else None)
        if meta is not None:
            skills.append(meta)
    return SkillCatalogResult(
        enabled=bool(data.get("enabled")),
        provider=data.get("provider"),
        tenant_id_hash=data.get("tenant_id_hash"),
        profile_snapshot_id=data.get("profile_snapshot_id"),
        skills=skills,
        conflicts=list(data.get("conflicts") or []),
        errors=list(data.get("errors") or []),
        degraded=bool(data.get("degraded")),
        omission_reason=data.get("omission_reason"),
    )


def company_profile_snapshot_from_dict(payload: Optional[Dict[str, Any]]) -> Optional[CompanyProfileSnapshot]:
    if not isinstance(payload, dict):
        return None
    snapshot_id = str(payload.get("snapshot_id") or "").strip()
    profile_id = str(payload.get("profile_id") or "").strip()
    tenant_id = str(payload.get("tenant_id") or "").strip()
    if not snapshot_id or not profile_id or not tenant_id:
        return None
    settings = payload.get("provider_settings") if isinstance(payload.get("provider_settings"), dict) else {}
    pins = payload.get("pins") if isinstance(payload.get("pins"), dict) else {}
    return CompanyProfileSnapshot(
        profile_id=profile_id,
        tenant_id=tenant_id,
        display_name=str(payload.get("display_name") or profile_id),
        version=str(payload.get("version") or ""),
        status=str(payload.get("status") or "active"),
        provider_id=str(payload.get("provider_id") or ""),
        source_id=str(payload.get("source_id") or ""),
        provider_settings={str(key): str(value) for key, value in settings.items()},
        required_skills=tuple(str(item) for item in (payload.get("required_skills") or [])),
        available_skills=tuple(str(item) for item in (payload.get("available_skills") or [])),
        disabled_skills=tuple(str(item) for item in (payload.get("disabled_skills") or [])),
        pins={str(key): str(value) for key, value in pins.items()},
        priority=tuple(str(item) for item in (payload.get("priority") or [])),
        policy_ref=payload.get("policy_ref"),
        knowledge_profile=payload.get("knowledge_profile"),
        memory_namespace=payload.get("memory_namespace"),
        snapshot_id=snapshot_id,
        artifact_policies=dict(payload.get("artifact_policies") or {}),
    )


def frozen_profile_snapshot_from_runtime(runtime: Optional[Dict[str, Any]]) -> Optional[CompanyProfileSnapshot]:
    profile = (runtime or {}).get("company_profile") if isinstance(runtime, dict) else None
    if not isinstance(profile, dict):
        return None
    snapshot = profile.get("snapshot") if isinstance(profile.get("snapshot"), dict) else None
    return company_profile_snapshot_from_dict(snapshot)

