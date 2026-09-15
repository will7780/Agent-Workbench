# -*- coding: utf-8 -*-
"""Skill Provider factory registry。调用链不得写 provider 条件分支。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .registry import ModuleCapabilityRegistry
from .skill_models import (
    CompanyProfileResolveResult,
    CompanyProfileSnapshot,
    SkillCatalogResult,
    empty_company_skill_pipeline_fields,
    empty_skill_catalog,
    hash_tenant_id,
)
from .skill_provider import SkillProvider

SkillProviderFactory = Callable[
    [CompanyProfileSnapshot, Mapping[str, str], Optional[ModuleCapabilityRegistry]],
    SkillProvider,
]

_PROVIDER_FACTORIES: Dict[str, SkillProviderFactory] = {}


def register_skill_provider(provider_id: str, factory: SkillProviderFactory) -> None:
    key = str(provider_id or "").strip()
    if not key:
        raise ValueError("provider_id is required")
    _PROVIDER_FACTORIES[key] = factory


def _env_mapping(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    if env is None:
        return os.environ
    return env


def _build_local_skill_provider(
    profile: CompanyProfileSnapshot,
    env: Mapping[str, str],
    registry: Optional[ModuleCapabilityRegistry],
) -> SkillProvider:
    from .local_skill_provider import LocalSkillProvider

    settings = dict(profile.provider_settings or {})
    root_kind = str(settings.get("root_kind") or "").strip()
    if root_kind == "builtin":
        # Preserve the legacy selector without distributing or discovering assets.
        root_value = str(env.get("AGENT_WORKBENCH_SKILLS_ROOT") or "").strip()
        if not root_value:
            raise SkillProviderUnavailable()
        root = Path(root_value).expanduser()
    elif root_kind == "env":
        env_name = str(settings.get("root_env") or "").strip()
        root_value = str(env.get(env_name) or "").strip() if env_name else ""
        if not env_name or not root_value:
            raise SkillProviderUnavailable()
        root = Path(root_value).expanduser()
    else:
        raise SkillProviderUnavailable()
    return LocalSkillProvider(
        root,
        source_id=profile.source_id,
        tenant_id=profile.tenant_id,
        registry=registry,
    )


class SkillProviderUnavailable(Exception):
    error_type = "skill_provider_unavailable"


register_skill_provider("local", _build_local_skill_provider)


def build_skill_provider(
    profile: CompanyProfileSnapshot,
    *,
    env: Optional[Mapping[str, str]] = None,
    registry: Optional[ModuleCapabilityRegistry] = None,
) -> Tuple[Optional[SkillProvider], Optional[str]]:
    factory = _PROVIDER_FACTORIES.get(str(profile.provider_id or "").strip())
    if factory is None:
        return None, "skill_provider_unavailable"
    try:
        provider = factory(profile, _env_mapping(env), registry)
    except SkillProviderUnavailable:
        return None, "skill_provider_unavailable"
    except Exception:
        return None, "skill_provider_unavailable"
    return provider, None


def discover_profile_skills(
    profile_result: CompanyProfileResolveResult,
    *,
    env: Optional[Mapping[str, str]] = None,
    registry: Optional[ModuleCapabilityRegistry] = None,
    timeout_ms: int = 1000,
) -> SkillCatalogResult:
    if not profile_result.enabled or profile_result.snapshot is None:
        reason = profile_result.omission_reason or profile_result.error_type or "company_profile_disabled"
        return empty_skill_catalog(
            reason=reason,
            error_type=profile_result.error_type,
            degraded=bool(profile_result.degraded or profile_result.error_type),
        )
    snapshot = profile_result.snapshot
    provider, error_type = build_skill_provider(snapshot, env=env, registry=registry)
    if provider is None:
        return empty_skill_catalog(
            reason=error_type or "skill_provider_unavailable",
            provider=snapshot.provider_id,
            tenant_id_hash=hash_tenant_id(snapshot.tenant_id),
            profile_snapshot_id=snapshot.snapshot_id,
            degraded=True,
            error_type=error_type or "skill_provider_unavailable",
        )
    deadline = time.monotonic() + max(1, int(timeout_ms)) / 1000.0
    try:
        catalog = provider.list_metadata(snapshot, deadline_monotonic=deadline)
    except Exception:
        return empty_skill_catalog(
            reason="skill_provider_timeout" if time.monotonic() >= deadline else "skill_provider_unavailable",
            provider=provider.provider_id,
            tenant_id_hash=hash_tenant_id(snapshot.tenant_id),
            profile_snapshot_id=snapshot.snapshot_id,
            degraded=True,
            error_type="skill_provider_timeout" if time.monotonic() >= deadline else "skill_provider_unavailable",
        )
    return catalog


def prepare_company_skill_runtime(
    company_profile_config: Optional[Dict[str, Any]],
    *,
    env: Optional[Mapping[str, str]] = None,
    registry: Optional[ModuleCapabilityRegistry] = None,
    skill_version_store: Optional[Any] = None,
) -> Dict[str, Any]:
    from .company_profile import resolve_company_profile

    fields = empty_company_skill_pipeline_fields("company_profile_disabled")
    profile_result = resolve_company_profile(company_profile_config, env=env)
    catalog = discover_profile_skills(
        profile_result,
        env=env,
        registry=registry,
        timeout_ms=1000,
    )
    errors = list(catalog.errors)
    if profile_result.error_type:
        errors = [{"error_type": profile_result.error_type}] + errors
    fields["company_profile"] = profile_result.to_dict()
    fields["company_profile_snapshot_id"] = (
        profile_result.snapshot.snapshot_id if profile_result.snapshot is not None else None
    )
    fields["skill_catalog"] = catalog.to_dict()
    fields["skill_conflicts"] = list(catalog.conflicts)
    fields["skill_errors"] = errors
    fields["skill_suggestions"] = []
    fields["loaded_skills"] = []
    fields["skill_resources"] = []
    fields["skill_context_manifest"] = []
    fields["deferred_business_tool_calls"] = []
    if not profile_result.enabled or profile_result.snapshot is None:
        return fields

    from .skill_governance import apply_governance_to_catalog, resolve_skill_governance_snapshot
    from .skill_store import SkillGovernanceStoreError, SqliteSkillVersionStore

    def _fail_closed_governance(governance_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        unavailable = governance_payload or {
            "enabled": False,
            "error_type": "skill_governance_unavailable",
            "legacy": False,
            "managed": False,
            "active": [],
            "last_known_good": [],
            "records": [],
            "omission_reason": "skill_governance_unavailable",
        }
        unavailable["error_type"] = "skill_governance_unavailable"
        unavailable["legacy"] = False
        unavailable["omission_reason"] = "skill_governance_unavailable"
        filtered = apply_governance_to_catalog(catalog.skills, unavailable)
        catalog.skills = []
        catalog.enabled = False
        catalog.degraded = True
        catalog.omission_reason = "skill_governance_unavailable"
        catalog.errors = list(catalog.errors) + [{"error_type": "skill_governance_unavailable"}]
        merged_errors = list(errors) + list(filtered.get("errors") or [{"error_type": "skill_governance_unavailable"}])
        if not any(item.get("error_type") == "skill_governance_unavailable" for item in merged_errors if isinstance(item, dict)):
            merged_errors.append({"error_type": "skill_governance_unavailable"})
        fields["skill_catalog"] = catalog.to_dict()
        fields["skill_conflicts"] = list(catalog.conflicts)
        fields["skill_errors"] = merged_errors
        fields["skill_governance"] = unavailable
        fields["skill_replay_manifest"] = {
            "replayed": False,
            "error_type": None,
            "profile_snapshot_id": fields["company_profile_snapshot_id"],
            "skills": [],
            "omission_reason": "not_replayed",
        }
        return fields

    store = skill_version_store
    if store is None:
        try:
            store = SqliteSkillVersionStore()
        except Exception:
            return _fail_closed_governance()

    refs = [item.ref.to_dict() for item in catalog.skills]
    try:
        governance = resolve_skill_governance_snapshot(store, refs)
    except SkillGovernanceStoreError:
        return _fail_closed_governance()
    if governance.get("error_type") == "skill_governance_unavailable":
        return _fail_closed_governance(governance)

    filtered = apply_governance_to_catalog(catalog.skills, governance)
    if filtered.get("error_type") == "skill_governance_unavailable":
        return _fail_closed_governance(governance)
    catalog.skills = list(filtered.get("skills") or [])
    errors.extend(list(filtered.get("errors") or []))
    fields["skill_catalog"] = catalog.to_dict()
    fields["skill_conflicts"] = list(catalog.conflicts)
    fields["skill_errors"] = errors
    fields["skill_governance"] = governance
    fields["skill_replay_manifest"] = {
        "replayed": False,
        "error_type": None,
        "profile_snapshot_id": fields["company_profile_snapshot_id"],
        "skills": [],
        "omission_reason": "not_replayed",
    }
    return fields
