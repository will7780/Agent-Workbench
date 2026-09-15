# -*- coding: utf-8 -*-
"""SkillGuard：按固定八步顺序校验，返回 Catalog 已有 SkillRef。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .skill_models import CompanyProfileSnapshot, SkillCatalogResult, SkillMetadata, SkillRef

DEFAULT_MAX_LOADED_SKILLS_PER_RUN = 3

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class SkillGuardResult:
    allowed: bool
    error_type: Optional[str]
    reason: str
    skill_ref: Optional[SkillRef]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "error_type": self.error_type,
            "reason": self.reason,
            "skill_ref": self.skill_ref.to_dict() if self.skill_ref is not None else None,
        }


def _deny(error_type: str, reason: str) -> SkillGuardResult:
    return SkillGuardResult(allowed=False, error_type=error_type, reason=reason, skill_ref=None)


def guard_skill_load(
    skill_name: str,
    *,
    profile: CompanyProfileSnapshot,
    catalog: SkillCatalogResult,
    requested_version: Optional[str] = None,
    resource_path: Optional[str] = None,
    loaded_stable_ids: Optional[Sequence[str]] = None,
    reserved_stable_ids: Optional[Sequence[str]] = None,
    max_loaded_skills: int = DEFAULT_MAX_LOADED_SKILLS_PER_RUN,
) -> SkillGuardResult:
    name = str(skill_name or "").strip()
    if not _NAME_RE.match(name):
        return _deny("skill_not_allowed", "invalid skill name")

    allowed_names = set(profile.required_skills) | set(profile.available_skills)
    if name not in allowed_names or name in set(profile.disabled_skills):
        return _deny("skill_not_allowed", "skill is not enabled for this profile")

    matches: List[SkillMetadata] = [item for item in catalog.skills if item.ref.name == name]
    if not matches:
        return _deny("skill_not_found", "skill is not present in catalog")
    if len(matches) != 1:
        return _deny("skill_conflict", "catalog has multiple candidates")
    candidate = matches[0]

    pin = str(profile.pins.get(name) or "").strip() or None
    requested = str(requested_version or "").strip() or None
    if requested and not _VERSION_RE.match(requested):
        return _deny("skill_version_mismatch", "requested version is invalid")
    if pin and requested and pin != requested:
        return _deny("skill_version_mismatch", "requested version does not match pin")
    expected_version = requested or pin
    if expected_version and candidate.ref.version != expected_version:
        return _deny("skill_version_mismatch", "catalog version does not match pin")

    if candidate.status != "active":
        return _deny("skill_status_denied", "skill status is not active")

    catalog_names = {item.ref.name for item in catalog.skills}
    unresolved = []
    for other in candidate.conflicts_with:
        if other not in catalog_names:
            continue
        if other in candidate.replaces:
            continue
        winner = None
        for item in profile.priority:
            if item == name:
                winner = name
                break
            if item == other:
                winner = other
                break
        if winner != name:
            unresolved.append(other)
    if unresolved:
        return _deny("skill_conflict", "skill conflict is not resolved")

    if resource_path is not None:
        relative = str(resource_path).strip().replace("\\", "/")
        if relative not in set(candidate.resources):
            return _deny("skill_resource_not_declared", "resource is not declared")
        return SkillGuardResult(
            allowed=True,
            error_type=None,
            reason="allowed",
            skill_ref=candidate.ref,
        )

    loaded_ids = {str(item).strip() for item in (loaded_stable_ids or []) if str(item).strip()}
    reserved_ids = {str(item).strip() for item in (reserved_stable_ids or []) if str(item).strip()}
    stable_id = str(candidate.ref.stable_id() or "").strip()
    if stable_id and stable_id not in loaded_ids and stable_id not in reserved_ids:
        occupied = len(loaded_ids | reserved_ids)
        if occupied >= int(max_loaded_skills or DEFAULT_MAX_LOADED_SKILLS_PER_RUN):
            return _deny("skill_load_limit_exceeded", "full skill load limit exceeded")

    return SkillGuardResult(
        allowed=True,
        error_type=None,
        reason="allowed",
        skill_ref=candidate.ref,
    )
