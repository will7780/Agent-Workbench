# -*- coding: utf-8 -*-
"""LocalSkillProvider：只读 SKILL.md，Catalog 不暴露 body，路径/symlink fail-closed。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import yaml

from .redaction import contains_secret_blob, redact_recursive, redact_text
from .registry import ModuleCapabilityRegistry
from .skill_models import (
    CompanyProfileSnapshot,
    SkillCatalogResult,
    SkillDocument,
    SkillLoadResult,
    SkillMetadata,
    SkillRef,
    hash_tenant_id,
)

ALLOWED_RESOURCE_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml"}
MAX_SKILL_CHARS = 40_000
MAX_RESOURCE_CHARS = 20_000
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_STATUS_RE = re.compile(r"^(active|draft|deprecated|archived|security_reviewed|eval_passed)$")
_SKILL_TYPE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_RISK_TIER_RE = re.compile(r"^(low|medium|high)$")
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MODE_RE = re.compile(r"^(dry_run|read_only|local_write|live)$")


class SkillAccessDenied(Exception):
    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


class SkillProviderTimeout(Exception):
    def __init__(self) -> None:
        super().__init__("skill_provider_timeout")
        self.error_type = "skill_provider_timeout"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_deadline(clock: Callable[[], float], deadline_monotonic: Optional[float]) -> None:
    if deadline_monotonic is not None and clock() >= deadline_monotonic:
        raise SkillProviderTimeout()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _deny_unsafe_part(part: str) -> None:
    if not part or part in (".", "..") or "/" in part or "\\" in part or "\x00" in part:
        raise SkillAccessDenied("skill_path_denied")


def resolve_skill_path(root: Path, *parts: str) -> Path:
    base = Path(root)
    for part in parts:
        _deny_unsafe_part(str(part))
    cursor = base
    if cursor.exists() and cursor.is_symlink():
        raise SkillAccessDenied("skill_symlink_denied")
    for part in parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise SkillAccessDenied("skill_symlink_denied")
    try:
        resolved = (base.joinpath(*parts)).resolve(strict=True)
        root_resolved = base.resolve(strict=True)
    except FileNotFoundError:
        raise SkillAccessDenied("skill_not_found") from None
    except OSError:
        raise SkillAccessDenied("skill_path_denied") from None
    if not _is_relative_to(resolved, root_resolved):
        raise SkillAccessDenied("skill_path_denied")
    return resolved


def compute_skill_checksum(frontmatter: Dict[str, Any], body: str) -> str:
    canonical = json.dumps(frontmatter, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(f"{canonical}\n{body}".encode("utf-8")).hexdigest()


def _split_frontmatter(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, None, "skill_frontmatter_invalid"
    closing = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing = index
            break
    if closing is None:
        return None, None, "skill_frontmatter_invalid"
    yaml_text = "".join(lines[1:closing])
    body = "".join(lines[closing + 1 :])
    return yaml_text, body, None


def _string_tuple(raw: Any) -> Tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return tuple()
    out: List[str] = []
    seen = set()
    for item in raw:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def _capability_exists(registry: Optional[ModuleCapabilityRegistry], cap: str) -> bool:
    if registry is None or "." not in cap:
        return registry is None
    module, action = cap.split(".", 1)
    return registry.action_exists(module, action)


def _value_has_secret(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return contains_secret_blob(value)
    if isinstance(value, (list, tuple)):
        return any(_value_has_secret(item) for item in value)
    if isinstance(value, dict):
        return any(_value_has_secret(key) or _value_has_secret(item) for key, item in value.items())
    return contains_secret_blob(str(value))


def _valid_resource_path(path: str) -> bool:
    if not path or "\\" in path or "\x00" in path or path.startswith("/") or ".." in path.split("/"):
        return False
    suffix = Path(path).suffix.lower()
    if suffix not in ALLOWED_RESOURCE_SUFFIXES:
        return False
    parts = path.split("/")
    return all(bool(part) and part not in (".", "..") for part in parts)


def _control_fields_valid(
    *,
    name: str,
    version: str,
    status: str,
    skill_type: str,
    risk_tier: str,
    preferred_capabilities: Tuple[str, ...],
    required_observations: Tuple[str, ...],
    applicable_modes: Tuple[str, ...],
    conflicts_with: Tuple[str, ...],
    replaces: Tuple[str, ...],
    resources: Tuple[str, ...],
) -> bool:
    if not _NAME_RE.match(name) or not _VERSION_RE.match(version):
        return False
    if not _STATUS_RE.match(status) or not _SKILL_TYPE_RE.match(skill_type) or not _RISK_TIER_RE.match(risk_tier):
        return False
    if any(not _CAPABILITY_RE.match(item) for item in preferred_capabilities):
        return False
    if any(not _TOKEN_RE.match(item) for item in required_observations):
        return False
    if any(not _MODE_RE.match(item) for item in applicable_modes):
        return False
    if any(not _NAME_RE.match(item) for item in conflicts_with + replaces):
        return False
    if any(not _valid_resource_path(item) for item in resources):
        return False
    return True


def parse_skill_markdown(
    text: str,
    *,
    source_id: str,
    tenant_id: str,
    expected_name: str,
    registry: Optional[ModuleCapabilityRegistry] = None,
) -> SkillLoadResult:
    if len(text) > MAX_SKILL_CHARS:
        return SkillLoadResult(
            loaded=False,
            error_type="skill_content_too_large",
            omission_reason="skill_content_too_large",
        )
    yaml_text, body, split_error = _split_frontmatter(text)
    if split_error or yaml_text is None or body is None:
        return SkillLoadResult(loaded=False, error_type="skill_frontmatter_invalid", omission_reason="skill_frontmatter_invalid")
    try:
        frontmatter = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return SkillLoadResult(loaded=False, error_type="skill_frontmatter_invalid", omission_reason="skill_frontmatter_invalid")
    if not isinstance(frontmatter, dict):
        return SkillLoadResult(loaded=False, error_type="skill_frontmatter_invalid", omission_reason="skill_frontmatter_invalid")
    name = str(frontmatter.get("name") or "").strip()
    if name != expected_name or not _NAME_RE.match(name):
        return SkillLoadResult(loaded=False, error_type="skill_frontmatter_invalid", omission_reason="skill_frontmatter_invalid")
    metadata_block = frontmatter.get("metadata") if isinstance(frontmatter.get("metadata"), dict) else {}
    version = str(metadata_block.get("version") or "").strip()
    if not _VERSION_RE.match(version):
        return SkillLoadResult(loaded=False, error_type="skill_frontmatter_invalid", omission_reason="skill_frontmatter_invalid")
    status = str(metadata_block.get("status") or "active").strip() or "active"
    skill_type = str(metadata_block.get("skill_type") or "operational").strip() or "operational"
    risk_tier = str(metadata_block.get("risk_tier") or "medium").strip() or "medium"
    preferred = _string_tuple(metadata_block.get("preferred_capabilities"))
    required_observations = _string_tuple(metadata_block.get("required_observations"))
    applicable_modes = _string_tuple(metadata_block.get("applicable_modes"))
    conflicts_with = _string_tuple(metadata_block.get("conflicts_with"))
    replaces = _string_tuple(metadata_block.get("replaces"))
    resources = _string_tuple(metadata_block.get("resources"))
    control_values = {
        "name": name,
        "version": version,
        "status": status,
        "skill_type": skill_type,
        "risk_tier": risk_tier,
        "preferred_capabilities": preferred,
        "required_observations": required_observations,
        "applicable_modes": applicable_modes,
        "conflicts_with": conflicts_with,
        "replaces": replaces,
        "resources": resources,
    }
    if _value_has_secret(control_values):
        return SkillLoadResult(loaded=False, error_type="skill_secret_detected", omission_reason="skill_secret_detected")
    if not _control_fields_valid(
        name=name,
        version=version,
        status=status,
        skill_type=skill_type,
        risk_tier=risk_tier,
        preferred_capabilities=preferred,
        required_observations=required_observations,
        applicable_modes=applicable_modes,
        conflicts_with=conflicts_with,
        replaces=replaces,
        resources=resources,
    ):
        return SkillLoadResult(loaded=False, error_type="skill_frontmatter_invalid", omission_reason="skill_frontmatter_invalid")
    text_fields, _ = redact_recursive(
        {
            "description": str(frontmatter.get("description") or ""),
            "compatibility": str(frontmatter.get("compatibility") or ""),
            "owner": str(metadata_block.get("owner") or "unknown"),
        }
    )
    description = str(text_fields.get("description") or "")[:1000]
    compatibility = str(text_fields.get("compatibility") or "")[:240]
    owner = str(text_fields.get("owner") or "unknown")[:120] or "unknown"
    if _value_has_secret({"description": description, "compatibility": compatibility, "owner": owner}):
        return SkillLoadResult(loaded=False, error_type="skill_secret_detected", omission_reason="skill_secret_detected")
    body_text, body_redacted = redact_text(body)
    checksum = compute_skill_checksum(frontmatter, body)
    ref = SkillRef(
        source_id=source_id,
        tenant_id=tenant_id,
        name=name,
        version=version,
        checksum=checksum,
    )
    meta = SkillMetadata(
        ref=ref,
        description=description,
        compatibility=compatibility,
        owner=owner,
        skill_type=skill_type,
        risk_tier=risk_tier,
        preferred_capabilities=preferred,
        required_observations=required_observations,
        applicable_modes=applicable_modes,
        conflicts_with=conflicts_with,
        replaces=replaces,
        resources=resources,
        status=status,
    )
    if _value_has_secret(meta.to_dict()):
        return SkillLoadResult(loaded=False, error_type="skill_secret_detected", omission_reason="skill_secret_detected")
    document = SkillDocument(metadata=meta, body=body_text, loaded_at=_utc_now())
    error_type = None
    if registry is not None:
        missing = [cap for cap in preferred if not _capability_exists(registry, cap)]
        if missing:
            error_type = "skill_capability_incompatible"
    return SkillLoadResult(
        loaded=True,
        skill=document,
        resource_path=None,
        resource_checksum=None,
        error_type=error_type,
        omission_reason=error_type,
        redacted=body_redacted,
    )


class LocalSkillProvider:
    provider_id = "local"

    def __init__(
        self,
        root: Union[str, Path],
        *,
        source_id: str,
        tenant_id: str,
        registry: Optional[ModuleCapabilityRegistry] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root)
        self.source_id = str(source_id or "").strip()
        self.tenant_id = str(tenant_id or "").strip()
        self.registry = registry
        self.clock = clock

    def _fail_load(self, error_type: str) -> SkillLoadResult:
        return SkillLoadResult(loaded=False, error_type=error_type, omission_reason=error_type)

    def _assert_profile(self, profile: CompanyProfileSnapshot) -> Optional[str]:
        if profile.tenant_id != self.tenant_id or profile.source_id != self.source_id:
            return "skill_not_allowed"
        return None

    def _read_skill_text(self, name: str, version: str, deadline_monotonic: Optional[float]) -> str:
        _ensure_deadline(self.clock, deadline_monotonic)
        path = resolve_skill_path(self.root, name, version, "SKILL.md")
        _ensure_deadline(self.clock, deadline_monotonic)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise SkillAccessDenied("skill_frontmatter_invalid") from None
        except OSError:
            raise SkillAccessDenied("skill_path_denied") from None
        _ensure_deadline(self.clock, deadline_monotonic)
        if len(text) > MAX_SKILL_CHARS:
            raise SkillAccessDenied("skill_content_too_large")
        return text

    def _list_versions(self, name: str, deadline_monotonic: Optional[float] = None) -> List[str]:
        if not _NAME_RE.match(name):
            raise SkillAccessDenied("skill_path_denied")
        skill_dir = resolve_skill_path(self.root, name)
        versions: List[str] = []
        try:
            children = sorted(skill_dir.iterdir(), key=lambda item: item.name)
        except OSError:
            raise SkillAccessDenied("skill_path_denied") from None
        _ensure_deadline(self.clock, deadline_monotonic)
        for child in children:
            _ensure_deadline(self.clock, deadline_monotonic)
            if not _VERSION_RE.match(child.name):
                continue
            try:
                resolve_skill_path(self.root, name, child.name, "SKILL.md")
            except SkillAccessDenied:
                continue
            versions.append(child.name)
        return versions

    def _load_metadata_from_directory(
        self,
        name: str,
        version: str,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> SkillLoadResult:
        parsed = self._parse_named(name, version, deadline_monotonic=deadline_monotonic)
        if not parsed.loaded or parsed.skill is None:
            return parsed
        return SkillLoadResult(
            loaded=True,
            skill=SkillDocument(
                metadata=parsed.skill.metadata,
                body="",
                loaded_at=parsed.skill.loaded_at,
            ),
            resource_path=None,
            resource_checksum=None,
            error_type=parsed.error_type,
            omission_reason=parsed.omission_reason,
            redacted=parsed.redacted,
        )

    def _parse_named(
        self,
        name: str,
        version: str,
        *,
        deadline_monotonic: Optional[float],
    ) -> SkillLoadResult:
        try:
            text = self._read_skill_text(name, version, deadline_monotonic)
        except SkillAccessDenied as exc:
            return self._fail_load(exc.error_type)
        parsed = parse_skill_markdown(
            text,
            source_id=self.source_id,
            tenant_id=self.tenant_id,
            expected_name=name,
            registry=self.registry,
        )
        if parsed.loaded and parsed.skill is not None and parsed.skill.metadata.ref.version != version:
            return self._fail_load("skill_version_mismatch")
        return parsed

    def list_metadata(
        self,
        profile: CompanyProfileSnapshot,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> SkillCatalogResult:
        errors: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []
        skills: List[SkillMetadata] = []
        degraded = False
        identity_error = self._assert_profile(profile)
        if identity_error:
            return SkillCatalogResult(
                enabled=False,
                provider=self.provider_id,
                tenant_id_hash=hash_tenant_id(self.tenant_id),
                profile_snapshot_id=profile.snapshot_id,
                errors=[{"error_type": identity_error}],
                degraded=True,
                omission_reason=identity_error,
            )
        overlap = (set(profile.required_skills) | set(profile.available_skills)) & set(profile.disabled_skills)
        if overlap:
            return SkillCatalogResult(
                enabled=False,
                provider=self.provider_id,
                tenant_id_hash=hash_tenant_id(self.tenant_id),
                profile_snapshot_id=profile.snapshot_id,
                errors=[{"error_type": "company_profile_invalid"}],
                degraded=True,
                omission_reason="company_profile_invalid",
            )
        disabled = set(profile.disabled_skills)
        names: List[str] = []
        seen = set()
        for name in list(profile.required_skills) + list(profile.available_skills):
            if name in disabled or name in seen:
                continue
            seen.add(name)
            names.append(name)
        try:
            for name in names:
                _ensure_deadline(self.clock, deadline_monotonic)
                pin = str(profile.pins.get(name) or "").strip() or None
                required = name in profile.required_skills
                try:
                    if pin:
                        if not _VERSION_RE.match(pin):
                            raise SkillAccessDenied("skill_version_mismatch")
                        versions = [pin]
                        resolve_skill_path(self.root, name, pin, "SKILL.md")
                    else:
                        versions = self._list_versions(name, deadline_monotonic)
                except SkillAccessDenied as exc:
                    errors.append({"error_type": exc.error_type, "skill_name": name})
                    if required:
                        degraded = True
                    continue
                if not pin and len(versions) > 1:
                    conflicts.append({"error_type": "skill_conflict", "skill_name": name})
                    errors.append({"error_type": "skill_conflict", "skill_name": name})
                    degraded = True
                    continue
                if not versions:
                    errors.append({"error_type": "skill_not_found", "skill_name": name})
                    if required:
                        degraded = True
                    continue
                version = versions[0] if not pin else pin
                parsed = self._load_metadata_from_directory(
                    name, version, deadline_monotonic=deadline_monotonic
                )
                if not parsed.loaded or parsed.skill is None:
                    error_type = parsed.error_type or "skill_not_found"
                    errors.append({"error_type": error_type, "skill_name": name})
                    if required:
                        degraded = True
                    continue
                expected_checksum = str(profile.provider_settings.get(f"checksum.{name}") or "").strip()
                if expected_checksum and parsed.skill.metadata.ref.checksum != expected_checksum:
                    errors.append({"error_type": "skill_checksum_mismatch", "skill_name": name})
                    if required:
                        degraded = True
                    continue
                if parsed.error_type == "skill_capability_incompatible":
                    errors.append({"error_type": "skill_capability_incompatible", "skill_name": name})
                    degraded = True
                skills.append(parsed.skill.metadata)
        except SkillProviderTimeout:
            return SkillCatalogResult(
                enabled=False,
                provider=self.provider_id,
                tenant_id_hash=hash_tenant_id(self.tenant_id),
                profile_snapshot_id=profile.snapshot_id,
                skills=[],
                conflicts=conflicts,
                errors=errors + [{"error_type": "skill_provider_timeout"}],
                degraded=True,
                omission_reason="skill_provider_timeout",
            )
        except SkillAccessDenied as exc:
            return SkillCatalogResult(
                enabled=False,
                provider=self.provider_id,
                tenant_id_hash=hash_tenant_id(self.tenant_id),
                profile_snapshot_id=profile.snapshot_id,
                errors=[{"error_type": exc.error_type}],
                degraded=True,
                omission_reason=exc.error_type,
            )
        except OSError:
            return SkillCatalogResult(
                enabled=False,
                provider=self.provider_id,
                tenant_id_hash=hash_tenant_id(self.tenant_id),
                profile_snapshot_id=profile.snapshot_id,
                errors=[{"error_type": "skill_provider_unavailable"}],
                degraded=True,
                omission_reason="skill_provider_unavailable",
            )
        catalog_conflicts = _catalog_conflicts(skills, profile)
        if catalog_conflicts:
            conflicts.extend(catalog_conflicts)
            degraded = True
        return SkillCatalogResult(
            enabled=True,
            provider=self.provider_id,
            tenant_id_hash=hash_tenant_id(profile.tenant_id),
            profile_snapshot_id=profile.snapshot_id,
            skills=skills,
            conflicts=conflicts,
            errors=errors,
            degraded=degraded,
            omission_reason="skill_catalog_degraded" if degraded else None,
        )

    def load_skill(
        self,
        skill_ref: SkillRef,
        profile: CompanyProfileSnapshot,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> SkillLoadResult:
        identity_error = self._assert_profile(profile)
        if identity_error:
            return self._fail_load(identity_error)
        if skill_ref.tenant_id != self.tenant_id or skill_ref.source_id != self.source_id:
            return self._fail_load("skill_not_allowed")
        if not _NAME_RE.match(skill_ref.name) or not _VERSION_RE.match(skill_ref.version):
            return self._fail_load("skill_not_found")
        try:
            parsed = self._parse_named(skill_ref.name, skill_ref.version, deadline_monotonic=deadline_monotonic)
        except SkillProviderTimeout:
            return self._fail_load("skill_provider_timeout")
        if not parsed.loaded or parsed.skill is None:
            return parsed
        if parsed.skill.metadata.ref.checksum != skill_ref.checksum:
            return self._fail_load("skill_checksum_mismatch")
        expected_checksum = str(profile.provider_settings.get(f"checksum.{skill_ref.name}") or "").strip()
        if expected_checksum and parsed.skill.metadata.ref.checksum != expected_checksum:
            return self._fail_load("skill_checksum_mismatch")
        pin = str(profile.pins.get(skill_ref.name) or "").strip()
        if pin and pin != skill_ref.version:
            return self._fail_load("skill_version_mismatch")
        if parsed.skill.metadata.status != "active":
            return self._fail_load("skill_status_denied")
        return SkillLoadResult(
            loaded=True,
            skill=parsed.skill,
            resource_path=None,
            resource_checksum=None,
            error_type=parsed.error_type,
            omission_reason=parsed.omission_reason,
            redacted=parsed.redacted,
        )

    def load_resource(
        self,
        skill_ref: SkillRef,
        resource_path: str,
        profile: CompanyProfileSnapshot,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> SkillLoadResult:
        loaded = self.load_skill(skill_ref, profile, deadline_monotonic=deadline_monotonic)
        if not loaded.loaded or loaded.skill is None:
            return loaded
        declared = set(loaded.skill.metadata.resources)
        relative = str(resource_path or "").strip().replace("\\", "/")
        if (
            not relative
            or relative.startswith("/")
            or ".." in relative.split("/")
            or "\\" in str(resource_path or "")
            or "\x00" in relative
            or Path(relative).suffix.lower() not in ALLOWED_RESOURCE_SUFFIXES
        ):
            return self._fail_load("skill_resource_type_denied" if relative and Path(relative).suffix.lower() not in ALLOWED_RESOURCE_SUFFIXES else "skill_path_denied")
        if relative not in declared:
            return self._fail_load("skill_resource_not_declared")
        parts = tuple(relative.split("/"))
        try:
            _ensure_deadline(self.clock, deadline_monotonic)
            path = resolve_skill_path(self.root, skill_ref.name, skill_ref.version, *parts)
            text = path.read_text(encoding="utf-8")
            _ensure_deadline(self.clock, deadline_monotonic)
        except SkillProviderTimeout:
            return self._fail_load("skill_provider_timeout")
        except SkillAccessDenied as exc:
            return self._fail_load(exc.error_type)
        except UnicodeDecodeError:
            return self._fail_load("skill_resource_type_denied")
        except OSError:
            return self._fail_load("skill_path_denied")
        if len(text) > MAX_RESOURCE_CHARS:
            return self._fail_load("skill_content_too_large")
        cleaned, redacted = redact_text(text)
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        return SkillLoadResult(
            loaded=True,
            skill=SkillDocument(
                metadata=loaded.skill.metadata,
                body=cleaned,
                loaded_at=_utc_now(),
            ),
            resource_path=relative,
            resource_checksum=digest,
            error_type=None,
            omission_reason=None,
            redacted=redacted,
        )

    def health(
        self,
        profile: CompanyProfileSnapshot,
        *,
        deadline_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            _ensure_deadline(self.clock, deadline_monotonic)
            identity_error = self._assert_profile(profile)
            if identity_error:
                return {"healthy": False, "provider_id": self.provider_id, "error_type": identity_error}
            root_resolved = self.root.resolve(strict=True)
            if self.root.exists() and self.root.is_symlink():
                return {"healthy": False, "provider_id": self.provider_id, "error_type": "skill_symlink_denied"}
            if not root_resolved.is_dir():
                return {"healthy": False, "provider_id": self.provider_id, "error_type": "skill_provider_unavailable"}
            return {"healthy": True, "provider_id": self.provider_id, "error_type": None}
        except SkillProviderTimeout:
            return {"healthy": False, "provider_id": self.provider_id, "error_type": "skill_provider_timeout"}
        except OSError:
            return {"healthy": False, "provider_id": self.provider_id, "error_type": "skill_provider_unavailable"}


def _catalog_conflicts(skills: Iterable[SkillMetadata], profile: CompanyProfileSnapshot) -> List[Dict[str, Any]]:
    by_name = {item.ref.name: item for item in skills}
    out: List[Dict[str, Any]] = []
    seen = set()
    for meta in skills:
        for other in meta.conflicts_with:
            if other not in by_name:
                continue
            pair = tuple(sorted((meta.ref.name, other)))
            if pair in seen:
                continue
            seen.add(pair)
            winner = _priority_winner(profile.priority, meta.ref.name, other)
            if winner is None:
                out.append({"error_type": "skill_conflict", "skills": list(pair)})
    return out


def _priority_winner(priority: Tuple[str, ...], left: str, right: str) -> Optional[str]:
    for name in priority:
        if name == left:
            return left
        if name == right:
            return right
    return None
