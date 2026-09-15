# -*- coding: utf-8 -*-
"""Company Profile 解析与 frozen snapshot。config=None 时不访问文件系统。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from .redaction import contains_secret_blob, redact_text
from .skill_models import (
    CompanyProfileResolveResult,
    CompanyProfileSnapshot,
    empty_company_profile_result,
)

PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_OPTIONAL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,127}$")
_ROOT_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_ALLOWED_STATUS = frozenset({"active"})
_ALLOWED_ROOT_KIND = frozenset({"builtin", "env"})
DEFAULT_PROFILES_ROOT = Path.home() / ".agent-workbench" / "profiles"


def _unique_keep_order(values: Any) -> Tuple[str, ...]:
    seen = set()
    out: List[str] = []
    if not isinstance(values, (list, tuple)):
        return tuple()
    for item in values:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def normalize_company_profile_config(
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not config or not isinstance(config, dict) or config.get("enabled") is not True:
        return {
            "enabled": False,
            "profile_id": None,
            "tenant_id": None,
            "profiles_root": None,
            "trusted_tenant_id": None,
        }
    profile_id = str(config.get("profile_id") or "").strip()
    tenant_id = str(config.get("tenant_id") or "").strip()
    trusted = str(config.get("trusted_tenant_id") or tenant_id).strip()
    profiles_root = str(config.get("profiles_root") or "").strip() or None
    return {
        "enabled": True,
        "profile_id": profile_id,
        "tenant_id": tenant_id,
        "profiles_root": profiles_root,
        "trusted_tenant_id": trusted,
    }


def _fail(error_type: str, *, degraded: bool = True) -> CompanyProfileResolveResult:
    return empty_company_profile_result(error_type, error_type=error_type, degraded=degraded)


def _locate_profile_file(profiles_root: Optional[str], profile_id: str) -> Tuple[Optional[Path], Optional[str]]:
    if not PROFILE_ID_RE.match(profile_id):
        return None, "company_profile_invalid"
    if profiles_root:
        root = Path(profiles_root).expanduser()
    else:
        home = os.environ.get("AGENT_WORKBENCH_HOME")
        root = Path(home).expanduser() / "profiles" if home else DEFAULT_PROFILES_ROOT
    try:
        root_resolved = root.resolve()
    except OSError:
        return None, "company_profile_path_denied"
    candidate = root / f"{profile_id}.yaml"
    try:
        resolved = candidate.resolve()
    except OSError:
        return None, "company_profile_path_denied"
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        return None, "company_profile_path_denied"
    if not resolved.is_file():
        return None, "company_profile_not_found"
    return resolved, None


def _read_yaml_mapping(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, "company_profile_not_found"
    except UnicodeDecodeError:
        return None, "company_profile_invalid"
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError:
        return None, "company_profile_invalid"
    if not isinstance(loaded, dict):
        return None, "company_profile_invalid"
    return loaded, None


def resolve_company_profile(
    config: Optional[Dict[str, Any]],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> CompanyProfileResolveResult:
    del env  # Profile 文件不读环境变量；root_env 只保存在 snapshot settings 中。
    normalized = normalize_company_profile_config(config)
    if not normalized["enabled"]:
        return empty_company_profile_result("company_profile_disabled", error_type=None, degraded=False)
    profile_id = str(normalized.get("profile_id") or "")
    tenant_id = str(normalized.get("tenant_id") or "")
    trusted_tenant_id = str(normalized.get("trusted_tenant_id") or "")
    if not PROFILE_ID_RE.match(profile_id) or not PROFILE_ID_RE.match(tenant_id) or not PROFILE_ID_RE.match(trusted_tenant_id):
        return _fail("company_profile_invalid")
    if tenant_id != trusted_tenant_id:
        return _fail("company_profile_tenant_mismatch")
    path, locate_error = _locate_profile_file(normalized.get("profiles_root"), profile_id)
    if locate_error:
        return _fail(locate_error)
    assert path is not None
    raw_profile, read_error = _read_yaml_mapping(path)
    if read_error:
        return _fail(read_error)
    assert raw_profile is not None
    try:
        snapshot = build_company_profile_snapshot(raw_profile, trusted_tenant_id=trusted_tenant_id)
    except ValueError as exc:
        error_type = str(exc) if str(exc) in {
            "company_profile_invalid",
            "company_profile_tenant_mismatch",
        } else "company_profile_invalid"
        return _fail(error_type)
    if snapshot.profile_id != profile_id:
        return _fail("company_profile_invalid")
    return CompanyProfileResolveResult(
        enabled=True,
        snapshot=snapshot,
        error_type=None,
        degraded=False,
        omission_reason=None,
    )


def _snapshot_id(payload: Dict[str, Any]) -> str:
    seed = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
    return f"profile_{digest}"


def _optional_ref(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if not _OPTIONAL_ID_RE.match(text) or contains_secret_blob(text):
        raise ValueError("company_profile_invalid")
    return text


def build_company_profile_snapshot(
    raw_profile: Dict[str, Any],
    *,
    trusted_tenant_id: str,
) -> CompanyProfileSnapshot:
    if not isinstance(raw_profile, dict):
        raise ValueError("company_profile_invalid")
    profile_id = str(raw_profile.get("profile_id") or "").strip()
    tenant_id = str(raw_profile.get("tenant_id") or "").strip()
    if not PROFILE_ID_RE.match(profile_id) or not PROFILE_ID_RE.match(tenant_id):
        raise ValueError("company_profile_invalid")
    if tenant_id != trusted_tenant_id:
        raise ValueError("company_profile_tenant_mismatch")
    version = str(raw_profile.get("version") or "").strip()
    if not _VERSION_RE.match(version):
        raise ValueError("company_profile_invalid")
    status = str(raw_profile.get("status") or "").strip() or "active"
    if status not in _ALLOWED_STATUS:
        raise ValueError("company_profile_invalid")
    display_raw = str(raw_profile.get("display_name") or profile_id)
    if contains_secret_blob(display_raw):
        raise ValueError("company_profile_invalid")
    display_name, _ = redact_text(display_raw)
    display_name = display_name.strip()[:120] or profile_id
    provider_block = raw_profile.get("skill_provider") if isinstance(raw_profile.get("skill_provider"), dict) else {}
    provider_id = str(provider_block.get("provider") or "").strip()
    source_id = str(provider_block.get("source_id") or "").strip()
    root_kind = str(provider_block.get("root_kind") or "").strip()
    root_env = str(provider_block.get("root_env") or "").strip()
    if not PROFILE_ID_RE.match(provider_id) or not PROFILE_ID_RE.match(source_id):
        raise ValueError("company_profile_invalid")
    if root_kind not in _ALLOWED_ROOT_KIND:
        raise ValueError("company_profile_invalid")
    if root_kind == "builtin" and root_env:
        raise ValueError("company_profile_invalid")
    provider_settings: Dict[str, str] = {"root_kind": root_kind}
    if root_kind == "env":
        if not _ROOT_ENV_RE.match(root_env):
            raise ValueError("company_profile_invalid")
        provider_settings["root_env"] = root_env
    skills_block = raw_profile.get("skills") if isinstance(raw_profile.get("skills"), dict) else {}
    required_skills = _unique_keep_order(skills_block.get("required"))
    available_skills = _unique_keep_order(skills_block.get("available"))
    disabled_skills = _unique_keep_order(skills_block.get("disabled"))
    if (set(required_skills) | set(available_skills)) & set(disabled_skills):
        raise ValueError("company_profile_invalid")
    for name in required_skills + available_skills + disabled_skills:
        if not PROFILE_ID_RE.match(name):
            raise ValueError("company_profile_invalid")
    pins_raw = skills_block.get("pins") if isinstance(skills_block.get("pins"), dict) else {}
    pins: Dict[str, str] = {}
    for key, value in pins_raw.items():
        name = str(key or "").strip()
        version_pin = str(value or "").strip()
        if not PROFILE_ID_RE.match(name) or not _VERSION_RE.match(version_pin):
            raise ValueError("company_profile_invalid")
        pins[name] = version_pin
    allowed = set(required_skills) | set(available_skills)
    priority_all = _unique_keep_order(skills_block.get("priority"))
    priority = tuple(name for name in priority_all if name in allowed)
    checksums = skills_block.get("checksums") if isinstance(skills_block.get("checksums"), dict) else {}
    for key, value in checksums.items():
        name = str(key or "").strip()
        digest = str(value or "").strip().lower()
        if not PROFILE_ID_RE.match(name) or not re.match(r"^[a-f0-9]{32,64}$", digest):
            raise ValueError("company_profile_invalid")
        provider_settings[f"checksum.{name}"] = digest
    from .artifact_policy import validate_company_artifact_policies
    canonical = {
        "artifact_policies": validate_company_artifact_policies(raw_profile.get("artifact_policies")),
        "available_skills": list(available_skills),
        "disabled_skills": list(disabled_skills),
        "display_name": display_name,
        "knowledge_profile": _optional_ref(raw_profile.get("knowledge_profile")),
        "memory_namespace": _optional_ref(raw_profile.get("memory_namespace")),
        "pins": dict(sorted(pins.items())),
        "policy_ref": _optional_ref(raw_profile.get("policy_ref")),
        "priority": list(priority),
        "profile_id": profile_id,
        "provider_id": provider_id,
        "provider_settings": dict(sorted(provider_settings.items())),
        "required_skills": list(required_skills),
        "source_id": source_id,
        "status": status,
        "tenant_id": tenant_id,
        "version": version,
    }
    return CompanyProfileSnapshot(
        profile_id=profile_id,
        tenant_id=tenant_id,
        display_name=display_name,
        version=version,
        status=status,
        provider_id=provider_id,
        source_id=source_id,
        provider_settings=dict(canonical["provider_settings"]),
        required_skills=required_skills,
        available_skills=available_skills,
        disabled_skills=disabled_skills,
        pins=dict(pins),
        priority=priority,
        policy_ref=canonical["policy_ref"],
        knowledge_profile=canonical["knowledge_profile"],
        memory_namespace=canonical["memory_namespace"],
        snapshot_id=_snapshot_id(canonical),
        artifact_policies=canonical["artifact_policies"],
    )
