# -*- coding: utf-8 -*-
"""Skill 生命周期治理：审批、promotion、rollback、改进候选与冻结快照重放。"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .redaction import contains_secret_blob
from .skill_models import (
    CompanyProfileResolveResult,
    company_profile_snapshot_from_dict,
    skill_ref_from_dict,
)
from .skill_store import (
    SkillGovernanceStoreError,
    SkillVersionStore,
    hash_governance_identity,
    validate_skill_identity_fields,
)


def empty_skill_governance(reason: str = "company_profile_disabled") -> Dict[str, Any]:
    return {
        "enabled": False,
        "error_type": None,
        "legacy": True,
        "managed": False,
        "active": [],
        "last_known_good": [],
        "omission_reason": reason,
    }


def empty_skill_replay_manifest(reason: str = "not_replayed") -> Dict[str, Any]:
    return {
        "replayed": False,
        "error_type": None,
        "profile_snapshot_id": None,
        "skills": [],
        "omission_reason": reason,
    }


def _required_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_strict_true(value: Any) -> bool:
    return value is True


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _eval_summary_fails_promotion_gate(summary: Dict[str, Any]) -> bool:
    required = (
        "promotion_gate_passed",
        "eval_valid",
        "case_count",
        "cross_company_leak_rate",
        "policy_violation_rate",
        "outcome_score_delta",
    )
    if any(key not in summary for key in required):
        return True
    if not _is_strict_true(summary.get("promotion_gate_passed")):
        return True
    if not _is_strict_true(summary.get("eval_valid")):
        return True
    if not _is_positive_int(summary.get("case_count")):
        return True
    leak_rate = _required_number(summary.get("cross_company_leak_rate"))
    violation_rate = _required_number(summary.get("policy_violation_rate"))
    outcome_delta = _required_number(summary.get("outcome_score_delta"))
    if leak_rate is None or violation_rate is None or outcome_delta is None:
        return True
    return leak_rate != 0.0 or violation_rate != 0.0 or outcome_delta < 0.0


def _loaded_skill_refs(state: Dict[str, Any], report: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    report = report or {}
    raw = state.get("loaded_skills")
    if raw is None:
        raw = report.get("loaded_skills") or []
    refs: List[Dict[str, Any]] = []
    seen = set()
    for item in raw or []:
        payload = item if isinstance(item, dict) else {}
        ref = payload.get("ref")
        if not isinstance(ref, dict):
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            ref = metadata.get("ref") if isinstance(metadata.get("ref"), dict) else payload
        if not isinstance(ref, dict):
            continue
        name = str(ref.get("name") or "").strip()
        version = str(ref.get("version") or "").strip()
        checksum = str(ref.get("checksum") or "").strip()
        stable_id = str(ref.get("stable_id") or payload.get("stable_id") or "").strip()
        key = (name, version, checksum, stable_id)
        if not name or key in seen:
            continue
        seen.add(key)
        refs.append(
            {
                "name": name,
                "version": version,
                "checksum": checksum,
                "stable_id": stable_id,
                "source_id": str(ref.get("source_id") or "").strip(),
                "tenant_id": str(ref.get("tenant_id") or "").strip(),
            }
        )
    return refs


def validate_skill_promotion(
    candidate: Dict[str, Any],
    *,
    eval_summary: Dict[str, Any],
    approver_id: str,
) -> Dict[str, Any]:
    record = candidate if isinstance(candidate, dict) else {}
    summary = eval_summary if isinstance(eval_summary, dict) else {}
    approver = str(approver_id or "").strip()
    if not record:
        return {"eligible": False, "error_type": "skill_not_found"}
    if not approver:
        return {"eligible": False, "error_type": "skill_self_approval_denied"}
    if str(record.get("lifecycle") or "") != "eval_passed":
        return {"eligible": False, "error_type": "skill_lifecycle_transition_denied"}
    identity_error = validate_skill_identity_fields(
        skill_id=str(record.get("skill_id") or ""),
        skill_name=str(record.get("skill_name") or record.get("name") or ""),
        version=str(record.get("version") or ""),
        checksum=str(record.get("checksum") or ""),
    )
    if identity_error:
        return {"eligible": False, "error_type": identity_error}
    author_hash = str(record.get("author_hash") or "").strip()
    if not author_hash:
        return {"eligible": False, "error_type": "skill_self_approval_denied"}
    approver_hash = hash_governance_identity("actor", approver)
    if author_hash == approver_hash:
        return {"eligible": False, "error_type": "skill_self_approval_denied"}
    if _eval_summary_fails_promotion_gate(summary):
        return {"eligible": False, "error_type": "skill_promotion_eval_failed"}
    blob_parts = [
        str(record.get("skill_id") or ""),
        str(record.get("version") or ""),
        str(record.get("checksum") or ""),
        str(record.get("skill_name") or ""),
        approver,
    ]
    if any(contains_secret_blob(part) for part in blob_parts if part):
        return {"eligible": False, "error_type": "skill_secret_detected"}
    return {"eligible": True, "error_type": None}


def promote_skill_version(
    store: SkillVersionStore,
    *,
    skill_id: str,
    version: str,
    eval_summary: Dict[str, Any],
    approver_id: str,
) -> Dict[str, Any]:
    try:
        candidate = store.get_version(str(skill_id or "").strip(), str(version or "").strip())
    except SkillGovernanceStoreError:
        return {"record": None, "error_type": "skill_governance_unavailable"}
    gate = validate_skill_promotion(candidate or {}, eval_summary=eval_summary, approver_id=approver_id)
    if gate.get("eligible") is not True:
        return {"record": None, "error_type": gate.get("error_type") or "skill_promotion_eval_failed"}
    promoter = getattr(store, "promote_version_atomic", None)
    if not callable(promoter):
        return {"record": None, "error_type": "skill_governance_unavailable"}
    try:
        return promoter(
            skill_id=str(skill_id or "").strip(),
            version=str(version or "").strip(),
            actor_id=str(approver_id or "").strip(),
            eval_summary=eval_summary,
        )
    except SkillGovernanceStoreError:
        return {"record": None, "error_type": "skill_governance_unavailable"}


def rollback_skill_version(
    store: SkillVersionStore,
    *,
    skill_name: str,
    tenant_id: str,
    actor_id: str,
) -> Dict[str, Any]:
    if not str(actor_id or "").strip():
        return {"record": None, "error_type": "skill_self_approval_denied"}
    if not str(skill_name or "").strip() or not str(tenant_id or "").strip():
        return {"record": None, "error_type": "skill_no_last_known_good"}
    rollbacker = getattr(store, "rollback_version_atomic", None)
    if not callable(rollbacker):
        return {"record": None, "error_type": "skill_governance_unavailable"}
    try:
        return rollbacker(
            skill_name=str(skill_name or "").strip(),
            tenant_id=str(tenant_id or "").strip(),
            actor_id=str(actor_id or "").strip(),
        )
    except SkillGovernanceStoreError:
        return {"record": None, "error_type": "skill_governance_unavailable"}


def build_skill_improvement_candidate(
    state: Dict[str, Any],
    report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report = report or {}
    refs = _loaded_skill_refs(state, report)
    if not refs:
        return {"candidate": None, "error_type": None}
    primary = refs[0]
    profile = state.get("company_profile") or report.get("company_profile") or {}
    snapshot = profile.get("snapshot") if isinstance(profile, dict) else None
    tenant_id = ""
    if isinstance(snapshot, dict):
        tenant_id = str(snapshot.get("tenant_id") or "").strip()
    if not tenant_id:
        tenant_id = str(primary.get("tenant_id") or "").strip()
    trigger_reasons = []
    memory_review = state.get("memory_review") or report.get("memory_review") or {}
    if isinstance(memory_review, dict):
        trigger_reasons = list(memory_review.get("trigger_reasons") or [])
    if not trigger_reasons:
        for key in ("user_correction", "tool_failure", "workflow_repeat"):
            if state.get(key) or report.get(key):
                trigger_reasons.append(key)
    failed = list(state.get("failed_steps") or report.get("failed_steps") or [])
    if failed and "tool_failure" not in trigger_reasons:
        trigger_reasons.append("tool_failure")
    reason = "skill_improvement_from_review"
    if "user_correction" in trigger_reasons:
        reason = "user_correction"
    elif "tool_failure" in trigger_reasons:
        reason = "tool_failure"
    elif "workflow_repeat" in trigger_reasons:
        reason = "workflow_repeat"
    candidate = {
        "tenant_id": tenant_id,
        "skill_name": primary.get("name"),
        "skill_id": primary.get("stable_id"),
        "version": primary.get("version"),
        "checksum": primary.get("checksum"),
        "reason": reason,
        "trigger_reasons": trigger_reasons,
        "run_id": str(state.get("run_id") or report.get("run_id") or ""),
        "loaded_skill_count": len(refs),
        "checksum_digest": str(primary.get("checksum") or "")[:12],
    }
    if contains_secret_blob(str(candidate.get("checksum") or "")):
        return {"candidate": None, "error_type": "skill_secret_detected"}
    return {"candidate": candidate, "error_type": None}


def resolve_skill_governance_snapshot(
    store: SkillVersionStore,
    refs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    snapshot = {
        "enabled": True,
        "error_type": None,
        "legacy": True,
        "managed": False,
        "active": [],
        "last_known_good": [],
        "records": [],
        "omission_reason": None,
    }
    try:
        seen_names = set()
        for ref in refs or []:
            if not isinstance(ref, dict):
                continue
            name = str(ref.get("name") or "").strip()
            tenant_id = str(ref.get("tenant_id") or "").strip()
            if not name or not tenant_id or name in seen_names:
                continue
            seen_names.add(name)
            versions = store.list_versions(name, tenant_id)
            if not versions:
                continue
            snapshot["legacy"] = False
            snapshot["managed"] = True
            snapshot["records"].extend(versions)
            active = store.resolve_active(name, tenant_id)
            if active:
                snapshot["active"].append(active)
            lkg = store.resolve_last_known_good(name, tenant_id)
            if lkg:
                snapshot["last_known_good"].append(lkg)
        return snapshot
    except SkillGovernanceStoreError:
        return {
            "enabled": False,
            "error_type": "skill_governance_unavailable",
            "legacy": True,
            "managed": False,
            "active": [],
            "last_known_good": [],
            "records": [],
            "omission_reason": "skill_governance_unavailable",
        }


def _replay_mismatch(
    *,
    profile_snapshot_id: Optional[str],
    reason: str,
    skills: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "replayed": False,
        "error_type": "skill_replay_snapshot_mismatch",
        "profile_snapshot_id": profile_snapshot_id,
        "skills": list(skills or []),
        "omission_reason": reason,
    }


def replay_skill_snapshot(
    run_bundle: Dict[str, Any],
    *,
    company_profile_config: Optional[Dict[str, Any]],
    registry: Optional[Any] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    from .company_profile import resolve_company_profile
    from .skill_registry import build_skill_provider

    bundle = run_bundle if isinstance(run_bundle, dict) else {}
    state = bundle.get("state") if isinstance(bundle.get("state"), dict) else {}
    report = bundle.get("report") if isinstance(bundle.get("report"), dict) else {}
    if not state and isinstance(bundle.get("report_bundle"), dict):
        state = bundle["report_bundle"].get("state") or {}
        report = bundle["report_bundle"].get("report") or report
    frozen_profile_id = (
        bundle.get("company_profile_snapshot_id")
        or report.get("company_profile_snapshot_id")
        or state.get("company_profile_snapshot_id")
    )
    profile_obj = (
        bundle.get("company_profile")
        or report.get("company_profile")
        or state.get("company_profile")
        or {}
    )
    frozen_dict = profile_obj.get("snapshot") if isinstance(profile_obj, dict) else None
    frozen = company_profile_snapshot_from_dict(frozen_dict if isinstance(frozen_dict, dict) else None)
    if frozen is None:
        return _replay_mismatch(profile_snapshot_id=frozen_profile_id, reason="missing_frozen_profile")
    if not frozen_profile_id or frozen.snapshot_id != str(frozen_profile_id):
        return _replay_mismatch(profile_snapshot_id=frozen_profile_id, reason="profile_snapshot_mismatch")

    current: CompanyProfileResolveResult = resolve_company_profile(company_profile_config, env=env)
    current_snapshot = current.snapshot
    if not current.enabled or current_snapshot is None:
        return _replay_mismatch(
            profile_snapshot_id=frozen.snapshot_id,
            reason=current.error_type or current.omission_reason or "company_profile_disabled",
        )
    trusted_tenant = str((company_profile_config or {}).get("trusted_tenant_id") or current_snapshot.tenant_id).strip()
    identity_ok = (
        current_snapshot.profile_id == frozen.profile_id
        and current_snapshot.tenant_id == frozen.tenant_id
        and trusted_tenant == frozen.tenant_id
        and current_snapshot.provider_id == frozen.provider_id
        and current_snapshot.source_id == frozen.source_id
    )
    if not identity_ok:
        return _replay_mismatch(profile_snapshot_id=frozen.snapshot_id, reason="profile_identity_mismatch")

    refs = _loaded_skill_refs(state, report)
    if not refs:
        catalog = state.get("skill_catalog") or report.get("skill_catalog") or {}
        for item in catalog.get("skills") or []:
            ref = item.get("ref") if isinstance(item, dict) else None
            if isinstance(ref, dict):
                refs.append(
                    {
                        "name": ref.get("name"),
                        "version": ref.get("version"),
                        "checksum": ref.get("checksum"),
                        "stable_id": ref.get("stable_id"),
                        "source_id": ref.get("source_id"),
                        "tenant_id": ref.get("tenant_id"),
                    }
                )
    provider, provider_error = build_skill_provider(frozen, env=env, registry=registry)
    if provider is None:
        return _replay_mismatch(
            profile_snapshot_id=frozen.snapshot_id,
            reason=provider_error or "skill_provider_unavailable",
        )
    replayed: List[Dict[str, Any]] = []
    for raw_ref in refs:
        skill_ref = skill_ref_from_dict(raw_ref)
        if skill_ref is None:
            return _replay_mismatch(
                profile_snapshot_id=frozen.snapshot_id,
                reason="skill_ref_invalid",
                skills=replayed,
            )
        loaded = provider.load_skill(skill_ref, frozen)
        matched = bool(
            loaded.loaded
            and loaded.skill is not None
            and loaded.skill.metadata.ref.checksum == skill_ref.checksum
            and loaded.skill.metadata.ref.version == skill_ref.version
            and loaded.skill.metadata.ref.stable_id() == skill_ref.stable_id()
        )
        replayed.append(
            {
                "name": skill_ref.name,
                "version": skill_ref.version,
                "checksum": skill_ref.checksum,
                "checksum_digest": skill_ref.checksum[:12],
                "stable_id": skill_ref.stable_id(),
                "matched": matched,
                "error_type": None if matched else (loaded.error_type or "skill_replay_snapshot_mismatch"),
            }
        )
        if not matched:
            return _replay_mismatch(
                profile_snapshot_id=frozen.snapshot_id,
                reason=loaded.error_type or "skill_replay_snapshot_mismatch",
                skills=replayed,
            )
    return {
        "replayed": True,
        "error_type": None,
        "profile_snapshot_id": frozen.snapshot_id,
        "skills": replayed,
        "omission_reason": None,
    }


def apply_governance_to_catalog(
    catalog_skills: Sequence[Any],
    governance: Dict[str, Any],
) -> Dict[str, Any]:
    """按治理快照过滤已管理 Skill。无记录保持原列表。"""
    errors: List[Dict[str, Any]] = []
    if governance.get("error_type") == "skill_governance_unavailable":
        return {
            "skills": [],
            "errors": [{"error_type": "skill_governance_unavailable"}],
            "filtered": True,
            "error_type": "skill_governance_unavailable",
        }
    records = list(governance.get("records") or [])
    if not records:
        return {"skills": list(catalog_skills), "errors": errors, "filtered": False}
    active_by_name = {
        str(item.get("skill_name") or ""): item
        for item in governance.get("active") or []
        if isinstance(item, dict)
    }
    kept = []
    for skill in catalog_skills:
        ref = getattr(skill, "ref", None)
        if ref is None and isinstance(skill, dict):
            ref_dict = skill.get("ref") if isinstance(skill.get("ref"), dict) else skill
            name = str(ref_dict.get("name") or "")
            version = str(ref_dict.get("version") or "")
            checksum = str(ref_dict.get("checksum") or "")
        else:
            name = str(getattr(ref, "name", "") or "")
            version = str(getattr(ref, "version", "") or "")
            checksum = str(getattr(ref, "checksum", "") or "")
        managed = any(str(item.get("skill_name") or "") == name for item in records)
        if not managed:
            kept.append(skill)
            continue
        active = active_by_name.get(name)
        if (
            active
            and str(active.get("version") or "") == version
            and str(active.get("checksum") or "") == checksum
            and str(active.get("lifecycle") or "") == "active"
        ):
            kept.append(skill)
        else:
            errors.append(
                {
                    "error_type": "skill_status_denied" if not active else "skill_checksum_mismatch",
                    "skill_name": name,
                }
            )
    return {"skills": kept, "errors": errors, "filtered": True}
