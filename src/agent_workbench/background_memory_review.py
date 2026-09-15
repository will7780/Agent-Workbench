# -*- coding: utf-8 -*-
"""Phase 3 — Background Memory Review：run 结束后生成 memory candidate（不自动 approve）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .memory_retriever import infer_workflow_keys_from_request
from .memory_store import MemoryStore, build_candidate_from_business_review
from .redaction import contains_secret_blob, redact_text

USER_CORRECTION_PATTERNS = (
    r"不是",
    r"不要用",
    r"请用",
    r"应该用",
    r"纠正",
    r"错了",
    r"不对",
    r"别用",
    r"改用",
)

PROFILE_HINT_PATTERNS = (
    r"简洁",
    r"详细",
    r"中文",
    r"英文",
    r"不要废话",
    r"沟通",
    r"偏好",
    r"习惯",
)


@dataclass
class MemoryReviewPolicy:
    """后台 review 策略（第一版同步执行，可配置触发频率）。"""

    enabled: bool = False
    every_n_turns: int = 0  # 0 = 不按周期触发
    min_workflow_repeat_count: int = 2
    trigger_on_business_review: bool = True
    trigger_on_tool_failure: bool = True
    trigger_on_guard_block: bool = True
    trigger_on_answer_verify_fail: bool = True
    trigger_on_user_correction: bool = True
    trigger_on_workflow_repeat: bool = True


def empty_memory_review(error_type: Optional[str] = None) -> Dict[str, Any]:
    return {
        "enabled": False,
        "triggered": False,
        "candidate_ids": [],
        "skill_improvement_candidate_ids": [],
        "skipped_reason": error_type or "disabled",
        "errors": [],
        "evidence_refs": [],
        "trigger_reasons": [],
        "error_type": error_type,
    }


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_text(text: Any, limit: int = 500) -> str:
    cleaned, _ = redact_text(str(text or ""))
    return _truncate(cleaned, limit)


def _workflow_scope_key(user_request: str, plan: Optional[Dict[str, Any]] = None) -> str:
    keys = infer_workflow_keys_from_request(user_request)
    if plan and plan.get("plan_type"):
        keys.add(str(plan.get("plan_type")))
    if keys:
        return sorted(keys)[0]
    return "general"


def _store_scope_from_config(module_config: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    for cfg in (module_config or {}).values():
        if not isinstance(cfg, dict):
            continue
        for sk in ("store_id", "shop_id", "data_source", "store_key"):
            val = cfg.get(sk)
            if val not in (None, ""):
                return "store", str(val)
    return "global", ""


def _evidence_ref(
    ref_type: str,
    *,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    detail: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"ref_type": ref_type}
    if run_id:
        entry["run_id"] = run_id
    if step_id:
        entry["step_id"] = step_id
    if detail:
        entry["detail"] = _safe_text(detail, 300)
    if extra:
        for k, v in extra.items():
            if v is not None:
                entry[k] = v
    return entry


def _has_tool_failures(state: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidences: List[Dict[str, Any]] = []
    run_id = str(state.get("run_id") or report.get("run_id") or "")
    failed_steps = list(state.get("failed_steps") or report.get("failed_steps") or [])
    for step in failed_steps:
        evidences.append(
            _evidence_ref(
                "tool_failure",
                run_id=run_id,
                step_id=str(step.get("step_id") or ""),
                detail=f"{step.get('module')}.{step.get('action')}: {step.get('error') or step.get('message') or 'failed'}",
            )
        )
    observations = list(state.get("observations") or report.get("observations") or [])
    for obs in observations:
        status = obs.get("status")
        if status in (None, "success", "dry_run"):
            continue
        evidences.append(
            _evidence_ref(
                "tool_failure",
                run_id=run_id,
                step_id=str(obs.get("step_id") or ""),
                detail=f"observation_status={status}; {obs.get('error_type') or obs.get('summary') or ''}",
            )
        )
    return evidences


def _has_guard_blocks(state: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidences: List[Dict[str, Any]] = []
    run_id = str(state.get("run_id") or report.get("run_id") or "")
    tool_calls = list(state.get("tool_calls") or report.get("tool_calls") or [])
    for tc in tool_calls:
        guard = tc.get("guard") or {}
        human = tc.get("human_confirm") or {}
        if guard.get("allowed") is False or human.get("allowed") is False:
            evidences.append(
                _evidence_ref(
                    "guard_block",
                    run_id=run_id,
                    step_id=str(tc.get("step_id") or tc.get("tool_call_id") or ""),
                    detail=f"tool={tc.get('tool_name')}; guard={guard.get('error_type') or guard.get('reason')}",
                    extra={"tool_name": tc.get("tool_name")},
                )
            )
    return evidences


def _answer_verify_issues(state: Dict[str, Any], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    verification = state.get("verification") or report.get("verification") or {}
    if verification.get("passed"):
        return []
    run_id = str(state.get("run_id") or report.get("run_id") or "")
    issues = list(verification.get("issues") or [])
    detail = "; ".join(str(i) for i in issues[:5]) or "answer_verification_failed"
    return [
        _evidence_ref(
            "answer_verify",
            run_id=run_id,
            detail=detail,
            extra={"issue_count": len(issues)},
        )
    ]


def _user_correction_signals(user_request: str) -> bool:
    text = user_request or ""
    for pat in USER_CORRECTION_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def _profile_style_signals(user_request: str) -> bool:
    text = user_request or ""
    for pat in PROFILE_HINT_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def _count_thread_turns(
    thread_id: Optional[str],
    ledger_context: Optional[Dict[str, Any]] = None,
) -> int:
    if ledger_context and ledger_context.get("thread_turn_count") is not None:
        return int(ledger_context["thread_turn_count"])
    if not thread_id:
        return 0
    try:
        ledger = (ledger_context or {}).get("ledger")
        if ledger is None or not ledger.healthy:
            return 0
        with ledger._lock:
            with ledger._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS c FROM sessions WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
                return int(row["c"] if row else 0)
    except Exception:
        return 0


def _count_workflow_repeats(
    thread_id: Optional[str],
    workflow_key: str,
    ledger_context: Optional[Dict[str, Any]] = None,
) -> int:
    if ledger_context and ledger_context.get("workflow_repeat_count") is not None:
        return int(ledger_context["workflow_repeat_count"])
    if not thread_id or workflow_key == "general":
        return 0
    try:
        ledger = (ledger_context or {}).get("ledger")
        if ledger is None or not ledger.healthy:
            return 0
        hits = ledger.search(workflow_key, thread_id=thread_id, limit=50)
        return len(hits)
    except Exception:
        return 0


def _collect_trigger_reasons(
    state: Dict[str, Any],
    report: Dict[str, Any],
    policy: MemoryReviewPolicy,
    ledger_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    evidence: List[Dict[str, Any]] = []

    business_review = state.get("business_review") or report.get("business_review") or {}
    if policy.trigger_on_business_review and business_review.get("status"):
        reasons.append("business_review")
        evidence.append(
            _evidence_ref(
                "business_review",
                run_id=str(state.get("run_id") or report.get("run_id") or ""),
                detail=f"status={business_review.get('status')}; reason={business_review.get('reason_code')}",
                extra={
                    "reason_code": business_review.get("reason_code"),
                    "status": business_review.get("status"),
                },
            )
        )

    tool_failures = _has_tool_failures(state, report)
    if policy.trigger_on_tool_failure and tool_failures:
        reasons.append("tool_failure")
        evidence.extend(tool_failures)

    guard_blocks = _has_guard_blocks(state, report)
    if policy.trigger_on_guard_block and guard_blocks:
        reasons.append("guard_block")
        evidence.extend(guard_blocks)

    verify_issues = _answer_verify_issues(state, report)
    if policy.trigger_on_answer_verify_fail and verify_issues:
        reasons.append("answer_verify_fail")
        evidence.extend(verify_issues)

    user_request = str(state.get("user_request") or report.get("goal") or "")
    if policy.trigger_on_user_correction and _user_correction_signals(user_request):
        reasons.append("user_correction")
        evidence.append(
            _evidence_ref(
                "user_correction",
                run_id=str(state.get("run_id") or report.get("run_id") or ""),
                detail=_safe_text(user_request, 200),
            )
        )

    workflow_key = _workflow_scope_key(user_request, state.get("plan") or report.get("plan"))
    thread_id = state.get("thread_id") or report.get("thread_id")
    repeat_count = _count_workflow_repeats(thread_id, workflow_key, ledger_context)
    if policy.trigger_on_workflow_repeat and repeat_count >= policy.min_workflow_repeat_count:
        reasons.append("workflow_repeat")
        evidence.append(
            _evidence_ref(
                "workflow_repeat",
                run_id=str(state.get("run_id") or report.get("run_id") or ""),
                detail=f"workflow={workflow_key}; repeat_count={repeat_count}",
                extra={"workflow_key": workflow_key, "repeat_count": repeat_count},
            )
        )

    turn_count = _count_thread_turns(thread_id, ledger_context)
    if policy.every_n_turns > 0 and turn_count > 0 and turn_count % policy.every_n_turns == 0:
        reasons.append("periodic_turn")
        evidence.append(
            _evidence_ref(
                "periodic_turn",
                run_id=str(state.get("run_id") or report.get("run_id") or ""),
                detail=f"turn_count={turn_count}; every_n={policy.every_n_turns}",
                extra={"turn_count": turn_count},
            )
        )

    return reasons, evidence


def _map_business_review_to_candidates(
    run_bundle: Dict[str, Any],
    evidence_refs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    review = run_bundle.get("business_review") or {}
    if not review.get("status"):
        return []
    reason_code = str(review.get("reason_code") or "other")
    base_fields = build_candidate_from_business_review(run_bundle)
    candidates: List[Dict[str, Any]] = []

    if reason_code in ("wrong_tool", "wrong_params"):
        fields = dict(base_fields)
        fields["memory_type"] = "tool_experience"
        fields["title"] = _truncate(f"工具经验: {reason_code}", 200)
        fields["why"] = _safe_text(
            f"用户业务验收指出工具/参数问题 ({reason_code}); evidence={len(evidence_refs)} refs",
            500,
        )
        fields["how_to_apply"] = "同类请求优先参考验收反馈中的工具与参数选择。"
        candidates.append(fields)
    elif reason_code in ("missing_steps", "extra_steps", "wrong_order", "as_expected"):
        fields = dict(base_fields)
        fields["memory_type"] = "workflow_preference"
        fields["title"] = _truncate(f"流程偏好: {reason_code}", 200)
        candidates.append(fields)
    elif review.get("status") == "FAIL":
        fields = dict(base_fields)
        fields["memory_type"] = "incident_lesson"
        fields["title"] = _truncate(f"失败教训: {reason_code}", 200)
        fields["why"] = _safe_text(
            f"业务验收 FAIL ({reason_code}); evidence={len(evidence_refs)} refs",
            500,
        )
        candidates.append(fields)
    else:
        candidates.append(base_fields)

    comment = str(review.get("comment") or "")
    if _profile_style_signals(comment):
        profile_fields = dict(base_fields)
        profile_fields["memory_type"] = "user_profile"
        profile_fields["scope_type"] = "user"
        profile_fields["scope_key"] = ""
        profile_fields["title"] = "用户沟通偏好"
        profile_fields["summary"] = _safe_text(comment, 500)
        profile_fields["why"] = "业务验收评论体现沟通/风格偏好"
        profile_fields["how_to_apply"] = "回复风格与沟通方式参考此偏好（不覆盖当前明确指令）。"
        candidates.append(profile_fields)

    return candidates


def _build_candidates_from_signals(
    state: Dict[str, Any],
    report: Dict[str, Any],
    trigger_reasons: List[str],
    evidence_refs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    run_id = str(state.get("run_id") or report.get("run_id") or "")
    thread_id = state.get("thread_id") or report.get("thread_id")
    user_request = str(state.get("user_request") or report.get("goal") or "")
    plan = state.get("plan") or report.get("plan") or {}
    module_config = state.get("module_config") or report.get("module_config")
    workflow_key = _workflow_scope_key(user_request, plan)
    store_scope_type, store_scope_key = _store_scope_from_config(module_config)

    run_bundle = {
        "run_id": run_id,
        "thread_id": thread_id,
        "user_request": user_request,
        "message": report.get("final_response") or state.get("final_response"),
        "plan": plan,
        "business_review": state.get("business_review") or report.get("business_review"),
        "session_snapshot": state.get("session_snapshot") or report.get("session_snapshot"),
    }

    candidates: List[Dict[str, Any]] = []

    if "business_review" in trigger_reasons:
        candidates.extend(_map_business_review_to_candidates(run_bundle, evidence_refs))

    if "tool_failure" in trigger_reasons or "guard_block" in trigger_reasons:
        failed = _has_tool_failures(state, report)
        guard = _has_guard_blocks(state, report)
        detail_parts = [e.get("detail") or "" for e in failed + guard][:3]
        fields = {
            "memory_type": "incident_lesson",
            "scope_type": "workflow",
            "scope_key": workflow_key,
            "title": _truncate(f"工具失败/阻断: {workflow_key}", 200),
            "summary": _safe_text("; ".join(detail_parts) or "工具执行或 guard 出现问题", 500),
            "why": _safe_text(
                f"run 出现 tool_failure/guard_block; triggers={','.join(trigger_reasons)}",
                500,
            ),
            "how_to_apply": "同类 workflow 规划时检查 guard 条件与失败恢复路径。",
            "source_run_id": run_id,
            "source_thread_id": thread_id,
            "confidence": 0.45,
        }
        candidates.append(fields)

    if "answer_verify_fail" in trigger_reasons:
        verification = state.get("verification") or report.get("verification") or {}
        issues = verification.get("issues") or []
        fields = {
            "memory_type": "incident_lesson",
            "scope_type": "workflow",
            "scope_key": workflow_key,
            "title": _truncate(f"回复校验问题: {workflow_key}", 200),
            "summary": _safe_text("; ".join(str(i) for i in issues[:5]), 500),
            "why": "answer_verify 未通过，需记录事实/遗漏风险",
            "how_to_apply": "最终回复需与 observation 一致，避免遗漏失败步骤。",
            "source_run_id": run_id,
            "source_thread_id": thread_id,
            "confidence": 0.5,
        }
        candidates.append(fields)

    if "user_correction" in trigger_reasons:
        if _profile_style_signals(user_request):
            candidates.append(
                {
                    "memory_type": "user_profile",
                    "scope_type": "user",
                    "scope_key": "",
                    "title": "用户纠正（沟通/偏好）",
                    "summary": _safe_text(user_request, 500),
                    "why": "用户请求包含纠正/偏好信号",
                    "how_to_apply": "优先遵循当前用户明确指令；本记忆仅作后续默认参考。",
                    "source_run_id": run_id,
                    "source_thread_id": thread_id,
                    "confidence": 0.55,
                }
            )
        else:
            candidates.append(
                {
                    "memory_type": "workflow_preference",
                    "scope_type": "workflow",
                    "scope_key": workflow_key,
                    "title": _truncate(f"用户纠正流程: {workflow_key}", 200),
                    "summary": _safe_text(user_request, 500),
                    "why": "用户请求包含纠正信号（工具/流程/参数）",
                    "how_to_apply": "同类请求参考用户纠正内容规划步骤顺序与工具选择。",
                    "source_run_id": run_id,
                    "source_thread_id": thread_id,
                    "confidence": 0.55,
                }
            )
            candidates.append(
                {
                    "memory_type": "tool_experience",
                    "scope_type": "workflow",
                    "scope_key": workflow_key,
                    "title": _truncate(f"用户纠正工具/参数: {workflow_key}", 200),
                    "summary": _safe_text(user_request, 500),
                    "why": "用户纠正可能涉及工具或参数选择",
                    "how_to_apply": "工具调用前核对用户纠正中的工具名与参数约束。",
                    "source_run_id": run_id,
                    "source_thread_id": thread_id,
                    "confidence": 0.5,
                }
            )

    if "workflow_repeat" in trigger_reasons and "business_review" not in trigger_reasons:
        candidates.append(
            {
                "memory_type": "workflow_preference",
                "scope_type": "workflow",
                "scope_key": workflow_key,
                "title": _truncate(f"重复 workflow 经验: {workflow_key}", 200),
                "summary": _safe_text(
                    f"线程内同类 workflow ({workflow_key}) 重复出现；最近请求: {user_request}",
                    500,
                ),
                "why": "同类 workflow 重复触发，沉淀默认流程偏好",
                "how_to_apply": "同类请求可复用已验证步骤顺序（仍服从当前明确指令）。",
                "source_run_id": run_id,
                "source_thread_id": thread_id,
                "confidence": 0.4,
            }
        )

    if store_scope_type == "store" and store_scope_key and (
        "tool_failure" in trigger_reasons or "workflow_repeat" in trigger_reasons
    ):
        tool_calls = list(state.get("tool_calls") or report.get("tool_calls") or [])
        tc_summary = "; ".join(
            f"{tc.get('tool_name')}" for tc in tool_calls[:3] if tc.get("tool_name")
        )
        if tc_summary:
            candidates.append(
                {
                    "memory_type": "store_knowledge",
                    "scope_type": "store",
                    "scope_key": store_scope_key,
                    "title": _truncate(f"Workspace context: {store_scope_key}", 200),
                    "summary": _safe_text(f"store={store_scope_key}; tools={tc_summary}", 500),
                    "why": "Stable tool/path context within the configured workspace scope",
                    "how_to_apply": f"仅当 scope 匹配 store:{store_scope_key} 时参考。",
                    "source_run_id": run_id,
                    "source_thread_id": thread_id,
                    "confidence": 0.35,
                }
            )

    deduped: List[Dict[str, Any]] = []
    seen_titles: Set[str] = set()
    for fields in candidates:
        key = f"{fields.get('memory_type')}::{fields.get('title')}"
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(fields)
    return deduped


def _create_candidates(
    candidate_fields_list: List[Dict[str, Any]],
    memory_store: MemoryStore,
    evidence_refs: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    created_ids: List[str] = []
    errors: List[Dict[str, Any]] = []
    for fields in candidate_fields_list:
        blob = " ".join(
            str(fields.get(k) or "") for k in ("title", "summary", "why", "how_to_apply")
        )
        if contains_secret_blob(blob):
            errors.append({"error_type": "memory_sensitive_content_skipped", "title": fields.get("title")})
            continue
        record, err = memory_store.create_candidate(fields)
        if err:
            errors.append({"error_type": err, "title": fields.get("title")})
            continue
        if record:
            created_ids.append(record.memory_id)
    return created_ids, errors


def review_run_for_memory_candidates(
    state: Dict[str, Any],
    report: Optional[Dict[str, Any]] = None,
    *,
    ledger_context: Optional[Dict[str, Any]] = None,
    memory_store: Optional[MemoryStore] = None,
    policy: Optional[MemoryReviewPolicy] = None,
    memory_dir: Optional[Union[str, Path]] = None,
    skill_version_store: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    读取 run 快照，按策略生成 memory candidate（status=candidate，不 approve）。
    Reviewer 不调用业务工具；失败 fail-open。
    """
    report = report or {}
    if memory_store is None and not memory_dir:
        return empty_memory_review("review_disabled")
    if policy is None:
        policy = MemoryReviewPolicy(enabled=True)

    if not policy.enabled:
        return empty_memory_review("review_disabled")

    try:
        if ledger_context is None:
            from .session_ledger import SessionLedger

            directory = Path(memory_dir) if memory_dir else memory_store.memory_dir
            ledger_path = directory / "state.db"
            if ledger_path.is_file():
                ledger_context = {"ledger": SessionLedger(ledger_path)}
        trigger_reasons, evidence_refs = _collect_trigger_reasons(
            state, report, policy, ledger_context
        )
        if not trigger_reasons:
            return {
                "enabled": True,
                "triggered": False,
                "candidate_ids": [],
                "skill_improvement_candidate_ids": [],
                "skipped_reason": "no_trigger",
                "errors": [],
                "evidence_refs": evidence_refs,
                "trigger_reasons": [],
                "error_type": None,
            }

        candidate_fields = _build_candidates_from_signals(
            state, report, trigger_reasons, evidence_refs
        )
        if not candidate_fields:
            payload = {
                "enabled": True,
                "triggered": True,
                "candidate_ids": [],
                "skill_improvement_candidate_ids": [],
                "skipped_reason": "no_candidate_payload",
                "errors": [],
                "evidence_refs": evidence_refs,
                "trigger_reasons": trigger_reasons,
                "error_type": None,
            }
            payload["skill_improvement_candidate_ids"] = _maybe_create_skill_improvement_candidates(
                state,
                report,
                trigger_reasons,
                skill_version_store=skill_version_store,
            )
            return payload

        store = memory_store
        if store is None:
            store = MemoryStore(memory_dir=Path(memory_dir) if memory_dir else None)

        created_ids, errors = _create_candidates(candidate_fields, store, evidence_refs)
        skill_ids = _maybe_create_skill_improvement_candidates(
            state,
            report,
            trigger_reasons,
            skill_version_store=skill_version_store,
        )

        return {
            "enabled": True,
            "triggered": True,
            "candidate_ids": created_ids,
            "skill_improvement_candidate_ids": skill_ids,
            "skipped_reason": None if created_ids else "create_failed_or_duplicate",
            "errors": errors,
            "evidence_refs": evidence_refs,
            "trigger_reasons": trigger_reasons,
            "error_type": errors[0]["error_type"] if errors and not created_ids else None,
        }
    except Exception as exc:  # noqa: BLE001 — fail-open boundary
        return {
            "enabled": True,
            "triggered": False,
            "candidate_ids": [],
            "skill_improvement_candidate_ids": [],
            "skipped_reason": "review_exception",
            "errors": [{"error_type": f"memory_review_failed:{type(exc).__name__}"}],
            "evidence_refs": [],
            "trigger_reasons": [],
            "error_type": f"memory_review_failed:{type(exc).__name__}",
        }


def _maybe_create_skill_improvement_candidates(
    state: Dict[str, Any],
    report: Dict[str, Any],
    trigger_reasons: Sequence[str],
    *,
    skill_version_store: Optional[Any] = None,
) -> List[str]:
    try:
        loaded = state.get("loaded_skills") or report.get("loaded_skills") or []
        skill_triggers = {"user_correction", "tool_failure", "workflow_repeat"}
        if not loaded or not (set(trigger_reasons or []) & skill_triggers):
            return []
        from .skill_governance import build_skill_improvement_candidate
        from .skill_store import SkillGovernanceStoreError, SqliteSkillVersionStore

        built = build_skill_improvement_candidate(state, report)
        candidate = built.get("candidate") if isinstance(built, dict) else None
        if not candidate or built.get("error_type"):
            return []
        store = skill_version_store
        if store is None:
            store = SqliteSkillVersionStore()
        created = store.create_improvement_candidate(candidate)
        candidate_id = created.get("candidate_id") if isinstance(created, dict) else None
        if candidate_id:
            return [str(candidate_id)]
        return []
    except Exception:  # noqa: BLE001 — skill candidate fail-open
        return []
