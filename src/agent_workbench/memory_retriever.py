# -*- coding: utf-8 -*-
"""Phase 13.3 — Approved Memory Recall（结构化过滤 + 确定性文本匹配）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .memory_store import MemoryRecord, MemoryStore
from .redaction import contains_secret_blob, redact_recursive, redact_text
from .schemas import Plan

DEFAULT_RECALL_TOP_K = 3
MAX_RECALL_TOP_K = 5

_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass
class ScoredMemory:
    record: MemoryRecord
    score: int
    retrieval_reason: str


@dataclass
class MemoryRecallResult:
    approved_memories: List[Dict[str, Any]] = field(default_factory=list)
    memory_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    omitted: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _is_expired(record: MemoryRecord, now: Optional[datetime] = None) -> bool:
    if not record.expires_at:
        return False
    exp = _parse_iso(record.expires_at)
    if exp is None:
        return False
    current = now or _utc_now()
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp < current


def _cjk_bigrams(text: str) -> Set[str]:
    run = re.sub(r"[^\u4e00-\u9fff]", "", text or "")
    if len(run) < 2:
        return {run} if run else set()
    return {run[i : i + 2] for i in range(len(run) - 1)}


def _tokenize(text: str) -> Set[str]:
    tokens: Set[str] = set()
    lowered = (text or "").lower()
    for word in re.findall(r"[a-z0-9_]+", lowered):
        if len(word) >= 2:
            tokens.add(word)
    tokens |= _cjk_bigrams(text)
    return {t for t in tokens if len(t) >= 2 or re.search(r"[\u4e00-\u9fff]", t)}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_workflow_keys(candidate_plan: Optional[Plan], candidate_intent: str) -> Set[str]:
    keys: Set[str] = set()
    if candidate_intent:
        keys.add(candidate_intent)
    keys |= infer_workflow_keys_from_request(candidate_intent or "")
    if candidate_plan is None:
        return keys
    if candidate_plan.plan_type:
        keys.add(candidate_plan.plan_type)
    for step in candidate_plan.steps:
        keys.add(step.module)
        keys.add(f"{step.module}.{step.action}")
    return keys


_WORKFLOW_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


def infer_workflow_keys_from_request(
    user_request: str,
    *,
    workflow_hints: Optional[Tuple[Tuple[str, Tuple[str, ...]], ...]] = None,
) -> Set[str]:
    """Match literal capability IDs and optional host-provided workflow aliases."""
    keys: Set[str] = set()
    text = user_request or ""
    lowered = text.lower()
    for capability in re.findall(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\b", lowered):
        keys.update((capability, capability.split(".", 1)[0]))
    for keyword, wf_keys in (_WORKFLOW_HINTS if workflow_hints is None else workflow_hints):
        if keyword in text or keyword in lowered:
            keys.update(wf_keys)
    return keys


def _extract_store_keys(
    module_config: Optional[Dict[str, Any]],
    session_record: Optional[Dict[str, Any]],
) -> Set[str]:
    keys: Set[str] = set()
    for cfg in (module_config or {}).values():
        if not isinstance(cfg, dict):
            continue
        for sk in ("store_id", "shop_id", "scope_key", "data_source", "store_key"):
            val = cfg.get(sk)
            if val not in (None, ""):
                keys.add(str(val))
    if session_record:
        kp = session_record.get("known_parameters") or {}
        if isinstance(kp, dict):
            for sk in ("store_id", "shop_id", "store_key", "data_source"):
                val = kp.get(sk)
                if val not in (None, ""):
                    keys.add(str(val))
    return keys


def _extract_user_keys(session_record: Optional[Dict[str, Any]]) -> Set[str]:
    keys: Set[str] = set()
    if not session_record:
        return keys
    kp = session_record.get("known_parameters") or {}
    if isinstance(kp, dict):
        for sk in ("user_id", "operator_id", "user_scope_key"):
            val = kp.get(sk)
            if val not in (None, ""):
                keys.add(str(val))
    explicit = session_record.get("user_scope_key")
    if explicit not in (None, ""):
        keys.add(str(explicit))
    return keys


def _scope_eligible(
    record: MemoryRecord,
    *,
    workflow_keys: Set[str],
    store_keys: Set[str],
    user_keys: Set[str],
) -> Tuple[bool, Optional[str]]:
    if record.scope_type == "global":
        return True, None
    if record.scope_type == "workflow":
        if not record.scope_key:
            return True, None
        if record.scope_key in workflow_keys:
            return True, None
        return False, "scope_workflow_mismatch"
    if record.scope_type == "store":
        if record.scope_key in store_keys:
            return True, None
        return False, "scope_store_unavailable"
    if record.scope_type == "user":
        if record.scope_key in user_keys:
            return True, None
        return False, "scope_user_unavailable"
    return False, "scope_unknown"


def _score_memory(
    record: MemoryRecord,
    *,
    query_tokens: Set[str],
    workflow_keys: Set[str],
    user_request: str,
) -> Tuple[int, str]:
    corpus = " ".join(
        [
            record.title,
            record.summary,
            record.why,
            record.how_to_apply,
            record.scope_key,
        ]
    )
    mem_tokens = _tokenize(corpus)
    overlap = query_tokens & mem_tokens
    score = len(overlap)
    reasons: List[str] = []
    if overlap:
        reasons.append("text_token_overlap")
    if record.scope_key and record.scope_key in workflow_keys:
        score += 2
        reasons.append("workflow_scope_match")
    req_norm = _normalize_text(user_request)
    title_norm = _normalize_text(record.title)
    summary_norm = _normalize_text(record.summary)
    if title_norm and title_norm in req_norm:
        score += 3
        reasons.append("title_in_request")
    if summary_norm and any(tok in req_norm for tok in _tokenize(summary_norm) if len(tok) >= 3):
        score += 1
        reasons.append("summary_partial_match")
    if not reasons:
        return 0, "no_match"
    return score, "+".join(reasons)


def _conflict_group_key(record: MemoryRecord) -> Tuple[str, str, str, str]:
    return (
        record.scope_type,
        record.scope_key or "",
        record.memory_type,
        _normalize_text(record.title)[:48],
    )


def _detect_conflicts(scored: List[ScoredMemory]) -> Tuple[List[ScoredMemory], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str, str], List[ScoredMemory]] = {}
    for item in scored:
        groups.setdefault(_conflict_group_key(item.record), []).append(item)

    conflicts: List[Dict[str, Any]] = []
    safe: List[ScoredMemory] = []
    for key, items in groups.items():
        applies = {_normalize_text(i.record.how_to_apply) for i in items}
        summaries = {_normalize_text(i.record.summary) for i in items}
        if len(items) > 1 and len(applies) > 1 and len(summaries) > 1:
            conflicts.append(
                {
                    "memory_ids": sorted(i.record.memory_id for i in items),
                    "scope_type": key[0],
                    "scope_key": key[1],
                    "memory_type": key[2],
                    "topic": key[3],
                    "reason": "contradictory_content",
                    "reviewer_hint": "ask_user",
                }
            )
            continue
        safe.extend(items)
    return safe, conflicts


def _slim_memory(record: MemoryRecord, retrieval_reason: str) -> Dict[str, Any]:
    payload = {
        "memory_id": record.memory_id,
        "memory_type": record.memory_type,
        "scope_type": record.scope_type,
        "scope_key": record.scope_key,
        "title": record.title[:200],
        "summary": record.summary[:500],
        "how_to_apply": record.how_to_apply[:500],
        "confidence": record.confidence,
        "updated_at": record.updated_at,
        "retrieval_reason": retrieval_reason,
    }
    safe, _ = redact_recursive(payload)
    return safe


def _sort_key(item: ScoredMemory) -> Tuple[int, float, str, str]:
    return (
        -item.score,
        -float(item.record.confidence or 0),
        item.record.updated_at or "",
        item.record.memory_id,
    )


def empty_recall_summary(*, enabled: bool = True, error_type: Optional[str] = None) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "used": False,
        "retrieved_memory_ids": [],
        "omitted": [],
        "conflicts": [],
        "error_type": error_type,
    }


def recall_approved_memories(
    *,
    user_request: str,
    candidate_plan: Optional[Plan],
    candidate_intent: str,
    module_config: Optional[Dict[str, Any]] = None,
    session_record: Optional[Dict[str, Any]] = None,
    memory_store: Optional[MemoryStore] = None,
    memory_dir: Optional[Union[str, Path]] = None,
    top_k: int = DEFAULT_RECALL_TOP_K,
    extra_workflow_keys: Optional[Set[str]] = None,
    enabled: Optional[bool] = None,
) -> MemoryRecallResult:
    """召回 approved 记忆；存储初始化/读取失败均 fail-open，不阻断规划。"""
    if enabled is False or (memory_store is None and not memory_dir):
        return MemoryRecallResult(summary=empty_recall_summary(enabled=False))
    limit = min(max(1, top_k), MAX_RECALL_TOP_K)
    summary = empty_recall_summary(enabled=True)
    result = MemoryRecallResult(summary=summary)

    store = memory_store
    if store is None:
        try:
            store = MemoryStore(memory_dir=Path(memory_dir) if memory_dir else None)
        except Exception as exc:  # noqa: BLE001 — fail-open boundary
            summary["error_type"] = f"memory_store_unavailable:{type(exc).__name__}"
            result.omitted.append({"reason": "store_unavailable"})
            summary["omitted"] = list(result.omitted)
            return result

    try:
        listed = store.list(status="approved")
    except Exception as exc:  # noqa: BLE001 — fail-open boundary
        summary["error_type"] = f"memory_recall_failed:{exc}"
        result.omitted.append({"reason": "list_failed"})
        summary["omitted"] = list(result.omitted)
        return result

    if listed.index_error_type:
        summary["error_type"] = listed.index_error_type
        result.omitted.append({"reason": "index_error", "error_type": listed.index_error_type})
        summary["omitted"] = list(result.omitted)
        return result

    if listed.load_errors:
        for err in listed.load_errors:
            result.omitted.append(
                {
                    "memory_id": err.get("memory_id"),
                    "reason": "load_error",
                    "error_type": err.get("error_type"),
                }
            )

    workflow_keys = _extract_workflow_keys(candidate_plan, candidate_intent)
    if extra_workflow_keys:
        workflow_keys |= set(extra_workflow_keys)
    workflow_keys |= infer_workflow_keys_from_request(user_request)
    store_keys = _extract_store_keys(module_config, session_record)
    user_keys = _extract_user_keys(session_record)
    query_tokens = _tokenize(user_request)
    now = _utc_now()

    scored: List[ScoredMemory] = []
    for record in listed.memories:
        if record.status != "approved":
            result.omitted.append({"memory_id": record.memory_id, "reason": "not_approved"})
            continue
        if record.title == "[FORGOTTEN]" or record.status == "forgotten":
            result.omitted.append({"memory_id": record.memory_id, "reason": "forgotten_tombstone"})
            continue
        if _is_expired(record, now):
            result.omitted.append({"memory_id": record.memory_id, "reason": "expired"})
            continue
        eligible, scope_reason = _scope_eligible(
            record,
            workflow_keys=workflow_keys,
            store_keys=store_keys,
            user_keys=user_keys,
        )
        if not eligible:
            result.omitted.append({"memory_id": record.memory_id, "reason": scope_reason})
            continue
        blob = " ".join([record.title, record.summary, record.why, record.how_to_apply])
        if contains_secret_blob(blob):
            result.omitted.append({"memory_id": record.memory_id, "reason": "sensitive_content"})
            continue
        score, reason = _score_memory(
            record,
            query_tokens=query_tokens,
            workflow_keys=workflow_keys,
            user_request=user_request,
        )
        if score <= 0:
            result.omitted.append({"memory_id": record.memory_id, "reason": "no_match"})
            continue
        scored.append(ScoredMemory(record=record, score=score, retrieval_reason=reason))

    scored.sort(key=_sort_key)
    safe_scored, conflicts = _detect_conflicts(scored)
    result.memory_conflicts = conflicts
    summary["conflicts"] = conflicts

    if conflicts:
        conflict_ids = {mid for c in conflicts for mid in c.get("memory_ids", [])}
        for item in scored:
            if item.record.memory_id in conflict_ids:
                result.omitted.append(
                    {
                        "memory_id": item.record.memory_id,
                        "reason": "conflict_excluded",
                    }
                )

    selected = safe_scored[:limit]
    truncated = safe_scored[limit:]
    for item in truncated:
        result.omitted.append({"memory_id": item.record.memory_id, "reason": "top_k_truncated"})

    result.approved_memories = [
        _slim_memory(item.record, item.retrieval_reason) for item in selected
    ]
    summary["used"] = bool(result.approved_memories)
    summary["retrieved_memory_ids"] = [m["memory_id"] for m in result.approved_memories]
    summary["omitted"] = list(result.omitted)
    return result


def recall_memories_for_langgraph(
    *,
    user_request: str,
    module_config: Optional[Dict[str, Any]] = None,
    session_record: Optional[Dict[str, Any]] = None,
    memory_store: Optional[MemoryStore] = None,
    memory_dir: Optional[Union[str, Path]] = None,
    top_k: int = DEFAULT_RECALL_TOP_K,
) -> MemoryRecallResult:
    """LangGraph retrieve_context_node 使用的 approved memory 召回（fail-open）。"""
    workflow_keys = infer_workflow_keys_from_request(user_request)
    intent_hint = next(iter(workflow_keys), "")
    try:
        return recall_approved_memories(
            user_request=user_request,
            candidate_plan=None,
            candidate_intent=intent_hint,
            module_config=module_config,
            session_record=session_record,
            memory_store=memory_store,
            memory_dir=memory_dir,
            top_k=top_k,
            extra_workflow_keys=workflow_keys,
        )
    except Exception as exc:  # noqa: BLE001 — fail-open boundary
        summary = empty_recall_summary(enabled=True, error_type=f"memory_recall_failed:{exc}")
        return MemoryRecallResult(summary=summary, omitted=[{"reason": "recall_exception"}])
