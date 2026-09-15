# -*- coding: utf-8 -*-
"""Phase 2 — 从 approved memories 构建 frozen profile snapshot 与 LLM 注入区块。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .context_contract import _manifest_entry, estimate_chars
from .memory_retriever import MemoryRecallResult
from .redaction import redact_recursive, redact_text

SECTION_BUDGETS: Dict[str, int] = {
    "user_profile": 800,
    "workflow_preferences": 900,
    "store_knowledge": 700,
    "tool_experience": 900,
    "incident_lessons": 800,
    "session_search": 500,
}

MEMORY_TYPE_TO_SECTION: Dict[str, str] = {
    "user_profile": "user_profile",
    "operator_preference": "user_profile",
    "workflow_preference": "workflow_preferences",
    "business_feedback": "workflow_preferences",
    "store_knowledge": "store_knowledge",
    "reference": "store_knowledge",
    "tool_experience": "tool_experience",
    "incident_lesson": "incident_lessons",
    "incident_case": "incident_lessons",
}

SECTION_LABELS: Dict[str, str] = {
    "user_profile": "User profile / operator preferences",
    "workflow_preferences": "Workflow preferences",
    "store_knowledge": "Store knowledge",
    "tool_experience": "Tool experience",
    "incident_lessons": "Incident lessons",
    "session_search": "Session history snippets (advisory)",
}

ADVISORY_PREFIX = (
    "Memory context (advisory only). Current user instructions in this turn always override "
    "memory suggestions. Do not claim tools were executed unless observations confirm execution."
)

MAX_SESSION_SNIPPETS = 2
MAX_SNIPPET_CHARS = 200


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _section_item_line(memory: Dict[str, Any]) -> str:
    title = memory.get("title") or memory.get("memory_id") or "memory"
    summary = memory.get("summary") or ""
    how = memory.get("how_to_apply") or ""
    conf = memory.get("confidence")
    scope = memory.get("scope_type") or "global"
    scope_key = memory.get("scope_key") or ""
    scope_label = f"{scope}:{scope_key}" if scope_key else scope
    line = f"- [{memory.get('memory_id')}] ({memory.get('memory_type')}, {scope_label}, conf={conf}) {title}: {summary}"
    if how:
        line += f" | apply: {how}"
    cleaned, _ = redact_text(line)
    return cleaned


def build_memory_profile_from_recall(
    recall_result: MemoryRecallResult,
    *,
    session_search: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """将 recall 结果整理为结构化 profile（含字符预算与 omission）。"""
    sections: Dict[str, Dict[str, Any]] = {}
    for section_id in (
        "user_profile",
        "workflow_preferences",
        "store_knowledge",
        "tool_experience",
        "incident_lessons",
    ):
        sections[section_id] = {
            "section_id": section_id,
            "label": SECTION_LABELS[section_id],
            "char_budget": SECTION_BUDGETS[section_id],
            "chars_used": 0,
            "included": False,
            "omission_reason": "no_matching_memories",
            "memory_ids": [],
            "items": [],
            "text": "",
        }

    injected_memories: List[Dict[str, Any]] = []
    for memory in recall_result.approved_memories:
        section_id = MEMORY_TYPE_TO_SECTION.get(str(memory.get("memory_type") or ""))
        if not section_id or section_id not in sections:
            continue
        bucket = sections[section_id]
        line = _section_item_line(memory)
        budget = bucket["char_budget"]
        used = bucket["chars_used"]
        if used + len(line) + 1 > budget:
            continue
        bucket["items"].append(memory)
        bucket["memory_ids"].append(memory.get("memory_id"))
        bucket["chars_used"] = used + len(line) + 1
        bucket["included"] = True
        bucket["omission_reason"] = None
        text_parts = bucket.get("_text_parts") or []
        text_parts.append(line)
        bucket["_text_parts"] = text_parts
        injected_memories.append(
            {
                **memory,
                "included_in_llm": True,
                "section_id": section_id,
            }
        )

    for section_id, bucket in sections.items():
        parts = bucket.pop("_text_parts", None) or []
        if parts:
            bucket["text"] = "\n".join(parts)

    session_block = _build_session_search_block(session_search or {})
    omitted = list(recall_result.omitted or [])
    conflicts = list(recall_result.memory_conflicts or [])

    return {
        "enabled": True,
        "sections": sections,
        "session_search": session_block,
        "injected_memories": injected_memories,
        "omitted": omitted,
        "conflicts": conflicts,
        "summary": dict(recall_result.summary or {}),
    }


def _build_session_search_block(session_search: Dict[str, Any]) -> Dict[str, Any]:
    hits = list(session_search.get("retrieved_sessions") or [])
    snippets: List[Dict[str, Any]] = []
    chars_used = 0
    budget = SECTION_BUDGETS["session_search"]
    for hit in hits[:MAX_SESSION_SNIPPETS]:
        snippet = _truncate(str(hit.get("snippet") or hit.get("user_request_preview") or ""), MAX_SNIPPET_CHARS)
        cleaned, _ = redact_text(snippet)
        if not cleaned:
            continue
        line = f"- [{hit.get('run_id')}] ({hit.get('match_reason')}) {cleaned}"
        if chars_used + len(line) + 1 > budget:
            break
        snippets.append(
            {
                "run_id": hit.get("run_id"),
                "match_reason": hit.get("match_reason"),
                "snippet": cleaned,
                "included_in_llm": True,
            }
        )
        chars_used += len(line) + 1

    included = bool(snippets)
    omission = None
    if session_search.get("error_type"):
        omission = session_search.get("error_type")
    elif not hits:
        omission = "no_session_hits"
    elif not snippets:
        omission = "session_snippet_budget_exceeded"

    text = ""
    if snippets:
        text = "\n".join(f"- [{s['run_id']}] ({s['match_reason']}) {s['snippet']}" for s in snippets)

    return {
        "section_id": "session_search",
        "label": SECTION_LABELS["session_search"],
        "char_budget": budget,
        "chars_used": chars_used,
        "included": included,
        "omission_reason": omission if not included else None,
        "enabled": session_search.get("enabled", True),
        "used": session_search.get("used", False),
        "query": session_search.get("query") or "",
        "error_type": session_search.get("error_type"),
        "retrieved_sessions": hits,
        "snippets": snippets,
        "text": text,
    }


def inject_memory_into_messages(
    messages: List[Dict[str, Any]],
    *,
    memory_profile: Dict[str, Any],
    user_request: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """
    将 profile/session snippet 作为独立 system 区块注入 messages（仅 advisory）。
    返回 (new_messages, manifest_entries, injected_flag)。
    """
    if not messages:
        return messages, [], False

    blocks: List[str] = []
    manifest: List[Dict[str, Any]] = []

    for section_id in (
        "user_profile",
        "workflow_preferences",
        "store_knowledge",
        "tool_experience",
        "incident_lessons",
    ):
        section = (memory_profile.get("sections") or {}).get(section_id) or {}
        text = section.get("text") or ""
        included = bool(section.get("included") and text)
        manifest.append(
            _manifest_entry(
                f"memory_profile.{section_id}",
                source="memory_profile.snapshot",
                included=included,
                payload={
                    "memory_ids": section.get("memory_ids") or [],
                    "items": section.get("items") or [],
                    "text_preview": _truncate(text, 400) if text else None,
                },
                truncated=len(text) > SECTION_BUDGETS.get(section_id, 800),
                omission_reason=section.get("omission_reason"),
                sent_to_llm=included,
            )
        )
        if included:
            blocks.append(f"{section.get('label') or section_id}:\n{text}")

    session_block = memory_profile.get("session_search") or {}
    session_text = session_block.get("text") or ""
    session_included = bool(session_block.get("included") and session_text)
    manifest.append(
        _manifest_entry(
            "session_search",
            source="session_ledger.search",
            included=session_included,
            payload={
                "retrieved_sessions": session_block.get("retrieved_sessions") or [],
                "snippets": session_block.get("snippets") or [],
                "text_preview": _truncate(session_text, 400) if session_text else None,
            },
            omission_reason=session_block.get("omission_reason"),
            sent_to_llm=session_included,
        )
    )
    if session_included:
        blocks.append(f"{session_block.get('label') or 'Session search'}:\n{session_text}")

    if not blocks:
        return messages, manifest, False

    user_line = _truncate(user_request, 500)
    cleaned_user, _ = redact_text(user_line)
    injection_body = ADVISORY_PREFIX + "\n\n" + "\n\n".join(blocks)
    if cleaned_user:
        injection_body += f"\n\nCurrent user request (authoritative): {cleaned_user}"

    injection_msg = {"role": "system", "content": injection_body}
    manifest.append(
        _manifest_entry(
            "memory_context_injection",
            source="memory_profile.inject_memory_into_messages",
            included=True,
            payload={"chars": len(injection_body)},
            sent_to_llm=True,
        )
    )

    new_messages = [messages[0], injection_msg] + messages[1:]
    return new_messages, manifest, True


def empty_memory_profile(*, error_type: Optional[str] = None) -> Dict[str, Any]:
    return {
        "enabled": False,
        "sections": {},
        "session_search": {
            "section_id": "session_search",
            "enabled": False,
            "used": False,
            "included": False,
            "omission_reason": error_type or "unavailable",
            "retrieved_sessions": [],
            "snippets": [],
        },
        "injected_memories": [],
        "omitted": [],
        "conflicts": [],
        "summary": {"enabled": False, "error_type": error_type},
    }
