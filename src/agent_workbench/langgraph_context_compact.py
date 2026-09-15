# -*- coding: utf-8 -*-
"""
LangGraph StateGraph 上下文压缩（Phase 2）。

策略：tool result budget、observation micro compact、history snip。
不切断 assistant tool_call 与 tool message 配对。
"""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .context_contract import estimate_chars
from .langgraph_shared import observation_to_tool_message
from .redaction import redact_recursive, redact_text

TOOL_RESULT_MAX_CHARS = 800
TOOL_RESULT_PREVIEW_CHARS = 240
FULL_OBSERVATION_KEEP_COUNT = 4
MAX_RECENT_SEGMENTS = 6
COMPACT_PLACEHOLDER_NOTE = "observation compacted; rerun/read artifact if needed"


def empty_context_budget() -> Dict[str, Any]:
    return {
        "chars_before": 0,
        "chars_after": 0,
        "messages_snipped": 0,
        "observations_compacted": 0,
        "artifacts_spilled": 0,
        "strategies_applied": [],
        "spilled_artifacts": [],
    }


def _parse_tool_observation_content(content: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _compact_observation_payload(
    obs: Dict[str, Any],
    *,
    tool_call_id: Optional[str],
    preview_chars: int = TOOL_RESULT_PREVIEW_CHARS,
) -> Dict[str, Any]:
    summary = str(obs.get("summary") or "")
    preview = summary[:preview_chars]
    if len(summary) > preview_chars:
        preview = preview + "…"
    compact: Dict[str, Any] = {
        "step_id": obs.get("step_id"),
        "status": obs.get("status"),
        "summary": preview,
        "compacted": True,
        "note": COMPACT_PLACEHOLDER_NOTE,
    }
    if tool_call_id:
        compact["tool_call_id"] = tool_call_id
    artifacts = obs.get("artifacts") or obs.get("artifact_refs") or []
    if artifacts:
        compact["artifact_refs"] = []
        for artifact in artifacts[:5]:
            if not isinstance(artifact, dict):
                continue
            keys = ("type", "path", "step", "status", "artifact_id", "version", "content_hash")
            reference = {k: artifact[k] for k in keys if k in artifact}
            if artifact.get("type") == "artifact.review":
                for key in ("decision", "review_id", "tool_call_id", "rule_version", "comment"):
                    if key in artifact:
                        text, _ = redact_text(str(artifact[key]))
                        reference[key] = text[:2000]
                        if len(text) > 2000:
                            reference[key + "_truncated"] = True
            compact["artifact_refs"].append(reference)
    if obs.get("raw_output_ref"):
        compact["raw_output_ref"] = obs.get("raw_output_ref")
    err = obs.get("error")
    if err:
        compact["error"] = {
            "type": (err or {}).get("type"),
            "message": str((err or {}).get("message") or "")[:160],
        }
    return compact


def _apply_tool_result_budget(
    messages: List[Dict[str, Any]],
    *,
    spill_dir: Optional[Path],
    round_idx: int,
    budget: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "tool":
            out.append(dict(msg))
            continue
        content = str(msg.get("content") or "")
        tool_call_id = msg.get("tool_call_id")
        if len(content) <= TOOL_RESULT_MAX_CHARS:
            out.append(dict(msg))
            continue

        obs = _parse_tool_observation_content(content) or {"summary": content[:TOOL_RESULT_PREVIEW_CHARS]}
        obs, _ = redact_recursive(obs)
        content = json.dumps(obs, ensure_ascii=False)
        compact_obs = _compact_observation_payload(obs, tool_call_id=tool_call_id)
        spill_path: Optional[str] = None
        if spill_dir is not None:
            spill_dir.mkdir(parents=True, exist_ok=True)
            identity = hashlib.sha256(str(tool_call_id or 'tool').encode()).hexdigest()[:16]
            digest = hashlib.sha256(content.encode()).hexdigest()[:16]
            fname = f"round{int(round_idx)}_{identity}_{digest}.json"
            target = spill_dir / fname
            if target.is_symlink():
                raise ValueError("unsafe_context_spill_path")
            target.write_text(content, encoding="utf-8")
            spill_path = str(target)
            compact_obs["spill_path"] = spill_path
            budget["artifacts_spilled"] = int(budget.get("artifacts_spilled") or 0) + 1
            budget.setdefault("spilled_artifacts", []).append(
                {"tool_call_id": tool_call_id, "path": spill_path, "chars": len(content)}
            )

        compact_content = json.dumps(compact_obs, ensure_ascii=False)
        out.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": compact_content,
            }
        )
        if "tool_result_budget" not in budget["strategies_applied"]:
            budget["strategies_applied"].append("tool_result_budget")
    return out


def _apply_micro_compact_observations(
    messages: List[Dict[str, Any]],
    budget: Dict[str, Any],
) -> List[Dict[str, Any]]:
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= FULL_OBSERVATION_KEEP_COUNT:
        return list(messages)

    keep_indices = set(tool_indices[-FULL_OBSERVATION_KEEP_COUNT:])
    out: List[Dict[str, Any]] = []
    compacted = 0
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool" or idx in keep_indices:
            out.append(dict(msg))
            continue
        content = str(msg.get("content") or "")
        obs = _parse_tool_observation_content(content) or {"summary": content[:120], "status": "unknown"}
        if obs.get("compacted"):
            out.append(dict(msg))
            continue
        compact_obs = _compact_observation_payload(obs, tool_call_id=msg.get("tool_call_id"))
        out.append(
            {
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id"),
                "content": json.dumps(compact_obs, ensure_ascii=False),
            }
        )
        compacted += 1

    if compacted:
        budget["observations_compacted"] = int(budget.get("observations_compacted") or 0) + compacted
        if "micro_compact_observations" not in budget["strategies_applied"]:
            budget["strategies_applied"].append("micro_compact_observations")
    return out


def _group_message_segments(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    segments: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = {tc.get("id") for tc in (msg.get("tool_calls") or []) if tc.get("id")}
            seg = [dict(msg)]
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                tcid = messages[i].get("tool_call_id")
                if not ids or tcid in ids:
                    seg.append(dict(messages[i]))
                    i += 1
                else:
                    break
            segments.append(seg)
        else:
            segments.append([dict(msg)])
            i += 1
    return segments


def _flatten_segments(segments: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in segments:
        out.extend(seg)
    return out


def _apply_history_snip(
    messages: List[Dict[str, Any]],
    *,
    user_request: str,
    running_summary: str,
    budget: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if len(messages) <= 2:
        return list(messages)

    prefix: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    seen_user = False
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            prefix.append(dict(msg))
        elif role == "user" and not seen_user:
            prefix.append(dict(msg))
            seen_user = True
        else:
            rest.append(dict(msg))

    if running_summary.strip():
        prefix.append(
            {
                "role": "system",
                "content": f"Session running summary:\n{running_summary[:1200]}",
            }
        )

    segments = _group_message_segments(rest)
    if len(segments) <= MAX_RECENT_SEGMENTS:
        return prefix + _flatten_segments(segments)

    snip_count = len(segments) - MAX_RECENT_SEGMENTS
    kept = segments[-MAX_RECENT_SEGMENTS:]
    placeholder = {
        "role": "assistant",
        "content": f"[history snipped {snip_count} message segment(s); recent tool call pairs preserved]",
    }
    budget["messages_snipped"] = int(budget.get("messages_snipped") or 0) + snip_count
    if "history_snip" not in budget["strategies_applied"]:
        budget["strategies_applied"].append("history_snip")
    return prefix + [placeholder] + _flatten_segments(kept)


@dataclass
class ContextCompactOptions:
    spill_dir: Optional[Path] = None
    round_idx: int = 0
    user_request: str = ""
    running_summary: str = ""
    enable_tool_result_budget: bool = True
    enable_micro_compact: bool = True
    enable_history_snip: bool = True


def compact_messages_for_llm(
    messages: List[Dict[str, Any]],
    *,
    options: Optional[ContextCompactOptions] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    将完整 messages 压缩为送入 LLM 的 llm_input_messages，并返回 context_budget。
    """
    opts = options or ContextCompactOptions()
    budget = empty_context_budget()
    budget["chars_before"] = estimate_chars(messages)

    working = [dict(m) for m in messages or []]
    if opts.enable_tool_result_budget:
        working = _apply_tool_result_budget(
            working,
            spill_dir=opts.spill_dir,
            round_idx=opts.round_idx,
            budget=budget,
        )
    if opts.enable_micro_compact:
        working = _apply_micro_compact_observations(working, budget)
    if opts.enable_history_snip:
        working = _apply_history_snip(
            working,
            user_request=opts.user_request,
            running_summary=opts.running_summary,
            budget=budget,
        )

    budget["chars_after"] = estimate_chars(working)
    return working, budget


def collect_spilled_artifacts(budget_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spilled: List[Dict[str, Any]] = []
    for budget in budget_history or []:
        spilled.extend(list(budget.get("spilled_artifacts") or []))
    return spilled


def default_context_spill_dir() -> Optional[Path]:
    base = os.environ.get("AGENT_WORKBENCH_CONTEXT_SPILL_DIR", "").strip()
    if base:
        return Path(base)
    return None
