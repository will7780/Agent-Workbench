# -*- coding: utf-8 -*-
"""LangGraph 每轮 LLM 调用前的上下文快照（脱敏/截断，不含 chain-of-thought）。"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .context_contract import _manifest_entry, _safe_module_config_summary, estimate_chars
from .llm_client import load_llm_config
from .redaction import redact_recursive, redact_text
from .registry import ModuleCapabilityRegistry, get_registry
from .tool_schema import capability_id, tool_name_to_capability

MAX_MESSAGE_CONTENT_CHARS = 4000
MAX_TOOL_DESCRIPTION_CHARS = 240
SNAPSHOT_SOURCE = "langgraph_runner.chat_completion_with_tools"


def _summarize_tools(
    tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    summaries: List[Dict[str, Any]] = []
    truncated = False
    for tool_def in tools or []:
        fn = tool_def.get("function") or {}
        name = str(fn.get("name") or "")
        parsed = tool_name_to_capability(name)
        cap_id = capability_id(parsed[0], parsed[1]) if parsed else name
        params = fn.get("parameters") or {}
        required = list(params.get("required") or [])
        desc = str(fn.get("description") or "")
        if len(desc) > MAX_TOOL_DESCRIPTION_CHARS:
            truncated = True
            desc = desc[:MAX_TOOL_DESCRIPTION_CHARS] + "…"
        summaries.append(
            {
                "tool_name": name,
                "capability_id": cap_id,
                "required_params": required,
                "description_preview": desc,
            }
        )
    return summaries, truncated


def _prepare_messages(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool, bool]:
    """脱敏并截断 messages；仅保留实际 role/content/tool_calls，不含隐藏推理。"""
    prepared: List[Dict[str, Any]] = []
    truncated = False
    redacted_any = False
    allowed_keys = frozenset({"role", "content", "tool_calls", "tool_call_id", "name", "type", "function", "id"})

    for msg in messages or []:
        raw = copy.deepcopy(msg)
        slim: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in allowed_keys and key not in ("function",):
                continue
            slim[key] = value

        content = slim.get("content")
        if isinstance(content, str):
            cleaned, was_redacted = redact_text(content)
            redacted_any = redacted_any or was_redacted
            if len(cleaned) > MAX_MESSAGE_CONTENT_CHARS:
                slim["content"] = cleaned[:MAX_MESSAGE_CONTENT_CHARS] + "…[truncated]"
                truncated = True
            else:
                slim["content"] = cleaned

        if slim.get("tool_calls"):
            redacted_tc, was = redact_recursive(slim["tool_calls"])
            slim["tool_calls"] = redacted_tc
            redacted_any = redacted_any or was

        prepared.append(slim)
    return prepared, truncated, redacted_any


def _split_message_sections(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    system_prompt = ""
    user_request = ""
    conversation_messages: List[Dict[str, Any]] = []
    tool_observations: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            content = str(msg.get("content") or "")
            if not system_prompt:
                system_prompt = content
            elif content:
                system_prompt = f"{system_prompt}\n\n{content}"
        elif role == "user" and not user_request:
            user_request = str(msg.get("content") or "")
        elif role == "assistant":
            conversation_messages.append(msg)
        elif role == "tool":
            tool_observations.append(msg)

    return {
        "system_prompt": system_prompt,
        "user_request": user_request,
        "conversation_messages": conversation_messages,
        "tool_observations": tool_observations,
    }


def build_langgraph_llm_context_snapshot(
    round_idx: int,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    module_config: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    registry: Optional[ModuleCapabilityRegistry] = None,
    context_budget: Optional[Dict[str, Any]] = None,
    raw_message_count: Optional[int] = None,
    extra_manifest: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    在 chat_completion_with_tools 调用前构建本轮 LLM 输入快照。
    不记录 chain-of-thought，仅 messages / tool schemas / observations / metadata。
    """
    reg = registry or get_registry()
    cfg = load_llm_config(model_override=model, load_credentials=False)
    resolved_model = model or cfg.model
    resolved_provider = provider or cfg.provider

    prepared_messages, msg_truncated, msg_redacted = _prepare_messages(messages)
    tool_summaries, tools_truncated = _summarize_tools(tools or [])
    sections = _split_message_sections(prepared_messages)

    mod_summary, mod_redacted, mod_omission = _safe_module_config_summary(module_config)
    manifest: List[Dict[str, Any]] = []

    manifest.append(
        _manifest_entry(
            "system_prompt",
            source=SNAPSHOT_SOURCE,
            included=bool(sections["system_prompt"]),
            payload={"content": sections["system_prompt"]} if sections["system_prompt"] else None,
            redacted=msg_redacted,
            sent_to_llm=bool(sections["system_prompt"]),
        )
    )
    manifest.append(
        _manifest_entry(
            "user_request",
            source=SNAPSHOT_SOURCE,
            included=bool(sections["user_request"]),
            payload={"content": sections["user_request"]} if sections["user_request"] else None,
            redacted=msg_redacted,
            sent_to_llm=bool(sections["user_request"]),
        )
    )
    manifest.append(
        _manifest_entry(
            "conversation_messages",
            source=SNAPSHOT_SOURCE,
            included=bool(sections["conversation_messages"]),
            payload=sections["conversation_messages"] or None,
            truncated=msg_truncated,
            redacted=msg_redacted,
            omission_reason=None if sections["conversation_messages"] else "no_assistant_messages_yet",
            sent_to_llm=bool(sections["conversation_messages"]),
        )
    )
    manifest.append(
        _manifest_entry(
            "tool_catalog",
            source="registry.tool_schemas",
            included=bool(tool_summaries),
            payload=tool_summaries,
            truncated=tools_truncated,
            sent_to_llm=bool(tool_summaries),
        )
    )
    manifest.append(
        _manifest_entry(
            "available_tools",
            source="registry.tool_schemas",
            included=bool(tool_summaries),
            payload={"count": len(tool_summaries), "tool_names": [t["tool_name"] for t in tool_summaries]},
            truncated=tools_truncated,
            sent_to_llm=bool(tool_summaries),
        )
    )
    manifest.append(
        _manifest_entry(
            "tool_observations",
            source="execute_adapter.observations",
            included=bool(sections["tool_observations"]),
            payload=sections["tool_observations"] or None,
            truncated=msg_truncated,
            redacted=msg_redacted,
            omission_reason=None if sections["tool_observations"] else "no_tool_observations_yet",
            sent_to_llm=bool(sections["tool_observations"]),
        )
    )
    manifest.append(
        _manifest_entry(
            "module_config_summary",
            source="module_config",
            included=bool(mod_summary),
            payload=mod_summary if mod_summary else None,
            truncated=mod_omission == "module_config_truncated",
            redacted=mod_redacted,
            omission_reason=mod_omission,
            sent_to_llm=bool(mod_summary),
        )
    )

    budget = context_budget or {}
    if budget:
        manifest.append(
            _manifest_entry(
                "context_budget",
                source="langgraph_context_compact.compact_messages_for_llm",
                included=True,
                payload=budget,
                truncated=bool(budget.get("messages_snipped") or budget.get("observations_compacted")),
                sent_to_llm=False,
            )
        )

    for entry in extra_manifest or []:
        if isinstance(entry, dict):
            manifest.append(entry)

    estimated_chars = estimate_chars(
        {
            "messages": prepared_messages,
            "tools": tool_summaries,
            "module_config": mod_summary,
        }
    )

    return {
        "round": round_idx,
        "snapshot_id": f"snapshot-{round_idx}",
        "model_call_id": f"model-{round_idx}",
        "captured": True,
        "truncated": msg_truncated or tools_truncated,
        "redacted": msg_redacted,
        "omission_reason": "bounded_capture" if msg_truncated or tools_truncated else None,
        "tool_schemas": redact_recursive(copy.deepcopy(tools or []))[0],
        "source": SNAPSHOT_SOURCE,
        "model": resolved_model,
        "provider": resolved_provider,
        "messages": prepared_messages,
        "tools": tool_summaries,
        "manifest": manifest,
        "estimated_chars": estimated_chars,
        "sent_to_llm": True,
        "context_budget": budget,
        "raw_message_count": raw_message_count if raw_message_count is not None else len(messages),
        "llm_message_count": len(prepared_messages),
    }
