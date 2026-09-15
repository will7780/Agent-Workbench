# -*- coding: utf-8 -*-
"""
LangGraph StateGraph ReAct runtime（Phase 1 + Phase 2 上下文压缩与 answer_verify）。
"""

from __future__ import annotations

import copy
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Annotated, Dict, List, Optional, Tuple, Union

from .eval_judge import CallableEvalJudge, EvalJudge, NativeEvalJudge
from .artifact_review import (
    artifact_event, artifact_blocked_observation, bind_artifact_review,
    prepare_artifact_review, resolve_artifact_policy, review_interaction,
    verify_adapter_binding, artifact_revision_matches,
)
from .execution_mode import ExecutionMode, resolve_pipeline_execution_mode
from .configuration import build_llm_config_availability_context
from .knowledge_citation import verify_knowledge_citations
from .langgraph_answer_verify import verify_final_answer
from .langgraph_context_compact import (
    ContextCompactOptions,
    compact_messages_for_llm,
    collect_spilled_artifacts,
    default_context_spill_dir,
)
from .langgraph_human_confirm import (
    human_confirm_blocked_observation,
    human_confirm_required,
)
from .langgraph_context_snapshot import build_langgraph_llm_context_snapshot
from .langgraph_interactions import (
    InteractionRuntimeStore,
    InteractionRuntimeHandle,
    build_clarification_interaction,
    build_confirmation_interaction,
    build_interaction_tool_schemas,
    extract_pending_interaction,
    is_interaction_tool,
    sanitize_interaction_response,
    validate_clarification_tool_call,
)
from .langgraph_shared import (
    DEFAULT_MAX_TOOL_ROUNDS,
    format_langgraph_report_text,
    guard_blocked_observation,
    langgraph_system_prompt,
    observation_to_tool_message,
    pipeline_response,
)
from .tool_schema import build_tool_schemas_from_registry, capability_id, tool_name_to_capability
from .skill_tool_schema import (
    SKILL_LOAD_TOOL,
    SKILL_RESOURCE_LOAD_TOOL,
    SKILLS_LIST_TOOL,
    build_skill_harness_tool_schemas,
    is_skill_harness_tool,
)
from .tools.adapter_contracts import AdapterInput
from .tools.execution_adapters import execute_adapter
from .llm_client import LLMChatWithToolsResult, LLMToolCall, ToolsCompleteFn, chat_completion_with_tools
from .observer import Observer
from .parameter_verification import (
    parameter_validation_observation,
    verify_tool_call_parameters,
)
from .redaction import redact_recursive, redact_text
from .registry import ModuleCapabilityRegistry, get_registry
from .reporter import resolve_execution_modes
from .runtime_telemetry import RuntimeTelemetryCollector
from .schemas import ExecutionStatus
from .state import create_initial_state
from .tool_guard import guard_tool_call

_STATE_GRAPH_AVAILABLE = False
_StateGraph = None
_END = None
_START = None

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore
    from langgraph.types import Overwrite

    _STATE_GRAPH_AVAILABLE = True
    _StateGraph = StateGraph
    _END = END
    _START = START
except ImportError:
    pass


def is_state_graph_available() -> bool:
    return _STATE_GRAPH_AVAILABLE


def _merge_lists(left: Optional[List[Any]], right: Optional[List[Any]]) -> List[Any]:
    return list(left or []) + list(right or [])


if _STATE_GRAPH_AVAILABLE:
    from typing_extensions import TypedDict

    class AgentGraphState(TypedDict, total=False):
        messages: Annotated[List[Dict[str, Any]], _merge_lists]
        llm_input_messages: List[Dict[str, Any]]
        user_request: str
        thread_id: Optional[str]
        execution_mode: str
        module_config: Optional[Dict[str, Any]]
        parameter_sources: Dict[str, str]
        user_confirmed: bool
        confirmation_source: Optional[str]
        available_tools: List[Dict[str, Any]]
        pending_tool_calls: List[Dict[str, Any]]
        artifact_gate: Dict[str, Any]
        artifact_revisions: Dict[str, Any]
        guard_results: List[Dict[str, Any]]
        round_parameter_verification_results: List[Dict[str, Any]]
        parameter_verification_results: Annotated[List[Dict[str, Any]], _merge_lists]
        human_confirm_results: List[Dict[str, Any]]
        round_tool_results: List[Dict[str, Any]]
        tool_calls: Annotated[List[Dict[str, Any]], _merge_lists]
        observations: Annotated[List[Dict[str, Any]], _merge_lists]
        completed_steps: Annotated[List[Dict[str, Any]], _merge_lists]
        failed_steps: Annotated[List[Dict[str, Any]], _merge_lists]
        agent_trace: Annotated[List[Dict[str, Any]], _merge_lists]
        llm_context_snapshots: Annotated[List[Dict[str, Any]], _merge_lists]
        memory_recall: Dict[str, Any]
        session_search: Dict[str, Any]
        memory_profile: Dict[str, Any]
        memory_review: Dict[str, Any]
        memory_context_injected: bool
        memory_context_manifest: List[Dict[str, Any]]
        knowledge_search: Dict[str, Any]
        knowledge_context_manifest: List[Dict[str, Any]]
        knowledge_context_injected: bool
        company_profile: Dict[str, Any]
        company_profile_snapshot_id: Optional[str]
        company_profile_binding_id: Optional[str]
        skill_catalog: Dict[str, Any]
        skill_suggestions: List[Dict[str, Any]]
        loaded_skills: List[Dict[str, Any]]
        skill_resources: List[Dict[str, Any]]
        skill_context_manifest: List[Dict[str, Any]]
        skill_conflicts: List[Dict[str, Any]]
        skill_errors: List[Dict[str, Any]]
        deferred_business_tool_calls: List[Dict[str, Any]]
        run_id: Optional[str]
        artifacts: Annotated[List[Dict[str, Any]], _merge_lists]
        errors: Annotated[List[Dict[str, Any]], _merge_lists]
        final_response: Optional[str]
        next_actions: List[str]
        execution_status: str
        running_summary: str
        round_idx: int
        llm_provider: Optional[str]
        llm_model: Optional[str]
        goal: str
        requested_execution_mode: str
        agent_mode: str
        graph_runtime: str
        context_budget: Dict[str, Any]
        context_budget_history: Annotated[List[Dict[str, Any]], _merge_lists]
        verification: Dict[str, Any]
        pending_interaction: Optional[Dict[str, Any]]
        interaction_history: Annotated[List[Dict[str, Any]], _merge_lists]
        interactions_enabled: bool
        checkpoint_thread_id: Optional[str]
        post_model_route: Optional[str]
        runtime_telemetry: Dict[str, Any]
        resource_usage: Dict[str, Any]


@dataclass
class LangGraphRuntimeConfig:
    registry: ModuleCapabilityRegistry
    execution_mode: ExecutionMode
    module_config: Optional[Dict[str, Any]]
    user_confirmed: bool
    llm_model: Optional[str]
    llm_complete_fn: Optional[ToolsCompleteFn]
    max_tool_rounds: int
    safe_request: str
    requested_mode: str
    confirmation_source: Optional[str] = None
    context_spill_dir: Optional[Any] = None
    knowledge_config: Optional[Dict[str, Any]] = None
    company_skill_runtime: Dict[str, Any] = field(default_factory=dict)
    observer: Observer = field(default_factory=Observer)
    interactions_enabled: bool = False
    interaction_runtime_store: Optional[Any] = None
    checkpoint_thread_id: Optional[str] = None
    progress_callback: Optional[Any] = None
    parameter_sources: Dict[str, str] = field(default_factory=dict)
    parameter_judge: Optional[EvalJudge] = None
    telemetry: RuntimeTelemetryCollector = field(default_factory=RuntimeTelemetryCollector)
    session_search_retriever: Optional[Any] = None
    session_ledger: Optional[Any] = None
    memory_dir: Optional[Union[str, Path]] = None
    history_messages: List[Dict[str, Any]] = field(default_factory=list)


def _emit_runtime_progress(config: LangGraphRuntimeConfig, event: Dict[str, Any]) -> None:
    callback = config.progress_callback
    if callback is None:
        return
    safe, _ = redact_recursive(event)
    try:
        callback(safe if isinstance(safe, dict) else {})
    except Exception:
        # Product progress reporting is observational and must never stop the Agent.
        return


def _trace_route(from_node: str, target: str, reason: str) -> Dict[str, Any]:
    return {
        "type": "route_decision",
        "from": from_node,
        "target": target,
        "reason": reason,
    }


def _trace_node(node_name: str, summary: str, **extra: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "type": "graph_node",
        "node": node_name,
        "summary": summary,
    }
    entry.update(extra)
    return entry


def _risk_level_label(registry: ModuleCapabilityRegistry, module: Any, action: Any) -> str:
    if not module or not action:
        return "L4"
    try:
        risk = registry.get_risk_level(str(module), str(action))
        return f"L{int(risk)}"
    except Exception:
        return "L4"


def _strip_interrupt(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(graph_state, dict):
        return {}
    return {key: value for key, value in graph_state.items() if key != "__interrupt__"}


def _interaction_error_pipeline(
    error_type: str,
    *,
    safe_request: str = "",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    error_entry = {"type": error_type, "message": error_type}
    state = create_initial_state(safe_request)
    state.update(
        {
            "agent_mode": "langgraph",
            "graph_runtime": "state_graph",
            "execution_status": ExecutionStatus.STOPPED.value,
            "errors": [error_entry],
            "run_id": run_id,
            "running_summary": error_type,
            "final_response": None,
            "next_actions": ["review_agent_trace"],
        }
    )
    report = {
        "goal": safe_request,
        "status": ExecutionStatus.STOPPED.value,
        "agent_mode": "langgraph",
        "graph_runtime": "state_graph",
        "running_summary": error_type,
        "errors": [error_entry],
        "final_response": None,
        "run_id": run_id,
        "next_actions": ["review_agent_trace"],
        "tool_calls": [],
        "observations": [],
        "agent_trace": [],
    }
    return pipeline_response(
        state=state,
        report=report,
        report_text=format_langgraph_report_text(report),
    )


def _tool_call_from_pending(entry: Dict[str, Any]) -> LLMToolCall:
    return LLMToolCall(
        id=str(entry.get("tool_call_id") or entry.get("id") or "call_unknown"),
        name=str(entry.get("tool_name") or ""),
        arguments=dict(entry.get("arguments") or {}),
    )


def _assistant_message_from_llm(llm_result: LLMChatWithToolsResult) -> Optional[Dict[str, Any]]:
    if llm_result.raw_message:
        return {key: llm_result.raw_message[key] for key in ("role", "content", "tool_calls") if key in llm_result.raw_message}
    if llm_result.has_tool_calls:
        return {
            "role": "assistant",
            "content": llm_result.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments or {}, ensure_ascii=False),
                    },
                }
                for tc in llm_result.tool_calls
            ],
        }
    if llm_result.content:
        return {"role": "assistant", "content": llm_result.content}
    return None


def _skill_dedupe_key(item: Dict[str, Any], *, resource_path: Optional[str] = None) -> Tuple[str, str, str, str]:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else item
    ref = meta.get("ref") if isinstance(meta.get("ref"), dict) else {}
    return (
        str(ref.get("stable_id") or ""),
        str(ref.get("version") or ""),
        str(ref.get("checksum") or ""),
        str(resource_path if resource_path is not None else item.get("resource_path") or ""),
    )


def _json_tool_content(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _build_compiled_graph(
    config: LangGraphRuntimeConfig,
    *,
    checkpointer: Optional[Any] = None,
) -> Any:
    if not _STATE_GRAPH_AVAILABLE or _StateGraph is None:
        raise ImportError("langgraph is not installed")

    graph = _StateGraph(AgentGraphState)

    def intake_node(state: AgentGraphState) -> Dict[str, Any]:
        from .session_checkpoint import opaque_company_profile_binding_id
        from .skill_models import select_company_skill_pipeline_fields

        reg = config.registry
        tools = build_tool_schemas_from_registry(
            reg,
            execution_mode=config.execution_mode,
            module_config=config.module_config,
        )
        skill_fields = select_company_skill_pipeline_fields(config.company_skill_runtime)
        profile = skill_fields.get("company_profile") or {}
        if profile.get("enabled"):
            tools = list(tools) + build_skill_harness_tool_schemas()
        if config.interactions_enabled:
            tools = list(tools) + build_interaction_tool_schemas()
        messages = [
            {
                "role": "system",
                "content": langgraph_system_prompt(reg, interactions_enabled=config.interactions_enabled),
            },
        ]
        context_builder = getattr(reg, "config_context_builder", None) or build_llm_config_availability_context
        runtime_config_hint = context_builder(config.module_config)
        if runtime_config_hint:
            messages.append({"role": "system", "content": runtime_config_hint})
        messages.extend(config.history_messages)
        messages.append({"role": "user", "content": state.get("user_request") or config.safe_request})
        binding_id = opaque_company_profile_binding_id(config.company_skill_runtime)
        return {
            "messages": messages,
            "llm_input_messages": list(messages),
            "available_tools": tools,
            "round_idx": 0,
            "execution_status": ExecutionStatus.RUNNING.value,
            "interactions_enabled": config.interactions_enabled,
            "checkpoint_thread_id": config.checkpoint_thread_id,
            "pending_interaction": None,
            "agent_trace": [
                _trace_node("intake_node", "initialized messages and tool catalog", tool_count=len(tools)),
            ],
            **skill_fields,
            "company_profile_binding_id": binding_id,
        }

    def resolve_company_profile_node(state: AgentGraphState) -> Dict[str, Any]:
        from .skill_models import select_company_skill_pipeline_fields

        skill_fields = select_company_skill_pipeline_fields(config.company_skill_runtime)
        profile = skill_fields.get("company_profile") or {}
        return {
            "company_profile": profile,
            "company_profile_snapshot_id": skill_fields.get("company_profile_snapshot_id"),
            "agent_trace": [
                _trace_node(
                    "resolve_company_profile_node",
                    "reused frozen company profile snapshot",
                    enabled=bool(profile.get("enabled")),
                    snapshot_id=skill_fields.get("company_profile_snapshot_id"),
                    error_type=profile.get("error_type"),
                ),
            ],
        }

    def skill_discover_node(state: AgentGraphState) -> Dict[str, Any]:
        from .skill_models import select_company_skill_pipeline_fields

        skill_fields = select_company_skill_pipeline_fields(config.company_skill_runtime)
        catalog = dict(skill_fields.get("skill_catalog") or {})
        suggestions: List[Dict[str, Any]] = []
        for item in catalog.get("skills") or []:
            if not isinstance(item, dict):
                continue
            ref = item.get("ref") if isinstance(item.get("ref"), dict) else {}
            suggestions.append(
                {
                    "name": ref.get("name"),
                    "version": ref.get("version"),
                    "checksum": ref.get("checksum"),
                    "stable_id": ref.get("stable_id"),
                    "description": item.get("description"),
                }
            )
        return {
            "skill_catalog": catalog,
            "skill_suggestions": suggestions,
            "skill_conflicts": list(skill_fields.get("skill_conflicts") or catalog.get("conflicts") or []),
            "skill_errors": list(skill_fields.get("skill_errors") or catalog.get("errors") or []),
            "agent_trace": [
                _trace_node(
                    "skill_discover_node",
                    f"catalog_skills={len(catalog.get('skills') or [])}",
                    enabled=bool(catalog.get("enabled")),
                    degraded=bool(catalog.get("degraded")),
                    suggestion_count=len(suggestions),
                ),
            ],
        }

    def retrieve_context_node(state: AgentGraphState) -> Dict[str, Any]:
        from .knowledge_provider import empty_knowledge_search
        from .knowledge_registry import retrieve_knowledge_for_langgraph
        from .memory_profile import build_memory_profile_from_recall, empty_memory_profile
        from .memory_retriever import recall_memories_for_langgraph
        from .session_ledger import retrieve_session_search

        user_request = state.get("user_request") or config.safe_request
        session_search = retrieve_session_search(
            user_request,
            thread_id=state.get("thread_id"),
            company_profile_binding_id=state.get("company_profile_binding_id"),
            exclude_run_id=state.get("run_id"),
            retriever=config.session_search_retriever,
        ).to_dict()

        try:
            recall_result = recall_memories_for_langgraph(
                user_request=user_request,
                module_config=state.get("module_config") or config.module_config,
                session_record=None,
                memory_dir=config.memory_dir,
            )
            memory_profile = build_memory_profile_from_recall(
                recall_result,
                session_search=session_search,
            )
            memory_recall = dict(recall_result.summary or {})
            memory_recall["session_search_used"] = session_search.get("used")
            memory_recall["profile_enabled"] = True
        except Exception as exc:
            memory_profile = empty_memory_profile(error_type=f"memory_recall_failed:{exc}")
            memory_recall = memory_profile.get("summary") or {
                "enabled": False,
                "used": False,
                "error_type": f"memory_recall_failed:{exc}",
            }
            memory_recall["session_search_used"] = session_search.get("used")

        try:
            knowledge_search = retrieve_knowledge_for_langgraph(
                user_request,
                config.knowledge_config,
                request_id=state.get("run_id"),
            )
        except Exception as exc:
            knowledge_search = empty_knowledge_search(
                enabled=bool(config.knowledge_config),
                query=user_request,
                error_type=f"knowledge_provider_failed:{type(exc).__name__}",
                omission_reason="provider_error",
                degraded=True,
            )

        retrieved_sessions = session_search.get("retrieved_sessions") or []
        injected_count = len(memory_profile.get("injected_memories") or [])
        knowledge_hits = knowledge_search.get("hits") or []
        return {
            "memory_recall": memory_recall,
            "session_search": session_search,
            "memory_profile": memory_profile,
            "knowledge_search": knowledge_search,
            "agent_trace": [
                _trace_node(
                    "retrieve_context_node",
                    (
                        f"session_hits={len(retrieved_sessions)} memory_inject={injected_count} "
                        f"knowledge_hits={len(knowledge_hits)}"
                    ),
                    session_search_used=session_search.get("used"),
                    retrieved_session_count=len(retrieved_sessions),
                    memory_injected_count=injected_count,
                    memory_conflicts=len(memory_profile.get("conflicts") or []),
                    knowledge_enabled=knowledge_search.get("enabled"),
                    knowledge_hit_count=len(knowledge_hits),
                    knowledge_error_type=knowledge_search.get("error_type"),
                    knowledge_degraded=knowledge_search.get("degraded"),
                ),
            ],
        }

    def context_prepare_node(state: AgentGraphState) -> Dict[str, Any]:
        from .skill_context import inject_loaded_skills_into_messages, inject_skill_catalog_into_messages

        messages = list(state.get("messages") or [])
        round_idx = int(state.get("round_idx") or 0)
        memory_manifest: List[Dict[str, Any]] = list(state.get("memory_context_manifest") or [])
        memory_injected = bool(state.get("memory_context_injected"))
        knowledge_manifest: List[Dict[str, Any]] = list(state.get("knowledge_context_manifest") or [])
        knowledge_injected = bool(state.get("knowledge_context_injected"))
        knowledge_search = dict(state.get("knowledge_search") or {})
        skill_manifest: List[Dict[str, Any]] = list(state.get("skill_context_manifest") or [])
        user_request = state.get("user_request") or config.safe_request

        if round_idx == 0 and not memory_injected:
            from .memory_profile import inject_memory_into_messages

            messages, injected_manifest, injected_flag = inject_memory_into_messages(
                messages,
                memory_profile=state.get("memory_profile") or {},
                user_request=user_request,
            )
            if injected_flag:
                memory_manifest = injected_manifest
                memory_injected = True

        if round_idx == 0 and not knowledge_injected:
            from .knowledge_provider import DEFAULT_MAX_CONTEXT_CHARS, inject_knowledge_into_messages, normalize_knowledge_config

            cfg = normalize_knowledge_config(config.knowledge_config)
            max_chars = int(cfg.get("max_context_chars") or DEFAULT_MAX_CONTEXT_CHARS)
            messages, k_manifest, k_flag, knowledge_search = inject_knowledge_into_messages(
                messages,
                knowledge_search=knowledge_search,
                max_context_chars=max_chars,
                user_request=user_request,
            )
            knowledge_manifest = k_manifest
            knowledge_injected = k_flag

        catalog_already = any(
            isinstance(item, dict) and item.get("section") == "skill_catalog" for item in skill_manifest
        )
        if round_idx == 0 and not catalog_already:
            messages, catalog_manifest, _catalog_flag = inject_skill_catalog_into_messages(
                messages,
                profile=state.get("company_profile") or {},
                catalog=state.get("skill_catalog") or {},
            )
            skill_manifest = list(skill_manifest) + list(catalog_manifest)

        messages, loaded_manifest, _loaded_flag = inject_loaded_skills_into_messages(
            messages,
            loaded_skills=list(state.get("loaded_skills") or []),
            skill_resources=list(state.get("skill_resources") or []),
            existing_manifest=skill_manifest,
        )
        if loaded_manifest:
            skill_manifest = list(skill_manifest) + list(loaded_manifest)

        spill_dir = config.context_spill_dir or default_context_spill_dir()
        compacted, budget = compact_messages_for_llm(
            messages,
            options=ContextCompactOptions(
                spill_dir=spill_dir,
                round_idx=round_idx,
                user_request=user_request,
                running_summary=state.get("running_summary") or "",
            ),
        )
        trace_extra: Dict[str, Any] = {
            "llm_message_count": len(compacted),
            "raw_message_count": len(messages),
            "memory_context_injected": memory_injected,
            "knowledge_context_injected": knowledge_injected,
            "skill_manifest_sections": len(skill_manifest),
        }
        if memory_manifest:
            trace_extra["memory_manifest_sections"] = len(memory_manifest)
        if knowledge_manifest:
            trace_extra["knowledge_manifest_sections"] = len(knowledge_manifest)
        return {
            "messages": Overwrite(messages),
            "memory_context_injected": memory_injected,
            "memory_context_manifest": memory_manifest,
            "knowledge_search": knowledge_search,
            "knowledge_context_injected": knowledge_injected,
            "knowledge_context_manifest": knowledge_manifest,
            "skill_context_manifest": skill_manifest,
            "llm_input_messages": compacted,
            "context_budget": budget,
            "context_budget_history": [budget],
            "agent_trace": [
                _trace_node(
                    "context_prepare_node",
                    f"compressed messages {budget.get('chars_before')} -> {budget.get('chars_after')} chars",
                    **{k: budget.get(k) for k in (
                        "messages_snipped",
                        "observations_compacted",
                        "artifacts_spilled",
                        "strategies_applied",
                    )},
                    **trace_extra,
                ),
            ],
        }

    def model_decide_node(state: AgentGraphState) -> Dict[str, Any]:
        round_idx = int(state.get("round_idx") or 0)
        raw_messages = list(state.get("messages") or [])
        llm_messages = list(state.get("llm_input_messages") or raw_messages)
        tools = state.get("available_tools") or []
        context_budget = dict(state.get("context_budget") or {})

        extra_manifest = (
            list(state.get("memory_context_manifest") or [])
            + list(state.get("knowledge_context_manifest") or [])
            + list(state.get("skill_context_manifest") or [])
        )
        snapshot = build_langgraph_llm_context_snapshot(
            round_idx,
            llm_messages,
            tools,
            module_config=config.module_config,
            model=config.llm_model or state.get("llm_model"),
            provider=state.get("llm_provider"),
            registry=config.registry,
            context_budget=context_budget,
            raw_message_count=len(raw_messages),
            extra_manifest=extra_manifest,
        )

        llm_result: LLMChatWithToolsResult = chat_completion_with_tools(
            llm_messages,
            tools=tools,
            model_override=config.llm_model,
            complete_fn=config.llm_complete_fn,
        )
        config.telemetry.record_llm_result("agent", llm_result)

        safe_content_preview, _ = redact_recursive(llm_result.content or "")
        model_trace: Dict[str, Any] = {
            "type": "model_call",
            "round": round_idx,
            "finish_reason": llm_result.finish_reason,
            "has_tool_calls": llm_result.has_tool_calls,
            "content_preview": safe_content_preview[:240],
            "tool_call_names": [tc.name for tc in llm_result.tool_calls],
            "error_type": llm_result.error_type,
        }

        updates: Dict[str, Any] = {
            "llm_context_snapshots": [snapshot],
            "agent_trace": [model_trace],
            "llm_provider": llm_result.provider,
            "llm_model": llm_result.model,
            "pending_tool_calls": [],
            "guard_results": [],
        }

        if llm_result.error_type:
            updates["execution_status"] = ExecutionStatus.STOPPED.value
            updates["errors"] = [
                {"type": llm_result.error_type, "message": llm_result.error_message},
            ]
            updates["running_summary"] = llm_result.error_message or llm_result.error_type
            return updates

        new_messages: List[Dict[str, Any]] = []
        assistant_message = _assistant_message_from_llm(llm_result)
        if assistant_message:
            new_messages.append(assistant_message)

        if not llm_result.has_tool_calls:
            updates["messages"] = new_messages
            updates["final_response"] = llm_result.content or ""
            updates["execution_status"] = ExecutionStatus.COMPLETED.value
            updates["running_summary"] = llm_result.content or "Agent completed without tool calls."
            return updates

        pending: List[Dict[str, Any]] = []
        for tc_idx, tc in enumerate(llm_result.tool_calls):
            pending.append(
                {
                    "round": round_idx,
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "step_id": f"lg_{round_idx}_{tc_idx}",
                }
            )
        updates["messages"] = new_messages
        updates["pending_tool_calls"] = pending
        _emit_runtime_progress(
            config,
            {
                "type": "tool_selected",
                "execution_mode": config.execution_mode.value,
                "tool_calls": pending,
            },
        )
        return updates

    def post_model_validate_node(state: AgentGraphState) -> Dict[str, Any]:
        pending = list(state.get("pending_tool_calls") or [])
        ask_calls = [entry for entry in pending if is_interaction_tool(entry.get("tool_name") or "")]
        trace_extra: Dict[str, Any] = {"pending_count": len(pending)}
        unknown_names = [
            entry.get("tool_name") or ""
            for entry in pending
            if not is_skill_harness_tool(entry.get("tool_name") or "")
            and not is_interaction_tool(entry.get("tool_name") or "")
            and not tool_name_to_capability(entry.get("tool_name") or "")
        ]
        if unknown_names:
            trace_extra["unknown_tool_names"] = unknown_names
        harness_count = sum(1 for entry in pending if is_skill_harness_tool(entry.get("tool_name") or ""))
        if ask_calls and (len(ask_calls) != 1 or len(pending) != 1):
            error_messages = [
                {
                    "role": "tool",
                    "tool_call_id": entry.get("tool_call_id"),
                    "content": json.dumps(
                        {
                            "error_type": "interaction_tool_must_be_single",
                            "message": "agent__ask_user must be the only tool call in a turn",
                        },
                        ensure_ascii=False,
                    ),
                }
                for entry in pending
            ]
            return {
                "messages": error_messages,
                "pending_tool_calls": [],
                "pending_interaction": None,
                "post_model_route": "context_prepare",
                "round_idx": int(state.get("round_idx") or 0) + 1,
                "execution_status": ExecutionStatus.RUNNING.value,
                "agent_trace": [
                    _trace_node(
                        "post_model_validate_node",
                        "rejected mixed or non-single interaction tool calls",
                        **trace_extra,
                        harness_count=harness_count,
                        error_type="interaction_tool_must_be_single",
                    ),
                ],
            }
        if len(ask_calls) == 1 and len(pending) == 1:
            interaction_error = validate_clarification_tool_call(ask_calls[0])
            if interaction_error:
                entry = ask_calls[0]
                return {
                    "messages": [
                        {
                            "role": "tool",
                            "tool_call_id": entry.get("tool_call_id"),
                            "content": json.dumps(
                                {
                                    "error_type": interaction_error,
                                    "message": "agent__ask_user arguments are invalid",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    ],
                    "pending_tool_calls": [],
                    "pending_interaction": None,
                    "post_model_route": "context_prepare",
                    "round_idx": int(state.get("round_idx") or 0) + 1,
                    "execution_status": ExecutionStatus.RUNNING.value,
                    "agent_trace": [
                        _trace_node(
                            "post_model_validate_node",
                            "rejected invalid interaction tool arguments",
                            **trace_extra,
                            error_type=interaction_error,
                        ),
                    ],
                }
        route = "user_input" if len(ask_calls) == 1 and len(pending) == 1 else None
        return {
            "pending_tool_calls": pending,
            "post_model_route": route,
            "agent_trace": [
                _trace_node(
                    "post_model_validate_node",
                    f"validated {len(pending)} tool calls harness={harness_count}",
                    **trace_extra,
                    harness_count=harness_count,
                ),
            ],
        }

    def user_input_node(state: AgentGraphState) -> Dict[str, Any]:
        from langgraph.types import interrupt

        pending = list(state.get("pending_tool_calls") or [])
        entry = pending[0] if pending else {}
        payload = build_clarification_interaction(str(state.get("run_id") or ""), entry)
        _emit_runtime_progress(
            config,
            {
                "type": "interaction_required",
                "execution_mode": config.execution_mode.value,
                "pending_interaction": payload,
            },
        )
        config.telemetry.pause()
        resume_value = interrupt(payload)
        config.telemetry.resume()
        sanitized, error_type = sanitize_interaction_response(
            resume_value if isinstance(resume_value, dict) else {},
            payload,
        )
        if error_type or not sanitized:
            raise RuntimeError(error_type or "interaction_response_invalid")
        answer = str(sanitized.get("answer") or "")
        history_item = {
            "interaction_id": payload.get("interaction_id"),
            "type": "clarification",
            "status": "answered",
            "question": payload.get("question"),
            "answer": answer,
        }
        return {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": entry.get("tool_call_id"),
                    "content": answer,
                }
            ],
            "interaction_history": [history_item],
            "pending_tool_calls": [],
            "pending_interaction": None,
            "post_model_route": None,
            "round_idx": int(state.get("round_idx") or 0) + 1,
            "execution_status": ExecutionStatus.RUNNING.value,
            "agent_trace": [
                _trace_node("user_input_node", "received clarification answer"),
            ],
        }

    def tool_guard_node(state: AgentGraphState) -> Dict[str, Any]:
        pending = [
            entry
            for entry in list(state.get("pending_tool_calls") or [])[:1]
            if not is_skill_harness_tool(entry.get("tool_name") or "")
        ]
        guard_results: List[Dict[str, Any]] = []
        trace_entries: List[Dict[str, Any]] = []
        for entry in pending:
            tc = _tool_call_from_pending(entry)
            guard = guard_tool_call(
                tc.name,
                tc.arguments,
                registry=config.registry,
                execution_mode=config.execution_mode,
                module_config=config.module_config,
            )
            guard_entry = {
                "type": "guard_result",
                "tool_call_id": tc.id,
                "tool_name": tc.name,
                "step_id": entry.get("step_id"),
                **guard.to_dict(),
            }
            guard_results.append(guard_entry)
            trace_entries.append(guard_entry)
        _emit_runtime_progress(
            config,
            {
                "type": "tool_guarded",
                "execution_mode": config.execution_mode.value,
                "tool_calls": pending,
                "guard_results": guard_results,
            },
        )
        return {"guard_results": guard_results, "agent_trace": trace_entries}

    def parameter_verify_node(state: AgentGraphState) -> Dict[str, Any]:
        pending = list(state.get("pending_tool_calls") or [])[:1]
        guard_results = list(state.get("guard_results") or [])
        results: List[Dict[str, Any]] = []
        trace_entries: List[Dict[str, Any]] = []
        for entry, guard_entry in zip(pending, guard_results):
            tc = _tool_call_from_pending(entry)
            step_id = str(entry.get("step_id") or "")
            parsed = tool_name_to_capability(tc.name)
            if not guard_entry.get("allowed") or parsed is None:
                verification = {
                    "type": "parameter_verification_result",
                    "tool_call_id": tc.id,
                    "step_id": step_id,
                    "tool_name": tc.name,
                    "capability_id": capability_id(parsed[0], parsed[1]) if parsed else "",
                    "applicable": False,
                    "schema_pass": False,
                    "allowed": False,
                    "requires_confirmation": False,
                    "semantic_check_required": False,
                    "intent_decision": None,
                    "intent_score": None,
                    "method": "guard_blocked",
                    "parameter_snapshot_hash": None,
                    "parameter_summary": {},
                    "parameter_sources": {},
                    "source_conflicts": [],
                    "source_conflict_count": 0,
                    "schema_errors": [],
                    "issue_codes": ["guard_blocked"],
                    "evidence_refs": [],
                    "confirmed": False,
                }
            else:
                verification = verify_tool_call_parameters(
                    tool_call_id=tc.id,
                    step_id=step_id,
                    tool_name=tc.name,
                    module=str(guard_entry.get("module") or parsed[0]),
                    action=str(guard_entry.get("action") or parsed[1]),
                    arguments=tc.arguments,
                    resolved_params=guard_entry.get("params") or tc.arguments,
                    user_request=state.get("user_request") or config.safe_request,
                    execution_mode=config.execution_mode,
                    registry=config.registry,
                    module_config=state.get("module_config") or config.module_config,
                    parameter_sources=state.get("parameter_sources") or config.parameter_sources,
                    recent_dialogue=list(state.get("messages") or [])[-8:],
                    semantic_mode="auto",
                    judge=config.parameter_judge,
                ).to_dict()
            results.append(verification)
            trace_entries.append(dict(verification))
        _emit_runtime_progress(
            config,
            {
                "type": "parameter_verified",
                "execution_mode": config.execution_mode.value,
                "parameter_verification_results": results,
            },
        )
        return {
            "round_parameter_verification_results": results,
            "agent_trace": trace_entries,
        }

    def current_artifact_policy(state, guard):
        profile = config.company_skill_runtime.get("company_profile") or {}
        company = (profile.get("snapshot") or profile).get("artifact_policies") or {}
        return resolve_artifact_policy(config.registry, guard.get("module", ""), guard.get("action", ""), company)

    def artifact_check_node(state: AgentGraphState) -> Dict[str, Any]:
        entry = (state.get("pending_tool_calls") or [{}])[0]
        guard = (state.get("guard_results") or [{}])[0]
        verification = (state.get("round_parameter_verification_results") or [{}])[0]
        policy = current_artifact_policy(state, guard)
        if not policy.get("required") or not guard.get("allowed") or not verification.get("schema_pass"):
            return {"artifact_gate": {}}
        revisions = dict(state.get("artifact_revisions") or {})
        gate = prepare_artifact_review(
            guard.get("params") or entry.get("arguments") or {}, state.get("artifacts") or [], policy,
            run_id=state.get("run_id"), tool_call_id=entry.get("tool_call_id"),
            step_id=entry.get("step_id"), execution_mode=config.execution_mode,
            staging_root=(Path(config.context_spill_dir) / "review") if config.context_spill_dir else None,
            registry=config.registry,
        )
        previous = revisions.get(gate["artifact_id"])
        if previous:
            gate["version"] = previous["version"] + int(previous.get("revision") != gate.get("revision"))
        revisions[gate["artifact_id"]] = {k: gate.get(k) for k in ("version", "revision")}
        events = [artifact_event("artifact.check", gate)]
        if gate.get("complete"):
            events.insert(0, artifact_event("artifact.created", gate))
            events.append(artifact_event("artifact.sample", gate, sample=gate["sample"]))
        for event in events:
            _emit_runtime_progress(config, event)
        return {"artifact_gate": gate, "artifact_revisions": revisions, "agent_trace": events}

    def artifact_review_node(state: AgentGraphState) -> Dict[str, Any]:
        gate = copy.deepcopy(state.get("artifact_gate") or {})
        if not gate or not gate.get("complete"):
            return {}
        if not config.interactions_enabled:
            gate["error_type"] = "artifact_review_runtime_required"
            return {"artifact_gate": gate}
        from langgraph.types import interrupt
        payload = review_interaction(gate)
        config.telemetry.pause()
        _emit_runtime_progress(config, {"type": "interaction_required", "pending_interaction": payload})
        response = interrupt(payload)
        config.telemetry.resume()
        sanitized, error = sanitize_interaction_response(response, payload)
        if error:
            raise RuntimeError(error)
        gate["decision"] = sanitized["decision"]
        gate["comment"] = sanitized.get("comment", "")
        guard = (state.get("guard_results") or [{}])[0]
        if gate["decision"] == "approve" and not artifact_revision_matches(
                gate, guard.get("params") or {}, current_artifact_policy(state, guard)):
            gate.update(decision="invalidated", error_type="artifact_approval_invalidated")
        event = artifact_event("artifact.review", gate, decision=gate["decision"], comment=gate["comment"])
        _emit_runtime_progress(config, event)
        return {"artifact_gate": gate, "agent_trace": [event], "interaction_history": [
            {"interaction_id": gate["review_id"], "type": "artifact_review", "decision": gate["decision"],
             "status": gate["decision"], "resolved": True}], "pending_interaction": None}

    def artifact_bind_node(state: AgentGraphState) -> Dict[str, Any]:
        gate = copy.deepcopy(state.get("artifact_gate") or {})
        if not gate or gate.get("decision") != "approve":
            return {}
        guard = (state.get("guard_results") or [{}])[0]
        entry = (state.get("pending_tool_calls") or [{}])[0]
        current_guard = guard_tool_call(entry["tool_name"], entry["arguments"], registry=config.registry,
                                       execution_mode=config.execution_mode, module_config=config.module_config)
        gate["capability"] = f"{guard.get('module')}.{guard.get('action')}"
        profile = config.company_skill_runtime.get("company_profile") or {}
        gate["company_policy"] = (profile.get("snapshot") or profile).get("artifact_policies") or {}
        gate = bind_artifact_review(gate, current_guard.params, current_artifact_policy(state, guard))
        if not current_guard.allowed:
            gate.update(error_type="artifact_approval_invalidated", decision=None)
        events = []
        if gate.get("error_type"):
            events.append(artifact_event("artifact.review", gate, decision="invalidated", error_type=gate["error_type"]))
        return {"artifact_gate": gate, "agent_trace": events}

    def human_confirm_node(state: AgentGraphState) -> Dict[str, Any]:
        pending = list(state.get("pending_tool_calls") or [])[:1]
        gate = state.get("artifact_gate") or {}
        if gate and (gate.get("decision") != "approve" or not gate.get("valid")):
            return {"human_confirm_results": [{"allowed": True, "skipped": True}],
                    "parameter_verification_results": list(state.get("round_parameter_verification_results") or [])}
        guard_results = list(state.get("guard_results") or [])
        verification_results = list(state.get("round_parameter_verification_results") or [])
        confirmation_source = state.get("confirmation_source") or config.confirmation_source
        already_confirmed = False if gate else bool(state.get("user_confirmed") or config.user_confirmed)

        computed: List[Dict[str, Any]] = []
        finalized_verifications: List[Dict[str, Any]] = []
        confirm_needed_ids: List[str] = []
        confirm_pending: List[Dict[str, Any]] = []
        confirm_guards: List[Dict[str, Any]] = []
        confirm_verifications: List[Dict[str, Any]] = []

        for entry, guard_entry, verification_entry in zip(
            pending,
            guard_results,
            verification_results,
        ):
            tc = _tool_call_from_pending(entry)
            step_id = str(entry.get("step_id") or "")
            verification = dict(verification_entry)
            finalized_verifications.append(verification)
            guard_blocked = not guard_entry.get("allowed")
            parameter_blocked = (
                bool(verification.get("applicable"))
                and not verification.get("allowed")
                and not verification.get("requires_confirmation")
            )
            module = guard_entry.get("module")
            action = guard_entry.get("action")
            risk_label = _risk_level_label(config.registry, module, action)
            guard_for_payload = dict(guard_entry)
            guard_for_payload["risk_level"] = risk_label

            if parameter_blocked or (
                guard_blocked
                and config.execution_mode not in (ExecutionMode.LOCAL_WRITE, ExecutionMode.LIVE)
            ):
                computed.append(
                    {
                        "type": "human_confirm_result",
                        "tool_call_id": tc.id,
                        "tool_name": tc.name,
                        "step_id": step_id,
                        "allowed": True,
                        "skipped": True,
                        "reason": "parameter_already_blocked" if parameter_blocked else "guard_already_blocked",
                        "requested_execution_mode": config.execution_mode.value,
                        "user_confirmed": already_confirmed,
                        "confirmation_kind": None,
                        "parameter_snapshot_hash": verification.get("parameter_snapshot_hash"),
                    }
                )
                continue

            risk_needs, risk_error_type, risk_error_message = human_confirm_required(
                execution_mode=config.execution_mode,
                user_confirmed=already_confirmed,
                confirmation_source=confirmation_source,
                module=module,
                action=action,
                registry=config.registry,
            )
            parameter_needs = bool(
                verification.get("requires_confirmation") and not verification.get("confirmed")
            )
            needs = risk_needs or parameter_needs
            if parameter_needs and risk_needs:
                confirmation_kind = "combined"
            elif parameter_needs:
                confirmation_kind = "parameter"
            elif risk_needs:
                confirmation_kind = "risk"
            else:
                confirmation_kind = None
            error_type = risk_error_type
            error_message = risk_error_message
            if parameter_needs and not risk_needs:
                error_type = "parameter_confirmation_required"
                error_message = "parameter intent verification is uncertain and requires explicit confirmation"

            confirm_entry: Dict[str, Any] = {
                "type": "human_confirm_result",
                "tool_call_id": tc.id,
                "tool_name": tc.name,
                "step_id": step_id,
                "allowed": not needs,
                "skipped": False,
                "error_type": error_type,
                "error_message": error_message,
                "requested_execution_mode": config.execution_mode.value,
                "confirmation_source": confirmation_source,
                "guard_blocked": False,
                "requires_human_confirm": bool(guard_entry.get("requires_human_confirm")),
                "requires_parameter_confirm": parameter_needs,
                "mode_status": guard_entry.get("mode_status"),
                "user_confirmed": already_confirmed if not needs else False,
                "risk_level": risk_label,
                "confirmation_kind": confirmation_kind,
                "parameter_snapshot_hash": verification.get("parameter_snapshot_hash"),
            }
            computed.append(confirm_entry)
            if needs and config.interactions_enabled:
                confirm_needed_ids.append(tc.id)
                confirm_pending.append(entry)
                confirm_guards.append(guard_for_payload)
                confirm_verifications.append(verification)

        if confirm_needed_ids:
            from langgraph.types import interrupt

            payload = build_confirmation_interaction(
                str(state.get("run_id") or ""),
                confirm_pending,
                confirm_guards,
                config.execution_mode.value,
                parameter_verification_results=confirm_verifications,
            )
            config.telemetry.pause()
            _emit_runtime_progress(
                config,
                {
                    "type": "interaction_required",
                    "execution_mode": config.execution_mode.value,
                    "pending_interaction": payload,
                },
            )
            resume_value = interrupt(payload)
            config.telemetry.resume()
            sanitized, error_type = sanitize_interaction_response(
                resume_value if isinstance(resume_value, dict) else {},
                payload,
            )
            if error_type or not sanitized:
                raise RuntimeError(error_type or "interaction_response_invalid")
            approved = sanitized.get("decision") == "approve"
            history_status = "approved" if approved else "rejected"

            confirm_results: List[Dict[str, Any]] = []
            for item in computed:
                updated = dict(item)
                if item.get("tool_call_id") in confirm_needed_ids:
                    if approved:
                        updated.update(
                            {
                                "allowed": True,
                                "user_confirmed": True,
                                "confirmation_source": "ui_explicit",
                                "error_type": None,
                                "error_message": None,
                            }
                        )
                    else:
                        updated.update(
                            {
                                "allowed": False,
                                "user_confirmed": False,
                                "error_type": "human_confirmation_rejected",
                                "error_message": f"operator rejected {config.execution_mode.value} tool execution",
                                "confirmation_source": "ui_explicit",
                            }
                        )
                confirm_results.append(updated)

            resolved_verifications: List[Dict[str, Any]] = []
            for item in finalized_verifications:
                updated = dict(item)
                if item.get("tool_call_id") in confirm_needed_ids and item.get("requires_confirmation"):
                    updated["confirmed"] = approved
                    updated["allowed"] = approved
                    if approved:
                        updated["evidence_refs"] = list(
                            dict.fromkeys(
                                list(updated.get("evidence_refs") or [])
                                + [f"interaction:{payload.get('interaction_id')}"]
                            )
                        )
                resolved_verifications.append(updated)
            history_item = {
                "interaction_id": payload.get("interaction_id"),
                "type": "confirmation",
                "confirmation_kind": payload.get("confirmation_kind"),
                "status": history_status,
                "decision": sanitized.get("decision"),
                "parameter_snapshot_hashes": payload.get("parameter_snapshot_hashes") or [],
            }
            return {
                "human_confirm_results": confirm_results,
                "round_parameter_verification_results": resolved_verifications,
                "parameter_verification_results": resolved_verifications,
                "pending_interaction": None,
                "interaction_history": [history_item],
                "agent_trace": [{"type": "interaction.request", **{k: v for k, v in payload.items() if k != "type"}, "interaction_type": "confirmation"},
                                {**history_item, "type": "interaction.response", "interaction_type": "confirmation"}] + confirm_results,
            }

        return {
            "human_confirm_results": computed,
            "round_parameter_verification_results": finalized_verifications,
            "parameter_verification_results": finalized_verifications,
            "agent_trace": [item for item in computed if not item.get("skipped")],
        }

    def tool_execute_node(state: AgentGraphState) -> Dict[str, Any]:
        pending = list(state.get("pending_tool_calls") or [])[:1]
        guard_results = list(state.get("guard_results") or [])
        parameter_results = list(state.get("round_parameter_verification_results") or [])
        human_confirm_results = list(state.get("human_confirm_results") or [])
        tool_calls_log: List[Dict[str, Any]] = []
        completed: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        round_tool_results: List[Dict[str, Any]] = []
        trace_entries: List[Dict[str, Any]] = []

        for entry, guard_entry, parameter_entry, confirm_entry in zip(
            pending,
            guard_results,
            parameter_results,
            human_confirm_results,
        ):
            tc = _tool_call_from_pending(entry)
            step_id = str(entry.get("step_id") or f"lg_{entry.get('round', 0)}_0")
            parsed = tool_name_to_capability(tc.name)
            cap = capability_id(parsed[0], parsed[1]) if parsed else None

            log_entry: Dict[str, Any] = {
                "round": entry.get("round"),
                "step_id": step_id,
                "tool_call_id": tc.id,
                "tool_name": tc.name,
                "capability_id": cap,
                "arguments": tc.arguments,
                "guard": guard_entry,
                "parameter_verification": parameter_entry,
                "human_confirm": confirm_entry,
                "adapter_called": False,
            }

            parameter_blocked = bool(parameter_entry.get("applicable")) and not parameter_entry.get("allowed")
            confirm_blocked = (
                not confirm_entry.get("allowed") and not confirm_entry.get("skipped")
            )
            gate = state.get("artifact_gate") or {}
            if gate and (not gate.get("seal") or gate.get("error_type")):
                error = gate.get("error_type") or ("artifact_review_" + str(gate.get("decision") or "unavailable"))
                obs = artifact_blocked_observation(step_id, error, gate)
                compact = config.observer.observe(obs)
                log_entry["observation_status"] = obs.status
                log_entry["adapter_called"] = False
                tool_calls_log.append(log_entry)
            elif confirm_blocked:
                obs = human_confirm_blocked_observation(step_id, confirm_entry)
                compact = config.observer.observe(obs)
                log_entry["observation_status"] = obs.status
                tool_calls_log.append(log_entry)
                trace_entries.append(
                    _trace_node("tool_execute_node", f"human_confirm blocked {tc.name}", step_id=step_id),
                )
            elif parameter_blocked:
                obs = parameter_validation_observation(step_id, parameter_entry)
                compact = config.observer.observe(obs)
                log_entry["observation_status"] = obs.status
                tool_calls_log.append(log_entry)
                trace_entries.append(
                    _trace_node(
                        "tool_execute_node",
                        f"parameter verification blocked {tc.name}",
                        step_id=step_id,
                        issue_codes=parameter_entry.get("issue_codes") or [],
                    ),
                )
            elif not guard_entry.get("allowed"):
                obs = guard_blocked_observation(step_id, guard_entry)
                compact = config.observer.observe(obs)
                log_entry["observation_status"] = obs.status
                tool_calls_log.append(log_entry)
                trace_entries.append(
                    _trace_node("tool_execute_node", f"guard blocked {tc.name}", step_id=step_id),
                )
            else:
                module = guard_entry.get("module") or (parsed[0] if parsed else "")
                action = guard_entry.get("action") or (parsed[1] if parsed else "")
                adapter_name = config.registry.get_execution_adapter(module, action)
                permission = config.registry.get_risk_level(module, action)
                adapter_input = AdapterInput(
                    artifact_binding=gate if gate else None,
                    run_id=state.get("run_id"),
                    step_id=step_id,
                    module=module,
                    action=action,
                    params=gate.get("bound_params") if gate else (guard_entry.get("params") or tc.arguments),
                    execution_mode=config.execution_mode,
                    dry_run=config.execution_mode == ExecutionMode.DRY_RUN,
                    user_confirmed=bool(confirm_entry.get("user_confirmed")),
                    permission_level=permission,
                    adapter_name=adapter_name,
                )
                _emit_runtime_progress(
                    config,
                    {
                        "type": "tool_running",
                        "execution_mode": config.execution_mode.value,
                        "tool_call": {**entry, "guard": guard_entry, "human_confirm": confirm_entry},
                    },
                )
                tool_span = config.telemetry.start_span(
                    "tool",
                    tc.name,
                    step_id=step_id,
                    capability_id=cap,
                )
                try:
                    binding_error = verify_adapter_binding(adapter_input, config.registry)
                    if binding_error:
                        obs = artifact_blocked_observation(step_id, binding_error)
                        log_entry["adapter_called"] = False
                    else:
                        obs = execute_adapter(adapter_input, registry=config.registry)
                        log_entry["adapter_called"] = True
                        if gate and obs.status in {"success", "dry_run"}:
                            event = artifact_event("artifact.consume", gate, decision="approve")
                            trace_entries.append(event)
                            _emit_runtime_progress(config, event)
                except Exception as exc:
                    config.telemetry.end_span(
                        tool_span,
                        status="failed",
                        error_type=type(exc).__name__,
                    )
                    raise
                config.telemetry.end_span(tool_span, status=obs.status)
                compact = config.observer.observe(obs)
                log_entry["observation_status"] = obs.status
                tool_calls_log.append(log_entry)
                trace_entries.append(
                    _trace_node(
                        "tool_execute_node",
                        f"execute_adapter {tc.name} status={obs.status}",
                        step_id=step_id,
                    ),
                )

            obs_dict = obs.to_dict()
            trace_entries.append({"type": "tool_result", "tool_call": log_entry, "observation": obs_dict})
            _emit_runtime_progress(
                config,
                {
                    "type": "tool_finished",
                    "execution_mode": config.execution_mode.value,
                    "tool_call": log_entry,
                    "observation": obs_dict,
                },
            )
            module = guard_entry.get("module") or (parsed[0] if parsed else None)
            action = guard_entry.get("action") or (parsed[1] if parsed else None)
            step_record = {
                "step_id": step_id,
                "module": module,
                "action": action,
                "tool_name": tc.name,
                "status": obs.status,
                "summary": obs.summary,
                "observation": compact,
                "error": obs.error,
            }
            if obs.status in ("success", "dry_run"):
                completed.append(step_record)
            else:
                failed.append(step_record)

            round_tool_results.append(
                {
                    "tool_call_id": tc.id,
                    "observation": obs_dict,
                    "observation_obj": obs,
                    "tool_call_log": log_entry,
                }
            )

        return {
            "tool_calls": tool_calls_log,
            "completed_steps": completed,
            "failed_steps": failed,
            "round_tool_results": round_tool_results,
            "agent_trace": trace_entries,
        }

    def observe_node(state: AgentGraphState) -> Dict[str, Any]:
        round_results = list(state.get("round_tool_results") or [])
        new_messages: List[Dict[str, Any]] = []
        observations: List[Dict[str, Any]] = []
        running_summary = state.get("running_summary") or ""
        artifacts = list(state.get("artifacts") or [])

        for item in round_results:
            obs_dict = item.get("observation") or {}
            observations.append(obs_dict)
            obs_obj = item.get("observation_obj")
            if obs_obj is not None:
                running_summary = config.observer.append_running_summary(running_summary, obs_obj.summary)
                artifacts = config.observer.collect_artifacts(artifacts, obs_obj)
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("tool_call_id"),
                    "content": observation_to_tool_message(obs_obj) if obs_obj is not None else json.dumps(obs_dict, ensure_ascii=False),
                }
            )

        return {
            "messages": new_messages,
            "observations": observations,
            "artifacts": [a for a in artifacts if a not in (state.get("artifacts") or [])],
            "running_summary": running_summary,
            "round_tool_results": [],
            "pending_tool_calls": list(state.get("pending_tool_calls") or [])[1:],
            "round_parameter_verification_results": [],
            "agent_trace": [
                _trace_node(
                    "observe_node",
                    f"observations={len(observations)}",
                    tool_messages=len(new_messages),
                ),
            ],
        }

    def context_compact_node(state: AgentGraphState) -> Dict[str, Any]:
        round_idx = int(state.get("round_idx") or 0) + 1
        budget = dict(state.get("context_budget") or {})
        updates: Dict[str, Any] = {
            "round_idx": round_idx,
            "pending_tool_calls": [],
            "guard_results": [],
            "agent_trace": [
                _trace_node(
                    "context_compact_node",
                    f"round advanced to {round_idx}",
                    message_count=len(state.get("messages") or []),
                    last_context_budget=budget,
                ),
            ],
        }
        if round_idx >= config.max_tool_rounds:
            updates["execution_status"] = ExecutionStatus.STOPPED.value
            summary = f"max tool rounds ({config.max_tool_rounds}) reached"
            updates["running_summary"] = summary
            updates["errors"] = [{"type": "max_rounds", "message": summary}]
        return updates

    def answer_verify_node(state: AgentGraphState) -> Dict[str, Any]:
        knowledge_search = dict(state.get("knowledge_search") or {})
        verifier = getattr(config.registry, "answer_verifier", None) or verify_final_answer
        verification = verifier(
            final_response=state.get("final_response"),
            tool_calls=list(state.get("tool_calls") or []),
            observations=list(state.get("observations") or []),
            failed_steps=list(state.get("failed_steps") or []),
            completed_steps=list(state.get("completed_steps") or []),
            knowledge_search=knowledge_search,
        )
        citation_verification = verification.get("citation_verification") or verify_knowledge_citations(
            state.get("final_response"),
            knowledge_search,
        )
        knowledge_search["citation_verification"] = citation_verification
        trace_entry: Dict[str, Any] = {
            "type": "answer_verify",
            "passed": verification.get("passed"),
            "issue_count": len(verification.get("issues") or []),
            "issues": verification.get("issues") or [],
            "evidence_step_ids": verification.get("evidence_step_ids") or [],
            "citation_verification": citation_verification,
        }
        next_actions = list(state.get("next_actions") or [])
        if not verification.get("passed"):
            next_actions = list(dict.fromkeys(next_actions + ["review_answer_verification"]))
        return {
            "verification": verification,
            "knowledge_search": knowledge_search,
            "next_actions": next_actions,
            "agent_trace": [trace_entry],
        }

    def finalize_node(state: AgentGraphState) -> Dict[str, Any]:
        from .background_memory_review import review_run_for_memory_candidates

        report_snapshot = {
            "run_id": state.get("run_id"),
            "thread_id": state.get("thread_id"),
            "goal": state.get("user_request") or state.get("goal"),
            "final_response": state.get("final_response"),
            "failed_steps": list(state.get("failed_steps") or []),
            "observations": list(state.get("observations") or []),
            "tool_calls": list(state.get("tool_calls") or []),
            "verification": state.get("verification") or {},
            "business_review": state.get("business_review"),
        }
        memory_review = review_run_for_memory_candidates(
            dict(state),
            report_snapshot,
            memory_dir=config.memory_dir,
        )
        trace_entries: List[Dict[str, Any]] = [
            _trace_node(
                "finalize_node",
                f"status={state.get('execution_status')}",
                final_response=bool(state.get("final_response")),
            ),
        ]
        if memory_review.get("triggered"):
            trace_entries.append(
                {
                    "type": "memory_review",
                    "triggered": True,
                    "candidate_count": len(memory_review.get("candidate_ids") or []),
                    "trigger_reasons": memory_review.get("trigger_reasons") or [],
                }
            )
        return {
            "memory_review": memory_review,
            "agent_trace": trace_entries,
        }

    def skill_guard_node(state: AgentGraphState) -> Dict[str, Any]:
        from .skill_guard import guard_skill_load
        from .skill_models import frozen_profile_snapshot_from_runtime, skill_catalog_result_from_dict

        pending = list(state.get("pending_tool_calls") or [])
        harness = [entry for entry in pending if is_skill_harness_tool(entry.get("tool_name") or "")]
        business = [entry for entry in pending if not is_skill_harness_tool(entry.get("tool_name") or "")]
        snapshot = frozen_profile_snapshot_from_runtime(state) or frozen_profile_snapshot_from_runtime(
            config.company_skill_runtime
        )
        catalog = skill_catalog_result_from_dict(state.get("skill_catalog") or {})
        profile_enabled = bool((state.get("company_profile") or {}).get("enabled"))
        guarded: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []
        errors = list(state.get("skill_errors") or [])
        loaded_stable_ids: List[str] = []
        for item in state.get("loaded_skills") or []:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else item
            ref = meta.get("ref") if isinstance(meta.get("ref"), dict) else {}
            sid = str(ref.get("stable_id") or "").strip()
            if sid and sid not in loaded_stable_ids:
                loaded_stable_ids.append(sid)
        reserved_stable_ids: List[str] = []
        for entry in harness:
            updated = dict(entry)
            tool_name = str(entry.get("tool_name") or "")
            args = dict(entry.get("arguments") or {})
            if tool_name == SKILLS_LIST_TOOL:
                allowed = profile_enabled and snapshot is not None
                updated["skill_guard"] = {
                    "allowed": allowed,
                    "error_type": None if allowed else "company_profile_disabled",
                    "reason": "profile enabled" if allowed else "company profile is disabled",
                    "skill_ref": None,
                }
            elif snapshot is None:
                updated["skill_guard"] = {
                    "allowed": False,
                    "error_type": "company_profile_disabled",
                    "reason": "frozen company profile snapshot is missing",
                    "skill_ref": None,
                }
            else:
                result = guard_skill_load(
                    str(args.get("skill_name") or ""),
                    profile=snapshot,
                    catalog=catalog,
                    requested_version=str(args.get("version") or "").strip() or None,
                    resource_path=str(args.get("resource_path") or "").strip() or None
                    if tool_name == SKILL_RESOURCE_LOAD_TOOL
                    else None,
                    loaded_stable_ids=loaded_stable_ids,
                    reserved_stable_ids=reserved_stable_ids,
                )
                updated["skill_guard"] = result.to_dict()
                if (
                    tool_name == SKILL_LOAD_TOOL
                    and result.allowed
                    and result.skill_ref is not None
                ):
                    sid = str(result.skill_ref.stable_id() or "").strip()
                    if sid and sid not in loaded_stable_ids and sid not in reserved_stable_ids:
                        reserved_stable_ids.append(sid)
            if not updated["skill_guard"].get("allowed") and updated["skill_guard"].get("error_type"):
                errors.append({"error_type": updated["skill_guard"]["error_type"], "tool_name": tool_name})
            traces.append(
                _trace_node(
                    "skill_guard_node",
                    f"{tool_name} allowed={updated['skill_guard'].get('allowed')}",
                    error_type=updated["skill_guard"].get("error_type"),
                    tool_call_id=entry.get("tool_call_id"),
                )
            )
            guarded.append(updated)

        deferred = []
        for entry in business:
            item = dict(entry)
            item["status"] = "deferred"
            item["reason"] = "skill_context_changed"
            item["error_type"] = "skill_load_deferred_business_calls"
            deferred.append(item)
        if deferred:
            traces.append(
                _trace_node(
                    "skill_guard_node",
                    f"deferred {len(deferred)} business tool calls",
                    deferred_count=len(deferred),
                )
            )
            errors.append({"error_type": "skill_load_deferred_business_calls", "count": len(deferred)})

        return {
            "pending_tool_calls": guarded,
            "deferred_business_tool_calls": list(state.get("deferred_business_tool_calls") or []) + deferred,
            "skill_errors": errors,
            "agent_trace": traces,
        }

    def skill_load_node(state: AgentGraphState) -> Dict[str, Any]:
        import time

        from .skill_models import frozen_profile_snapshot_from_runtime, skill_ref_from_dict
        from .skill_registry import build_skill_provider

        pending = list(state.get("pending_tool_calls") or [])
        snapshot = frozen_profile_snapshot_from_runtime(state) or frozen_profile_snapshot_from_runtime(
            config.company_skill_runtime
        )
        catalog = dict(state.get("skill_catalog") or {})
        loaded_skills = list(state.get("loaded_skills") or [])
        skill_resources = list(state.get("skill_resources") or [])
        errors = list(state.get("skill_errors") or [])
        new_messages: List[Dict[str, Any]] = []
        traces: List[Dict[str, Any]] = []
        loaded_keys = {_skill_dedupe_key(item) for item in loaded_skills}
        resource_keys = {_skill_dedupe_key(item) for item in skill_resources}
        provider = None
        if snapshot is not None:
            provider, provider_error = build_skill_provider(snapshot, registry=config.registry)
            if provider_error and provider is None:
                errors.append({"error_type": provider_error})

        def _existing_skill(ref_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            key = (
                str((ref_dict or {}).get("stable_id") or ""),
                str((ref_dict or {}).get("version") or ""),
                str((ref_dict or {}).get("checksum") or ""),
                "",
            )
            for item in loaded_skills:
                if _skill_dedupe_key(item) == key:
                    return item
            return None

        deadline = time.monotonic() + 1.0
        for entry in pending:
            tool_name = str(entry.get("tool_name") or "")
            args = dict(entry.get("arguments") or {})
            guard = dict(entry.get("skill_guard") or {})
            payload: Dict[str, Any]
            if tool_name == SKILLS_LIST_TOOL:
                if not guard.get("allowed"):
                    payload = {
                        "loaded": False,
                        "error_type": guard.get("error_type") or "company_profile_disabled",
                    }
                else:
                    skills_meta = []
                    for item in catalog.get("skills") or []:
                        if not isinstance(item, dict):
                            continue
                        public = dict(item)
                        public.pop("body", None)
                        ref = dict(public.get("ref") or {})
                        ref.pop("tenant_id", None)
                        public["ref"] = ref
                        skills_meta.append(public)
                    payload = {
                        "loaded": True,
                        "skills": skills_meta,
                        "degraded": bool(catalog.get("degraded")),
                        "omission_reason": catalog.get("omission_reason"),
                    }
            elif not guard.get("allowed"):
                payload = {
                    "loaded": False,
                    "error_type": guard.get("error_type") or "skill_not_allowed",
                }
            elif provider is None or snapshot is None:
                payload = {"loaded": False, "error_type": "skill_provider_unavailable"}
            else:
                skill_ref = skill_ref_from_dict((guard.get("skill_ref") or None))
                if skill_ref is None:
                    payload = {"loaded": False, "error_type": "skill_not_found"}
                elif tool_name == SKILL_LOAD_TOOL:
                    existing = _existing_skill(guard.get("skill_ref") if isinstance(guard.get("skill_ref"), dict) else None)
                    if existing is not None:
                        payload = {
                            "loaded": True,
                            "reused": True,
                            "skill": existing.get("metadata") or existing,
                        }
                    else:
                        result = provider.load_skill(skill_ref, snapshot, deadline_monotonic=deadline)
                        if result.loaded and result.skill is not None:
                            stored = result.skill.to_dict(include_body=True)
                            key = _skill_dedupe_key(stored)
                            if key not in loaded_keys:
                                loaded_skills.append(stored)
                                loaded_keys.add(key)
                            payload = {"loaded": True, "skill": stored.get("metadata"), "reused": False}
                        else:
                            payload = {
                                "loaded": False,
                                "error_type": result.error_type or "skill_not_found",
                            }
                            errors.append({"error_type": payload["error_type"], "skill_name": skill_ref.name})
                elif tool_name == SKILL_RESOURCE_LOAD_TOOL:
                    resource_path = str(args.get("resource_path") or "").strip()
                    existing_res = None
                    key = (
                        str((guard.get("skill_ref") or {}).get("stable_id") or ""),
                        str((guard.get("skill_ref") or {}).get("version") or ""),
                        str((guard.get("skill_ref") or {}).get("checksum") or ""),
                        resource_path,
                    )
                    for item in skill_resources:
                        if _skill_dedupe_key(item) == key:
                            existing_res = item
                            break
                    if existing_res is not None:
                        payload = {
                            "loaded": True,
                            "reused": True,
                            "resource_path": resource_path,
                        }
                    else:
                        result = provider.load_resource(
                            skill_ref, resource_path, snapshot, deadline_monotonic=deadline
                        )
                        if result.loaded and result.skill is not None:
                            stored = result.skill.to_dict(include_body=True)
                            stored["resource_path"] = result.resource_path
                            stored["resource_checksum"] = result.resource_checksum
                            res_key = _skill_dedupe_key(stored)
                            if res_key not in resource_keys:
                                skill_resources.append(stored)
                                resource_keys.add(res_key)
                            payload = {
                                "loaded": True,
                                "resource_path": result.resource_path,
                                "reused": False,
                            }
                        else:
                            payload = {
                                "loaded": False,
                                "error_type": result.error_type or "skill_resource_not_declared",
                            }
                            errors.append({"error_type": payload["error_type"], "skill_name": skill_ref.name})
                else:
                    payload = {"loaded": False, "error_type": "skill_not_allowed"}
            payload["tool_name"] = tool_name
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": entry.get("tool_call_id"),
                    "content": _json_tool_content(payload),
                }
            )
            traces.append(
                _trace_node(
                    "skill_load_node",
                    f"{tool_name} loaded={payload.get('loaded')}",
                    error_type=payload.get("error_type"),
                    reused=payload.get("reused"),
                )
            )

        existing_tool_ids = {
            str(msg.get("tool_call_id") or "")
            for msg in (state.get("messages") or [])
            if isinstance(msg, dict) and msg.get("role") == "tool"
        }
        seen_ids = {str(msg.get("tool_call_id") or "") for msg in new_messages} | existing_tool_ids
        for item in list(state.get("deferred_business_tool_calls") or []):
            call_id = str(item.get("tool_call_id") or "")
            if not call_id or call_id in seen_ids:
                continue
            if item.get("reason") != "skill_context_changed":
                continue
            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _json_tool_content(
                        {
                            "status": "deferred",
                            "reason": "skill_context_changed",
                            "error_type": "skill_load_deferred_business_calls",
                        }
                    ),
                }
            )
            seen_ids.add(call_id)

        round_idx = int(state.get("round_idx") or 0) + 1
        return {
            "messages": new_messages,
            "loaded_skills": loaded_skills,
            "skill_resources": skill_resources,
            "skill_errors": errors,
            "pending_tool_calls": [],
            "round_idx": round_idx,
            "agent_trace": traces
            + [
                _trace_node(
                    "skill_load_node",
                    "cleared pending tool calls and returned to context_prepare",
                    loaded_count=len(loaded_skills),
                    resource_count=len(skill_resources),
                )
            ],
        }

    def route_after_model_decide(state: AgentGraphState) -> str:
        hinted = state.get("post_model_route")
        if hinted in {"user_input", "context_prepare"}:
            return hinted
        if state.get("execution_status") == ExecutionStatus.STOPPED.value:
            return "finalize"
        pending = list(state.get("pending_tool_calls") or [])
        if not pending:
            if state.get("final_response"):
                return "answer_verify"
            return "finalize"
        if any(is_skill_harness_tool(entry.get("tool_name") or "") for entry in pending):
            return "skill_guard"
        return "tool_guard"

    def route_after_post_model(state: AgentGraphState) -> str:
        return route_after_model_decide(state)

    def post_model_route_trace_node(state: AgentGraphState) -> Dict[str, Any]:
        route = route_after_model_decide(state)
        return {
            "agent_trace": [
                _trace_route("post_model_validate_node", route, "post_model routing"),
            ],
        }

    def route_after_post_model_trace(state: AgentGraphState) -> str:
        return route_after_model_decide(state)

    def route_after_skill_load(state: AgentGraphState) -> str:
        if state.get("execution_status") == ExecutionStatus.STOPPED.value:
            return "finalize"
        return "context_prepare"

    def route_after_compact(state: AgentGraphState) -> str:
        if state.get("execution_status") == ExecutionStatus.STOPPED.value:
            return "finalize"
        return "context_prepare"

    def instrument_node(node_name: str, node_fn: Any) -> Any:
        """Emit safe lifecycle metadata without exposing graph state or changing routing."""

        def instrumented(state: AgentGraphState) -> Dict[str, Any]:
            cancelled = getattr(config.registry, "cancel_requested", None)
            if cancelled and cancelled():
                from .runtime_errors import RuntimeCancelled
                raise RuntimeCancelled(dict(state))
            round_idx = int(state.get("round_idx") or 0)
            span_id = config.telemetry.start_span(
                "node",
                node_name,
                round_idx=round_idx,
            )
            _emit_runtime_progress(
                config,
                {
                    "type": "graph_node",
                    "node": node_name,
                    "phase": "started",
                    "status": "running",
                    "round_idx": round_idx,
                },
            )
            try:
                result = node_fn(state)
            except BaseException as exc:
                error_name = type(exc).__name__
                status = "interrupted" if "interrupt" in error_name.lower() else "failed"
                config.telemetry.end_span(span_id, status=status, error_type=error_name)
                raise
            config.telemetry.end_span(span_id, status="completed")
            _emit_runtime_progress(
                config,
                {
                    "type": "graph_node",
                    "node": node_name,
                    "phase": "completed",
                    "status": "completed",
                    "round_idx": round_idx,
                },
            )
            return result

        return instrumented

    graph_nodes = (
        ("intake_node", intake_node),
        ("resolve_company_profile_node", resolve_company_profile_node),
        ("skill_discover_node", skill_discover_node),
        ("retrieve_context_node", retrieve_context_node),
        ("context_prepare_node", context_prepare_node),
        ("model_decide_node", model_decide_node),
        ("post_model_validate_node", post_model_validate_node),
        ("post_model_route_trace_node", post_model_route_trace_node),
        ("skill_guard_node", skill_guard_node),
        ("skill_load_node", skill_load_node),
        ("tool_guard_node", tool_guard_node),
        ("parameter_verify_node", parameter_verify_node),
        ("artifact_check_node", artifact_check_node),
        ("artifact_review_node", artifact_review_node),
        ("artifact_bind_node", artifact_bind_node),
        ("human_confirm_node", human_confirm_node),
        ("user_input_node", user_input_node),
        ("tool_execute_node", tool_execute_node),
        ("observe_node", observe_node),
        ("context_compact_node", context_compact_node),
        ("answer_verify_node", answer_verify_node),
        ("finalize_node", finalize_node),
    )
    for node_name, node_fn in graph_nodes:
        graph.add_node(node_name, instrument_node(node_name, node_fn))

    graph.add_edge(_START, "intake_node")
    graph.add_edge("intake_node", "resolve_company_profile_node")
    graph.add_edge("resolve_company_profile_node", "skill_discover_node")
    graph.add_edge("skill_discover_node", "retrieve_context_node")
    graph.add_edge("retrieve_context_node", "context_prepare_node")
    graph.add_edge("context_prepare_node", "model_decide_node")
    graph.add_edge("model_decide_node", "post_model_validate_node")
    graph.add_edge("post_model_validate_node", "post_model_route_trace_node")
    graph.add_conditional_edges(
        "post_model_route_trace_node",
        route_after_post_model_trace,
        {
            "skill_guard": "skill_guard_node",
            "tool_guard": "tool_guard_node",
            "user_input": "user_input_node",
            "context_prepare": "context_prepare_node",
            "answer_verify": "answer_verify_node",
            "finalize": "finalize_node",
        },
    )
    graph.add_edge("user_input_node", "context_prepare_node")
    graph.add_edge("skill_guard_node", "skill_load_node")
    graph.add_conditional_edges(
        "skill_load_node",
        route_after_skill_load,
        {
            "context_prepare": "context_prepare_node",
            "finalize": "finalize_node",
        },
    )
    graph.add_edge("tool_guard_node", "parameter_verify_node")
    graph.add_edge("parameter_verify_node", "artifact_check_node")
    graph.add_edge("artifact_check_node", "artifact_review_node")
    graph.add_edge("artifact_review_node", "human_confirm_node")
    graph.add_edge("human_confirm_node", "artifact_bind_node")
    graph.add_edge("artifact_bind_node", "tool_execute_node")
    graph.add_edge("tool_execute_node", "observe_node")
    graph.add_conditional_edges("observe_node",
                                lambda s: "tool_guard" if s.get("pending_tool_calls") else "context_compact",
                                {"tool_guard": "tool_guard_node", "context_compact": "context_compact_node"})
    graph.add_conditional_edges(
        "context_compact_node",
        route_after_compact,
        {
            "context_prepare": "context_prepare_node",
            "finalize": "finalize_node",
        },
    )
    graph.add_edge("answer_verify_node", "finalize_node")
    graph.add_edge("finalize_node", _END)

    return graph.compile(checkpointer=checkpointer)


def _copy_company_skill_fields(graph_state: Dict[str, Any]) -> Dict[str, Any]:
    from .skill_models import select_company_skill_pipeline_fields

    return select_company_skill_pipeline_fields(graph_state)


def _graph_state_to_pipeline(
    graph_state: Dict[str, Any],
    config: LangGraphRuntimeConfig,
    *,
    pending_interaction: Optional[Dict[str, Any]] = None,
    terminal: bool = True,
) -> Dict[str, Any]:
    graph_state = _strip_interrupt(graph_state)
    execution_status = graph_state.get("execution_status") or ExecutionStatus.STOPPED.value
    completed_steps = list(graph_state.get("completed_steps") or [])
    failed_steps = list(graph_state.get("failed_steps") or [])
    interaction_history = list(graph_state.get("interaction_history") or [])
    pending = pending_interaction if pending_interaction is not None else graph_state.get("pending_interaction")
    final_response = graph_state.get("final_response")
    verification = graph_state.get("verification") or {}
    next_actions: List[str] = list(graph_state.get("next_actions") or [])
    if pending and not terminal:
        pending_type = str(pending.get("type") or "")
        if pending_type in {"confirmation", "artifact_review"}:
            execution_status = ExecutionStatus.AWAITING_CONFIRMATION.value
            next_actions = ["confirm_or_reject"]
        else:
            execution_status = ExecutionStatus.AWAITING_INPUT.value
            next_actions = ["provide_requested_information"]
        final_response = None
        verification = {}

    modes = resolve_execution_modes(
        requested_execution_mode=config.requested_mode,
        execution_status=execution_status,
        completed_steps=completed_steps,
        failed_steps=failed_steps,
    )

    if not next_actions and execution_status == ExecutionStatus.STOPPED.value:
        errors = graph_state.get("errors") or []
        err_type = (errors[0] or {}).get("type") if errors else None
        if err_type == "llm_unavailable":
            next_actions = ["configure_llm_api_key"]
        elif err_type == "max_rounds":
            next_actions = ["retry_or_simplify_request"]
        else:
            next_actions = ["review_agent_trace"]

    budget_history = list(graph_state.get("context_budget_history") or [])
    spilled = collect_spilled_artifacts(budget_history)
    if pending and not terminal:
        config.telemetry.pause()
    runtime_evidence = config.telemetry.snapshot()

    base_state = create_initial_state(config.safe_request, module_config=config.module_config)
    base_state.update(
        {
            "agent_mode": "langgraph",
            "requested_execution_mode": config.requested_mode,
            "user_confirmed": config.user_confirmed,
            "execution_status": execution_status,
            "agent_trace": list(graph_state.get("agent_trace") or []),
            "tool_calls": list(graph_state.get("tool_calls") or []),
            "observations": list(graph_state.get("observations") or []),
            "parameter_sources": dict(graph_state.get("parameter_sources") or config.parameter_sources),
            "parameter_verification_results": list(graph_state.get("parameter_verification_results") or []),
            "runtime_telemetry": runtime_evidence["runtime_telemetry"],
            "resource_usage": runtime_evidence["resource_usage"],
            "final_response": final_response,
            "pending_interaction": pending if pending and not terminal else None,
            "interaction_history": interaction_history,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "artifacts": list(graph_state.get("artifacts") or []),
            "running_summary": graph_state.get("running_summary") or "",
            "llm_used": True,
            "llm_provider": graph_state.get("llm_provider"),
            "llm_model": graph_state.get("llm_model"),
            "llm_context_snapshots": list(graph_state.get("llm_context_snapshots") or []),
            "messages": list(graph_state.get("messages") or []),
            "llm_input_messages": list(graph_state.get("llm_input_messages") or []),
            "memory_recall": graph_state.get("memory_recall") or {},
            "session_search": graph_state.get("session_search") or {},
            "memory_profile": graph_state.get("memory_profile") or {},
            "memory_review": graph_state.get("memory_review") or {},
            "memory_context_manifest": list(graph_state.get("memory_context_manifest") or []),
            "knowledge_search": graph_state.get("knowledge_search") or {},
            "knowledge_context_manifest": list(graph_state.get("knowledge_context_manifest") or []),
            "knowledge_context_injected": bool(graph_state.get("knowledge_context_injected")),
            "run_id": graph_state.get("run_id"),
            "thread_id": graph_state.get("thread_id"),
            "company_profile_binding_id": graph_state.get("company_profile_binding_id"),
            "errors": list(graph_state.get("errors") or []),
            "graph_runtime": "state_graph",
            "context_budget": graph_state.get("context_budget") or {},
            "context_budget_history": list(graph_state.get("context_budget_history") or []),
            "verification": verification,
            "spilled_artifacts": spilled,
            **_copy_company_skill_fields(graph_state),
        }
    )

    report = {
        "goal": config.safe_request,
        "status": execution_status,
        "agent_mode": "langgraph",
        "graph_runtime": "state_graph",
        "requested_execution_mode": modes["requested_execution_mode"],
        "execution_mode": modes["execution_mode"],
        "plan_id": None,
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "artifacts": list(graph_state.get("artifacts") or []),
        "running_summary": graph_state.get("running_summary") or "",
        "next_actions": next_actions,
        "agent_trace": list(graph_state.get("agent_trace") or []),
        "tool_calls": list(graph_state.get("tool_calls") or []),
        "observations": list(graph_state.get("observations") or []),
        "parameter_sources": dict(graph_state.get("parameter_sources") or config.parameter_sources),
        "parameter_verification_results": list(graph_state.get("parameter_verification_results") or []),
        "runtime_telemetry": runtime_evidence["runtime_telemetry"],
        "resource_usage": runtime_evidence["resource_usage"],
        "final_response": final_response,
        "pending_interaction": pending if pending and not terminal else None,
        "interaction_history": interaction_history,
        "llm_provider": graph_state.get("llm_provider"),
        "llm_model": graph_state.get("llm_model"),
        "llm_context_snapshots": list(graph_state.get("llm_context_snapshots") or []),
        "context_budget_history": list(graph_state.get("context_budget_history") or []),
        "verification": verification,
        "spilled_artifacts": spilled,
        "memory_recall": graph_state.get("memory_recall") or {},
        "session_search": graph_state.get("session_search") or {},
        "memory_profile": graph_state.get("memory_profile") or {},
        "memory_review": graph_state.get("memory_review") or {},
        "knowledge_search": graph_state.get("knowledge_search") or {},
        "knowledge_context_injected": bool(graph_state.get("knowledge_context_injected")),
        "run_id": graph_state.get("run_id"),
        "thread_id": graph_state.get("thread_id"),
        "company_profile_binding_id": graph_state.get("company_profile_binding_id"),
    }
    report.update(_copy_company_skill_fields(graph_state))

    if terminal:
        from .session_ledger import record_langgraph_run_to_ledger

        try:
            recorded_run_id = record_langgraph_run_to_ledger(
                state=dict(base_state),
                report=report,
                run_id=graph_state.get("run_id"),
                thread_id=graph_state.get("thread_id"),
                company_profile_binding_id=graph_state.get("company_profile_binding_id"),
                ledger=config.session_ledger,
            )
            if recorded_run_id:
                base_state["run_id"] = recorded_run_id
                report["run_id"] = recorded_run_id
        except Exception:
            pass

    return pipeline_response(
        state=dict(base_state),
        report=report,
        report_text=format_langgraph_report_text(report),
    )


def run_react_state_graph_runtime(
    user_request: str,
    *,
    module_config: Optional[Dict[str, Any]] = None,
    user_confirmed: bool = True,
    confirmation_source: Optional[str] = None,
    dry_run: bool = True,
    execution_mode: Optional[Union[str, ExecutionMode]] = None,
    llm_model: Optional[str] = None,
    llm_complete_fn: Optional[ToolsCompleteFn] = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    registry: Optional[ModuleCapabilityRegistry] = None,
    context_spill_dir: Optional[Union[str, Path]] = None,
    thread_id: Optional[str] = None,
    run_id: Optional[str] = None,
    knowledge_config: Optional[Dict[str, Any]] = None,
    company_skill_runtime: Optional[Dict[str, Any]] = None,
    interactions_enabled: bool = False,
    interaction_runtime_store: Optional[InteractionRuntimeStore] = None,
    progress_callback: Optional[Any] = None,
    parameter_sources: Optional[Dict[str, str]] = None,
    parameter_judge: Optional[EvalJudge] = None,
    parameter_judge_fn: Optional[Any] = None,
    runtime_telemetry_collector: Optional[RuntimeTelemetryCollector] = None,
    price_card_path: Optional[Union[str, Path]] = None,
    price_card_catalog: Optional[Dict[str, Any]] = None,
    session_search_retriever: Optional[Any] = None,
    session_ledger: Optional[Any] = None,
    memory_dir: Optional[Union[str, Path]] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not is_state_graph_available():
        raise ImportError("langgraph is not installed")

    reg = registry or get_registry()
    mode = resolve_pipeline_execution_mode(execution_mode=execution_mode, dry_run=dry_run)
    safe_request, _ = redact_text(user_request)

    spill_path = Path(context_spill_dir) if context_spill_dir else None
    telemetry = runtime_telemetry_collector or RuntimeTelemetryCollector(
        price_card_path=Path(price_card_path) if price_card_path else None,
        price_card_catalog=price_card_catalog,
    )
    effective_parameter_judge = parameter_judge
    if effective_parameter_judge is None and parameter_judge_fn is not None:
        effective_parameter_judge = CallableEvalJudge(
            parameter_judge_fn,
            telemetry=telemetry,
        )
    # A native judge is opt-in. Missing semantic service leads to confirmation.

    from .session_checkpoint import opaque_company_profile_binding_id
    from .skill_models import empty_company_skill_pipeline_fields

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    effective_run_id = run_id or f"run_{ts}_{secrets.token_hex(4)}"
    checkpoint_thread_id = f"acceptance:{effective_run_id}" if interactions_enabled else None

    if interactions_enabled:
        if interaction_runtime_store is None:
            return _interaction_error_pipeline(
                "interaction_runtime_unavailable",
                safe_request=safe_request,
                run_id=effective_run_id,
            )
        checkpointer = interaction_runtime_store.create_checkpointer()
        if checkpointer is None:
            return _interaction_error_pipeline(
                "interaction_runtime_unavailable",
                safe_request=safe_request,
                run_id=effective_run_id,
            )
    else:
        checkpointer = None

    frozen_runtime = copy.deepcopy(company_skill_runtime or empty_company_skill_pipeline_fields())
    runtime_config = LangGraphRuntimeConfig(
        registry=reg,
        execution_mode=mode,
        module_config=module_config,
        user_confirmed=user_confirmed,
        confirmation_source=confirmation_source,
        llm_model=llm_model,
        llm_complete_fn=llm_complete_fn,
        max_tool_rounds=max_tool_rounds,
        safe_request=safe_request,
        requested_mode=mode.value,
        context_spill_dir=spill_path,
        knowledge_config=knowledge_config,
        company_skill_runtime=frozen_runtime,
        interactions_enabled=interactions_enabled,
        interaction_runtime_store=interaction_runtime_store,
        checkpoint_thread_id=checkpoint_thread_id,
        progress_callback=progress_callback,
        parameter_sources=dict(parameter_sources or {}),
        parameter_judge=effective_parameter_judge,
        telemetry=telemetry,
        session_search_retriever=session_search_retriever,
        session_ledger=session_ledger,
        memory_dir=memory_dir,
        history_messages=list(history_messages or []),
    )

    compiled = _build_compiled_graph(runtime_config, checkpointer=checkpointer)
    initial: Dict[str, Any] = {
        "user_request": safe_request,
        "goal": safe_request,
        "run_id": effective_run_id,
        "thread_id": thread_id,
        "checkpoint_thread_id": checkpoint_thread_id,
        "module_config": module_config,
        "parameter_sources": dict(parameter_sources or {}),
        "user_confirmed": user_confirmed,
        "confirmation_source": confirmation_source,
        "execution_mode": mode.value,
        "requested_execution_mode": mode.value,
        "agent_mode": "langgraph",
        "graph_runtime": "state_graph",
        "interactions_enabled": interactions_enabled,
        "pending_interaction": None,
        "interaction_history": [],
        "post_model_route": None,
        "messages": [],
        "tool_calls": [],
        "observations": [],
        "completed_steps": [],
        "failed_steps": [],
        "agent_trace": [],
        "parameter_verification_results": [],
        "round_parameter_verification_results": [],
        "llm_context_snapshots": [],
        "artifacts": [],
        "errors": [],
        "memory_recall": {},
        "session_search": {},
        "memory_profile": {},
        "memory_review": {},
        "memory_context_injected": False,
        "memory_context_manifest": [],
        "knowledge_search": {},
        "knowledge_context_manifest": [],
        "knowledge_context_injected": False,
        "running_summary": "",
        "round_idx": 0,
        "context_budget": {},
        "context_budget_history": [],
        "verification": {},
        "runtime_telemetry": {},
        "resource_usage": {},
        **empty_company_skill_pipeline_fields(),
        "company_profile_binding_id": opaque_company_profile_binding_id(frozen_runtime),
    }

    invoke_config: Dict[str, Any] = {"recursion_limit": max(max_tool_rounds * 128, 128)}
    if checkpoint_thread_id:
        invoke_config["configurable"] = {"thread_id": checkpoint_thread_id}

    if interactions_enabled and interaction_runtime_store is not None:
        registration_error = interaction_runtime_store.register(
            InteractionRuntimeHandle(
                run_id=effective_run_id,
                compiled_graph=compiled,
                runtime_config=runtime_config,
                invoke_config=invoke_config,
                pending_interaction={},
                status="pending",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        if registration_error:
            return _interaction_error_pipeline(
                registration_error,
                safe_request=safe_request,
                run_id=effective_run_id,
            )

    try:
        final_state = compiled.invoke(initial, config=invoke_config)
    except Exception:
        if interactions_enabled and interaction_runtime_store is not None:
            interaction_runtime_store.fail_resume(effective_run_id)
        raise
    pending = extract_pending_interaction(final_state) if interactions_enabled else None
    if pending and interaction_runtime_store is not None:
        interaction_runtime_store.update_pending(effective_run_id, pending)
        return _graph_state_to_pipeline(
            final_state,
            runtime_config,
            pending_interaction=pending,
            terminal=False,
        )
    if interactions_enabled and interaction_runtime_store is not None:
        interaction_runtime_store.complete(effective_run_id)
    return _graph_state_to_pipeline(final_state, runtime_config, terminal=True)


def resume_react_state_graph_runtime(
    run_id: str,
    interaction_response: Dict[str, Any],
    *,
    interaction_runtime_store: InteractionRuntimeStore,
) -> Dict[str, Any]:
    handle, sanitized, error_type = interaction_runtime_store.begin_resume(run_id, interaction_response)
    if error_type or handle is None or sanitized is None:
        return _interaction_error_pipeline(error_type or "interaction_expired", run_id=run_id)
    try:
        from langgraph.types import Command
        from .runtime_errors import RuntimeCancelled

        handle.runtime_config.telemetry.resume()
        final_state = handle.compiled_graph.invoke(Command(resume=sanitized), handle.invoke_config)
    except RuntimeCancelled as exc:
        interaction_runtime_store.complete(run_id)
        stopped = dict(exc.state)
        stopped.update(execution_status=ExecutionStatus.STOPPED.value, pending_interaction=None)
        stopped["errors"] = list(stopped.get("errors") or []) + [
            {"type": "run_cancelled", "message": "run_cancelled"}]
        return _graph_state_to_pipeline(stopped, handle.runtime_config, terminal=True)
    except Exception:
        interaction_runtime_store.fail_resume(run_id)
        return _interaction_error_pipeline("interaction_resume_failed", run_id=run_id)
    pending = extract_pending_interaction(final_state)
    if pending:
        interaction_runtime_store.update_pending(run_id, pending)
        return _graph_state_to_pipeline(
            final_state,
            handle.runtime_config,
            pending_interaction=pending,
            terminal=False,
        )
    interaction_runtime_store.complete(run_id)
    return _graph_state_to_pipeline(final_state, handle.runtime_config, terminal=True)
