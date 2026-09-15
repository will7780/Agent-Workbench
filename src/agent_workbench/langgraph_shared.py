# -*- coding: utf-8 -*-
"""LangGraph runner 与 StateGraph runtime 共享工具（避免循环导入）。"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .eval_judge import EvalJudge
from .execution_mode import ExecutionMode
from .llm_client import LLMToolCall
from .observer import Observer
from .observation_capabilities import observation_system_hint
from .parameter_verification import parameter_validation_observation, verify_tool_call_parameters
from .registry import ModuleCapabilityRegistry, get_registry
from .runtime_telemetry import RuntimeTelemetryCollector
from .schemas import Observation
from .tool_guard import guard_tool_call
from .tools.adapter_contracts import AdapterInput
from .tools.execution_adapters import execute_adapter

DEFAULT_MAX_TOOL_ROUNDS = 8

_LANGGRAPH_SYSTEM_BASE = (
    "You are a general-purpose assistant. Use only the provided tools from the capability registry. "
    "Use actual tool feedback and respect the execution mode. "
    "When finished, respond in the user's language with a concise factual summary. "
    "Do not invent tools or parameters not in the schema."
)


_LANGGRAPH_INTERACTION_RULES = (
    "When required execution parameters are missing and cannot be obtained from "
    "context or read-only tools, you MUST call agent__ask_user by itself. "
    "Do not mix agent__ask_user with business tool calls in the same turn. "
    "Do not ask for optional information. Do not invent missing parameters."
)


def langgraph_system_prompt(
    registry: Optional[ModuleCapabilityRegistry] = None,
    *,
    interactions_enabled: bool = False,
) -> str:
    hint = observation_system_hint(registry)
    parts = [getattr(registry, "system_prompt", None) or _LANGGRAPH_SYSTEM_BASE]
    if hint:
        parts.append(hint)
    if interactions_enabled:
        parts.append(_LANGGRAPH_INTERACTION_RULES)
    return "\n".join(parts)


def pipeline_response(
    *,
    state: Dict[str, Any],
    report: Dict[str, Any],
    report_text: str,
) -> Dict[str, Any]:
    return {
        "state": state,
        "report": report,
        "report_text": report_text,
        "plan": None,
        "preflight_result": None,
    }


def format_langgraph_report_text(report: Dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "Agent Workbench — LangGraph Tool-Calling Report",
        "=" * 60,
        f"Goal: {report.get('goal', '')}",
        f"Status: {report.get('status', '')}",
        f"Agent mode: {report.get('agent_mode', 'langgraph')}",
        f"Execution mode: {report.get('execution_mode', 'dry_run')}",
        f"Graph runtime: {report.get('graph_runtime', 'local_loop')}",
    ]
    fallback_reason = report.get("fallback_reason")
    if fallback_reason:
        lines.append(f"Runtime fallback: {fallback_reason}")
    lines.extend([
        "",
        f"Tool calls: {len(report.get('tool_calls') or [])}",
        f"Observations: {len(report.get('observations') or [])}",
        "",
        "Running summary:",
        report.get("running_summary") or "(empty)",
        "",
    ])
    final = report.get("final_response")
    if final:
        lines.append("Final response:")
        lines.append(str(final))
        lines.append("")
    next_actions = report.get("next_actions") or []
    if next_actions:
        lines.append("Next actions:")
        for action in next_actions:
            lines.append(f"- {action}")
    lines.append("=" * 60)
    return "\n".join(lines)


def observation_to_tool_message(obs: Observation) -> str:
    from .redaction import redact_recursive
    safe, _ = redact_recursive(obs.to_dict())
    return json.dumps(safe, ensure_ascii=False)


def guard_blocked_observation(step_id: str, guard_dict: Dict[str, Any]) -> Observation:
    return Observation(
        step_id=step_id,
        status="failed",
        summary=guard_dict.get("error_message") or "tool guard blocked",
        error={
            "type": guard_dict.get("error_type") or "guard_blocked",
            "message": guard_dict.get("error_message") or "",
            "missing_params": guard_dict.get("missing_params"),
        },
        suggested_next_action="stop",
    )


def execute_guarded_tool(
    tool_call: LLMToolCall,
    *,
    step_id: str,
    registry: ModuleCapabilityRegistry,
    execution_mode: ExecutionMode,
    module_config: Optional[Dict[str, Any]],
    user_confirmed: bool,
    observer: Observer,
    user_request: str = "",
    parameter_sources: Optional[Dict[str, str]] = None,
    parameter_judge: Optional[EvalJudge] = None,
    telemetry: Optional[RuntimeTelemetryCollector] = None,
) -> tuple[Observation, Dict[str, Any], Dict[str, Any]]:
    guard = guard_tool_call(
        tool_call.name,
        tool_call.arguments,
        registry=registry,
        execution_mode=execution_mode,
        module_config=module_config,
    )
    guard_entry: Dict[str, Any] = {
        "type": "guard_result",
        "tool_call_id": tool_call.id,
        "tool_name": tool_call.name,
        **guard.to_dict(),
    }
    if not guard.allowed:
        obs = guard_blocked_observation(step_id, guard.to_dict())
        return obs, guard_entry, observer.observe(obs)

    verification = verify_tool_call_parameters(
        tool_call_id=tool_call.id,
        step_id=step_id,
        tool_name=tool_call.name,
        module=str(guard.module or ""),
        action=str(guard.action or ""),
        arguments=tool_call.arguments,
        resolved_params=guard.params,
        user_request=user_request,
        execution_mode=execution_mode,
        registry=registry,
        module_config=module_config,
        parameter_sources=parameter_sources,
        semantic_mode="auto",
        judge=parameter_judge,
    ).to_dict()
    guard_entry["parameter_verification"] = verification
    if not verification.get("allowed") or verification.get("requires_confirmation"):
        obs = parameter_validation_observation(step_id, verification)
        return obs, guard_entry, observer.observe(obs)

    adapter_name = registry.get_execution_adapter(guard.module, guard.action)
    if (registry.get_action(guard.module, guard.action) or {}).get("artifact_policy", {}).get("required"):
        from .artifact_review import artifact_blocked_observation
        obs = artifact_blocked_observation(step_id, "artifact_review_runtime_required")
        guard_entry["adapter_called"] = False
        return obs, guard_entry, observer.observe(obs)
    permission = registry.get_risk_level(guard.module, guard.action)
    adapter_input = AdapterInput(
        step_id=step_id,
        module=guard.module,
        action=guard.action,
        params=guard.params,
        execution_mode=execution_mode,
        dry_run=execution_mode == ExecutionMode.DRY_RUN,
        user_confirmed=user_confirmed,
        permission_level=permission,
        adapter_name=adapter_name,
    )
    span_id = (
        telemetry.start_span("tool", tool_call.name, step_id=step_id)
        if telemetry is not None
        else None
    )
    try:
        obs = execute_adapter(adapter_input, registry=registry)
    except Exception as exc:
        if telemetry is not None and span_id is not None:
            telemetry.end_span(span_id, status="failed", error_type=type(exc).__name__)
        raise
    if telemetry is not None and span_id is not None:
        telemetry.end_span(span_id, status=obs.status)
    return obs, guard_entry, observer.observe(obs)
