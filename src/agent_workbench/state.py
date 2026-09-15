# -*- coding: utf-8 -*-
"""AgentState — Plan-First Agent 执行账本。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    user_request: str

    intent: Optional[str]
    selected_playbook_id: Optional[str]
    selected_case_ids: List[str]

    plan: Optional[Dict[str, Any]]
    plan_status: str
    current_step_id: Optional[str]
    completed_steps: List[Dict[str, Any]]
    pending_steps: List[Dict[str, Any]]

    store_scope: Dict[str, Any]
    module_config: Dict[str, Any]
    parameter_sources: Dict[str, str]
    parameter_verification_results: List[Dict[str, Any]]
    runtime_telemetry: Dict[str, Any]
    resource_usage: Dict[str, Any]
    selected_module: Optional[str]
    selected_tool_ids: List[str]
    selected_capability_ids: List[str]
    selection_reason: str
    context_bundle: Dict[str, Any]

    preflight_result: Optional[Dict[str, Any]]
    risk_level: str
    requires_confirmation: bool
    user_confirmed: bool

    execution_status: str
    last_observation: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    verification_result: Optional[Dict[str, Any]]

    running_summary: str
    artifacts: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]

    llm_used: bool
    llm_provider: Optional[str]
    llm_model: Optional[str]
    llm_error_type: Optional[str]
    llm_plan_source: Optional[str]

    llm_review_required: bool
    llm_review_used: bool
    llm_review_decision: Optional[str]
    llm_review_summary: Optional[str]
    llm_review_error_type: Optional[str]
    candidate_plan: Optional[Dict[str, Any]]
    candidate_plan_source: Optional[str]
    final_plan_source: Optional[str]
    plan_changes: Dict[str, Any]

    thread_id: Optional[str]
    planning_context: Dict[str, Any]
    execution_context: Dict[str, Any]
    context_manifest: List[Dict[str, Any]]
    session_snapshot: Optional[Dict[str, Any]]
    session_error_type: Optional[str]


def create_initial_state(user_request: str, module_config: Optional[Dict[str, Any]] = None) -> AgentState:
    """创建初始 Agent state。"""

    return AgentState(
        messages=[],
        user_request=user_request,
        intent=None,
        selected_playbook_id=None,
        selected_case_ids=[],
        plan=None,
        plan_status="draft",
        current_step_id=None,
        completed_steps=[],
        pending_steps=[],
        store_scope={},
        module_config=module_config or {},
        parameter_sources={},
        parameter_verification_results=[],
        runtime_telemetry={},
        resource_usage={},
        selected_module=None,
        selected_tool_ids=[],
        selected_capability_ids=[],
        selection_reason="",
        context_bundle={},
        preflight_result=None,
        risk_level="low",
        requires_confirmation=False,
        user_confirmed=False,
        execution_status="pending",
        last_observation=None,
        execution_result=None,
        verification_result=None,
        running_summary="",
        artifacts=[],
        errors=[],
        llm_used=False,
        llm_provider=None,
        llm_model=None,
        llm_error_type=None,
        llm_plan_source=None,
        llm_review_required=False,
        llm_review_used=False,
        llm_review_decision=None,
        llm_review_summary=None,
        llm_review_error_type=None,
        candidate_plan=None,
        candidate_plan_source=None,
        final_plan_source=None,
        plan_changes={},
        thread_id=None,
        planning_context={},
        execution_context={},
        context_manifest=[],
        session_snapshot=None,
        session_error_type=None,
    )
