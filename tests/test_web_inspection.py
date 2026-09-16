"""Deterministic inspection projections; no runtime invocation."""
from types import SimpleNamespace

from agent_workbench.registry import ToolRegistry
from agent_workbench.web.inspection import catalogue, recorded_flow
from agent_workbench.web.service import WebAppService


def fixture_registry():
    return ToolRegistry(payload={"version": "fixture", "modules": {"notes": {
        "display_name": "Notes", "actions": {"read": {
            "display_name": "Read notes", "risk_level": "L1",
            "dry_run_status": "available", "live_status": "unavailable",
            "parameter_contract": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]},
            "execution_adapter": "not_for_public_inspection",
            "workflow_file": "not_for_public_inspection",
        }, "archive": {"risk_level": "L3", "dry_run_status": "available"}}}}})


def test_catalogue_matches_registry_and_keeps_unavailable_modes():
    registry = fixture_registry()
    result = catalogue(registry)
    assert result["total"] == 2 and len(result["tools"]) == 2
    item = result["tools"][0]
    assert item["name"] == "notes__read"
    assert item["declared_modes"]["live"] == "unavailable"
    assert "execution_adapter" not in item
    assert "parameter_contract" not in item
    assert catalogue(registry, tool_name="notes__read")["parameter_contract"]["required"] == ["path"]
    assert catalogue(registry, tool_name="missing") is None
    assert catalogue(registry, limit=1)["truncated"] is True


def test_absent_and_empty_catalogue_are_distinct():
    assert catalogue(None)["available"] is False
    assert catalogue(ToolRegistry(payload={"modules": {}}))["total"] == 0


def test_inspection_does_not_read_configuration_or_execute_and_is_redacted():
    def forbidden(*args, **kwargs):
        raise AssertionError("Inspection must be read-only")
    registry = fixture_registry()
    registry.get_action("notes", "read")["description"] = "password=" + "synthetic-secret-value"
    runtime = SimpleNamespace(services=SimpleNamespace(registry=registry, event_sink=None,
                              config_provider=SimpleNamespace(snapshot=forbidden)),
                              start=forbidden, resume=forbidden)
    service = WebAppService(runtime)
    try:
        result = service.tool_catalogue()
        assert "synthetic-secret-value" not in str(result)
        assert result["total"] == 2
    finally:
        service.close()


def test_lifecycle_pairing_preserves_repeated_invocations_and_event_references():
    events = [
        {"type": "graph_node", "node": "model_decide_node", "round_idx": 1, "phase": "started", "status": "running"},
        {"type": "model_call", "round": 1},
        {"type": "graph_node", "node": "model_decide_node", "round_idx": 1, "phase": "completed"},
        {"type": "graph_node", "node": "model_decide_node", "round_idx": 2, "phase": "started", "status": "running"},
    ]
    flow = recorded_flow(events, run_status="running")
    assert len(flow["nodes"]) == 3
    assert flow["nodes"][0]["event_refs"] == [0, 2]
    assert flow["nodes"][0]["status"] == "completed"
    assert flow["nodes"][-1]["status"] == "running"
    assert flow["edges"] == [{"source": "event_0", "target": "event_1"}, {"source": "event_1", "target": "event_3"}]


def test_terminal_or_pause_does_not_invent_missing_completion():
    events = [{"type": "graph_node", "node": "review", "phase": "started", "status": "running"}]
    pending = {"type": "artifact_review", "status": "pending"}
    flow = recorded_flow(events, run_status="awaiting_confirmation", pending=pending)
    assert [n["status"] for n in flow["nodes"]] == ["completion_unrecorded", "waiting"]
    assert flow["nodes"][-1]["event_refs"] == []
    assert recorded_flow(events, run_status="completed")["nodes"][0]["status"] == "completion_unrecorded"


def test_business_and_guard_statuses_are_not_inferred_from_node_completion():
    events = [
        {"type": "tool_guarded", "guard_results": [{"allowed": False}]},
        {"type": "tool_finished", "observation": {"status": "blocked"}},
        {"type": "graph_node", "node": "tool_execute_node", "phase": "completed"},
        {"type": "artifact.check", "valid": False},
        {"type": "future_event", "detail": "unknown event remains visible"},
    ]
    flow = recorded_flow(events, run_status="completed")
    assert [n["status"] for n in flow["nodes"]] == ["blocked", "blocked", "completed", "failed", "recorded"]
    assert flow["nodes"][-1]["type"] == "future_event"


def test_empty_or_bounded_evidence_is_explicit():
    assert recorded_flow([], run_status="completed")["nodes"] == []
    flow = recorded_flow([{"type": "observed"}] * 3, limit=3)
    assert flow["history_may_be_truncated"] is True
    assert recorded_flow([], pending={"type": "confirmation", "status": "pending"})["has_event_evidence"] is False
