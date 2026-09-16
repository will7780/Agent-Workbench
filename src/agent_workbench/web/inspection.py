"""Read-only projections of registered contracts and observed execution events."""
from collections.abc import Mapping

from ..tool_schema import capability_to_tool_name


def catalogue(registry, *, tool_name=None, limit=200):
    if registry is None:
        return {"available": False, "reason": "catalogue_not_exposed", "tools": [], "total": None}
    actions = registry.iter_actions()
    items = []
    for module, action, definition in actions:
        name = capability_to_tool_name(module, action)
        if tool_name is not None and name != tool_name:
            continue
        group = registry.get_module(module) or {}
        item = {
            "name": name, "capability_id": f"{module}.{action}",
            "group": group.get("display_name") or module,
            "label": definition.get("display_name") or action,
            "description": definition.get("description") or definition.get("notes"),
            "risk": definition.get("risk_level"),
            "declared_modes": registry.get_execution_mode_statuses(module, action),
        }
        if tool_name is not None:
            item["parameter_contract"] = registry.get_parameter_contract(module, action)
            return item
        if len(items) < limit:
            items.append(item)
    if tool_name is not None:
        return None
    return {"available": True, "version": registry.version, "tools": items,
            "total": len(actions), "truncated": len(actions) > limit,
            "scope": "registered_contracts_not_run_authorization"}


def _status(event):
    observation = event.get("observation")
    if isinstance(observation, Mapping) and observation.get("status"):
        return str(observation["status"])
    if event.get("allowed") is False:
        return "blocked"
    if event.get("valid") is False or event.get("error_type"):
        return "failed"
    for key in ("guard_results", "parameter_verification_results"):
        checks = event.get(key)
        if isinstance(checks, list) and any(isinstance(c, Mapping) and c.get("allowed") is False for c in checks):
            return "blocked"
    return str(event.get("status") or event.get("phase") or "recorded")


def recorded_flow(events, *, run_status=None, pending=None, limit=200):
    """Edges are observation/start order, not fabricated business dependencies.

    Repeated node invocations remain distinct. A start/completion pair is a
    single node with both event references; no completion is inferred from a
    later event or a terminal run. The caller already bounds/redacts events.
    """
    events = events if isinstance(events, list) else []
    nodes, open_nodes = [], {}
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        kind, phase = event.get("type") or "event", event.get("phase")
        key = (str(event.get("node")), str(event.get("round_idx")))
        if kind == "graph_node" and phase == "completed" and open_nodes.get(key):
            item = open_nodes[key].pop()
            item["status"] = "completed"
            item["event_refs"].append(index)
            continue
        tool = event.get("tool_call")
        tool = tool if isinstance(tool, Mapping) else {}
        item = {"id": f"event_{index}", "type": kind,
                "label": event.get("node") or event.get("tool_name") or tool.get("tool_name") or kind,
                "round": event.get("round_idx", event.get("round")),
                "status": _status(event), "event_refs": [index]}
        if kind == "graph_node" and phase == "started":
            open_nodes.setdefault(key, []).append(item)
        nodes.append(item)
    for items in open_nodes.values():
        for item in items:
            if run_status not in {"running", "in_progress"}:
                item["status"] = "completion_unrecorded"
    if isinstance(pending, Mapping) and pending.get("status") == "pending":
        nodes.append({"id": "pending", "type": "pending_interaction",
                      "label": pending.get("type") or "interaction", "status": "waiting",
                      "event_refs": [], "round": None})
    return {"nodes": nodes,
            "edges": [{"source": a["id"], "target": b["id"]} for a, b in zip(nodes, nodes[1:])],
            "order": "observed_start_order", "history_limit": limit,
            "history_may_be_truncated": len(events) >= limit,
            "has_event_evidence": any(node["event_refs"] for node in nodes)}
