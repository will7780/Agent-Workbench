"""Opt-in JSON 1.2 export. No Eval imports, uploads or provenance assertions.

export_trace accepts an AgentRuntime result (or its state). write_trace writes
one standalone JSON envelope suitable for Eval's trace ImportService.
"""
from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from .redaction import redact_recursive

_PRIVATE = {"reasoning_content", "chain_of_thought", "hidden_reasoning", "thinking", "analysis",
            "seal", "snapshot_seal", "artifact_binding", "bound_params", "original_params",
            "stage_dir", "source_root", "trusted"}
_MESSAGE_KEYS = {"role", "content", "name", "tool_calls", "tool_call_id"}
_LOCAL_PATH = "[LOCAL_PATH]"
_PATH_FIELD = re.compile(r"(?:^|_)(?:paths?|dirs?|directories|directory|folders?|files?|filenames?|filepaths?)$")
_PATH_START = re.compile(r"(?<![\w:/\\])(?:file://|[A-Za-z]:[/\\]|\\\\|//|~[/\\]|/(?![/\s]))", re.IGNORECASE)
_REMOTE_URL = re.compile(r"\b(?!file:)[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"'`]+", re.IGNORECASE)
_PROSE_END = re.compile(r"[\r\n\"'`<>|,;{}\[\]()]|\.(?=\s|$)|[\u3002\uff1b\uff0c]")
_FILENAME_END = re.compile(r".*?\.[A-Za-z0-9]{1,16}(?:\.[A-Za-z0-9]{1,16})*(?=$|[\s.])")
_NUMERIC_USAGE = {
    "agent_llm_calls", "judge_llm_calls", "tool_calls", "prompt_tokens", "completion_tokens",
    "total_tokens", "reasoning_tokens", "cache_hit_tokens", "cache_miss_tokens",
    "agent_llm_latency_ms", "judge_llm_latency_ms", "tool_latency_ms", "active_runtime_ms",
    "wall_runtime_ms", "user_wait_ms", "estimated_cost",
    *(f"{kind}_{field}_tokens" for kind in ("agent", "judge")
      for field in ("prompt", "completion", "total", "reasoning", "cache_hit", "cache_miss")),
}


def _path_field(key):
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key)).lower()
    return name in {"cwd", "raw_output_ref"} or bool(_PATH_FIELD.search(name))


def _redact_prose_paths(text):
    """Recognize local paths without treating roles, tool IDs or URLs as paths.

    Quotes delimit a complete path, including spaces and punctuation. Bare
    paths include spaced directory components and filenames; an ambiguous
    extensionless tail is conservatively hidden through the next delimiter.
    This is text recognition, not filesystem access or a privacy guarantee.
    """
    urls = list(_REMOTE_URL.finditer(text))
    starts = list(_PATH_START.finditer(text))
    chunks, cursor = [], 0
    for index, start in enumerate(starts):
        begin = start.start()
        if begin < cursor or any(url.start() <= begin < url.end() for url in urls):
            continue
        quote = text[begin - 1] if begin else ""
        end = text.find(quote, start.end()) if quote in {"\"", "'", "`"} else -1
        if end < 0:
            delimiter = _PROSE_END.search(text, start.end())
            end = delimiter.start() if delimiter else len(text)
            if index + 1 < len(starts):
                end = min(end, starts[index + 1].start())
            end = min([end] + [url.start() for url in urls if url.start() > begin])
            candidate = text[begin:end].rstrip()
            end = begin + len(candidate)
            final_component = max(candidate.rfind("/"), candidate.rfind("\\")) + 1
            filename = _FILENAME_END.match(candidate[final_component:])
            if filename:
                end = begin + final_component + filename.end()
        chunks.extend((text[cursor:begin], _LOCAL_PATH))
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def _clean(value, *, path_field=False):
    if isinstance(value, dict):
        return {key: _clean(item, path_field=_path_field(key)) for key, item in value.items()
                if str(key).lower() not in _PRIVATE}
    if isinstance(value, (list, tuple)):
        return [_clean(item, path_field=path_field) for item in value]
    if isinstance(value, str):
        if path_field:
            return _LOCAL_PATH if value else value
        # Tool observations and function arguments can themselves contain JSON.
        if value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except (ValueError, TypeError):
                pass
            else:
                if isinstance(decoded, (dict, list)):
                    cleaned, _ = redact_recursive(_clean(decoded), max_depth=32, max_list_items=10000)
                    return json.dumps(cleaned, ensure_ascii=False, allow_nan=False)
        return _redact_prose_paths(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _messages(value):
    return [{k: copy.deepcopy(v) for k, v in message.items() if k in _MESSAGE_KEYS}
            for message in value if isinstance(message, dict) and isinstance(message.get("role"), str)]


def _identifier(value, label, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "::" in value:
        raise ValueError("invalid_" + label)
    if _clean(value) != value or redact_recursive(value)[0] != value:
        raise ValueError("sensitive_" + label)
    return value


def _event_status(status, *, blocked=False):
    if blocked:
        return "blocked"
    return {"success": "ok", "dry_run": "ok", "completed": "ok", "failed": "error",
            "stopped": "error", "pending": "pending", "skipped": "skipped"}.get(status, "ok")


def export_trace(result: dict, *, project_id="agent-workbench", target_id="agent-workbench",
                 target_version="0.1.0a1", trace_id=None) -> dict:
    """Return a detached, redacted TraceEnvelopeV1 (contract_version 1.2).

    Only captured message arrays qualify as model input. Legacy split prompt
    fields are never reconstructed. Missing usage remains null. A timestamp
    fallback describes export time explicitly, not a fabricated runtime start.
    """
    state = result.get("state", result)
    if not isinstance(state, dict):
        raise ValueError("trace_state_required")
    report = result.get("report") or {}
    events, omissions = [], []

    def add(kind, attributes, *, name=None, status="ok", parent=None):
        sequence = len(events)
        if sequence >= 10000:
            raise ValueError("trace_event_limit_exceeded")
        event = {"contract_version": "1.2", "event_id": f"event-{sequence:05d}",
                 "sequence": sequence, "kind": kind, "status": status, "attributes": attributes}
        if name is not None:
            event["name"] = name
        if parent:
            event["parent_event_id"] = parent
        events.append(event)
        return event["event_id"]

    snapshots = state.get("llm_context_snapshots", report.get("llm_context_snapshots", [])) or []
    by_round = {s.get("round"): s for s in snapshots if isinstance(s, dict)}
    captured = set()

    def model_input(snapshot):
        if "messages" not in snapshot or not isinstance(snapshot["messages"], list):
            omissions.append("model_input_messages_unavailable")
            return
        fields = ("round", "snapshot_id", "model_call_id", "captured", "truncated", "redacted",
                  "omission_reason", "source", "model", "provider", "sent_to_llm", "tool_schemas")
        slim = {key: copy.deepcopy(snapshot[key]) for key in fields if key in snapshot}
        slim["messages"] = _messages(snapshot["messages"])
        add("model.input", {"snapshot": slim})
        captured.add(snapshot.get("round"))

    for entry in state.get("agent_trace", report.get("agent_trace", [])) or []:
        kind = entry.get("type")
        if kind == "model_call":
            snapshot = by_round.get(entry.get("round"))
            if snapshot is not None:
                model_input(snapshot)
            else:
                omissions.append("model_input_messages_unavailable")
            add("model.output", copy.deepcopy(entry), status="error" if entry.get("error_type") else "ok")
        elif kind == "tool_result":
            call, observation = entry.get("tool_call") or {}, entry.get("observation") or {}
            name = call.get("tool_name")
            attributes = {"tool_id": name, "tool_call_id": call.get("tool_call_id"),
                          "arguments": copy.deepcopy(call.get("arguments") or {}),
                          "adapter_called": call.get("adapter_called"),
                          "guard": copy.deepcopy(call.get("guard") or {})}
            parent = add("tool.call", attributes, name=name,
                         status=_event_status(observation.get("status"), blocked=call.get("adapter_called") is False))
            add("tool.result", {"tool_id": name, "tool_call_id": call.get("tool_call_id"),
                                "output": copy.deepcopy(observation)}, name=name, parent=parent,
                status=_event_status(observation.get("status"), blocked=call.get("adapter_called") is False))
        elif isinstance(kind, str) and kind.startswith("artifact."):
            add(kind, copy.deepcopy(entry), status="blocked" if entry.get("decision") in
                {"reject", "request_changes", "invalidated"} else "ok")
    for snapshot in snapshots:
        if snapshot.get("round") not in captured:
            model_input(snapshot)
    for interaction in state.get("interaction_history", []) or []:
        add("interaction.response", copy.deepcopy(interaction))
    pending = state.get("pending_interaction", report.get("pending_interaction"))
    if pending:
        add("interaction.request", copy.deepcopy(pending), status="pending")

    usage = {}
    raw_usage = state.get("resource_usage", report.get("resource_usage", {})) or {}
    aliases = {"agent_llm_calls": "agent_llm_call_count", "judge_llm_calls": "judge_llm_call_count",
               "tool_calls": "tool_call_count"}
    for key in _NUMERIC_USAGE:
        value = raw_usage.get(key, raw_usage.get(aliases.get(key)))
        integer = key.endswith(("_tokens", "_calls"))
        usage[key] = value if (type(value) in ((int,) if integer else (int, float))
                               and math.isfinite(value) and value >= 0) else None
    for key in ("currency", "price_card_version", "cost_status"):
        value = raw_usage.get(key)
        usage[key] = value if isinstance(value, str) else None
    raw_status = state.get("execution_status", report.get("status"))
    if pending:
        status = "awaiting_input" if pending.get("type") == "clarification" else "awaiting_confirmation"
    elif any(error.get("type") == "run_cancelled" for error in state.get("errors", [])):
        status = "cancelled"
    else:
        status = {"completed": "completed", "success": "completed", "running": "running",
                  "failed": "failed", "stopped": "interrupted", "cancelled": "cancelled"}.get(raw_status, "interrupted")
    telemetry = state.get("runtime_telemetry", report.get("runtime_telemetry", {})) or {}
    started_at = telemetry.get("started_at")
    if not started_at:
        started_at = datetime.now(timezone.utc).isoformat()
        omissions.append("runtime_start_unavailable_export_time_used")
    actual_input = next((s["messages"] for s in snapshots if isinstance(s.get("messages"), list)), None)
    if actual_input is None:
        omissions.append("initial_model_input_unavailable")
    input_data = {"messages": _messages(actual_input)} if actual_input is not None else {}
    if "user_request" in state:
        input_data["message"] = state["user_request"]
    envelope = {
        "contract_version": "1.2",
        "trace_id": _identifier(trace_id or state.get("run_id"), "trace_id", 160),
        "project_id": _identifier(project_id, "project_id", 120),
        "target_id": _identifier(target_id, "target_id", 120),
        "target_version": _identifier(target_version, "target_version", 80),
        "started_at": started_at, "status": status, "input": input_data,
        "output": {"response": state.get("final_response", report.get("final_response")),
                   "pending_interaction": pending, "artifacts": state.get("artifacts", [])},
        "resource_usage": usage, "events": events,
        "metadata": {"source": "agent_workbench.trace_export", "provenance": "unverified_local_export",
                     "automatic_upload": False, "business_evidence": False,
                     "omission_reasons": sorted(set(omissions)),
                     "path_redaction_limit": "Unquoted prose path boundaries can be ambiguous; review before sharing.",
                     "event_order": "runtime_log_order; interaction_history_appended",
                     "notice": "Runtime observations only; not independent business proof."},
        "tags": {"model": str(state.get("llm_model") or "unknown")},
    }
    safe, _ = redact_recursive(_clean(envelope), max_depth=32, max_list_items=10000)
    safe["metadata"]["trusted"] = False
    safe["metadata"]["local_paths"] = "structured_fields_and_detected_prose_paths_redacted"
    # Validate JSON serializability here, without requiring commerce_eval.
    json.dumps(safe, allow_nan=False)
    return safe


def write_trace(result: dict, path, **options) -> Path:
    """Write only to the caller-selected local path; never upload or overwrite."""
    payload = export_trace(result, **options)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return destination.resolve()


def build_trace_envelope(result: dict, project_id="default", **options) -> dict:
    """CLI/host entry point for the standalone Eval 1.2 envelope."""
    return export_trace(result, project_id=project_id, **options)
