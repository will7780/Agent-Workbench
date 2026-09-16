"""Bounded HTTP operation adapter; the injected runtime owns every execution.

The host supplies start(request), resume(run_id, interaction_id, response),
cancel(run_id), list_runs() and get_run(run_id). No graph, checkpoint, plugin,
credential loading or persistence belongs here. Operations are transport jobs,
not runs; only the runtime may assign a run ID or authorize an interaction.
"""

from __future__ import annotations

import math
import re
import threading
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from ..redaction import redact_recursive, redact_text
from .inspection import catalogue, recorded_flow

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_HIDDEN = frozenset({"reasoning", "reasoning_content", "chain_of_thought", "scratchpad", "hidden_reasoning", "thinking", "analysis",
                     "api_key", "launch_command", "plugins", "credentials"})


class WebError(Exception):
    def __init__(self, code: str, status: int = 400):
        self.code, self.status = code, status
        super().__init__(code)


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise WebError("invalid_identifier")
    return value


def _bounded(value: Any, depth: int = 0, budget: list | None = None) -> Any:
    # Bound the tree before recursive redaction; never stringify arbitrary objects.
    budget = [12000] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 12:
        return "[truncated]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return redact_text(value)[0][:16000]
    if isinstance(value, Mapping):
        return {redact_text(str(k))[0][:200]: _bounded(v, depth + 1, budget)
                for k, v in list(value.items())[:200]
                if str(k).lower() not in _HIDDEN}
    if isinstance(value, (list, tuple)):
        return [_bounded(item, depth + 1, budget) for item in value[:200]]
    return None


def public_data(value: Any) -> Any:
    return redact_recursive(_bounded(value), max_depth=14, max_list_items=200)[0]


def _dict(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _run_id(result: dict) -> str | None:
    value = result.get("run_id") or _dict(result.get("state")).get("run_id") or _dict(result.get("report")).get("run_id")
    return value if isinstance(value, str) and IDENTIFIER.fullmatch(value) else None


def project_result(result: Any, *, summary_only: bool = False) -> dict:
    bundle = _dict(result)
    state, report = _dict(bundle.get("state")), _dict(bundle.get("report"))

    def field(name: str, default: Any = None) -> Any:
        for source in (state, report, bundle):
            if name in source:
                return source[name]
        return default

    if summary_only:
        return public_data({
            "run_id": _run_id(bundle), "thread_id": field("thread_id"),
            "status": field("execution_status", field("status", "unknown")),
            "user_request": field("user_request", field("goal")),
            "execution_mode": field("execution_mode", field("requested_execution_mode")),
            "model_label": field("model_label", field("llm_model")),
            "offline": field("offline"), "summary_only": True,
        })

    pending = field("pending_interaction")
    if not isinstance(pending, Mapping) or pending.get("status") != "pending":
        pending = None
    usage = _dict(field("usage", field("resource_usage", {})))
    telemetry = _dict(field("runtime_telemetry", field("telemetry", {})))
    metrics = {}
    metric_sources = {"prompt_tokens": "agent_prompt_tokens", "completion_tokens": "agent_completion_tokens",
                      "total_tokens": "agent_total_tokens", "latency_ms": "agent_llm_latency_ms",
                      "tool_latency_ms": "tool_latency_ms", "active_runtime_ms": "active_runtime_ms",
                      "wall_runtime_ms": "wall_runtime_ms", "user_wait_ms": "user_wait_ms"}
    for key, source_key in metric_sources.items():
        value = usage.get(key, usage.get(source_key, telemetry.get(key, field(key))))
        metrics[key] = value if (isinstance(value, (int, float)) and not isinstance(value, bool)
                                 and math.isfinite(value) and value >= 0
                                 and (not key.endswith("_tokens") or isinstance(value, int))) else None
    messages = field("messages", [])
    normalized = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role") or message.get("type") or "unknown"
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        normalized.append({"role": role, "content": message.get("content"),
                           "tool_calls": message.get("tool_calls"), "tool_call_id": message.get("tool_call_id"),
                           "name": message.get("name")})
    view = {
        "run_id": _run_id(bundle), "thread_id": field("thread_id"),
        "status": field("execution_status", field("status", "unknown")), "user_request": field("user_request", field("goal")),
        "execution_mode": field("execution_mode", field("requested_execution_mode")), "model_label": field("model_label", field("llm_model")),
        "offline": field("offline"), "messages": normalized,
        "pending_interaction": pending, "final_response": field("final_response"),
        "tools": field("tool_calls", field("tool_call_history", [])),
        "parameter_sources": field("parameter_sources"),
        "parameter_verification": field("parameter_verification_results"),
        "observations": field("observations"),
        "context": field("context_manifest", field("context", {
            "snapshots": field("llm_context_snapshots"), "budget": field("context_budget"),
            "budget_history": field("context_budget_history"),
            "memory_manifest": field("memory_context_manifest"),
            "knowledge_manifest": field("knowledge_context_manifest"),
            "skill_manifest": field("skill_context_manifest"),
        })),
        "artifacts": field("artifacts", field("artifact_evidence")),
        "metrics": metrics, "error": field("error", field("error_type", field("errors"))),
        "events": field("live_events", field("agent_trace", telemetry.get("spans", []))),
        "cancellation_requested": field("cancellation_requested"),
    }
    return public_data(view)


class WebAppService:
    """Share ONE instance between both servers. No execution policy defaults.

    Pass offline=True only when the host actually uses a simulated model.
    services.event_sink is chained when available; otherwise the host can wire
    service.event_sink itself. Polling still works without event callbacks.
    """

    def __init__(self, runtime: Any, *, offline: bool | None = None,
                 model_label: str | None = None, max_jobs: int = 8,
                 max_history: int = 128, max_events: int = 200):
        self.runtime = runtime
        self.offline, self.model_label = offline, model_label
        self.max_jobs, self.max_history, self.max_events = max_jobs, max_history, max_events
        if min(max_jobs, max_history, max_events) < 1 or max_history < max_jobs + 1:
            raise ValueError("invalid_web_limits")
        self._lock = threading.RLock()
        self._local = threading.local()
        self._jobs: OrderedDict[str, dict] = OrderedDict()
        self._events: OrderedDict[str, deque] = OrderedDict()
        self._active: dict[str, str] = {}
        self._threads: set[threading.Thread] = set()
        self._closed = False
        self._services = getattr(runtime, "services", None)
        self._previous_sink = getattr(self._services, "event_sink", None)
        self._sink = self.event_sink
        if self._services is not None:
            self._services.event_sink = self._sink

    def event_sink(self, event: Any) -> None:
        try:
            data = public_data(event)
            if isinstance(data, dict):
                with self._lock:
                    operation = self._jobs.get(getattr(self._local, "operation_id", ""))
                    run_id = data.get("run_id") or (operation or {}).get("run_id")
                    if isinstance(run_id, str) and IDENTIFIER.fullmatch(run_id):
                        if operation is not None:
                            operation["run_id"] = run_id
                        if run_id not in self._events:
                            self._events[run_id] = deque(maxlen=self.max_events)
                        self._events.move_to_end(run_id)
                        data["received_at"] = datetime.now(timezone.utc).isoformat()
                        self._events[run_id].append(data)
                        while len(self._events) > self.max_history:
                            self._events.popitem(last=False)
        except Exception:
            pass  # Presentation must never alter runtime execution.
        if callable(self._previous_sink):
            self._previous_sink(event)

    def metadata(self) -> dict:
        return public_data({"offline": self.offline, "model_label": self.model_label,
                            "event_limit": self.max_events, "run_limit": self.max_history})

    def tool_catalogue(self, tool_name: str | None = None) -> dict:
        if tool_name is not None:
            identifier(tool_name)
        result = catalogue(getattr(self._services, "registry", None), tool_name=tool_name)
        if result is None:
            raise WebError("tool_not_found", 404)
        return public_data(result)

    def _decorate(self, result: Any) -> dict:
        view = project_result(result)
        with self._lock:
            observed = self._events.get(view["run_id"])
            view["events"] = list(observed) if observed else (view["events"] or [])[-self.max_events:]
            view["event_limit"] = self.max_events
            view["operations"] = [self._operation_view(job) for job in self._jobs.values()
                                  if job.get("run_id") == view["run_id"] and job["status"] == "running"]
        view["flow"] = recorded_flow(view["events"], run_status=view["status"],
                                     pending=view["pending_interaction"], limit=self.max_events)
        return view

    def list_runs(self) -> dict:
        results = self.runtime.list_runs()
        if not isinstance(results, list):
            raise WebError("runtime_result_invalid", 502)
        # Lists are navigation summaries; evidence is fetched per run. Repeating
        # entire histories here can exhaust the response budget and corrupt flags.
        views = [project_result(result, summary_only=True) for result in results[:self.max_history]]
        with self._lock:
            active = [self._operation_view(job) for job in self._jobs.values() if job["status"] == "running"]
        return {"runs": views, "operations": active, "truncated": len(results) > self.max_history}

    def get_run(self, run_id: str) -> dict:
        try:
            result = self.runtime.get_run(identifier(run_id))
        except KeyError:
            raise WebError("run_not_found", 404) from None
        if not result:
            raise WebError("run_not_found", 404)
        return self._decorate(result)

    @staticmethod
    def _operation_view(job: dict) -> dict:
        return {k: v for k, v in job.items() if k != "result"}

    def get_operation(self, operation_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(identifier(operation_id))
            if job is None:
                raise WebError("operation_not_found", 404)
            return public_data(dict(job))

    def start(self, request: dict) -> dict:
        if set(request) - {"user_request", "thread_id"}:
            raise WebError("unsupported_request_field")
        text = request.get("user_request")
        if not isinstance(text, str) or not text.strip() or len(text) > 12000:
            raise WebError("invalid_message")
        clean, redacted = redact_text(text)
        if redacted:
            raise WebError("sensitive_input_rejected")
        payload = {"user_request": clean.strip()}
        thread_id = request.get("thread_id")
        if thread_id is not None:
            payload["thread_id"] = identifier(thread_id)
        key = f"thread:{thread_id}" if thread_id else f"start:{uuid.uuid4().hex}"
        return self._submit("start", key, lambda: self.runtime.start(payload))

    def resume(self, run_id: str, payload: dict) -> dict:
        run_id = identifier(run_id)
        interaction_id = identifier(payload.get("interaction_id"))
        response = payload.get("response")
        if set(payload) != {"interaction_id", "response"} or not isinstance(response, dict):
            raise WebError("invalid_interaction_response")
        pending = self.get_run(run_id).get("pending_interaction")
        if not pending or pending.get("interaction_id") != interaction_id:
            raise WebError("interaction_mismatch", 409)
        kind = pending.get("type")
        allowed = {"type", "interaction_id", "answer"} if kind == "clarification" else {"type", "interaction_id", "decision"}
        if kind == "artifact_review":
            allowed.add("comment")
        if kind not in {"clarification", "confirmation", "artifact_review"} or set(response) - allowed:
            raise WebError("invalid_interaction_response")
        if response.get("type") != kind or response.get("interaction_id", interaction_id) != interaction_id:
            raise WebError("interaction_mismatch", 409)
        safe = {**response, "interaction_id": interaction_id}
        if kind == "clarification":
            answer = safe.get("answer")
            if not isinstance(answer, str) or not answer.strip() or len(answer) > 4000:
                raise WebError("invalid_answer")
        else:
            choices = {"approve", "reject", "request_changes"} if kind == "artifact_review" else {"approve", "reject"}
            if safe.get("decision") not in choices:
                raise WebError("invalid_decision")
            if kind == "artifact_review" and safe["decision"] == "approve" and pending.get("can_approve") is not True:
                raise WebError("artifact_has_errors", 409)
            comment = safe.get("comment", "")
            if not isinstance(comment, str) or len(comment) > 4000:
                raise WebError("invalid_comment")
        for field in ("answer", "comment"):
            if field in safe and redact_text(safe[field])[1]:
                raise WebError("sensitive_input_rejected")
        return self._submit("resume", f"run:{run_id}",
                            lambda: self.runtime.resume(run_id, interaction_id, safe), run_id)

    def cancel(self, run_id: str) -> dict:
        run_id = identifier(run_id)
        self.get_run(run_id)
        return self._submit("cancel", f"cancel:{run_id}", lambda: self.runtime.cancel(run_id), run_id)

    def _submit(self, kind: str, key: str, call: Any, run_id: str | None = None) -> dict:
        with self._lock:
            if self._closed:
                raise WebError("web_service_closed", 503)
            if key in self._active or (kind == "resume" and f"cancel:{run_id}" in self._active):
                raise WebError("operation_in_progress", 409)
            ordinary = sum(job["kind"] != "cancel" and job["status"] == "running" for job in self._jobs.values())
            cancelling = any(job["kind"] == "cancel" and job["status"] == "running" for job in self._jobs.values())
            if (kind != "cancel" and ordinary >= self.max_jobs) or (kind == "cancel" and cancelling):
                raise WebError("web_capacity_exceeded", 429)
            while len(self._jobs) >= self.max_history:
                old = next((k for k, v in self._jobs.items() if v["status"] != "running"), None)
                if old is None:
                    raise WebError("web_capacity_exceeded", 429)
                del self._jobs[old]
            operation_id = "op_" + uuid.uuid4().hex
            job = {"operation_id": operation_id, "kind": kind, "run_id": run_id, "status": "running"}
            self._jobs[operation_id] = job
            self._active[key] = operation_id
            worker = threading.Thread(target=self._execute, args=(job, key, call), daemon=True,
                                      name="workbench-web-operation")
            self._threads.add(worker)
            try:
                worker.start()
            except Exception:
                self._threads.discard(worker)
                self._active.pop(key, None)
                self._jobs.pop(operation_id, None)
                raise WebError("web_worker_unavailable", 503) from None
            return self._operation_view(job)

    def _execute(self, job: dict, key: str, call: Any) -> None:
        self._local.operation_id = job["operation_id"]
        try:
            result = call()
            if not isinstance(result, dict):
                raise ValueError("runtime_result_invalid")
            view = project_result(result)
            with self._lock:
                job.update(status="completed", run_id=view["run_id"] or job["run_id"], result=view)
        except Exception:
            # Never expose exception repr, tracebacks, request data or local paths.
            with self._lock:
                job.update(status="failed", error_type="runtime_operation_failed")
        finally:
            with self._lock:
                self._active.pop(key, None)
                self._threads.discard(threading.current_thread())
            self._local.operation_id = None

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            threads = list(self._threads)
        if wait:
            for worker in threads:
                worker.join()
        if self._services is not None and self._services.event_sink == self._sink:
            self._services.event_sink = self._previous_sink
