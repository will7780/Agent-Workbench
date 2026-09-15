"""Loopback-only web contract tests. No model, credentials, plugins or paid I/O."""

from __future__ import annotations

import copy
import http.client
import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_workbench.web.server import MAX_BODY_BYTES, create_servers
from agent_workbench.web.service import WebAppService, project_result, public_data


class FakeRuntime:
    def __init__(self):
        self.previous_events = []
        self.services = SimpleNamespace(event_sink=self.previous_events.append)
        self.runs = {}
        self.calls = []
        self.lock = threading.RLock()
        self.release = threading.Event()
        self.started = threading.Event()

    def add(self, run_id="run_fixture", *, kind=None, can_approve=True):
        pending = None
        if kind:
            pending = {"interaction_id": "int_fixture", "type": kind, "status": "pending",
                       "question": "Review this synthetic draft", "reason": "Human review required",
                       "can_approve": can_approve, "sample": [{"row_id": "row_1", "title": "Synthetic note"}],
                       "evidence": {"artifact_id": "artifact_fixture", "version": 1, "rows": 1,
                                    "content_hash": "fixture_hash", "errors": []},
                       "actions": [{"capability_id": "notes.archive", "risk_level": "L2",
                                    "execution_mode": "local_write", "params_summary": "{}"}]}
        state = {"run_id": run_id, "thread_id": "thread_fixture", "status": "awaiting_input" if kind else "completed",
                 "user_request": "Read synthetic notes", "messages": [
                     {"role": "user", "content": "Read synthetic notes"},
                     {"role": "assistant", "content": "", "tool_calls": [{"name": "notes.read"}]},
                     {"role": "tool", "content": "Synthetic note", "tool_call_id": "call_fixture"}],
                 "pending_interaction": pending, "execution_mode": "local_write",
                 "parameter_sources": {"title": "user"}, "context_manifest": {"items": []},
                 "resource_usage": {"prompt_tokens": 0, "completion_tokens": None, "latency_ms": None}}
        bundle = {"state": state, "report": {}, "report_text": "Synthetic report"}
        with self.lock:
            self.runs[run_id] = bundle
        return copy.deepcopy(bundle)

    def start(self, request):
        with self.lock:
            self.calls.append(("start", copy.deepcopy(request)))
            run_id = f"run_{len(self.runs) + 1}"
            self.add(run_id)
            self.runs[run_id]["state"].update(status="running", user_request=request["user_request"],
                                            thread_id=request.get("thread_id"))
        self.services.event_sink({"run_id": run_id, "type": "tool_running", "tool_call": {"tool_name": "notes.read"}})
        self.started.set()
        if request["user_request"] == "hold":
            self.release.wait(5)
        if request["user_request"] == "fail":
            raise ValueError("password=" + "synthetic-error-value")
        with self.lock:
            if self.runs[run_id]["state"]["status"] != "cancelled":
                self.runs[run_id]["state"]["status"] = "completed"
            return copy.deepcopy(self.runs[run_id])

    def resume(self, run_id, interaction_id, response):
        with self.lock:
            self.calls.append(("resume", run_id, interaction_id, copy.deepcopy(response)))
            state = self.runs[run_id]["state"]
            assert state["pending_interaction"]["interaction_id"] == interaction_id
            if response.get("decision") == "request_changes":
                state["pending_interaction"]["interaction_id"] = "int_revision"
                state["pending_interaction"]["evidence"]["version"] = 2
            else:
                state["pending_interaction"] = None
                state["status"] = "rejected" if response.get("decision") == "reject" else "completed"
            return copy.deepcopy(self.runs[run_id])

    def cancel(self, run_id):
        with self.lock:
            self.calls.append(("cancel", run_id))
            self.runs[run_id]["state"].update(status="cancelled", pending_interaction=None)
            result = copy.deepcopy(self.runs[run_id])
        self.release.set()
        return result

    def get_run(self, run_id):
        with self.lock:
            return copy.deepcopy(self.runs.get(run_id))

    def list_runs(self):
        with self.lock:
            return copy.deepcopy(list(self.runs.values()))


class WebTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AGENT_WORKBENCH_DISABLE_CENTRAL_ENV": "1"})
        self.env.start()
        self.runtime = FakeRuntime()
        self.servers = create_servers(self.runtime, chat_port=0, diagnostics_port=0,
                                      offline=True, model_label="Synthetic test model")
        self.service = self.servers[0].RequestHandlerClass.service
        self.threads = [threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}) for server in self.servers]
        for thread in self.threads:
            thread.start()

    def tearDown(self):
        self.runtime.release.set()
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join()
        self.service.close()
        self.env.stop()

    def request(self, method, path, body=None, *, port_index=0, headers=None, raw=None):
        port = self.servers[port_index].server_address[1]
        values = {"Origin": f"http://127.0.0.1:{port}", "Content-Type": "application/json"}
        values.update(headers or {})
        values = {key: value for key, value in values.items() if value is not None}
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request(method, path, body=data, headers=values)
            response = connection.getresponse()
            content = response.read()
            result = json.loads(content) if response.getheader("Content-Type", "").startswith("application/json") else content.decode()
            return response.status, result, dict(response.getheaders())
        finally:
            connection.close()

    def done(self, operation):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status, result, _ = self.request("GET", "/api/operations/" + operation["operation_id"])
            self.assertEqual(status, 200)
            if result["status"] != "running":
                return result
            time.sleep(.01)
        self.fail("Operation did not finish")

    def resume(self, kind, **response):
        return self.request("POST", "/api/runs/run_fixture/resume", {
            "interaction_id": "int_fixture", "response": {"type": kind, **response}})

    def test_both_servers_share_runtime_and_history(self):
        self.assertIs(self.servers[1].RequestHandlerClass.service, self.service)
        status, operation, _ = self.request("POST", "/api/runs", {"user_request": "hello", "thread_id": "thread_test"})
        self.assertEqual(status, 202)
        result = self.done(operation)
        self.assertEqual(result["status"], "completed")
        status, listing, _ = self.request("GET", "/api/runs", port_index=1)
        self.assertEqual(status, 200)
        self.assertEqual(listing["runs"][0]["run_id"], result["run_id"])
        self.assertEqual(self.runtime.calls[0][1], {"user_request": "hello", "thread_id": "thread_test"})
        self.assertEqual(len(self.runtime.previous_events), 1)

    def test_long_start_does_not_block_health_and_can_cancel(self):
        status, operation, _ = self.request("POST", "/api/runs", {"user_request": "hold"})
        self.assertEqual(status, 202)
        self.assertTrue(self.runtime.started.wait(1))
        self.assertEqual(self.request("GET", "/api/health", port_index=1)[0], 200)
        self.assertFalse(self.runtime.release.is_set())
        listing = self.request("GET", "/api/runs")[1]
        run_id = listing["runs"][0]["run_id"]
        self.assertEqual(self.service.get_operation(operation["operation_id"])["run_id"], run_id)
        status, cancel, _ = self.request("POST", f"/api/runs/{run_id}/cancel", {})
        self.assertEqual(status, 202)
        self.assertEqual(self.done(cancel)["result"]["status"], "cancelled")
        self.assertEqual(self.done(operation)["result"]["status"], "cancelled")

    def test_duplicate_start_same_thread_is_busy(self):
        _, operation, _ = self.request("POST", "/api/runs", {"user_request": "hold", "thread_id": "one"})
        self.assertTrue(self.runtime.started.wait(1))
        self.assertEqual(self.request("POST", "/api/runs", {"user_request": "hold", "thread_id": "one"})[0], 409)
        self.runtime.release.set()
        self.done(operation)

    def test_clarification_uses_canonical_response(self):
        self.runtime.add(kind="clarification")
        status, operation, _ = self.resume("clarification", answer="Use the short title")
        self.assertEqual(status, 202)
        self.done(operation)
        call = self.runtime.calls[-1]
        self.assertEqual(call[2], "int_fixture")
        self.assertEqual(call[3], {"interaction_id": "int_fixture", "type": "clarification", "answer": "Use the short title"})

    def test_parameter_confirmation_rejection(self):
        self.runtime.add(kind="confirmation")
        status, operation, _ = self.resume("confirmation", decision="reject")
        self.assertEqual(status, 202)
        self.assertEqual(self.done(operation)["result"]["status"], "rejected")

    def test_approval_is_not_replayed(self):
        self.runtime.add(kind="confirmation")
        status, operation, _ = self.resume("confirmation", decision="approve")
        self.assertEqual(status, 202)
        self.done(operation)
        self.assertEqual(self.resume("confirmation", decision="approve")[0], 409)
        self.assertEqual(len(self.runtime.calls), 1)

    def test_artifact_approval_requires_explicit_true(self):
        for value in (False, None, "true", 1):
            self.runtime.add(kind="artifact_review", can_approve=value)
            self.assertEqual(self.resume("artifact_review", decision="approve")[0], 409)
        self.assertEqual(self.runtime.calls, [])

    def test_artifact_reject_and_revise_forward_comment(self):
        for decision in ("reject", "request_changes"):
            self.runtime.add(kind="artifact_review", can_approve=False)
            status, operation, _ = self.resume("artifact_review", decision=decision, comment="Revise the synthetic title")
            self.assertEqual(status, 202)
            result = self.done(operation)["result"]
            self.assertEqual(self.runtime.calls[-1][3]["comment"], "Revise the synthetic title")
            if decision == "request_changes":
                self.assertEqual(result["pending_interaction"]["interaction_id"], "int_revision")

    def test_resume_rejects_parameter_tampering(self):
        self.runtime.add(kind="confirmation")
        for field in ("params", "risk_level", "parameter_snapshot_hash", "execution_mode", "arguments"):
            self.assertEqual(self.resume("confirmation", decision="approve", **{field: "tampered"})[0], 400)
        self.assertEqual(self.runtime.calls, [])

    def test_resume_rejects_mismatch_and_unknown_decisions(self):
        self.runtime.add(kind="confirmation")
        self.assertEqual(self.resume("artifact_review", decision="approve")[0], 409)
        self.assertEqual(self.resume("confirmation", decision="request_changes")[0], 400)
        self.assertEqual(self.resume("confirmation", decision="approve", interaction_id="other")[0], 409)

    def test_events_do_not_manufacture_pending_interactions(self):
        self.runtime.add()
        self.service.event_sink({"run_id": "run_fixture", "type": "interaction_required", "pending_interaction": {"status": "pending"}})
        view = self.service.get_run("run_fixture")
        self.assertIsNone(view["pending_interaction"])
        self.assertEqual(len(view["events"]), 1)

    def test_real_roles_and_unknown_usage(self):
        self.runtime.add()
        view = self.request("GET", "/api/runs/run_fixture")[1]
        self.assertEqual([message["role"] for message in view["messages"]], ["user", "assistant", "tool"])
        self.assertEqual(view["metrics"]["prompt_tokens"], 0)
        self.assertIsNone(view["metrics"]["completion_tokens"])
        self.assertIsNone(view["metrics"]["latency_ms"])

    def test_redaction_and_hidden_reasoning(self):
        secret = "synthetic-fixture-only"
        self.runtime.add()
        self.runtime.runs["run_fixture"]["state"]["context_manifest"] = {
            "password": secret, "authorization": secret, "reasoning": "hidden-internal-text"}
        self.service.event_sink({"run_id": "run_fixture", "type": "tool_finished", "message": "password=" + secret})
        serialized = json.dumps(self.request("GET", "/api/runs/run_fixture")[1])
        self.assertNotIn(secret, serialized)
        self.assertNotIn("hidden-internal-text", serialized)

    def test_worker_error_never_echoes_exception(self):
        _, operation, _ = self.request("POST", "/api/runs", {"user_request": "fail"})
        result = self.done(operation)
        self.assertEqual(result["error_type"], "runtime_operation_failed")
        self.assertNotIn("synthetic-error-value", json.dumps(result))

    def test_get_error_never_echoes_exception(self):
        with patch.object(self.runtime, "list_runs", side_effect=ValueError("sensitive-fixture")):
            status, result, _ = self.request("GET", "/api/runs")
        self.assertEqual(status, 500)
        self.assertNotIn("sensitive-fixture", json.dumps(result))

    def test_unknown_routes_and_path_traversal(self):
        for path in ("/../redaction.py", "/%2e%2e/redaction.py", "/static/../../redaction.py", "/styles.css/anything", "/api/plugins", "/api/launch-command", "/favicon.ico"):
            self.assertEqual(self.request("GET", path)[0], 404, path)
        for path in ("/api/plugins", "/api/launch-command", "/api/upload"):
            self.assertEqual(self.request("POST", path, {})[0], 404)

    def test_origin_and_host_are_checked(self):
        other_port = self.servers[1].server_address[1]
        for headers in ({"Origin": None}, {"Origin": "null"}, {"Origin": "https://example.invalid"},
                        {"Origin": f"http://127.0.0.1:{other_port}"}, {"Host": "example.invalid"},
                        {"Sec-Fetch-Site": "same-site"}, {"Sec-Fetch-Site": "cross-site"}):
            self.assertEqual(self.request("POST", "/api/runs", {"user_request": "hello"}, headers=headers)[0], 403)
        self.assertEqual(self.runtime.calls, [])
        self.assertEqual(self.request("GET", "/api/runs", headers={"Origin": "https://example.invalid"})[0], 403)

    def test_invalid_bodies_and_body_limits(self):
        for raw in (b"[]", b"null", b"{", b'{"user_request":"a","user_request":"b"}', b'{"user_request":NaN}'):
            self.assertEqual(self.request("POST", "/api/runs", raw=raw)[0], 400)
        self.assertEqual(self.request("POST", "/api/runs", raw=b"x" * (MAX_BODY_BYTES + 1))[0], 413)
        self.assertEqual(self.request("POST", "/api/runs", {}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/runs", {}, headers={"Transfer-Encoding": "chunked"})[0], 400)

    def test_no_config_plugin_command_or_secret_input(self):
        for key in ("api_key", "plugins", "launch_command", "execution_mode", "module_config", "run_id"):
            status, _, _ = self.request("POST", "/api/runs", {"user_request": "hello", key: "fixture"})
            self.assertEqual(status, 400)
        self.assertEqual(self.request("POST", "/api/runs", {"user_request": "password=" + "fixture-value"})[0], 400)
        self.assertEqual(self.runtime.calls, [])

    def test_static_pages_security_headers_and_no_remote_assets(self):
        for index, title in enumerate(("Chat", "Diagnostics")):
            status, html, headers = self.request("GET", "/", port_index=index)
            self.assertEqual(status, 200)
            self.assertIn(f"Agent Workbench | {title}", html)
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertNotIn("https://", html)
            self.assertNotIn('type="file"', html)
        script = self.request("GET", "/app.js")[1]
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("insertAdjacentHTML", script)

    def test_metadata_never_claims_offline_by_default(self):
        other = WebAppService(FakeRuntime())
        try:
            self.assertIsNone(other.metadata()["offline"])
        finally:
            other.close()
        self.assertIs(self.request("GET", "/api/meta")[1]["offline"], True)

    def test_bounded_event_buffers_and_host_sink_restoration(self):
        runtime = FakeRuntime()
        service = WebAppService(runtime, max_jobs=1, max_history=3, max_events=2)
        try:
            for run_id in range(8):
                for index in range(6):
                    service.event_sink({"run_id": f"run_{run_id}", "index": index})
            self.assertEqual(len(service._events), 3)
            self.assertEqual(len(service._events["run_7"]), 2)
        finally:
            service.close()
        self.assertEqual(runtime.services.event_sink, runtime.previous_events.append)

    def test_public_data_bounds_cycles_and_non_json_values(self):
        cyclic = []; cyclic.append(cyclic)
        encoded = json.dumps(public_data({"loop": cyclic, "object": object(), "number": float("nan")}))
        self.assertIn("truncated", encoded)
        self.assertNotIn("object at", encoded)

    def test_no_non_loopback_bind(self):
        with self.assertRaisesRegex(ValueError, "loopback_only"):
            create_servers(self.runtime, host="0.0.0.0")

    def test_public_projection_omits_eval_and_private_report(self):
        view = project_result({"state": {"run_id": "run_x"}, "report": {"eval_scores": [1, 2]}, "report_text": "private-text"})
        self.assertNotIn("eval_scores", json.dumps(view))
        self.assertNotIn("private-text", json.dumps(view))

    def test_main_runtime_projection_keys(self):
        view = project_result({"state": {"run_id": "run_main", "execution_status": "stopped",
                              "resource_usage": {"agent_prompt_tokens": 0, "agent_total_tokens": None,
                                                 "agent_llm_latency_ms": 2.5, "wall_runtime_ms": 10},
                              "llm_context_snapshots": [{"messages": [{"role": "system", "content": "Synthetic policy"}]}],
                              "agent_trace": [{"type": "tool_selected"}], "errors": [{"type": "model_not_configured"}]},
                              "report": {"execution_mode": "dry_run"}})
        self.assertEqual(view["status"], "stopped")
        self.assertEqual(view["metrics"]["prompt_tokens"], 0)
        self.assertEqual(view["metrics"]["latency_ms"], 2.5)
        self.assertEqual(view["metrics"]["wall_runtime_ms"], 10)
        self.assertIsNone(view["metrics"]["total_tokens"])
        self.assertEqual(view["context"]["snapshots"][0]["messages"][0]["role"], "system")
        self.assertEqual(view["events"][0]["type"], "tool_selected")
        self.assertEqual(view["error"][0]["type"], "model_not_configured")

    def test_rich_run_list_does_not_truncate_flags_or_replace_evidence(self):
        for index in range(5):
            run_id = f"run_rich_{index}"
            self.runtime.add(run_id)
            self.runtime.runs[run_id]["state"]["context_manifest"] = {
                "snapshots": [{"items": [{"value": i} for i in range(100)]} for _ in range(100)]}
        status, listing, _ = self.request("GET", "/api/runs")
        self.assertEqual(status, 200)
        self.assertIs(listing["truncated"], False)
        self.assertEqual(len(listing["runs"]), 5)
        self.assertTrue(all(item["summary_only"] is True for item in listing["runs"]))
        self.assertTrue(all("context" not in item for item in listing["runs"]))
        self.assertTrue(all(isinstance(item["run_id"], str) for item in listing["runs"]))
        self.assertEqual(self.runtime.get_run("run_rich_0")["state"]["context_manifest"]["snapshots"][0]["items"][0]["value"], 0)

    def test_runtime_keyerror_is_not_found(self):
        with patch.object(self.runtime, "get_run", side_effect=KeyError("run_not_found")):
            self.assertEqual(self.request("GET", "/api/runs/missing")[0], 404)

    def test_busy_resume_does_not_block_diagnostics(self):
        self.runtime.add(kind="clarification")
        original = self.runtime.resume

        def hold_resume(*args):
            self.runtime.started.set()
            self.runtime.release.wait(3)
            return original(*args)

        with patch.object(self.runtime, "resume", side_effect=hold_resume):
            status, operation, _ = self.resume("clarification", answer="Hold for test")
            self.assertEqual(status, 202)
            self.assertTrue(self.runtime.started.wait(1))
            self.assertEqual(self.request("GET", "/api/runs/run_fixture", port_index=1)[0], 200)
            self.assertEqual(self.resume("clarification", answer="Repeated")[0], 409)
            self.runtime.release.set()
            self.assertEqual(self.done(operation)["status"], "completed")

    def test_capacity_has_a_reserved_cancel_slot(self):
        self.service.max_jobs = 1
        _, operation, _ = self.request("POST", "/api/runs", {"user_request": "hold"})
        self.assertTrue(self.runtime.started.wait(1))
        self.assertEqual(self.request("POST", "/api/runs", {"user_request": "hello"})[0], 429)
        status, cancel, _ = self.request("POST", "/api/runs/run_1/cancel", {})
        self.assertEqual(status, 202)
        self.done(cancel)
        self.done(operation)

    def test_runtime_without_services_is_duck_typed(self):
        runtime = SimpleNamespace(start=lambda request: {"state": {"run_id": "external", "status": "completed"}},
                                  list_runs=lambda: [], get_run=lambda run_id: None)
        service = WebAppService(runtime)
        try:
            self.assertEqual(service.list_runs()["runs"], [])
            operation = service.start({"user_request": "Synthetic request"})
        finally:
            service.close()
        self.assertEqual(service.get_operation(operation["operation_id"])["run_id"], "external")


if __name__ == "__main__":
    unittest.main()
