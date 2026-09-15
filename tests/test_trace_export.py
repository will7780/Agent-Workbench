"""Set AGENT_WORKBENCH_EVAL_PYTHON to an installed Eval venv for ImportService QA."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess

import pytest
import requests

from agent_workbench import central_env, llm_client
from agent_workbench.demo import DEMO_REQUEST, build_demo_runtime
from agent_workbench.trace_export import build_trace_envelope, export_trace, write_trace


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_API_ENV_FILE", str(tmp_path / "disabled.env"))

    def forbidden(*args, **kwargs):
        pytest.fail("Trace export attempted network or credential loading")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(central_env, "load_central_agent_env", forbidden)
    monkeypatch.setattr(llm_client, "load_central_agent_env", forbidden)


@pytest.fixture
def result():
    return {"state": {
        "run_id": "trace-fixture", "execution_status": "completed", "user_request": "Inspect examples",
        "llm_model": "offline-reference-policy", "final_response": "Examples inspected",
        "runtime_telemetry": {"started_at": "2026-09-15T01:02:03+00:00"},
        "llm_context_snapshots": [{"round": 0, "captured": True, "truncated": False, "messages": [
            {"role": "system", "content": "Actual system input"},
            {"role": "developer", "content": "Actual developer input"},
            {"role": "user", "content": "Inspect examples"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "original-call", "type": "function", "function": {
                    "name": "Report/Read:v2", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "original-call", "content": "Original reply"},
        ]}],
        "agent_trace": [{"type": "model_call", "round": 0}, {
            "type": "tool_result", "tool_call": {"tool_name": "Report/Read:v2",
                "tool_call_id": "original-call", "adapter_called": True, "arguments": {}},
            "observation": {"status": "success", "summary": "Original reply"}}],
        "resource_usage": {"agent_llm_call_count": 1, "tool_call_count": 1,
                           "agent_prompt_tokens": None, "estimated_cost": None},
    }}


def test_actual_roles_original_names_and_no_business_provenance(result):
    original = copy.deepcopy(result)
    trace = build_trace_envelope(result)
    assert trace["contract_version"] == "1.2" and trace["project_id"] == "default"
    messages = trace["events"][0]["attributes"]["snapshot"]["messages"]
    assert messages == result["state"]["llm_context_snapshots"][0]["messages"]
    assert [m["role"] for m in messages] == ["system", "developer", "user", "assistant", "tool"]
    call = next(e for e in trace["events"] if e["kind"] == "tool.call")
    assert call["name"] == call["attributes"]["tool_id"] == "Report/Read:v2"
    assert trace["metadata"]["trusted"] is False and trace["metadata"]["business_evidence"] is False
    assert trace["metadata"]["automatic_upload"] is False
    assert trace["resource_usage"]["agent_llm_calls"] == 1
    assert trace["resource_usage"]["agent_prompt_tokens"] is None
    assert trace["resource_usage"]["estimated_cost"] is None
    assert result == original


def test_legacy_split_prompt_is_not_reconstructed(result):
    result["state"]["llm_context_snapshots"] = [{"round": 0, "system_prompt": "DO NOT RECONSTRUCT"}]
    trace = export_trace(result)
    assert not any(e["kind"] == "model.input" for e in trace["events"])
    assert "DO NOT RECONSTRUCT" not in json.dumps(trace)
    assert "messages" not in trace["input"]
    assert "model_input_messages_unavailable" in trace["metadata"]["omission_reasons"]


def test_redaction_in_nested_json_and_no_engine_seals(result):
    private = {"api_key": "PRIVATE_SENTINEL", "hidden_reasoning": "PRIVATE_SENTINEL",
               "path": "C:/Users/anonymous/private/report.json", "seal": "PRIVATE_SENTINEL",
               "source_root": "/home/anonymous/private", "trusted": True}
    result["state"]["llm_context_snapshots"][0]["messages"][-1]["content"] = json.dumps(private)
    result["state"]["artifacts"] = [private]
    serialized = json.dumps(export_trace(result))
    assert "PRIVATE_SENTINEL" not in serialized and "anonymous/private" not in serialized
    assert '"trusted": true' not in serialized and "[LOCAL_PATH]" in serialized


_PATH_CASES = [
    pytest.param(r"\\fixture-host\Fixture Share\Private Suffix\report draft.json", id="unc"),
    pytest.param("/srv/fixture-private/Private Suffix/report draft.json", id="posix-root"),
    pytest.param(r"Q:\Fixture Workspace\Private Suffix\report draft.json", id="windows-spaces"),
]


def _path_fixture(result, path):
    artifact = {"path": path, "inputPath": path, "filename": path, "output_dir": path,
                "raw_output_ref": path, "artifact_files": [path], "local_paths": [path],
                "summary": "Keep the checked row count: 12.", "row_count": 12}
    result["state"]["artifacts"] = [artifact]
    result["state"]["agent_trace"][1]["observation"]["artifacts"] = [copy.deepcopy(artifact)]
    messages = result["state"]["llm_context_snapshots"][0]["messages"]
    messages[-1]["content"] = json.dumps({"artifacts": [artifact], "conclusion": "Keep the conclusion."})
    messages[-2]["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "root_dir": path, "artifact_files": [path], "goal": "Keep all 12 rows."})
    result["state"]["final_response"] = f"Read {path}; checked 12 rows. Keep this conclusion."
    return result


@pytest.mark.parametrize("path", _PATH_CASES + [
    pytest.param(r"\\?\UNC\fixture-host\Fixture Share\Private Suffix", id="extended-unc-directory"),
    pytest.param("//fixture-host/Fixture Share/Private Suffix", id="forward-unc-directory"),
    pytest.param("/opt/fixture-private/Private Suffix", id="opt-directory"),
    pytest.param("/arbitrary-fixture/Private Suffix", id="arbitrary-posix-directory"),
    pytest.param("Q:/Fixture Workspace/Private Suffix", id="windows-forward-directory"),
    pytest.param("relative fixture/Private Suffix", id="relative-structured-path"),
])
def test_structured_paths_redacted_in_full_including_nested_json(result, path):
    _path_fixture(result, path)
    original = copy.deepcopy(result)
    trace = export_trace(result)
    expected = {key: "[LOCAL_PATH]" for key in (
        "path", "inputPath", "filename", "output_dir", "raw_output_ref")}
    expected.update(artifact_files=["[LOCAL_PATH]"], local_paths=["[LOCAL_PATH]"],
                    summary="Keep the checked row count: 12.", row_count=12)
    assert trace["output"]["artifacts"] == [expected]
    observation = next(e["attributes"]["output"] for e in trace["events"] if e["kind"] == "tool.result")
    assert observation["artifacts"] == [expected]
    for messages in (trace["input"]["messages"], trace["events"][0]["attributes"]["snapshot"]["messages"]):
        assert [m["role"] for m in messages] == ["system", "developer", "user", "assistant", "tool"]
        assert json.loads(messages[-1]["content"]) == {"artifacts": [expected], "conclusion": "Keep the conclusion."}
        function = messages[-2]["tool_calls"][0]["function"]
        assert function["name"] == "Report/Read:v2"
        assert json.loads(function["arguments"]) == {
            "root_dir": "[LOCAL_PATH]", "artifact_files": ["[LOCAL_PATH]"], "goal": "Keep all 12 rows."}
    assert result == original


@pytest.mark.parametrize("path", _PATH_CASES)
@pytest.mark.parametrize("quote", ["", '"', "'", "`"])
def test_prose_paths_remove_complete_suffix_but_keep_surrounding_text(result, path, quote):
    prose = f"Read {quote}{path}{quote}; checked 12 rows. Keep this conclusion."
    expected = f"Read {quote}[LOCAL_PATH]{quote}; checked 12 rows. Keep this conclusion."
    result["state"]["user_request"] = prose
    result["state"]["final_response"] = prose
    messages = result["state"]["llm_context_snapshots"][0]["messages"]
    for message in messages:
        message["content"] = prose
    trace = export_trace(result)
    assert trace["input"]["message"] == trace["output"]["response"] == expected
    for output_messages in (trace["input"]["messages"], trace["events"][0]["attributes"]["snapshot"]["messages"]):
        assert [m["role"] for m in output_messages] == [m["role"] for m in messages]
        assert all(m["content"] == expected for m in output_messages)
    assert "Private Suffix" not in json.dumps(trace)
    assert trace["metadata"]["local_paths"] == "structured_fields_and_detected_prose_paths_redacted"
    assert "ambiguous" in trace["metadata"]["path_redaction_limit"]


@pytest.mark.parametrize("text, expected", [
    (r"Saved Q:\Fixture Workspace\report draft.json and checked 12 rows.",
     "Saved [LOCAL_PATH] and checked 12 rows."),
    ('Read "/srv/fixture-private/Report (draft), final.json"; checked 12 rows.',
     'Read "[LOCAL_PATH]"; checked 12 rows.'),
    ("Read file:///srv/fixture-private/report.json; checked 12 rows.",
     "Read [LOCAL_PATH]; checked 12 rows."),
    ("Compare /srv/fixture/a.json and /opt/fixture/b.json; keep both counts.",
     "Compare [LOCAL_PATH] and [LOCAL_PATH]; keep both counts."),
    ("See https://example.invalid/srv/guide and Tool/Read:v2. Ratio 3 / 4; 12 rows.",
     "See https://example.invalid/srv/guide and Tool/Read:v2. Ratio 3 / 4; 12 rows."),
])
def test_prose_boundaries_and_nonpaths_remain_useful(result, text, expected):
    result["state"]["final_response"] = text
    assert export_trace(result)["output"]["response"] == expected


def test_path_schema_metadata_is_not_a_path_value(result):
    schema = {"type": "object", "properties": {"path": {
        "type": "string", "description": "Choose the report.", "minLength": 1}}, "required": ["path"]}
    result["state"]["llm_context_snapshots"][0]["tool_schemas"] = [schema]
    snapshot = export_trace(result)["events"][0]["attributes"]["snapshot"]
    assert snapshot["tool_schemas"] == [schema]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "unavailable"])
def test_unknown_or_invalid_usage_not_zero(result, value):
    result["state"]["resource_usage"]["agent_prompt_tokens"] = value
    trace = export_trace(result)
    assert trace["resource_usage"]["agent_prompt_tokens"] is None
    json.dumps(trace, allow_nan=False)


def test_missing_time_explicitly_marked_and_event_links_valid(result):
    result["state"].pop("runtime_telemetry")
    trace = export_trace(result)
    assert "runtime_start_unavailable_export_time_used" in trace["metadata"]["omission_reasons"]
    assert trace["started_at"] and "ended_at" not in trace
    ids = {e["event_id"] for e in trace["events"]}
    assert len(ids) == len(trace["events"])
    assert all(e.get("parent_event_id") is None or e["parent_event_id"] in ids for e in trace["events"])


def test_write_only_selected_file_never_overwrite(tmp_path, result):
    path = write_trace(result, tmp_path / "exports" / "trace.json")
    assert json.loads(path.read_text(encoding="utf-8"))["trace_id"] == "trace-fixture"
    with pytest.raises(FileExistsError):
        write_trace(result, path)


def test_demo_trace_paused_then_reviewed_with_original_tools(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    result = runtime.start({"message": DEMO_REQUEST})
    trace = export_trace(result)
    assert trace["status"] == "awaiting_confirmation"
    pending = result["state"]["pending_interaction"]
    assert trace["events"][-1]["kind"] == "interaction.request"
    result = runtime.resume(result["state"]["run_id"], pending["interaction_id"], {"decision": "approve"})
    trace = export_trace(result)
    assert trace["status"] == "completed"
    assert [e["name"] for e in trace["events"] if e["kind"] == "tool.call"] == [
        "report__read_materials", "report__draft", "report__archive"]
    assert any(e["kind"] == "artifact.consume" for e in trace["events"])
    assert str(tmp_path) not in json.dumps(trace)


_IMPORT_SCRIPT = r'''
import json, pathlib, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError("network forbidden in Eval import QA")
socket.socket.connect = forbidden
from commerce_eval.contracts import TraceEnvelopeV1
from commerce_eval.services.imports import ImportService
from commerce_eval.storage.database import Database
from commerce_eval.storage.repository import Repository
path, database_path = map(pathlib.Path, sys.argv[1:])
payload = json.loads(path.read_text(encoding="utf-8"))
TraceEnvelopeV1.model_validate(payload)
database = Database(database_path)
try:
    database.initialize()
    repository = Repository(database)
    service = ImportService(repository)
    preview = service.preview("isolated-demo", "trace", [{"name": "trace.json", "content": path.read_text(encoding="utf-8")}])
    assert preview["status"] == "ready", preview["errors"]
    committed = service.commit(preview["import_id"])
    assert committed["status"] == "committed" and committed["count"] == 1
    assert service.commit(preview["import_id"]) == committed
    stored = repository.get_trace(payload["trace_id"])["trace"]
    assert stored["contract_version"] == "1.2"
    expected = [e["name"] for e in payload["events"] if e["kind"] == "tool.call"]
    assert [e["name"] for e in stored["events"] if e["kind"] == "tool.call"] == expected
    assert stored["input"]["messages"] == payload["input"]["messages"]
    assert stored["output"] == payload["output"]
    assert stored["metadata"]["trusted"] is False
    assert stored["resource_usage"]["agent_prompt_tokens"] is None
    assert len(repository.list_projects()) == 1
    print("EVAL_IMPORT_OK 1.2 preview+commit+roundtrip isolated-db")
finally:
    database.dispose()
'''


def _eval_import(tmp_path, result):
    executable = os.environ.get("AGENT_WORKBENCH_EVAL_PYTHON")
    if not executable:
        pytest.skip("Set AGENT_WORKBENCH_EVAL_PYTHON to an installed Eval venv for import QA")
    assert Path(executable).is_file(), "Configured Eval interpreter not found"
    path = write_trace(result, tmp_path / "trace.json")
    # Explicit environment allowlist and -I: no credentials or PYTHONPATH, and
    # no source-tree injection into the separately installed Eval interpreter.
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "USERPROFILE", "HOME", "LOCALAPPDATA"}
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(COMMERCE_EVAL_HOME=str(tmp_path / "eval-home"), AGENT_API_ENV_FILE=str(tmp_path / "disabled.env"))
    completed = subprocess.run([executable, "-I", "-c", _IMPORT_SCRIPT, str(path), str(tmp_path / "import.db")],
                               env=env, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "EVAL_IMPORT_OK 1.2" in completed.stdout
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("scenario", ["paused", "approve", "reject", "request_changes"])
def test_real_eval_import_service_in_separate_process(tmp_path, scenario):
    if not os.environ.get("AGENT_WORKBENCH_EVAL_PYTHON"):
        pytest.skip("Set AGENT_WORKBENCH_EVAL_PYTHON to an installed Eval venv for import QA")
    runtime = build_demo_runtime(tmp_path / "demo")
    result = runtime.start({"message": DEMO_REQUEST})
    if scenario != "paused":
        result = runtime.resume(result["state"]["run_id"], result["state"]["pending_interaction"]["interaction_id"],
                                {"decision": scenario, "comment": "Clearly label the synthetic observations."})
    _eval_import(tmp_path, result)


@pytest.mark.parametrize("path", _PATH_CASES)
def test_real_eval_import_of_redacted_synthetic_paths(tmp_path, result, path):
    trace = _eval_import(tmp_path, _path_fixture(result, path))
    assert trace["output"]["response"] == "Read [LOCAL_PATH]; checked 12 rows. Keep this conclusion."
    assert "Private Suffix" not in json.dumps(trace)
