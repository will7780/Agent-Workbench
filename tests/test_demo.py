import copy
import json
import os
import socket
import subprocess

import pytest
import requests

from agent_workbench import central_env, llm_client
from agent_workbench.demo import (DEMO_MODEL, DEMO_NOTICE, DEMO_REQUEST,
                                  DemoArtifactValidator, build_demo_runtime)
from agent_workbench.execution_mode import ExecutionMode
from agent_workbench.tools.adapter_contracts import AdapterInput


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_API_ENV_FILE", str(tmp_path / "disabled.env"))
    monkeypatch.setenv("AGENT_WORKBENCH_LLM_PRICE_CARDS_FILE", str(tmp_path / "missing-prices.json"))

    def forbidden(*args, **kwargs):
        pytest.fail("Offline demo attempted network or credential loading")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(central_env, "load_central_agent_env", forbidden)
    monkeypatch.setattr(llm_client, "load_central_agent_env", forbidden)


def start(runtime):
    result = runtime.start({"message": DEMO_REQUEST})
    assert result["state"]["pending_interaction"], result["state"].get("errors")
    assert result["state"]["pending_interaction"]["type"] == "artifact_review"
    return result


def resume(runtime, result, decision, comment=""):
    state = result["state"]
    return runtime.resume(state["run_id"], state["pending_interaction"]["interaction_id"],
                          {"decision": decision, "comment": comment})


def paths(directory, kind):
    pattern = "offline-demo-*/archives/*.json" if kind == "archive" else "offline-demo-*/*/draft/report.json"
    return list(directory.glob(pattern))


def test_first_run_pauses_with_all_rows_checked_and_real_files(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    result = start(runtime)
    assert result["state"]["pending_interaction"]["can_approve"]
    checks = [e for e in result["state"]["agent_trace"] if e.get("type") == "artifact.check"]
    assert checks[-1]["row_count"] == 12
    assert set(checks[-1]["checked_rules"]) == set(DemoArtifactValidator.RULES)
    assert len(paths(tmp_path, "draft")) == 1 and not paths(tmp_path, "archive")
    assert result["state"]["llm_model"] == DEMO_MODEL
    assert [t["tool_name"] for t in result["state"]["tool_calls"]] == ["report__read_materials", "report__draft"]
    rows = json.loads(paths(tmp_path, "draft")[0].read_text(encoding="utf-8"))
    assert len(rows) == 12 and all(r["notice"] == DEMO_NOTICE for r in rows)


def test_approval_archives_exact_reviewed_bytes_once(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    result = start(runtime)
    draft = paths(tmp_path, "draft")[0].read_bytes()
    completed = resume(runtime, result, "approve")
    assert completed["state"]["pending_interaction"] is None
    assert len(paths(tmp_path, "archive")) == 1
    assert paths(tmp_path, "archive")[0].read_bytes() == draft
    assert len([e for e in completed["state"]["agent_trace"] if e.get("type") == "artifact.consume"]) == 1
    with pytest.raises(ValueError):
        resume(runtime, result, "approve")
    assert len(paths(tmp_path, "archive")) == 1


def test_refusal_zero_archive(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    result = resume(runtime, start(runtime), "reject", "Do not archive this example.")
    assert not paths(tmp_path, "archive") and result["state"]["pending_interaction"] is None
    archive = [c for c in result["state"]["tool_calls"] if c["tool_name"] == "report__archive"]
    assert len(archive) == 1 and archive[0]["adapter_called"] is False


def test_revision_uses_real_feedback_and_requires_fresh_review(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    first = start(runtime)
    original = paths(tmp_path, "draft")[0].read_bytes()
    feedback = "Emphasize Reading area 12 and explain that the observations are invented."
    second = resume(runtime, first, "request_changes", feedback)
    assert second["state"]["pending_interaction"]
    assert second["state"]["pending_interaction"]["interaction_id"] != first["state"]["pending_interaction"]["interaction_id"]
    assert not paths(tmp_path, "archive")
    rows = json.loads(paths(tmp_path, "draft")[0].read_text(encoding="utf-8"))
    assert all(feedback in row["note"] for row in rows)
    assert paths(tmp_path, "draft")[0].read_bytes() != original
    checks = [e for e in second["state"]["agent_trace"] if e.get("type") == "artifact.check"]
    assert checks[-1]["version"] == 2 and checks[-1]["content_hash"] != checks[0]["content_hash"]
    drafts = [c for c in second["state"]["tool_calls"] if c["tool_name"] == "report__draft"]
    assert len(drafts) == 2 and drafts[-1]["arguments"]["feedback"] == feedback
    with pytest.raises(ValueError):
        resume(runtime, first, "approve")
    completed = resume(runtime, second, "approve")
    assert completed["state"]["pending_interaction"] is None and len(paths(tmp_path, "archive")) == 1


def test_changed_file_invalidates_approval(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    result = start(runtime)
    path = paths(tmp_path, "draft")[0]
    path.write_bytes(path.read_bytes() + b"\n")
    completed = resume(runtime, result, "approve")
    assert not paths(tmp_path, "archive")
    assert any(e.get("decision") == "invalidated" for e in completed["state"]["agent_trace"])


def test_validator_checks_invalid_rows_outside_five_row_sample(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    start(runtime)
    rows = json.loads(paths(tmp_path, "draft")[0].read_text(encoding="utf-8"))
    for i, row in enumerate(rows):
        row["row_id"] = "row-" + str(i)
    snapshot = {"rows": rows, "row_count": len(rows)}
    policy = runtime.services.registry.get_action("report", "archive")["artifact_policy"]
    rows[-1]["minutes"] += 1
    unchanged = copy.deepcopy(snapshot)
    checked = runtime.services.artifact_validator.validate(snapshot, policy)
    assert checked["complete"] and not checked["valid"] and checked["row_count"] == 12
    assert any(e["row_id"] == "row-11" for e in checked["errors"])
    assert snapshot == unchanged


def test_content_failure_never_archives(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    actual = runtime.services.artifact_validator

    class FailLastRow:
        def validate(self, snapshot, policy):
            result = actual.validate(snapshot, policy)
            result.update(valid=False, errors=[{"row_id": snapshot["rows"][-1]["row_id"],
                                                "field": "minutes", "code": "injected_content_failure"}])
            return result

    runtime.services.artifact_validator = FailLastRow()
    first = start(runtime)
    assert first["state"]["pending_interaction"]["can_approve"] is False
    with pytest.raises(ValueError):
        resume(runtime, first, "approve")
    assert not paths(tmp_path, "archive")
    resume(runtime, first, "reject")


def test_env_key_cannot_enable_real_model_or_live_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-placeholder")
    runtime = build_demo_runtime(tmp_path)
    start(runtime)
    assert runtime.services.allow_real_model is False
    output = runtime.services.executor.execute(AdapterInput(
        step_id="live-denied", module="report", action="read_materials",
        adapter_name="offline_demo.read_materials", execution_mode=ExecutionMode.LIVE))
    assert output.error["type"] == "missing_adapter"


def test_two_runtimes_isolate_materials_and_archives(tmp_path):
    first, second = build_demo_runtime(tmp_path), build_demo_runtime(tmp_path)
    a, b = start(first), start(second)
    assert len(paths(tmp_path, "draft")) == 2
    first_material = a["state"]["tool_calls"][1]["arguments"]["material_id"]
    output = second.services.executor.execute(AdapterInput(
        step_id="cross-runtime", module="report", action="draft", adapter_name="offline_demo.draft",
        params={"material_id": first_material}))
    assert output.error["type"] == "demo_material_missing"
    resume(first, a, "approve")
    resume(second, b, "reject")
    assert len(paths(tmp_path, "archive")) == 1


def _directory_link(link, target):
    if os.name == "nt":
        completed = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                                   capture_output=True, text=True, timeout=10)
        if completed.returncode:
            pytest.skip("Windows junction creation unavailable")
    else:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("Directory symlinks unavailable")


def test_archive_junction_cannot_escape_even_after_approval(tmp_path):
    data = tmp_path / "demo"
    runtime = build_demo_runtime(data)
    first = start(runtime)
    root = next(data.glob("offline-demo-*"))
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    _directory_link(root / "archives", external)
    result = resume(runtime, first, "approve")
    assert list(external.iterdir()) == [sentinel]
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    assert not any(a.get("type") == "demo.archive" for a in result["state"]["artifacts"])
    assert not any(e.get("type") == "artifact.consume" for e in result["state"]["agent_trace"])


def test_junction_data_root_rejected_before_creating_files(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    link = tmp_path / "linked-data"
    _directory_link(link, external)
    with pytest.raises(ValueError, match="demo_path_unsafe"):
        build_demo_runtime(link)
    assert list(external.iterdir()) == []


def test_demo_memory_off_and_long_revision_feedback_preserved(tmp_path):
    runtime = build_demo_runtime(tmp_path)
    assert runtime.services.config_provider.snapshot()["memory_dir"] is None
    first = start(runtime)
    feedback = "Explain the synthetic nature. " * 35 + "Keep the final reviewer sentence."
    second = resume(runtime, first, "request_changes", feedback)
    assert second["state"]["pending_interaction"]["can_approve"]
    rows = json.loads(paths(tmp_path, "draft")[0].read_text(encoding="utf-8"))
    assert all(row["note"] == "Reviewer revision: " + feedback for row in rows)
    assert not list(tmp_path.rglob("memory"))
    resume(runtime, second, "reject")
    assert not paths(tmp_path, "archive")
