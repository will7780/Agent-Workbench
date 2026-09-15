"""Anonymous report fixtures: real temporary bytes, no models or network calls."""

import builtins
import copy
import csv
import json
import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_workbench import artifact_review as review
from agent_workbench.artifact_policy import (
    merge_artifact_policies,
    validate_artifact_policies,
    validate_company_artifact_policies,
)


class ReportValidator:
    """An example plugin, not a built-in engine rule or a simulated model."""

    def __init__(self):
        self.calls = []

    def validate(self, snapshot, policy):
        self.calls.append(copy.deepcopy(snapshot))
        errors = []
        expected = policy.get("rules", {}).get("report_ready", {}).get("status", "ready")
        for row in snapshot["rows"]:
            if not row.get("section"):
                errors.append({"row_id": row["row_id"], "field": "section", "code": "required"})
            if row.get("status") != expected:
                errors.append({"row_id": row["row_id"], "field": "status", "code": "report_not_ready"})
        if not snapshot["rows"]:
            errors.append({"row_id": None, "field": "rows", "code": "empty_report"})
        return {"complete": True, "valid": not errors, "errors": errors,
                "row_count": snapshot["row_count"], "checked_rules": ["report_ready", "report_present"]}


class Registry:
    def __init__(self, validator=None):
        self.artifact_validator = validator
        self.artifact_policies = {}
        self.action = {"risk_level": "L4", "artifact_policy": {
            "required": True, "rule_version": "report-v1", "scope": "input_contract",
            "required_rules": ["report_ready"], "rules": {"report_ready": {"status": "ready"}},
            "columns": ["section", "status"],
        }}

    def get_action(self, module, action):
        return self.action if (module, action) == ("reports", "archive") else None


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("AGENT_WORKBENCH_DISABLE_CENTRAL_ENV", "1")
    monkeypatch.setenv("AGENT_DISABLE_CENTRAL_ENV", "1")
    monkeypatch.setenv("AGENT_API_ENV_FILE", os.devnull)

    def forbidden(*args, **kwargs):
        pytest.fail("artifact tests must not make network calls")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def write_report(path, rows=None):
    if rows is None:
        rows = [{"section": f"Section {i + 1}", "status": "ready"} for i in range(20)]
    if path.suffix == ".json":
        path.write_text(json.dumps(rows), encoding="utf-8")
    elif path.suffix == ".xlsx":
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        workbook.active.append(["section", "status"])
        for row in rows:
            workbook.active.append([row["section"], row["status"]])
        workbook.save(path)
        workbook.close()
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["section", "status"])
            writer.writeheader()
            writer.writerows(rows)
    return path


@pytest.fixture
def case(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = write_report(source / "report.csv")
    registry = Registry(ReportValidator())
    return SimpleNamespace(root=source, path=path, registry=registry, stage=tmp_path / "stages",
                           params={"root_dir": str(source), "artifact_files": [str(path)]})


def prepare(case, *, params=None, policy=None, previous=None, **kwargs):
    params = case.params if params is None else params
    policy = policy or review.resolve_artifact_policy(case.registry, "reports", "archive")
    kwargs.setdefault("step_id", "archive-step")
    kwargs.setdefault("execution_mode", "dry_run")
    return review.prepare_artifact_review(
        params, [{"type": "report", "path": str(case.path)}], policy,
        run_id="anonymous-report", tool_call_id="archive-call", staging_root=case.stage,
        previous=previous, registry=case.registry, **kwargs)


def approved(case, **kwargs):
    policy = review.resolve_artifact_policy(case.registry, "reports", "archive")
    gate = prepare(case, policy=policy, **kwargs)
    assert gate["complete"] and gate["valid"], gate["errors"]
    gate["decision"] = "approve"
    bound = review.bind_artifact_review(gate, case.params, policy)
    assert bound.get("seal"), bound.get("error_type")
    return gate, bound, policy


def call(bound, **overrides):
    values = {"module": "reports", "action": "archive", "step_id": "archive-step",
              "run_id": "anonymous-report", "execution_mode": "dry_run",
              "params": copy.deepcopy(bound.get("bound_params", {})), "artifact_binding": bound}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("suffix", [".csv", ".json", ".xlsx"])
def test_pass_and_frozen_consume(case, suffix):
    case.path.unlink()
    case.path = write_report(case.root / ("report" + suffix))
    case.params["artifact_files"] = [str(case.path)]
    gate, bound, policy = approved(case)
    interaction = review.review_interaction(gate)
    assert interaction["can_approve"]
    assert interaction["interaction_id"] == gate["review_id"]
    assert gate["rows"] == gate["row_count"] == 20
    assert len(gate["sample"]) == 5
    assert len(gate["checked_rows"]) == 20
    assert review.artifact_revision_matches(gate, case.params, policy)
    assert review.verify_adapter_binding(call(bound), case.registry) is None
    assert Path(bound["bound_params"]["root_dir"]) != case.root
    frozen = Path(bound["bound_params"]["artifact_files"][0])
    assert frozen.parent == Path(bound["bound_params"]["root_dir"])
    assert frozen.read_bytes() == case.path.read_bytes()
    # The adapter consumes the exact validated bytes, not the original root.
    consumed = [Path(path).read_bytes() for path in bound["bound_params"]["artifact_files"]]
    assert consumed == [case.path.read_bytes()]
    assert len(case.registry.artifact_validator.calls) == 1
    assert case.registry.artifact_validator.calls[0]["row_count"] == 20


@pytest.mark.parametrize("decision", [None, "reject", "request_changes", "user_confirmed"])
def test_refusal_never_binds(case, decision):
    gate = prepare(case)
    gate["decision"] = decision
    case.params["user_confirmed"] = True
    bound = review.bind_artifact_review(gate, case.params, gate["policy"])
    assert bound["error_type"] == "artifact_review_required"
    assert "seal" not in bound and "bound_params" not in bound


def test_invalid_rows_all_checked_including_outside_sample(case):
    rows = [{"section": f"Section {i}", "status": "ready"} for i in range(20)]
    rows[16]["status"] = "draft"
    rows[19]["section"] = ""
    write_report(case.path, rows)
    gate = prepare(case)
    assert gate["complete"] and not gate["valid"]
    assert len(gate["errors"]) == 2 and len(gate["sample"]) == 7
    assert {e["row_id"] for e in gate["errors"]}.issubset(gate["sample_row_ids"])
    assert not review.review_interaction(gate)["can_approve"]
    gate["decision"] = "approve"
    assert "seal" not in review.bind_artifact_review(gate, case.params, gate["policy"])


def test_validator_required_no_fake_success_and_explicit_injection(case):
    case.registry.artifact_validator = None
    gate = prepare(case)
    assert not gate["complete"] and not gate["valid"]
    assert gate["errors"][0]["code"] == "artifact_validator_unavailable"
    assert gate["validation_status"] == "unavailable"
    assert "stage_dir" not in gate
    assert prepare(case, validator=ReportValidator())["valid"]


@pytest.mark.parametrize("result", [
    None,
    {"complete": True, "valid": True, "errors": []},
    {"complete": True, "valid": True, "errors": [], "row_count": 19},
    {"complete": True, "valid": True, "errors": [], "row_count": 20},
    {"complete": "yes", "valid": True, "errors": [], "row_count": 20},
    {"complete": False, "valid": True, "errors": [], "row_count": 20},
    {"complete": True, "valid": True, "errors": [{"code": "failure"}], "row_count": 20},
])
def test_malformed_or_unchecked_validator_result_blocks(case, result):
    validator = SimpleNamespace(validate=lambda *args: copy.deepcopy(result))
    gate = prepare(case, validator=validator)
    assert not gate["complete"] and not gate["valid"] and gate["errors"]


def test_validator_failure_does_not_emit_exception_details(case):
    def broken(*args):
        raise RuntimeError(str(case.path))

    gate = prepare(case, validator=SimpleNamespace(validate=broken))
    assert gate["errors"][0]["code"] == "artifact_validator_failed"
    assert str(case.path) not in json.dumps(review.artifact_event("artifact.check", gate))


def test_validator_cannot_overwrite_identity_or_mutate_engine_snapshot(case):
    def mutate(snapshot, policy):
        result = ReportValidator().validate(snapshot, policy)
        snapshot["rows"][0]["section"] = "Mutated projection"
        policy["required"] = False
        return {**result, "artifact_id": "forged", "files": [], "seal": "forged", "sample": []}

    gate = prepare(case, validator=SimpleNamespace(validate=mutate))
    assert gate["valid"] and gate["artifact_id"] != "forged" and gate["files"]
    assert gate["checked_rows"][0]["section"] == "Section 1"
    assert len(gate["sample"]) == 5 and "seal" not in gate


def test_versions_and_stale_approval(case):
    first, bound, policy = approved(case)
    same = prepare(case, previous=first)
    assert same["version"] == first["version"] and same["revision"] == first["revision"]
    assert same["review_id"] != first["review_id"]
    write_report(case.path, [{"section": "Revised", "status": "ready"}])
    changed = prepare(case, previous=first)
    assert changed["artifact_id"] == first["artifact_id"]
    assert changed["version"] == 2 and changed["review_id"] != first["review_id"]
    assert changed["decision"] is None
    assert not review.artifact_revision_matches(first, case.params, policy)
    assert review.verify_adapter_binding(call(bound), case.registry) == "artifact_approval_invalidated"
    forged = copy.deepcopy(changed)
    forged["review_id"] = first["review_id"]
    forged["decision"] = "approve"
    assert "seal" not in review.bind_artifact_review(forged, case.params, policy)


@pytest.mark.parametrize("change", ["source", "staged", "stage_add", "stage_remove", "params", "policy", "host_policy"])
def test_mutations_invalidate_before_binding_and_dispatch(case, change):
    gate, bound, policy = approved(case)
    params = case.params
    if change == "source":
        write_report(case.path, [{"section": "Changed", "status": "ready"}])
    elif change == "staged":
        Path(gate["files"][0]["staged"]).write_bytes(b"changed")
    elif change == "stage_add":
        (Path(gate["stage_dir"]) / "extra.txt").write_text("extra", encoding="utf-8")
    elif change == "stage_remove":
        Path(gate["files"][0]["staged"]).unlink()
    elif change == "params":
        bound["bound_params"]["mode"] = "changed"
        params = {**params, "mode": "changed"}
    elif change == "policy":
        case.registry.action["artifact_policy"]["rule_version"] = "report-v2"
    else:
        case.registry.artifact_policies = {"reports.archive": {"required_rules": ["report_present"]}}
    current = review.resolve_artifact_policy(case.registry, "reports", "archive")
    assert not review.artifact_revision_matches(gate, params, current)
    assert "seal" not in review.bind_artifact_review(gate, params, current)
    assert review.verify_adapter_binding(call(bound), case.registry) is not None


@pytest.mark.parametrize("change", ["add", "remove", "add_directory", "remove_directory", "nested_add", "nested_remove", "rename"])
def test_full_directory_manifest_add_remove(case, change):
    case.params.pop("artifact_files")
    (case.root / "notes.txt").write_text("metadata only", encoding="utf-8")
    (case.root / "empty").mkdir()
    (case.root / "nested").mkdir()
    (case.root / "nested" / "notes.bin").write_bytes(b"metadata only")
    gate, bound, policy = approved(case)
    assert gate["artifact_scope"] == "directory_root" and len(gate["source_manifest"]) == 5
    assert len(gate["files"]) == 1
    if change == "add":
        (case.root / "extra.txt").write_text("added", encoding="utf-8")
    elif change == "remove":
        (case.root / "notes.txt").unlink()
    elif change == "add_directory":
        (case.root / "added").mkdir()
    elif change == "remove_directory":
        (case.root / "empty").rmdir()
    elif change == "nested_add":
        (case.root / "nested" / "added.txt").write_text("added", encoding="utf-8")
    elif change == "nested_remove":
        (case.root / "nested" / "notes.bin").unlink()
    else:
        (case.root / "notes.txt").rename(case.root / "renamed.txt")
    assert not review.artifact_revision_matches(gate, case.params, policy)
    assert review.verify_adapter_binding(call(bound), case.registry) == "artifact_approval_invalidated"
    revised = prepare(case, previous=gate)
    assert revised["version"] == 2 and revised["manifest_hash"] != gate["manifest_hash"]


def test_explicit_bundle_does_not_consume_unselected_files(case, monkeypatch):
    original = review._read_checked
    reads = []

    def selected_only(path):
        assert path.name in {"report.csv", "file-001.csv"}
        reads.append(path.name)
        return original(path)

    monkeypatch.setattr(review, "_read_checked", selected_only)
    (case.root / "notes.bin").write_bytes(b"not selected")
    gate, bound, policy = approved(case)
    (case.root / "extra.txt").write_text("not selected", encoding="utf-8")
    assert review.artifact_revision_matches(gate, case.params, policy)
    assert review.verify_adapter_binding(call(bound), case.registry) is None
    assert reads and len(bound["bound_params"]["artifact_files"]) == 1


def test_directory_metadata_never_opens_incidental_bytes(case, monkeypatch):
    case.params.pop("artifact_files")
    (case.root / "notes.bin").write_bytes(b"unselected")
    original = review.os.open

    def no_incidental(path, *args, **kwargs):
        assert Path(path).name != "notes.bin"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(review.os, "open", no_incidental)
    gate, bound, policy = approved(case)
    assert len(gate["source_manifest"]) == 2
    event = review.artifact_event("artifact.check", gate)
    assert str(case.root) not in json.dumps(event) and "notes.bin" not in json.dumps(event)
    assert review.artifact_revision_matches(gate, case.params, policy)


@pytest.mark.parametrize("when", ["read", "validate", "staging"])
def test_toctou_during_snapshot_blocks(case, monkeypatch, when):
    if when == "read":
        original = review._read_checked
        modified = False

        def changing(path):
            nonlocal modified
            result = original(path)
            if path == case.path and not modified:
                modified = True
                write_report(case.path, [{"section": "Changed", "status": "ready"}])
            return result

        monkeypatch.setattr(review, "_read_checked", changing)
    elif when == "validate":
        original = case.registry.artifact_validator.validate

        def changing(snapshot, policy):
            result = original(snapshot, policy)
            write_report(case.path, [{"section": "Changed", "status": "ready"}])
            return result

        monkeypatch.setattr(case.registry.artifact_validator, "validate", changing)
    else:
        case.params.pop("artifact_files")
        original = Path.open

        def changing(path, mode="r", *args, **kwargs):
            if mode == "xb":
                (case.root / "added.txt").write_text("added", encoding="utf-8")
            return original(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", changing)
    gate = prepare(case)
    assert not gate["complete"] and not gate["valid"] and "snapshot_seal" not in gate
    assert gate["errors"][0]["code"] in {"artifact_changed_during_read", "source_membership_changed"}


def test_descriptor_replacement_is_detected(case, monkeypatch):
    original = review.os.open

    def replace_before_open(path, *args, **kwargs):
        if Path(path) == case.path:
            replacement = case.root / "replacement.csv"
            replacement.write_bytes(case.path.read_bytes())
            replacement.replace(case.path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(review.os, "open", replace_before_open)
    gate = prepare(case)
    assert not gate["valid"]
    assert gate["errors"][0]["code"] == "artifact_changed_during_read"


@pytest.mark.parametrize("kind", ["file", "root", "directory_entry", "stage_parent"])
def test_symlink_is_never_followed(case, tmp_path, kind):
    target = tmp_path / "external"
    target.mkdir()
    external = write_report(target / "external.csv")
    link = case.root / ("linked.csv" if kind == "file" else "linked")
    try:
        link.symlink_to(external if kind == "file" else target, target_is_directory=kind != "file")
    except OSError:
        pytest.skip("OS does not permit creating symlinks")
    if kind == "file":
        case.params["artifact_files"] = [str(link)]
    elif kind == "root":
        case.params = {"root_dir": str(link), "artifact_files": [str(link / "external.csv")]}
    elif kind == "stage_parent":
        case.stage = link
    else:
        case.params.pop("artifact_files")
    gate = prepare(case)
    assert not gate["valid"]
    assert gate["errors"][0]["code"] == "symlink_forbidden"


def test_reparse_attribute_is_rejected():
    import stat
    assert review._is_link(SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400))


@pytest.mark.skipif(os.name != "nt", reason="Windows junction boundary")
@pytest.mark.parametrize("kind", ["root", "directory_entry", "stage_parent", "staged_after_review"])
def test_real_windows_junction_fails_closed(case, tmp_path, kind):
    target = tmp_path / "junction-target"
    target.mkdir()
    write_report(target / "report.csv")
    link = tmp_path / "junction"
    gate = bound = policy = None
    if kind == "directory_entry":
        link = case.root / "junction"
        case.params.pop("artifact_files")
    elif kind == "staged_after_review":
        gate, bound, policy = approved(case)
        link = Path(gate["stage_dir"])
        target = link.with_name("frozen-original")
        link.rename(target)
    completed = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                               capture_output=True, check=False)
    assert completed.returncode == 0, "temporary junction creation failed"
    try:
        if kind == "root":
            case.params = {"root_dir": str(link), "artifact_files": [str(link / "report.csv")]}
        elif kind == "stage_parent":
            case.stage = link
        if gate:
            assert not review.artifact_revision_matches(gate, case.params, policy)
            assert review.verify_adapter_binding(call(bound), case.registry) == "artifact_approval_invalidated"
        else:
            gate = prepare(case)
            assert not gate["complete"] and gate["errors"][0]["code"] == "symlink_forbidden"
    finally:
        # Remove only the junction itself; never recurse into the target.
        assert link.parent.resolve().is_relative_to(tmp_path.resolve()) or link.parent.resolve() == tmp_path.resolve()
        os.rmdir(link)


def test_in_place_mutation_during_descriptor_read(case, monkeypatch):
    original = review.os.fdopen

    class MutatingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, count):
            data = self.stream.read(count)
            write_report(case.path, [{"section": "Changed", "status": "ready"}])
            return data

    monkeypatch.setattr(review.os, "fdopen", lambda *args: MutatingReader(original(*args)))
    gate = prepare(case)
    assert not gate["complete"] and gate["errors"][0]["code"] == "artifact_changed_during_read"


@pytest.mark.parametrize("bound_name,limit,code", [
    ("_MAX_BYTES", 4, "file_limit"), ("_MAX_TOTAL_BYTES", 4, "total_file_limit"),
    ("_MAX_ROWS", 2, "row_limit"), ("_MAX_FILES", 0, "explicit_files_required"),
    ("_MAX_SOURCE_ENTRIES", 0, "source_entry_limit"),
    ("_MAX_SOURCE_DEPTH", 0, "source_depth_limit"),
])
def test_bounded_work_fails_closed(case, monkeypatch, bound_name, limit, code):
    case.params.pop("artifact_files")
    monkeypatch.setattr(review, bound_name, limit)
    gate = prepare(case)
    assert not gate["complete"] and gate["errors"][0]["code"] == code


@pytest.mark.parametrize("name,code", [(".hidden", "hidden_file_forbidden"),
                                      ("credentials", "sensitive_file_forbidden")])
def test_unsafe_incidental_entries_fail_before_reads(case, monkeypatch, name, code):
    case.params.pop("artifact_files")
    (case.root / name).mkdir()
    monkeypatch.setattr(review, "_read_checked", lambda *args: pytest.fail("unexpected read"))
    gate = prepare(case)
    assert not gate["complete"] and gate["errors"][0]["code"] == code


@pytest.mark.parametrize("raw,suffix", [(b"section,section\na,b\n", ".csv"),
                                        (b"section,status\na,b,c\n", ".csv"),
                                        (b'{"section":"not an array"}', ".json"),
                                        (b'[{"section":"a","section":"b"}]', ".json"),
                                        (b'[NaN]', ".json"), (b'[1,2]', ".json")])
def test_malformed_content_is_not_success(case, raw, suffix):
    path = case.root / ("malformed" + suffix)
    path.write_bytes(raw)
    case.params["artifact_files"] = [str(path)]
    gate = prepare(case)
    assert not gate["complete"] and not gate["valid"]


def test_missing_xlsx_support_is_explicit(case, monkeypatch):
    path = case.root / "report.xlsx"
    path.write_bytes(b"unavailable-reader-fixture")
    case.params["artifact_files"] = [str(path)]
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "openpyxl":
            raise ImportError("optional reader disabled")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    gate = prepare(case)
    assert not gate["complete"] and gate["validation_status"] == "unavailable"
    assert gate["errors"][0]["code"] == "xlsx_reader_unavailable"


def test_xlsx_expansion_limit(case, monkeypatch):
    path = write_report(case.root / "report.xlsx")
    case.params["artifact_files"] = [str(path)]
    monkeypatch.setattr(review, "_MAX_TOTAL_BYTES", path.stat().st_size + 1)
    gate = prepare(case)
    assert not gate["complete"] and gate["errors"][0]["code"] == "file_limit"


def test_xlsx_date_projection_is_json_compatible(case):
    from datetime import date
    path = write_report(case.root / "dated.xlsx", [{"section": date(2024, 1, 1), "status": "ready"}])
    case.params["artifact_files"] = [str(path)]
    gate = prepare(case)
    assert gate["valid"]
    assert json.loads(json.dumps(review.artifact_event("artifact.check", gate)))["complete"]


def test_legacy_row_count_result_is_supported(case):
    def legacy(snapshot, policy):
        result = ReportValidator().validate(snapshot, policy)
        result["rows"] = result.pop("row_count")
        return result

    assert prepare(case, validator=SimpleNamespace(validate=legacy))["valid"]


def test_undeclared_rule_details_are_not_evidence(case):
    gate = prepare(case)
    assert gate["checked_rules"] == ["report_ready"]

    def invalid(snapshot, policy):
        result = ReportValidator().validate(snapshot, policy)
        result["checked_rules"].append(str(case.path))
        return result

    failed = prepare(case, validator=SimpleNamespace(validate=invalid))
    assert not failed["complete"]
    assert str(case.path) not in json.dumps(review.artifact_event("artifact.check", failed))


@pytest.mark.parametrize("kind", ["empty_refs", "duplicates", "outside_root", "url", "unsupported", "missing"])
def test_only_actual_explicit_files_are_read(case, tmp_path, kind):
    if kind == "empty_refs":
        case.params["artifact_files"] = []
    elif kind == "duplicates":
        case.params["artifact_files"] *= 2
    elif kind == "outside_root":
        case.params["artifact_files"] = [str(write_report(tmp_path / "outside.csv"))]
    elif kind == "url":
        case.params["artifact_files"] = ["https://example.invalid/report.csv"]
    elif kind == "unsupported":
        path = case.root / "report.txt"
        path.write_text("not an artifact format", encoding="utf-8")
        case.params["artifact_files"] = [str(path)]
    else:
        case.path.unlink()
    gate = prepare(case)
    assert not gate["complete"] and not gate["valid"]
    assert not case.registry.artifact_validator.calls


def test_mock_refs_never_trigger_directory_content_discovery(case):
    params = {"root_dir": str(case.root)}
    policy = review.resolve_artifact_policy(case.registry, "reports", "archive")
    gate = review.prepare_artifact_review(params, [{"type": "mock_artifact", "path": str(case.path)}],
        policy, run_id="mock-ref", tool_call_id="archive-call", registry=case.registry)
    assert not gate["complete"] and gate["errors"][0]["code"] == "explicit_files_required"


def test_selected_multifile_manifest_and_row_ids(case):
    extra = write_report(case.root / "additional.json", [{"section": "Appendix", "status": "ready"}])
    case.params["artifact_files"].append(str(extra))
    gate, bound, policy = approved(case)
    assert len(gate["manifest"]) == 2 and gate["row_count"] == 21
    assert len(set(gate["checked_row_ids"])) == 21
    assert review.verify_adapter_binding(call(bound), case.registry) is None
    assert all(Path(path).parent != case.root for path in bound["bound_params"]["artifact_files"])


def test_relocates_staging_outside_source(case):
    case.params.pop("artifact_files")
    case.stage = case.root / "stages"
    gate = prepare(case)
    assert gate["valid"] and not Path(gate["stage_dir"]).is_relative_to(case.root)
    assert not case.stage.exists()
    # This test deliberately exercises the default staging fallback.
    for item in gate["files"]:
        Path(item["staged"]).unlink()
    Path(gate["stage_dir"]).rmdir()


def test_required_union_and_conflicts(case):
    policy = review.resolve_artifact_policy(case.registry, "reports", "archive", {
        "reports.archive": {"required": False, "required_rules": ["report_present"],
                            "scope": "explicit_file_set"}})
    assert policy["required"] and policy["required_rules"] == ["report_present", "report_ready"]
    assert prepare(case, policy=policy)["valid"]
    bad = review.resolve_artifact_policy(case.registry, "reports", "archive", {
        "reports.archive": {"rules": {"report_ready": {"status": "draft"}}}})
    assert bad["required"] and bad["policy_error"]
    assert not prepare(case, policy=bad)["valid"]
    with pytest.raises(ValueError):
        merge_artifact_policies({"required": "false"})
    strict = merge_artifact_policies({"scope": "directory_root"}, {"scope": "explicit_file_set"})
    assert strict["scope"] == "directory_root"


def test_policies_are_generic_bounded_and_detached():
    value = {"reports.archive": {"required": True, "rules": {"report_ready": {"status": "ready"}}}}
    copied = validate_artifact_policies(value)
    copied["reports.archive"]["rules"]["report_ready"]["status"] = "draft"
    assert value["reports.archive"]["rules"]["report_ready"]["status"] == "ready"
    assert validate_company_artifact_policies(value) == value
    for invalid in ({"bad name": {}}, {"reports.archive": {"unknown": True}},
                    {"reports.archive": {"columns": {"fixed": "field"}}}):
        with pytest.raises(ValueError):
            validate_artifact_policies(invalid)


def test_legacy_policy_metadata_and_display_version_remain_compatible(case):
    extension = {"reports.archive": {"rule_version": "report-host-v2"}}
    policy = review.resolve_artifact_policy(case.registry, "reports", "archive", extension)
    gate = prepare(case, policy=policy)
    gate.update(company_policy=extension, capability="reports.archive", decision="approve", version=3)
    bound = review.bind_artifact_review(gate, case.params, policy)
    assert bound["seal"] and review.verify_adapter_binding(call(bound), case.registry) is None


@pytest.mark.parametrize("key,value", [("valid", False), ("review_id", "different"),
                                      ("content_hash", "different"), ("capability", "reports.other")])
def test_binding_tampering_fails(case, key, value):
    gate, bound, policy = approved(case)
    bound[key] = value
    assert review.verify_adapter_binding(call(bound), case.registry) == "artifact_binding_invalid"


def test_missing_binding_and_unrequired_actions(case):
    adapter = call({})
    assert review.verify_adapter_binding(adapter, case.registry) == "artifact_review_runtime_required"
    case.registry.action["artifact_policy"] = {"required": False}
    assert review.verify_adapter_binding(adapter, case.registry) is None
    case.registry.artifact_policies = {"reports.archive": {"required": True}}
    assert review.verify_adapter_binding(adapter, case.registry) == "artifact_review_runtime_required"


def test_forged_completed_gate_cannot_acquire_binding(case):
    gate = prepare(case)
    gate.update(valid=True, complete=True, decision="approve", content_hash="forged")
    assert "seal" not in review.bind_artifact_review(gate, case.params, gate["policy"])


def test_evidence_default_is_ids_only_and_samples_deterministic(case):
    case.registry.action["artifact_policy"].pop("columns")
    first = prepare(case)
    second = prepare(case)
    assert first["sample"] == second["sample"]
    assert all(set(row) == {"row_id"} for row in first["checked_rows"])
    event = review.artifact_event("artifact.check", first, files=first["files"], policy=first["policy"])
    serialized = json.dumps(event)
    assert str(case.root) not in serialized and "snapshot_seal" not in serialized and "files" not in event


def test_blocked_observation_compatible(case):
    gate = prepare(case)
    result = review.artifact_blocked_observation("archive-step", "artifact_review_required", gate)
    assert result.status == "failed" and result.error["retryable"] is False
    assert result.artifacts[0]["type"] == "artifact.check"


@pytest.mark.parametrize("decision", ["request_changes", "reject", "invalidated"])
def test_blocked_observation_retains_actual_review_feedback(case, decision):
    from agent_workbench.tools.observation_tools import compact_observation

    gate = prepare(case)
    comment = "Revise Section 2 to include the missing reference."
    gate.update(decision=decision, comment=comment)
    before = copy.deepcopy(gate)
    result = review.artifact_blocked_observation("archive-step", "artifact_review_" + decision, gate)
    # This is the same serialized Observation contract sent to the next turn.
    payload = json.loads(json.dumps(result.to_dict()))
    events = [event for event in payload["artifacts"] if event["type"] == "artifact.review"]
    assert len(events) == 1
    assert events[0]["decision"] == decision and events[0]["comment"] == comment
    assert events[0]["review_id"] == gate["review_id"] and events[0]["version"] == gate["version"]
    assert events[0]["content_hash"] == gate["content_hash"]
    assert comment in compact_observation(result)["summary"]
    assert decision in result.summary and result.error["type"] == "artifact_review_" + decision
    assert gate == before


def test_review_event_defaults_to_gate_feedback_and_bounds_comment(case):
    gate = prepare(case)
    gate.update(decision="request_changes", comment="Revise the appendix. " + "x" * 5000)
    event = review.artifact_event("artifact.review", gate)
    assert event["decision"] == "request_changes" and len(event["comment"]) == 4000
    assert event["comment"].startswith("Revise the appendix.")
    result = review.artifact_blocked_observation("archive-step", "artifact_review_request_changes", gate)
    assert result.artifacts[-1]["comment"] == event["comment"]
    assert result.summary.endswith(event["comment"])


def test_blocked_summary_uses_redacted_feedback(case, monkeypatch):
    gate = prepare(case)
    gate.update(decision="request_changes", comment="reviewer detail withheld by policy")
    original = review.redact_recursive

    def redact_feedback(value, **kwargs):
        result, changed = original(value, **kwargs)
        if isinstance(result, dict) and "comment" in result:
            result["comment"] = "[REDACTED]"
        return result, changed

    monkeypatch.setattr(review, "redact_recursive", redact_feedback)
    result = review.artifact_blocked_observation("archive-step", "artifact_review_request_changes", gate)
    assert gate["comment"] not in json.dumps(result.to_dict())
    assert result.artifacts[-1]["comment"] == "[REDACTED]" and result.summary.endswith("[REDACTED]")


def test_unreviewed_failure_does_not_fabricate_feedback(case):
    gate = prepare(case)
    result = review.artifact_blocked_observation("archive-step", "artifact_review_unavailable", gate)
    assert [event["type"] for event in result.artifacts] == ["artifact.check"]
    assert result.summary == "artifact_review_unavailable"


def _approval_response(gate):
    return {"interaction_id": gate["review_id"], "type": "artifact_review", "decision": "approve"}


def test_repeated_call_id_cannot_replay_prior_artifact_approval(case):
    from agent_workbench.langgraph_interactions import sanitize_interaction_response

    first = prepare(case)
    later = prepare(case, previous=first)
    assert first["revision"] == later["revision"]
    result, error = sanitize_interaction_response(_approval_response(first), review.review_interaction(later))
    assert error == "interaction_mismatch" and result is None


def test_aba_revision_cannot_replay_prior_artifact_approval(case):
    from agent_workbench.langgraph_interactions import sanitize_interaction_response

    original = case.path.read_bytes()
    first = prepare(case)
    write_report(case.path, [{"section": "Intermediate revision", "status": "ready"}])
    intermediate = prepare(case, previous=first)
    case.path.write_bytes(original)
    restored = prepare(case, previous=intermediate)
    assert restored["version"] == 3 and first["revision"] == restored["revision"]
    result, error = sanitize_interaction_response(_approval_response(first), review.review_interaction(restored))
    assert error == "interaction_mismatch" and result is None


def test_binding_cannot_cross_engine_steps(case):
    _, bound, _ = approved(case)
    assert review.verify_adapter_binding(call(bound, step_id="later-step"), case.registry) == "artifact_approval_invalidated"


def test_binding_cannot_escalate_execution_mode(case):
    _, bound, _ = approved(case)
    assert review.verify_adapter_binding(call(bound, execution_mode="live"), case.registry) == "artifact_approval_invalidated"


def test_same_call_id_changed_params_already_invalidates_approval(case):
    from agent_workbench.langgraph_interactions import sanitize_interaction_response

    first = prepare(case)
    later = prepare(case, params={**case.params, "archive_name": "second-report"})
    assert first["revision"] != later["revision"]
    result, error = sanitize_interaction_response(_approval_response(first), review.review_interaction(later))
    assert error == "interaction_mismatch" and result is None


def test_binding_cannot_cross_runs_with_the_same_step(case):
    _, bound, _ = approved(case)
    error = review.verify_adapter_binding(call(bound, run_id="later-run"), case.registry)
    assert error == "artifact_approval_invalidated"


@pytest.mark.parametrize("field", ["step_id", "execution_mode"])
def test_missing_preparation_context_cannot_authorize(case, field):
    gate = prepare(case, **{field: None})
    assert gate["complete"] and gate["valid"]
    assert not review.review_interaction(gate)["can_approve"]
    gate["decision"] = "approve"
    bound = review.bind_artifact_review(gate, case.params, gate["policy"])
    assert bound["error_type"] == "artifact_binding_context_required" and "seal" not in bound


@pytest.mark.parametrize("field", ["run_id", "step_id", "execution_mode"])
def test_binding_requires_independent_current_context(case, field):
    _, bound, _ = approved(case)
    error = review.verify_adapter_binding(call(bound, **{field: None}), case.registry)
    assert error == "artifact_binding_context_required"


def test_explicit_verifier_run_id_and_enum_mode(case):
    from agent_workbench.execution_mode import ExecutionMode

    _, bound, _ = approved(case)
    adapter = call(bound, execution_mode=ExecutionMode.DRY_RUN)
    del adapter.run_id
    error = review.verify_adapter_binding(adapter, case.registry, run_id="anonymous-report")
    assert error is None
    error = review.verify_adapter_binding(adapter, case.registry, run_id="later-run")
    assert error == "artifact_approval_invalidated"
    adapter.run_id = "conflicting-run"
    error = review.verify_adapter_binding(adapter, case.registry, run_id="anonymous-report")
    assert error == "artifact_approval_invalidated"


def test_pending_review_id_is_stable_across_resume_and_render(case):
    from agent_workbench.langgraph_interactions import sanitize_interaction_response

    gate = prepare(case)
    pending = review.review_interaction(gate)
    resumed = review.review_interaction(copy.deepcopy(gate))
    assert pending["interaction_id"] == resumed["interaction_id"]
    result, error = sanitize_interaction_response(_approval_response(gate), resumed)
    assert error is None and result["decision"] == "approve"
    assert review.artifact_revision_matches(gate, case.params, gate["policy"])


@pytest.mark.parametrize("field,value", [("step_id", "later-step"), ("run_id", "later-run"),
                                        ("execution_mode", "live"), ("review_nonce", "reused")])
def test_approval_context_cannot_be_rewritten_before_binding(case, field, value):
    gate = prepare(case)
    gate.update(decision="approve")
    gate[field] = value
    bound = review.bind_artifact_review(gate, case.params, gate["policy"])
    assert bound["error_type"] == "artifact_approval_invalidated" and "seal" not in bound


def test_legacy_binding_without_execution_context_fails_closed(case):
    _, bound, _ = approved(case)
    # Model the old process-authenticated shape, not an unsealed payload.
    for key in ("run_id", "step_id", "execution_mode"):
        bound.pop(key)
    bound["seal"] = review._sign({key: value for key, value in bound.items() if key != "seal"})
    error = review.verify_adapter_binding(call(bound), case.registry)
    assert error == "artifact_binding_context_required"


def test_model_params_cannot_supply_binding_context(case):
    params = {**case.params, "step_id": "archive-step", "execution_mode": "dry_run",
              "run_id": "anonymous-report", "user_confirmed": True}
    gate = prepare(case, params=params, step_id=None, execution_mode=None)
    assert gate["valid"] and not review.review_interaction(gate)["can_approve"]
    gate["decision"] = "approve"
    bound = review.bind_artifact_review(gate, params, gate["policy"])
    assert bound["error_type"] == "artifact_binding_context_required" and "seal" not in bound


def test_malformed_binding_context_refuses_without_raising(case):
    gate = prepare(case)
    gate.update(decision="approve", execution_mode={})
    assert not review.review_interaction(gate)["can_approve"]
    bound = review.bind_artifact_review(gate, case.params, gate["policy"])
    assert bound["error_type"] == "artifact_binding_context_required" and "seal" not in bound


def test_verifier_is_repeatable_for_two_dispatch_boundaries(case):
    _, bound, _ = approved(case)
    adapter = call(bound)
    errors = [review.verify_adapter_binding(adapter, case.registry) for _ in range(2)]
    assert errors == [None, None]
