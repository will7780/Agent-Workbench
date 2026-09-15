"""Anonymous, non-commerce reference plugin. No model or network is contacted.

The dry-run adapters deliberately create real *temporary simulated example*
files beneath data_dir. They do not prove a business outcome. The shared runtime
owns content snapshots, approvals, revisions and consumption bindings.
"""
from __future__ import annotations

import json
import stat
import tempfile
import uuid
from pathlib import Path

from .configuration import StaticConfigProvider
from .execution_mode import ExecutionMode
from .llm_client import LLMChatWithToolsResult, LLMToolCall
from .registry import ModuleCapabilityRegistry
from .runtime import AgentRuntime, RuntimeServices
from .tools.adapter_contracts import AdapterInput, AdapterOutput, blocked_output
from .tools.execution_adapters import RegisteredToolExecutor

DEMO_MODEL = "offline-reference-policy"
DEMO_NOTICE = "Local temporary simulated example; synthetic facts, not business proof."
DEMO_REQUEST = "Read the synthetic materials, draft a report, and ask me to review before archiving."


def _facts():
    return [{"section": f"Reading area {i:02d}", "observations": 3 * i + 4,
             "minutes": 5 * i + 10} for i in range(1, 13)]


def _require_local(path: Path, root: Path):
    """Refuse links/reparse points before touching any example file or parent."""
    path.relative_to(root)
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("demo_path_unsafe")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("demo_path_unsafe")


def _artifacts(observation):
    return observation.get("artifacts") or observation.get("artifact_refs") or []


def _material_id(artifact):
    if artifact.get("material_id"):
        return artifact["material_id"]
    path = Path(artifact["path"])
    return path.parent.name if artifact["type"] == "demo.materials" else path.parent.parent.name


class DemoArtifactValidator:
    """Only content rules: inspect every parsed row, never approve or bind."""

    RULES = ["synthetic_facts", "complete_report", "example_notice", "revision_note"]

    def validate(self, snapshot: dict, policy: dict) -> dict:
        rows = snapshot["rows"]
        expected = {row["section"]: row for row in _facts()}
        seen, errors = set(), []
        for row in rows:
            row_id = row["row_id"]
            section = row.get("section")
            fact = expected.get(section) if isinstance(section, str) else None
            if fact is None or section in seen:
                errors.append({"row_id": row_id, "field": "section", "code": "unknown_or_duplicate_section"})
            else:
                seen.add(section)
                for field in ("observations", "minutes"):
                    if type(row.get(field)) is not int or row[field] != fact[field]:
                        errors.append({"row_id": row_id, "field": field, "code": "synthetic_fact_mismatch"})
            if row.get("notice") != DEMO_NOTICE:
                errors.append({"row_id": row_id, "field": "notice", "code": "example_notice_required"})
            if not isinstance(row.get("note"), str) or not row["note"].strip():
                errors.append({"row_id": row_id, "field": "note", "code": "revision_note_required"})
        if seen != set(expected) or len(rows) != len(expected):
            errors.append({"row_id": None, "field": "section", "code": "report_incomplete"})
        for rule in set(policy.get("required_rules", [])) - set(self.RULES):
            errors.append({"row_id": None, "field": "policy", "code": "unsupported_content_rule"})
        return {"complete": True, "valid": not errors, "errors": errors,
                "row_count": len(rows), "checked_rules": list(self.RULES)}


def _observation(messages):
    # Restrict policy decisions to the current user turn and actual tool replies.
    start = max((i for i, item in enumerate(messages) if item.get("role") == "user"), default=-1)
    calls = {}
    for item in messages[start + 1:]:
        for call in item.get("tool_calls") or []:
            calls[call["id"]] = call.get("function", {}).get("name", call.get("name"))
    for item in reversed(messages[start + 1:]):
        if item.get("role") == "tool":
            content = item.get("content")
            try:
                payload = json.loads(content) if isinstance(content, str) else content
            except (ValueError, TypeError):
                payload = {}
            return calls.get(item.get("tool_call_id")), payload if isinstance(payload, dict) else {}
    return None, {}


class OfflineReferenceModel:
    """A stateless scripted policy, not an LLM or a table of user answers.

    Decisions follow observations. request_changes copies the actual reviewer's
    comment into a regenerated report; the runtime must obtain a fresh review.
    """

    def __call__(self, config, messages, tools):
        del config  # Explicitly ignore keys, endpoints and provider environment.
        name, observation = _observation(messages)
        artifacts = _artifacts(observation)
        result = LLMChatWithToolsResult(model=DEMO_MODEL, provider="offline-reference",
                                       content=DEMO_NOTICE, finish_reason="stop")

        def call(tool, arguments):
            if tool not in {item.get("function", {}).get("name") for item in tools or []}:
                return LLMChatWithToolsResult(model=DEMO_MODEL, provider="offline-reference",
                                             error_type="demo_tool_unavailable")
            result.tool_calls = [LLMToolCall(id="demo_" + uuid.uuid4().hex, name=tool, arguments=arguments)]
            result.finish_reason = "tool_calls"
            return result

        if name is None:
            return call("report__read_materials", {})
        error = (observation.get("error") or {}).get("type", "")
        if error == "artifact_review_request_changes":
            review = next((a for a in reversed(artifacts) if a.get("type") == "artifact.review"), {})
            feedback = review.get("comment", "")
            if not isinstance(feedback, str) or not feedback.strip():
                result.content = "Revision needs a reviewer comment; no example archived. " + DEMO_NOTICE
                return result
            # Recover the latest successful draft from the real observation log.
            for item in reversed(messages):
                if item.get("role") != "tool":
                    continue
                try:
                    previous = json.loads(item.get("content") or "{}")
                except (ValueError, TypeError):
                    continue
                draft = next((a for a in _artifacts(previous) if a.get("type") == "demo.draft"), None)
                if draft:
                    return call("report__draft", {"material_id": _material_id(draft), "feedback": feedback})
            result.content = "Draft reference unavailable; no example archived. " + DEMO_NOTICE
            return result
        if error or observation.get("status") == "failed":
            result.content = "Stopped after refusal or failure; no example archived. " + DEMO_NOTICE
            return result
        if name == "report__read_materials":
            material = next((a for a in artifacts if a.get("type") == "demo.materials"), None)
            if material:
                return call("report__draft", {"material_id": _material_id(material)})
        if name == "report__draft":
            draft = next((a for a in artifacts if a.get("type") == "demo.draft"), None)
            if draft:
                return call("report__archive", {"root_dir": str(Path(draft["path"]).parent),
                                                 "artifact_files": [draft["path"]]})
        result.content = ("Reviewed example archived locally. " if name == "report__archive"
                          else "Example observation unavailable; stopped. ") + DEMO_NOTICE
        return result


class _DemoFiles:
    def __init__(self, root):
        self.root = root
        self.materials = {}

    def read_materials(self, request: AdapterInput) -> AdapterOutput:
        identifier = uuid.uuid4().hex
        directory = self.root / identifier
        _require_local(directory, self.root)
        directory.mkdir()
        path = directory / "materials.json"
        _require_local(path, self.root)
        path.write_text(json.dumps(_facts(), indent=2), encoding="utf-8")
        (directory / "draft").mkdir()
        self.materials[identifier] = directory
        facts = json.loads(path.read_text(encoding="utf-8"))
        return AdapterOutput(status="success", summary="Read synthetic reading-area observations. " + DEMO_NOTICE,
                             artifacts=[{"type": "demo.materials", "material_id": identifier,
                                         "path": str(path), "facts": facts, "notice": DEMO_NOTICE}])

    def draft(self, request: AdapterInput) -> AdapterOutput:
        directory = self.materials.get(request.params.get("material_id"))
        if directory is None:
            return blocked_output(summary="Unknown demo material", error_type="demo_material_missing",
                                  message="Read this runtime's synthetic materials first.")
        _require_local(directory / "materials.json", self.root)
        facts = json.loads((directory / "materials.json").read_text(encoding="utf-8"))
        feedback = request.params.get("feedback", "").strip()
        note = "Reviewer revision: " + feedback if feedback else "Initial synthetic observation summary."
        rows = [{**fact, "note": note, "notice": DEMO_NOTICE} for fact in facts]
        path = directory / "draft" / "report.json"
        _require_local(path, self.root)
        temporary = path.with_name(uuid.uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
        _require_local(path, self.root)
        temporary.replace(path)
        return AdapterOutput(status="success", summary="Regenerated example draft. " + DEMO_NOTICE,
                             artifacts=[{"type": "demo.draft", "path": str(path), "notice": DEMO_NOTICE,
                                         "material_id": request.params["material_id"], "feedback": feedback}])

    def archive(self, request: AdapterInput) -> AdapterOutput:
        # execute_adapter has verified the engine-owned binding; consume only
        # its staged input, never rediscover a draft from the original directory.
        if not request.artifact_binding:
            return blocked_output(summary="Review required", error_type="artifact_review_runtime_required",
                                  message="Use the runtime review gate before archiving.")
        sources = request.params["artifact_files"]
        if len(sources) != 1:
            return blocked_output(summary="One example required", error_type="demo_archive_input_invalid",
                                  message="Expected one reviewed example report.")
        content = Path(sources[0]).read_bytes()
        directory = self.root / "archives"
        _require_local(directory, self.root)
        directory.mkdir(exist_ok=True)
        path = directory / (uuid.uuid4().hex + ".json")
        _require_local(path, self.root)
        with path.open("xb") as stream:
            stream.write(content)
        return AdapterOutput(status="success", summary="Archived reviewed temporary example. " + DEMO_NOTICE,
                             artifacts=[{"type": "demo.archive", "path": str(path), "notice": DEMO_NOTICE}])


def build_demo_runtime(data_dir) -> AgentRuntime:
    """Build an isolated offline runtime; first start pauses for artifact review.

    resume(run_id, interaction_id, {"decision": "approve" | "reject" |
    "request_changes", "comment": "..."}) uses the ordinary AgentRuntime API.
    The caller owns cleanup of data_dir and its temporary example files.
    """
    data_dir = Path(data_dir).absolute()
    _require_local(data_dir, data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = _DemoFiles(Path(tempfile.mkdtemp(prefix="offline-demo-", dir=data_dir)))
    executor = RegisteredToolExecutor()
    schemas = {
        "read_materials": ({}, []),
        "draft": ({"material_id": {"type": "string", "pattern": "^[a-f0-9]{32}$"},
                   "feedback": {"type": "string", "maxLength": 2000}}, ["material_id"]),
        "archive": ({"root_dir": {"type": "string", "minLength": 1},
                     "artifact_files": {"type": "array", "items": {"type": "string"},
                                        "minItems": 1, "maxItems": 1}}, ["root_dir", "artifact_files"]),
    }
    actions = {}
    for action, (properties, required) in schemas.items():
        adapter = "offline_demo." + action
        executor.register(adapter, getattr(files, action), modes=(ExecutionMode.DRY_RUN,))
        actions[action] = {
            "description": action.replace("_", " ") + ". " + DEMO_NOTICE,
            "risk_level": "L1" if action == "read_materials" else "L2",
            "required_params": required, "execution_adapter": adapter,
            "parameter_contract": {"type": "object", "properties": properties,
                                   "required": required, "additionalProperties": False},
            "dry_run_status": "available", "read_only_status": "unavailable",
            "local_write_status": "unavailable", "live_status": "unavailable",
        }
    actions["archive"]["artifact_policy"] = {
        "required": True, "scope": "explicit_file_set", "rule_version": "demo-content-v1",
        "required_rules": list(DemoArtifactValidator.RULES),
        "columns": ["section", "observations", "minutes", "note", "notice"],
    }
    validator = DemoArtifactValidator()
    registry = ModuleCapabilityRegistry(
        payload={"version": "offline-demo-v1", "modules": {
            "report": {"display_name": "Synthetic reading-area report", "actions": actions}}},
        artifact_validator=validator, system_prompt=DEMO_NOTICE + " Use the offline reference tools.",
    )
    config = StaticConfigProvider({"execution_mode": "dry_run", "model": DEMO_MODEL,
                                   "module_config": {}, "parameter_sources": {},
                                   "memory_dir": None, "max_tool_rounds": 20})
    return AgentRuntime(RuntimeServices(registry=registry, executor=executor,
                                       model_client=OfflineReferenceModel(), config_provider=config,
                                       artifact_validator=validator, allow_real_model=False), data_dir=data_dir)
