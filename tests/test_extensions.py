"""Offline extension contracts with anonymous, temporary local fixtures only."""

import builtins
import importlib
import io
import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_extensions(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKBENCH_DISABLE_CENTRAL_ENV", "1")
    monkeypatch.setenv("AGENT_WORKBENCH_HOME", str(tmp_path / "storage"))
    for name in (
        "AGENT_WORKBENCH_MEMORY_ENABLED",
        "AGENT_WORKBENCH_MEMORY_REVIEW_ENABLED",
        "AGENT_WORKBENCH_SESSION_LEDGER_ENABLED",
        "AGENT_WORKBENCH_SKILLS_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("Extension tests must remain offline")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    for module in (builtins, io):
        original = module.open

        def safe_open(file, *args, _open=original, **kwargs):
            if not isinstance(file, int) and Path(file).name == ".env":
                raise AssertionError("Extension tests must never read credential files")
            return _open(file, *args, **kwargs)

        monkeypatch.setattr(module, "open", safe_open)


def _memory_fields(scope="global", key=""):
    return {
        "memory_type": "workflow_preference",
        "scope_type": scope,
        "scope_key": key,
        "title": "Document review order",
        "summary": "Check document evidence before writing a summary.",
        "why": "Confirmed preference in an anonymous test.",
        "how_to_apply": "Review the document when the current request permits it.",
    }


def _profile(tmp_path, *, root_kind="env"):
    import yaml

    from agent_workbench.company_profile import resolve_company_profile

    profiles = tmp_path / "profiles"
    profiles.mkdir(exist_ok=True)
    provider = {"provider": "local", "source_id": "fixture", "root_kind": root_kind}
    if root_kind == "env":
        provider["root_env"] = "AGENT_WORKBENCH_SKILLS_ROOT"
    raw = {
        "profile_id": "fixture",
        "tenant_id": "tenant-one",
        "display_name": "Fixture",
        "version": "1.0.0",
        "skill_provider": provider,
        "skills": {"available": ["document-review"], "pins": {"document-review": "1.0.0"}},
        "memory_namespace": "fixture-memory",
    }
    (profiles / "fixture.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = {
        "enabled": True,
        "profile_id": "fixture",
        "tenant_id": "tenant-one",
        "profiles_root": str(profiles),
    }
    result = resolve_company_profile(config, env={})
    assert result.error_type is None
    assert result.snapshot is not None
    return config, result


def _skill(tmp_path):
    root = tmp_path / "skills"
    directory = root / "document-review" / "1.0.0"
    directory.mkdir(parents=True)
    text = (
        "---\nname: document-review\ndescription: Review synthetic documents.\n"
        "metadata:\n  version: '1.0.0'\n  status: active\n"
        "  resources: [references/checklist.md]\n---\n"
        "Check the document evidence before drafting a summary.\n"
    )
    (directory / "SKILL.md").write_text(text, encoding="utf-8")
    (directory / "references").mkdir()
    (directory / "references" / "checklist.md").write_text("Verify all cited evidence.\n", encoding="utf-8")
    return root, directory


def test_defaults_do_not_touch_filesystem(tmp_path, monkeypatch):
    from agent_workbench.background_memory_review import MemoryReviewPolicy, review_run_for_memory_candidates
    from agent_workbench.company_profile import resolve_company_profile
    from agent_workbench.knowledge_registry import retrieve_knowledge_for_langgraph
    from agent_workbench.memory_retriever import recall_approved_memories, recall_memories_for_langgraph
    from agent_workbench.session_ledger import SessionSearchRetriever, record_langgraph_run_to_ledger, retrieve_session_search
    from agent_workbench.skill_registry import prepare_company_skill_runtime

    def unexpected(*args, **kwargs):
        raise AssertionError("Disabled extensions must not access the filesystem")

    for name in ("MEMORY_ENABLED", "MEMORY_REVIEW_ENABLED", "SESSION_LEDGER_ENABLED"):
        monkeypatch.setenv("AGENT_WORKBENCH_" + name, "1")
    with monkeypatch.context() as guard:
        for method in ("read_text", "read_bytes", "mkdir", "exists", "is_file", "resolve", "glob", "rglob", "iterdir"):
            guard.setattr(Path, method, unexpected)
        guard.setattr(builtins, "open", unexpected)
        guard.setattr(io, "open", unexpected)
        assert resolve_company_profile(None).enabled is False
        runtime = prepare_company_skill_runtime(None)
        assert runtime["loaded_skills"] == []
        assert runtime["company_profile"]["enabled"] is False
        assert retrieve_knowledge_for_langgraph("document")["enabled"] is False
        recall = recall_approved_memories(user_request="document", candidate_plan=None, candidate_intent="")
        assert recall.summary["enabled"] is False
        assert recall_memories_for_langgraph(user_request="document", memory_dir=None).summary["enabled"] is False
        assert review_run_for_memory_candidates({"thread_id": "fixture"})["enabled"] is False
        assert review_run_for_memory_candidates({}, policy=MemoryReviewPolicy(enabled=True))["enabled"] is False
        assert record_langgraph_run_to_ledger(state={}, report={}) is None
        assert retrieve_session_search("document").enabled is False
        assert SessionSearchRetriever().search("document").enabled is False
    assert not (tmp_path / "storage").exists()


def test_storage_defaults_are_user_relative_and_overridable(tmp_path, monkeypatch):
    from agent_workbench.local_knowledge_provider import default_knowledge_index_path
    from agent_workbench.memory_store import default_memory_dir
    from agent_workbench.session_checkpoint import default_session_dir
    from agent_workbench.session_ledger import default_ledger_db_path
    from agent_workbench.skill_store import default_skill_governance_db

    expected = tmp_path / "storage"
    for path in (default_memory_dir(), default_session_dir(), default_ledger_db_path(),
                 default_skill_governance_db(), default_knowledge_index_path("tenant-one")):
        assert path.is_relative_to(expected)
    monkeypatch.delenv("AGENT_WORKBENCH_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    expected = tmp_path / "home" / ".agent-workbench"
    for path in (default_memory_dir(), default_session_dir(), default_ledger_db_path(),
                 default_skill_governance_db(), default_knowledge_index_path("tenant-one")):
        assert path.is_relative_to(expected)
    assert default_memory_dir(tmp_path) == tmp_path / "runs" / "agent_memory"
    assert not expected.exists()


def test_memory_approval_scope_isolation_and_forgetting(tmp_path):
    from agent_workbench.memory_retriever import recall_approved_memories
    from agent_workbench.memory_store import MemoryStore

    store = MemoryStore(tmp_path / "one")
    other = MemoryStore(tmp_path / "two")
    candidate, error = store.create_candidate(_memory_fields("user", "reader-one"))
    assert error is None
    args = dict(user_request="document review", candidate_plan=None, candidate_intent="", memory_store=store)
    assert recall_approved_memories(**args).approved_memories == []
    approved, error = store.approve(candidate.memory_id, approved_by="reviewer")
    assert error is None and approved.status == "approved"
    assert recall_approved_memories(**args).approved_memories == []
    recalled = recall_approved_memories(**args, session_record={"known_parameters": {"user_id": "reader-one"}})
    assert [item["memory_id"] for item in recalled.approved_memories] == [candidate.memory_id]
    assert other.get(candidate.memory_id)[0] is None
    forgotten, error = store.forget(candidate.memory_id)
    assert error is None and forgotten.status == "forgotten"
    assert forgotten.summary == ""
    assert recall_approved_memories(**args).approved_memories == []


def test_memory_explicit_enable_and_disable(tmp_path, monkeypatch):
    from agent_workbench.memory_retriever import recall_approved_memories
    from agent_workbench.memory_store import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    candidate, error = store.create_candidate(_memory_fields())
    assert error is None
    assert store.approve(candidate.memory_id, approved_by="reviewer")[1] is None
    args = dict(user_request="document", candidate_plan=None, candidate_intent="", memory_store=store)
    assert recall_approved_memories(**args).summary["used"] is True
    assert recall_approved_memories(**args, enabled=False).summary["enabled"] is False
    monkeypatch.setenv("AGENT_WORKBENCH_MEMORY_ENABLED", "1")
    assert recall_approved_memories(user_request="document", candidate_plan=None, candidate_intent="").summary["enabled"] is False


def test_workflow_inference_uses_explicit_ids_or_injected_aliases():
    from agent_workbench.memory_retriever import infer_workflow_keys_from_request

    assert infer_workflow_keys_from_request("ordinary prose") == set()
    assert infer_workflow_keys_from_request("use document.review") == {"document", "document.review"}
    assert infer_workflow_keys_from_request("summarize", workflow_hints=(("summarize", ("document.review",)),)) == {"document.review"}


def test_background_review_creates_unapproved_candidates(tmp_path):
    from agent_workbench.background_memory_review import MemoryReviewPolicy, review_run_for_memory_candidates
    from agent_workbench.memory_store import MemoryStore

    store = MemoryStore(tmp_path / "memory")
    result = review_run_for_memory_candidates(
        {"run_id": "run-one", "user_request": "document.review", "verification": {"passed": True},
         "failed_steps": [{"module": "document", "action": "review", "error": "fixture failure"}]},
        policy=MemoryReviewPolicy(enabled=True), memory_store=store,
        ledger_context={"thread_turn_count": 1, "workflow_repeat_count": 0},
    )
    assert result["triggered"] is True
    assert result["candidate_ids"]
    assert all(store.get(key)[0].status == "candidate" for key in result["candidate_ids"])


@pytest.mark.parametrize("use_fts", [True, False])
def test_session_ledger_thread_and_profile_isolation(tmp_path, monkeypatch, use_fts):
    from agent_workbench.session_ledger import SessionLedger

    ledger = SessionLedger(tmp_path / "state.db")
    assert ledger.healthy
    if not use_fts:
        monkeypatch.setattr(ledger, "_search_fts", lambda *args, **kwargs: [])
    for run, thread, binding in (("one", "thread-one", "profile-one"), ("two", "thread-two", "profile-one"),
                                 ("three", "thread-one", "profile-two"), ("legacy", "thread-one", None)):
        assert ledger.record_run(run_id=run, thread_id=thread, company_profile_binding_id=binding,
                                 user_request="document evidence") == run
    hits = ledger.search("document", thread_id="thread-one", company_profile_binding_id="profile-one")
    assert [hit.run_id for hit in hits] == ["one"]
    if not use_fts:
        assert hits[0].match_reason == "keyword_overlap"


def test_checkpoint_roundtrip_and_path_denial(tmp_path):
    from agent_workbench.session_checkpoint import SessionCheckpointStore, SessionRecord

    store = SessionCheckpointStore(tmp_path / "sessions")
    record = SessionRecord.new_thread("thread-one")
    record.company_profile_binding_id = "profile-one"
    record.known_parameters = {"document.format": "text"}
    assert store.save(record) is None
    restored, error = store.load("thread-one")
    assert error is None
    assert restored.company_profile_binding_id == "profile-one"
    assert restored.known_parameters == record.known_parameters
    assert store.load("../outside")[1] == "invalid_thread_id"


def test_checkpoint_direct_save_redacts_sensitive_fields(tmp_path):
    from agent_workbench.session_checkpoint import SessionCheckpointStore, SessionRecord

    store = SessionCheckpointStore(tmp_path / "sessions")
    record = SessionRecord.new_thread("thread-one")
    record.known_parameters = {"api_key": "synthetic-fixture-marker"}
    record.artifact_refs = [{"token": "synthetic-fixture-marker"}]
    assert store.save(record) is None
    persisted = (tmp_path / "sessions" / "thread-one.json").read_text(encoding="utf-8")
    assert "synthetic-fixture-marker" not in persisted


@pytest.mark.parametrize("provider,settings,error", [
    ("missing-fixture-provider", {}, "knowledge_provider_unavailable"),
    ("local", {}, "knowledge_root_missing"),
    ("http_rag", {}, "http_rag_config_missing"),
    ("opensearch", {}, "opensearch_config_missing"),
    ("mcp", {}, "mcp_config_missing"),
    ("mcp", {"server_id": "fixture"}, "mcp_client_missing"),
])
def test_missing_knowledge_connectors_are_structured(provider, settings, error):
    from agent_workbench.knowledge_registry import retrieve_knowledge_for_langgraph

    result = retrieve_knowledge_for_langgraph("document", {
        "enabled": True, "provider": provider, "tenant_id": "tenant-one",
        "providers": {provider: settings},
    })
    assert result["hits"] == []
    assert result["error_type"] == error


def test_local_knowledge_real_search_fetch_and_tenant_isolation(tmp_path):
    from agent_workbench.knowledge_provider import KnowledgeFetchRequest, KnowledgeSearchRequest
    from agent_workbench.local_knowledge_provider import LocalKnowledgeProvider, sanitize_tenant_id

    root = tmp_path / "documents"
    root.mkdir()
    (root / "notes.md").write_text("# Evidence\nDocument evidence includes three observations.\n", encoding="utf-8")
    provider = LocalKnowledgeProvider(root=root, tenant_id="tenant-one", index_dir=tmp_path / "index")
    response = provider.search(KnowledgeSearchRequest("document evidence", tenant_id="tenant-one"))
    assert response.error_type is None and response.hits
    hit = response.hits[0]
    document = provider.fetch(KnowledgeFetchRequest(hit.document_id, tenant_id="tenant-one"))
    assert document.error_type is None and "three observations" in document.text
    denied = provider.search(KnowledgeSearchRequest("document", tenant_id="tenant-two"))
    assert denied.hits == [] and denied.error_type
    denied_document = provider.fetch(KnowledgeFetchRequest(hit.document_id, tenant_id="tenant-two"))
    assert denied_document.text == "" and denied_document.error_type
    assert sanitize_tenant_id("scope/a") != sanitize_tenant_id("scope_a")
    assert sanitize_tenant_id("x" * 65) != sanitize_tenant_id("x" * 64)


def test_knowledge_acl_missing_and_deny_override_allow():
    from agent_workbench.knowledge_acl import KnowledgeIdentity, check_knowledge_acl

    identity = KnowledgeIdentity("tenant-one", user_id="reader")
    assert check_knowledge_acl(None, identity, "tenant-one").allowed is False
    assert check_knowledge_acl({"visibility": "tenant"}, identity, "tenant-two").allowed is False
    acl = {"visibility": "restricted", "allowed_users": ["reader"], "denied_users": ["reader"]}
    assert check_knowledge_acl(acl, identity, "tenant-one").reason == "deny_user"


def test_profile_requires_explicit_identity_and_skill_root(tmp_path):
    from agent_workbench.company_profile import resolve_company_profile
    from agent_workbench.skill_registry import build_skill_provider

    config, resolved = _profile(tmp_path, root_kind="builtin")
    provider, error = build_skill_provider(resolved.snapshot, env={})
    assert provider is None and error == "skill_provider_unavailable"
    mismatch = resolve_company_profile({**config, "trusted_tenant_id": "tenant-two"})
    assert mismatch.error_type == "company_profile_tenant_mismatch"
    invalid = resolve_company_profile({**config, "profile_id": "../outside"})
    assert invalid.error_type == "company_profile_invalid"
    root, _ = _skill(tmp_path)
    provider, error = build_skill_provider(resolved.snapshot, env={"AGENT_WORKBENCH_SKILLS_ROOT": str(root)})
    assert error is None and provider.list_metadata(resolved.snapshot).skills


def test_skill_loading_metadata_resources_checksum_and_isolation(tmp_path):
    from agent_workbench.skill_guard import guard_skill_load
    from agent_workbench.skill_registry import build_skill_provider, prepare_company_skill_runtime
    from agent_workbench.skill_store import SqliteSkillVersionStore

    config, resolved = _profile(tmp_path)
    root, directory = _skill(tmp_path)
    env = {"AGENT_WORKBENCH_SKILLS_ROOT": str(root)}
    provider, error = build_skill_provider(resolved.snapshot, env=env)
    assert error is None
    catalog = provider.list_metadata(resolved.snapshot)
    assert len(catalog.skills) == 1
    assert "body" not in json.dumps(catalog.to_dict())
    guarded = guard_skill_load("document-review", profile=resolved.snapshot, catalog=catalog)
    assert guarded.allowed
    ref = guarded.skill_ref
    loaded = provider.load_skill(ref, resolved.snapshot)
    assert loaded.loaded and "document evidence" in loaded.skill.body
    resource = provider.load_resource(ref, "references/checklist.md", resolved.snapshot)
    assert resource.loaded and "cited evidence" in resource.skill.body
    assert provider.load_resource(ref, "../outside.md", resolved.snapshot).error_type == "skill_path_denied"
    assert provider.load_resource(ref, "references/missing.md", resolved.snapshot).error_type == "skill_resource_not_declared"
    other = replace(resolved.snapshot, tenant_id="tenant-two")
    assert provider.load_skill(ref, other).error_type == "skill_not_allowed"
    assert provider.load_skill(ref, resolved.snapshot, deadline_monotonic=0).error_type == "skill_provider_timeout"
    store = SqliteSkillVersionStore(tmp_path / "governance.db")
    runtime = prepare_company_skill_runtime(config, env=env, skill_version_store=store)
    assert runtime["company_profile_snapshot_id"] == resolved.snapshot.snapshot_id
    assert len(runtime["skill_catalog"]["skills"]) == 1
    assert runtime["loaded_skills"] == []
    (directory / "SKILL.md").write_text((directory / "SKILL.md").read_text(encoding="utf-8") + "Changed.\n", encoding="utf-8")
    assert provider.load_skill(ref, resolved.snapshot).error_type == "skill_checksum_mismatch"


def test_governance_preserves_immutable_version_records(tmp_path):
    from agent_workbench.skill_models import SkillRef
    from agent_workbench.skill_store import SqliteSkillVersionStore

    ref = SkillRef(source_id="fixture", tenant_id="tenant-one", name="document-review", version="1.0.0", checksum="a" * 64)
    store = SqliteSkillVersionStore(tmp_path / "governance.db")
    payload = {"skill_id": ref.stable_id(), "skill_name": ref.name, "version": ref.version,
               "checksum": ref.checksum, "tenant_id": ref.tenant_id, "source_id": ref.source_id,
               "author_id": "author"}
    result = store.register_version(payload)
    assert result["error_type"] is None
    assert result["record"]["lifecycle"] == "draft"
    assert store.list_versions(ref.name, "tenant-two") == []
    assert store.register_version({**payload, "checksum": "b" * 64})["error_type"] == "skill_version_immutable_conflict"
    assert store.register_version({**payload, "lifecycle": "active"})["error_type"] == "skill_lifecycle_transition_denied"


def test_explicit_memory_directory_preserves_graph_review_behavior(tmp_path):
    from agent_workbench.background_memory_review import review_run_for_memory_candidates
    from agent_workbench.memory_retriever import recall_memories_for_langgraph
    from agent_workbench.memory_store import MemoryStore

    directory = tmp_path / "memory"
    result = review_run_for_memory_candidates(
        {"run_id": "fixture", "user_request": "document.review", "verification": {"passed": True},
         "failed_steps": [{"module": "document", "action": "review", "error": "fixture failure"}]},
        memory_dir=directory,
    )
    assert result["candidate_ids"]
    store = MemoryStore(directory)
    key = result["candidate_ids"][0]
    assert store.approve(key, approved_by="reviewer")[1] is None
    recalled = recall_memories_for_langgraph(user_request="document.review", memory_dir=directory)
    assert key in recalled.summary["retrieved_memory_ids"]


def test_owned_modules_import_offline_without_private_dependencies():
    import ast

    root = Path(__file__).resolve().parents[1] / "src" / "agent_workbench"
    names = {"background_memory_review", "session_checkpoint", "session_ledger", "company_profile",
             "local_skill_provider", "local_knowledge_provider"}
    names.update("skill_" + name for name in ("context", "governance", "guard", "models", "provider", "registry", "store", "tool_schema"))
    paths = [path for path in root.glob("*.py") if path.stem in names or path.stem.startswith(("memory_", "knowledge_"))]
    paths = [path for path in paths if path.stem != "knowledge_sync_cli"]
    assert len(paths) == 32
    for path in paths:
        module = importlib.import_module("agent_workbench." + path.stem)
        assert Path(module.__file__).resolve() == path.resolve()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                dependencies = [node.module or ""]
            else:
                continue
            assert all(name.split(".")[0] not in {"agent", "modules", "core", "scripts", "commerce_eval"}
                       for name in dependencies), path.name
