"""Public dependency/privacy gates; all fixtures and runtime writes are temporary.

The installed-package probe never builds or installs anything. Set
AGENT_WORKBENCH_BOUNDARY_PYTHON to a wheel-installed interpreter to require it;
otherwise it skips when the package is absent or its installation is physically
inside the checkout. AGENT_WORKBENCH_REQUIRE_INSTALLED=1 makes either condition
a failure. An external installation always receives the full isolation probe.
"""

import ast
import builtins
import io
import json
import os
import re
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "agent_workbench"
PRIVATE_IMPORT_ROOTS = frozenset({"agent", "modules", "core", "scripts", "commerce_eval"})
STORAGE_MODULES = (
    "company_profile.py", "memory_store.py", "session_checkpoint.py",
    "session_ledger.py", "skill_store.py", "skill_registry.py",
    "local_knowledge_provider.py", "registry.py", "runtime.py",
)
DOCUMENT_SUFFIXES = frozenset({".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".tsv", ".pdf", ".docx", ".xlsx"})


def _private_imports(tree):
    violations = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            is_import = (
                isinstance(function, ast.Name) and function.id in {"__import__", "import_module"}
                or isinstance(function, ast.Attribute) and function.attr == "import_module"
            )
            if is_import and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                names = [node.args[0].value]
        for name in names:
            if name.split(".", 1)[0] in PRIVATE_IMPORT_ROOTS:
                violations.append((node.lineno, name))
    return violations


@pytest.fixture(autouse=True)
def boundary_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_WORKBENCH_DISABLE_CENTRAL_ENV", "1")
    monkeypatch.setenv("AGENT_WORKBENCH_HOME", str(tmp_path / "storage"))
    for name in ("AGENT_WORKBENCH_MEMORY_ENABLED", "AGENT_WORKBENCH_MEMORY_REVIEW_ENABLED",
                 "AGENT_WORKBENCH_SESSION_LEDGER_ENABLED", "AGENT_WORKBENCH_SKILLS_ROOT"):
        monkeypatch.delenv(name, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("Public-boundary tests must stay offline")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    for module in (builtins, io):
        original = module.open

        def no_credentials(file, *args, _open=original, **kwargs):
            if not isinstance(file, int) and Path(file).name.lower().endswith(".env"):
                raise AssertionError("Credential files are outside the public-boundary test")
            return _open(file, *args, **kwargs)

        monkeypatch.setattr(module, "open", no_credentials)


@pytest.mark.parametrize("statement", [
    "import agent", "from modules.part import value", "import core.part as helper",
    "from scripts import helper", "from commerce_eval.part import value",
    "__import__('agent.part')", "importlib.import_module('core.part')",
])
def test_private_import_scanner_rejects_direct_and_literal_dynamic_imports(statement):
    assert _private_imports(ast.parse(statement))


def test_public_package_has_no_private_import_dependencies():
    violations = []
    paths = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert paths
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        violations.extend(f"{path.relative_to(PACKAGE_ROOT)}:{line}: {name}"
                          for line, name in _private_imports(tree))
    assert violations == []
    assert _private_imports(ast.parse("from . import state\nfrom agent_workbench import registry")) == []


def test_no_private_default_catalog_or_project_directory_discovery():
    violations = []
    absolute_user_path = re.compile(r"(?:[A-Za-z]:[\\/]|/)(?:Users|home)[\\/]", re.IGNORECASE)
    private_catalogs = {"company_profiles", "module_capabilities.json", "playbooks.json"}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                normalized = node.value.replace("\\", "/")
                if absolute_user_path.search(normalized) or any(part in private_catalogs for part in normalized.split("/")):
                    violations.append(f"{path.name}:{node.lineno}: private default path")
            if path.name in STORAGE_MODULES and isinstance(node, ast.Name) and node.id == "__file__":
                violations.append(f"{path.name}:{node.lineno}: source-relative storage discovery")
    assert violations == []


def test_no_bundled_profile_skill_or_business_documents():
    documents = [str(path.relative_to(PACKAGE_ROOT)) for path in PACKAGE_ROOT.rglob("*")
                 if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES]
    assert documents == [], "Documents and catalogs must be supplied by an explicit host configuration"
    for directory in ("company_profiles", "skills", "data", "knowledge", "memory"):
        assert not (PACKAGE_ROOT / directory).exists()


def test_defaults_require_explicit_host_catalog_and_extension_config(tmp_path, monkeypatch):
    from agent_workbench.background_memory_review import review_run_for_memory_candidates
    from agent_workbench.knowledge_registry import retrieve_knowledge_for_langgraph
    from agent_workbench.memory_retriever import recall_memories_for_langgraph
    from agent_workbench.registry import ModuleCapabilityRegistry, get_registry
    from agent_workbench.session_ledger import retrieve_session_search
    from agent_workbench.skill_registry import prepare_company_skill_runtime

    def no_storage(*args, **kwargs):
        raise AssertionError("Default-off mechanisms must not access storage")

    with monkeypatch.context() as guard:
        for method in ("read_text", "read_bytes", "mkdir", "iterdir", "glob", "rglob", "resolve", "exists", "is_file"):
            guard.setattr(Path, method, no_storage)
        guard.setattr(builtins, "open", no_storage)
        guard.setattr(io, "open", no_storage)
        with pytest.raises((RuntimeError, ValueError)):
            get_registry()
        with pytest.raises((RuntimeError, ValueError)):
            ModuleCapabilityRegistry()
        assert prepare_company_skill_runtime(None)["company_profile"]["enabled"] is False
        assert recall_memories_for_langgraph(user_request="fixture").summary["enabled"] is False
        assert review_run_for_memory_candidates({"user_request": "fixture"})["enabled"] is False
        assert retrieve_session_search("fixture").enabled is False
        assert retrieve_knowledge_for_langgraph("fixture")["enabled"] is False
    assert not (tmp_path / "storage").exists()


def test_memory_defaults_and_runtime_data_dir_use_user_home(tmp_path, monkeypatch):
    from agent_workbench.local_knowledge_provider import default_knowledge_index_path
    from agent_workbench.memory_store import default_memory_dir
    from agent_workbench.registry import ModuleCapabilityRegistry
    from agent_workbench.runtime import AgentRuntime, RuntimeServices
    from agent_workbench.session_checkpoint import default_session_dir
    from agent_workbench.session_ledger import default_ledger_db_path
    from agent_workbench.skill_store import default_skill_governance_db

    home = tmp_path / "home"
    monkeypatch.delenv("AGENT_WORKBENCH_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    expected = home / ".agent-workbench"
    for path in (default_memory_dir(), default_session_dir(), default_ledger_db_path(),
                 default_skill_governance_db(), default_knowledge_index_path("fixture")):
        assert path.is_relative_to(expected)

    class NoExecution:
        def execute(self, *args, **kwargs):
            raise AssertionError("The storage-boundary test must not execute tools")

    services = RuntimeServices(registry=ModuleCapabilityRegistry(payload={"modules": {}}), executor=NoExecution())
    runtime = AgentRuntime(services)
    assert runtime.data_dir == expected.resolve()
    explicit = AgentRuntime(services, data_dir=tmp_path / "explicit")
    assert explicit.data_dir == (tmp_path / "explicit").resolve()


def test_profile_default_root_is_generic_and_configurable(tmp_path, monkeypatch):
    from agent_workbench.company_profile import DEFAULT_PROFILES_ROOT, _locate_profile_file

    assert DEFAULT_PROFILES_ROOT == Path.home() / ".agent-workbench" / "profiles"
    root = tmp_path / "storage" / "profiles"
    root.mkdir(parents=True)
    path = root / "fixture.yaml"
    path.write_text("profile_id: fixture\n", encoding="utf-8")
    assert _locate_profile_file(None, "fixture") == (path.resolve(), None)
    explicit = tmp_path / "explicit-profiles"
    explicit.mkdir()
    explicit_path = explicit / "fixture.yaml"
    explicit_path.write_text("profile_id: fixture\n", encoding="utf-8")
    assert _locate_profile_file(str(explicit), "fixture") == (explicit_path.resolve(), None)


def test_workspace_wording_preserves_legacy_scope_fields():
    from agent_workbench.background_memory_review import _build_candidates_from_signals

    candidates = _build_candidates_from_signals(
        {"run_id": "fixture", "module_config": {"document": {"store_id": "scope-one"}},
         "tool_calls": [{"tool_name": "document.review"}]},
        {}, ["tool_failure"], [],
    )
    scoped = next(item for item in candidates if item["memory_type"] == "store_knowledge")
    assert scoped["scope_type"] == "store"
    assert scoped["scope_key"] == "scope-one"
    assert scoped["title"] == "Workspace context: scope-one"
    assert "workspace scope" in scoped["why"]


_INSTALLED_PROBE = textwrap.dedent(r'''
    import importlib
    import importlib.util
    import json
    import os
    import pkgutil
    import sys
    from pathlib import Path

    checkout = Path(sys.argv[1]).resolve()
    home = Path(sys.argv[2]).resolve()
    assert sys.flags.isolated == 1
    assert not any(Path(item).resolve() == checkout for item in sys.path if item)
    spec = importlib.util.find_spec("agent_workbench")
    if spec is None:
        print(json.dumps({"status": "not_installed"}))
        raise SystemExit(77)
    assert spec.origin, "Expected an installed package with a concrete origin"
    package = Path(spec.origin).resolve().parent
    if package.is_relative_to(checkout):
        print(json.dumps({"status": "installation_inside_checkout",
                          "reason": "Physical isolation requires an installation outside the checkout"}))
        raise SystemExit(78)
    assert any(package.is_relative_to(Path(item).resolve()) for item in sys.path
               if item and Path(item).name in {"site-packages", "dist-packages"}), "Package must load from site-packages"

    forbidden = {"agent", "modules", "core", "scripts", "commerce_eval"}
    denied_events = []

    class DenyPrivateImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in forbidden:
                denied_events.append("private_import")
                raise AssertionError("Private dependency requested")

    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen"}:
            denied_events.append(event)
            raise AssertionError("Network/process operations are forbidden")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.name.lower().endswith(".env"):
                denied_events.append("credential_file")
                raise AssertionError("Credential file access is forbidden")
            if path.resolve().is_relative_to(checkout):
                denied_events.append("checkout_read")
                raise AssertionError("Installed package must not read the checkout")

    sys.meta_path.insert(0, DenyPrivateImports())
    sys.addaudithook(audit)
    imported = importlib.import_module("agent_workbench")
    count = 0
    for item in pkgutil.walk_packages(imported.__path__, imported.__name__ + "."):
        if item.name.rsplit(".", 1)[-1] != "__main__":
            importlib.import_module(item.name)
            count += 1
    assert count > 0

    from agent_workbench.background_memory_review import review_run_for_memory_candidates
    from agent_workbench.company_profile import DEFAULT_PROFILES_ROOT
    from agent_workbench.knowledge_registry import retrieve_knowledge_for_langgraph
    from agent_workbench.local_knowledge_provider import default_knowledge_index_path
    from agent_workbench.memory_retriever import recall_memories_for_langgraph
    from agent_workbench.memory_store import default_memory_dir
    from agent_workbench.registry import get_registry
    from agent_workbench.session_checkpoint import default_session_dir
    from agent_workbench.session_ledger import default_ledger_db_path, retrieve_session_search
    from agent_workbench.skill_registry import prepare_company_skill_runtime
    from agent_workbench.skill_store import default_skill_governance_db

    expected = home / ".agent-workbench"
    assert Path.home().resolve() == home
    for path in (DEFAULT_PROFILES_ROOT, default_memory_dir(), default_session_dir(),
                 default_ledger_db_path(), default_skill_governance_db(), default_knowledge_index_path("fixture")):
        assert path.resolve().is_relative_to(expected)
    assert prepare_company_skill_runtime(None)["company_profile"]["enabled"] is False
    assert recall_memories_for_langgraph(user_request="fixture").summary["enabled"] is False
    assert review_run_for_memory_candidates({})["enabled"] is False
    assert retrieve_session_search("fixture").enabled is False
    assert retrieve_knowledge_for_langgraph("fixture")["enabled"] is False
    try:
        get_registry()
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("Default registry must require an explicit host")
    assert not expected.exists(), "Default-disabled extensions must not create storage"
    assert not forbidden.intersection(name.split(".", 1)[0] for name in sys.modules)
    for directory in ("company_profiles", "skills", "data", "knowledge", "memory"):
        assert not (package / directory).exists()
    assert not denied_events, denied_events
    print(json.dumps({"status": "passed", "isolated": True, "module_count": count}))
''')


def _probe_installed_package(tmp_path):
    interpreter = os.environ.get("AGENT_WORKBENCH_BOUNDARY_PYTHON") or sys.executable
    required = bool(os.environ.get("AGENT_WORKBENCH_BOUNDARY_PYTHON")) or os.environ.get("AGENT_WORKBENCH_REQUIRE_INSTALLED") == "1"
    home = tmp_path / "isolated-home"
    home.mkdir()
    # Inherit only OS essentials, never provider credentials or Python path overrides.
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "SYSTEMDRIVE") if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), TEMP=str(tmp_path), TMP=str(tmp_path),
               AGENT_WORKBENCH_DISABLE_CENTRAL_ENV="1")
    result = subprocess.run(
        [interpreter, "-I", "-B", "-c", _INSTALLED_PROBE, str(PROJECT_ROOT), str(home)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90, check=False,
    )
    if result.returncode == 77 and not required:
        pytest.skip("No installed wheel in the isolated interpreter; parent owns installation")
    if result.returncode == 78 and not required:
        pytest.skip("Installation is inside the checkout; physical isolation requires an external wheel-installed interpreter")
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["status"] == "passed" and payload["isolated"] is True


def test_installed_package_boundary_in_isolated_interpreter(tmp_path):
    _probe_installed_package(tmp_path)


@pytest.mark.parametrize("mode", ["default", "explicit", "required"])
@pytest.mark.parametrize("returncode", [77, 78, 1])
def test_installed_probe_skip_is_limited_to_optional_unavailable_isolation(tmp_path, monkeypatch, mode, returncode):
    monkeypatch.delenv("AGENT_WORKBENCH_BOUNDARY_PYTHON", raising=False)
    monkeypatch.delenv("AGENT_WORKBENCH_REQUIRE_INSTALLED", raising=False)
    if mode == "explicit":
        monkeypatch.setenv("AGENT_WORKBENCH_BOUNDARY_PYTHON", sys.executable)
    elif mode == "required":
        monkeypatch.setenv("AGENT_WORKBENCH_REQUIRE_INSTALLED", "1")
    result = subprocess.CompletedProcess(args=[], returncode=returncode, stdout="fixture probe result", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)
    if mode == "default" and returncode in {77, 78}:
        reason = "No installed wheel" if returncode == 77 else "Installation is inside the checkout"
        with pytest.raises(pytest.skip.Exception, match=reason):
            _probe_installed_package(tmp_path)
    else:
        with pytest.raises(AssertionError, match="fixture probe result"):
            _probe_installed_package(tmp_path)
