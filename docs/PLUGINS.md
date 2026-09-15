# Plugin Boundary

An explicitly installed Python plugin provides a factory returning `AgentRuntime`.
All entrypoints use this runtime: Python calls, CLI, chat and diagnostics.

```python
from agent_workbench.runtime import AgentRuntime, RuntimeServices
from agent_workbench.configuration import StaticConfigProvider
from agent_workbench.registry import ToolRegistry
from agent_workbench.tools.execution_adapters import RegisteredToolExecutor
from agent_workbench.tools.adapter_contracts import AdapterOutput

def create_runtime(*, data_dir):
    executor = RegisteredToolExecutor()
    executor.register("notes.read", lambda request: AdapterOutput(
        status="success", summary="A synthetic note was read."), modes=("dry_run",))
    registry = ToolRegistry(payload={"version": "1", "modules": {
        "notes": {"actions": {"read": {
            "risk_level": "L1", "execution_adapter": "notes.read",
            "dry_run_status": "available",
            "parameter_contract": {"type": "object", "properties": {},
                                   "additionalProperties": False}
        }}}
    }})
    return AgentRuntime(RuntimeServices(
        registry=registry, executor=executor,
        config_provider=StaticConfigProvider({"execution_mode": "dry_run"}),
    ), data_dir=data_dir)
```

This minimal factory intentionally has no model. Starting a run returns
`model_not_configured` until the host supplies `model_client` (the existing
`ToolsCompleteFn` signature) or explicitly enables a configured real model.
It does not fabricate a successful run.

After installing your plugin package, invoke its factory explicitly:

```bash
agent-workbench serve --plugin my_plugin:create_runtime
agent-workbench run --plugin my_plugin:create_runtime --message "Read a note"
```

`--real-model` allows the configured compatible client and may incur charges.
The host's Python code is trusted and can perform arbitrary actions; command
flags and registry gates are not a Python security sandbox.

## Responsibilities

| Host supplies | Engine supplies |
| --- | --- |
| Actual tools, JSON Schema and explicit execution modes | Registered-tool routing and deterministic parameter validation |
| Mandatory business policy and optional semantic parameter reviewer | Frozen parameters, uncertain-result confirmation and fail-closed execution |
| `ArtifactValidator.validate(snapshot, policy)` | Full file snapshot, manifest, sample, immutable approval and consumption binding |
| Model client and central configuration reference | Ordered roles, context management, tool feedback and resource events |
| Profile/Skill/knowledge sources and authorization | Scoped loading, governance mechanisms and unavailable-source diagnostics |

`ArtifactValidator` receives actual parsed rows and file metadata. It returns
`complete`, `valid`, `errors`, `row_count` and `checked_rules`. Required rules
must be covered; absence, partial inspection or unsupported formats cannot pass.
The validator does not approve or bind artifacts. Use the anonymous
[demo plugin](../src/agent_workbench/demo.py) as a complete runnable example.

Execution bindings also require the engine's current `run_id`, `step_id` and
execution mode. Adapter wrappers must preserve that independent request context;
never reconstruct it from an old approval, signed binding or model argument.
An unchanged file does not make a prior task's approval valid for a new task.

`ConfigProvider.snapshot()` supplies `module_config`, `parameter_sources`,
execution mode, model, and optional `company_profile_config`, `knowledge_config`
and `memory_dir`. Configuration is host-owned, not accepted from a web run body.
Company-named compatibility fields represent generic organization/Profile scope;
no company catalog or policy content is distributed.

## Optional Extensions

Memory/Skill governance and local knowledge storage are implemented with local
files/SQLite. They remain disabled until configured. HTTP RAG and OpenSearch
connectors require explicit endpoints, schemas and permissions. MCP knowledge
access requires an injected client; it is not an automatic MCP transport
discovery system. Embedding interfaces and offline fixtures do not constitute a
bundled production embedding provider. Connector support is based on local
contract tests, not claims of live deployment certification.

## Trace Export

```bash
agent-workbench export RUN_ID --output trace.json
```

Export uses contract `1.2`, preserving captured roles and tool names, and marks
missing/truncated input explicitly. Local paths and protected binding material
are removed. This is a host diagnostic record, not trusted evidence that no
external side effects occurred. The Eval platform is a separate optional
consumer, never a dependency or automatic upload destination.
