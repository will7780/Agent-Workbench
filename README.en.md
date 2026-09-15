# Agent Workbench

A local Agent workbench built on LangGraph ReAct: tool execution, parameter
validation, human confirmation, artifact version binding and diagnostic traces.

**One generic runtime, with tools and rules explicitly supplied by the host.**

[简体中文](README.md) · [Plugin guide](docs/PLUGINS.md) · [Security](SECURITY.md)

> `0.1.0a1` · MIT · Local alpha. The offline demo uses no real model and contains
> no company business tools. This project is not published on PyPI.

## Try It Locally

Python 3.10 or later is required. Create a separate virtual environment:

```bash
git clone https://github.com/will7780/Agent-Workbench.git
cd Agent-Workbench
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv/Scripts/python.exe -m pip install .
.venv/Scripts/agent-workbench.exe demo
```

macOS / Linux:

```bash
.venv/bin/python -m pip install .
.venv/bin/agent-workbench demo
```

Open [chat](http://127.0.0.1:8786) and
[diagnostics](http://127.0.0.1:8785). Both pages share one process and runtime.
Use `--chat-port` and `--diagnostics-port` when those ports are occupied.

Ask: "Read the synthetic materials, draft a report, and ask me to review before
archiving." The example creates an actual temporary report and pauses for
review. Approve, reject, or request changes and review the new version. Inspect
the same run's inputs, tool observations and artifact events in diagnostics.

This is a labeled offline reference policy, not a demonstration of real model
ability or production success rates. Files live in an isolated example
directory under `~/.agent-workbench/demo/`; override it with `--data-dir`.

## Runtime Capabilities

- LangGraph tool-calling loop, context management and actual input-role snapshots.
- JSON Schema validation, injectable parameter-intent review, permissions and confirmation.
- Artifact checks, human review, re-review after changes and version-bound consumption.
- Structured observations, events, latency and tokens; unknown usage stays unknown.
- Opt-in memory governance, Profile isolation, Skill loading and knowledge connectors.
- Optional Trace export without an Eval dependency or automatic upload.

## Connect Your Tools

A registry describes input schemas, risk and execution modes. Your installed
plugin supplies implementation, model/configuration and content-validation
services through `RuntimeServices`. All callers use
`AgentRuntime.start/resume/cancel`.

Start an installed factory with
`agent-workbench serve --plugin my_plugin:create_runtime`.
Real model use requires explicit activation. Credentials come from the central
configuration referenced by `AGENT_API_ENV_FILE` or explicit process variables;
the web application does not install plugins, accept code uploads or manage keys.

See the [plugin guide](docs/PLUGINS.md) for contracts, the complete anonymous
example and connector limitations. JSON/CSV artifact support is included;
Excel parsing is optional: `python -m pip install '.[excel]'`.

## Separate From Eval

[E-commerce Eval](https://github.com/will7780/Ecommerce-eval) is a separate
business acceptance platform. This repository provides an Agent runtime.
Neither bundles the other. Export diagnostic Trace contract 1.2 with:

```bash
agent-workbench export RUN_ID --output trace.json
```

Successful execution is not proof of business correctness. Tool feedback,
artifact versions and permission checks are diagnostic evidence, not independent
proof that no external side effects occurred.

## Scope and Limitations

No private GUI, company implementations, business data, store connectors,
product templates or company rules are included. LangGraph supplies the graph
engine; this project does not claim to implement one from scratch.

Python plugins run with host permissions and are **not sandboxed**. This alpha
does not promise remote multi-user deployment or cross-process recovery of
paused runs. Historical reports survive a restart; old pending interactions
cannot resume. Cancellation is cooperative between nodes and cannot undo an
operation already performed.

Default tests are offline with temporary files. See
[CONTRIBUTING](CONTRIBUTING.md) and [SECURITY](SECURITY.md) before extending the
runtime or submitting an anonymous reproduction.
