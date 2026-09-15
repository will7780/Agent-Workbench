# Security

This alpha is for local, single-user operation. Bind the web application only to loopback. Do not expose it through a public proxy or use it as a multi-tenant service.

Python plugins execute with the host process permissions. They are not sandboxed. Install only reviewed plugins; a registry contract does not make arbitrary Python safe. Web requests cannot install plugins or launch commands.

Use the existing central environment file through `AGENT_API_ENV_FILE`; explicit process variables override it. The Windows fallback is the current user's `Desktop/api/.env`, not a path to a particular developer's account. Never commit keys or put them in project data. This application does not provide a new web secret store.

Default examples and tests are offline. Real model access must be explicitly enabled by the host. Missing semantic parameter verification requests user confirmation; missing required artifact validation blocks execution.

Review approves the frozen artifact, rules, parameters and operation scope. Modifications require a fresh check and approval. Checkpoint handles for paused runs are in-process only; a restart preserves historical reports but cannot resume those pauses. Cancellation is cooperative between graph nodes and does not undo an adapter already in progress.

Trace export is diagnostic evidence from the host, not an independent attestation of business side effects. Hosts remain responsible for field-sensitive redaction, access control, rule correctness and trustworthy adapters.

Do not include secrets, private data or exploit payloads in public bug reports. Use the repository's private vulnerability reporting facility when enabled; otherwise first report only a non-sensitive description to the maintainer.
