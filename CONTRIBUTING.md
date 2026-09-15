# Contributing

This alpha focuses on reliable local execution and explicit plugin boundaries.
Keep private business implementations, real credentials and historical runs out
of issues and pull requests. Use small anonymous fixtures.

Install from source with `python -m pip install -e '.[dev,excel]'`, then run
`python -m pytest --confcutdir=. -q`. Default tests cannot make model requests.
Optional browser QA uses an independently installed Playwright/Chromium runtime;
`tools/browser_web_qa.py` provides the runnable UI scenarios.

The installed-package boundary probe needs an installation outside the checkout.
Set `AGENT_WORKBENCH_BOUNDARY_PYTHON` to that interpreter to require the isolated
probe. Optional Eval import interoperability tests use
`AGENT_WORKBENCH_EVAL_PYTHON`, pointing to a separately installed Eval environment.
Both use temporary data and never import into your actual project.

For safety or lifecycle changes, include invalid-input, rejection, repeated
resume, changed-artifact and missing-evidence tests as applicable. Do not weaken
a gate to improve demo pass rates. Keep diagnostic checks distinct from business
acceptance. Public code must never discover a neighboring private project.
