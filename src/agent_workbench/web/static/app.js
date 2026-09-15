"use strict";

const $ = (id) => document.getElementById(id);
const page = document.body.dataset.page;
const query = new URLSearchParams(location.search);
const state = {
  runId: query.get("run_id"), operationId: query.get("operation_id"),
  threadId: crypto.randomUUID(), run: null, runs: [], meta: {}, busy: false,
  generation: 0, eventIndex: 0, pendingKey: "", signatures: new Map(),
};
const terminal = new Set(["completed", "failed", "cancelled", "canceled", "rejected", "error", "stopped"]);
const labels = {runtime_operation_failed: "Runtime operation failed. Review the run before retrying.",
  same_origin_required: "This request must come from this local page.",
  interaction_mismatch: "This interaction is no longer current. Refreshing the run.",
  operation_in_progress: "An operation is already in progress.",
  artifact_has_errors: "The runtime has not allowed approval of this artifact.",
  sensitive_input_rejected: "Sensitive input was rejected.",
  web_capacity_exceeded: "The local service is busy. Try again after an operation finishes.",
  run_not_found: "Run not found in this runtime session.",
  operation_not_found: "Operation history expired. Select the run to inspect its current state."};

function node(tag, className = "", text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function display(value) {
  return value === null || value === undefined ? "Unknown" : typeof value === "string" ? value : JSON.stringify(value, null, 2);
}
function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
async function api(path, body) {
  const options = {cache: "no-store", credentials: "same-origin", signal: AbortSignal.timeout(10000)};
  if (body !== undefined) Object.assign(options, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(labels[payload.error_type] || "The local service could not complete this request.");
  return payload;
}
function replaceWhenChanged(id, value, render) {
  const key = JSON.stringify(value);
  if (state.signatures.get(id) === key) return;
  state.signatures.set(id, key);
  const element = $(id);
  if (element) element.replaceChildren(...render(value));
}
function jsonBlock(value) { return node("pre", "", display(value)); }
function details(label, value) {
  const element = node("details", "tool-event");
  element.append(node("summary", "", label), jsonBlock(value));
  return element;
}
function locationState() {
  const params = new URLSearchParams();
  if (state.runId) params.set("run_id", state.runId);
  if (state.operationId) params.set("operation_id", state.operationId);
  history.replaceState(null, "", params.size ? `?${params}` : location.pathname);
  updatePeer();
}
function updatePeer() {
  const link = $("peerLink");
  if (!Number.isInteger(state.meta.peer_port)) return;
  const url = new URL(location.href);
  url.port = state.meta.peer_port;
  url.search = "";
  url.hash = "";
  if (state.runId) url.searchParams.set("run_id", state.runId);
  link.href = url.href;
  link.hidden = false;
}
function renderList() {
  const filter = $("runFilter").value.toLowerCase();
  const summaries = state.runs.map((run) => [run.run_id, run.user_request, run.status]);
  replaceWhenChanged("runList", [summaries, state.runId, filter], () => {
    const runs = state.runs.filter((run) => `${run.user_request || ""} ${run.run_id || ""}`.toLowerCase().includes(filter));
    if (!runs.length) return [node("p", "muted", "No runs")];
    return runs.map((run) => {
      const button = node("button", `run-item${run.run_id === state.runId ? " active" : ""}`);
      button.type = "button";
      button.setAttribute("aria-current", run.run_id === state.runId ? "true" : "false");
      button.append(node("strong", "", run.user_request || run.run_id || "Run"), node("small", "", run.status || "Unknown"));
      button.addEventListener("click", () => selectRun(run.run_id));
      return button;
    });
  });
}
function selectRun(runId) {
  state.generation++;
  state.runId = runId;
  state.operationId = null;
  state.busy = false;
  state.eventIndex = 0;
  state.pendingKey = "";
  state.signatures.clear();
  const run = state.runs.find((item) => item.run_id === runId) || null;
  if (run?.thread_id) state.threadId = run.thread_id;
  showError("");
  locationState();
  renderRun(run);
  renderList();
}
function messageNodes(messages) {
  if (!Array.isArray(messages) || !messages.length) return [node("p", "muted", "No recorded messages")];
  return messages.map((message) => {
    const role = message.role || "unknown";
    const known = ["user", "assistant", "tool", "system", "developer"].includes(role);
    const article = node("article", `message ${known ? role : "unknown"}`);
    article.append(node("div", "message-role", `${role}${message.name ? ` / ${message.name}` : ""}`), node("div", "message-content", display(message.content)));
    if (message.tool_calls?.length) article.append(details("Tool calls", message.tool_calls));
    if (message.tool_call_id) article.append(node("p", "muted", `Tool call: ${message.tool_call_id}`));
    return article;
  });
}
function sampleTable(sample) {
  if (!Array.isArray(sample) || !sample.length) return node("p", "muted", "No sample provided");
  const rows = sample.filter((row) => row && typeof row === "object");
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const scroll = node("div", "table-scroll");
  scroll.tabIndex = 0;
  scroll.setAttribute("role", "region");
  scroll.setAttribute("aria-label", "Artifact sample");
  const table = node("table");
  table.append(node("caption", "", "Runtime-provided sample"));
  const head = node("thead"), header = node("tr");
  columns.forEach((column) => { const cell = node("th", "", column); cell.scope = "col"; header.append(cell); });
  head.append(header);
  const body = node("tbody");
  rows.forEach((row) => { const tr = node("tr"); columns.forEach((column) => tr.append(node("td", "", display(row[column])))); body.append(tr); });
  table.append(head, body);
  scroll.append(table);
  return scroll;
}
async function submitOperation(path, payload, {clearComposer = false} = {}) {
  if (state.busy) return;
  const generation = state.generation;
  state.busy = true;
  showError("");
  updateControls();
  try {
    const operation = await api(path, payload);
    if (generation !== state.generation) return;
    state.operationId = operation.operation_id;
    if (operation.run_id) state.runId = operation.run_id;
    if (clearComposer && $("messageInput")) $("messageInput").value = "";
    locationState();
  } catch (error) {
    if (generation === state.generation) showError(error.message || "Request failed. Check the current run before retrying.");
  } finally {
    if (generation === state.generation) { state.busy = false; updateControls(); }
  }
}
function renderInteraction(pending) {
  if (!$("interaction")) return;
  const key = JSON.stringify([state.runId, pending]);
  if (key === state.pendingKey) return;
  state.pendingKey = key;
  $("interaction").replaceChildren();
  if (!pending) return;
  const panel = node("section", "interaction-panel");
  panel.dataset.interactionId = pending.interaction_id;
  const titles = {clarification: "Clarification", artifact_review: "Artifact Review", confirmation: ({parameter: "Parameter Review", risk: "Risk Review", combined: "Parameter & Risk Review"}[pending.confirmation_kind] || "Confirmation")};
  panel.append(node("h2", "", titles[pending.type] || "Pending interaction"));
  if (pending.question) panel.append(node("p", "", pending.question));
  if (pending.reason) panel.append(node("p", "", pending.reason));
  const runId = state.runId;
  const resume = (payload) => submitOperation(`/api/runs/${encodeURIComponent(runId)}/resume`, {interaction_id: pending.interaction_id, response: {type: pending.type, ...payload}});
  if (pending.type === "clarification") {
    if (pending.missing_fields?.length) panel.append(node("p", "muted", `Required: ${pending.missing_fields.join(", ")}`));
    const form = node("form");
    const label = node("label", "", "Answer"); label.htmlFor = "clarificationAnswer";
    const answer = node("textarea"); answer.id = "clarificationAnswer"; answer.maxLength = 4000; answer.required = true;
    const controls = node("div", "interaction-actions"), submit = node("button", "primary", "Submit answer"); submit.type = "submit";
    controls.append(submit); form.append(label, answer, controls);
    form.addEventListener("submit", (event) => { event.preventDefault(); if (answer.value.trim()) resume({answer: answer.value.trim()}); });
    panel.append(form);
  } else if (["confirmation", "artifact_review"].includes(pending.type)) {
    for (const action of pending.actions || []) {
      const target = node("div", "confirmation-target");
      target.append(node("h3", "", action.capability_id || action.tool_name || "Tool"), node("p", "muted", `Risk: ${display(action.risk_level)} | Mode: ${display(action.execution_mode)}`), jsonBlock(action.params_summary ?? action.parameters));
      if (action.parameter_snapshot_hash) target.append(node("pre", "", `Parameter snapshot: ${action.parameter_snapshot_hash}`));
      panel.append(target);
    }
    let comment;
    if (pending.type === "artifact_review") {
      const evidence = pending.evidence || {};
      panel.append(node("p", "", `Artifact: ${display(evidence.artifact_id)} | Revision: ${display(evidence.version)} | Rows: ${display(evidence.rows)}`));
      panel.append(sampleTable(pending.sample), details("Artifact proof", evidence));
      for (const error of evidence.errors || []) panel.append(node("p", "danger", display(error)));
      if (pending.can_approve !== true) panel.append(node("p", "danger", "Approval unavailable"));
      const label = node("label", "", "Review comment"); label.htmlFor = "reviewComment";
      comment = node("textarea"); comment.id = "reviewComment"; comment.maxLength = 4000;
      panel.append(label, comment);
    }
    const actions = node("div", "interaction-actions");
    const choices = [["approve", "Approve"], ["reject", "Reject"]];
    if (pending.type === "artifact_review") choices.push(["request_changes", "Request changes"]);
    choices.forEach(([decision, label]) => {
      const button = node("button", decision === "approve" ? "primary" : decision === "reject" ? "danger" : "", label);
      button.type = "button";
      button.dataset.blocked = String(decision === "approve" && pending.type === "artifact_review" && pending.can_approve !== true);
      button.addEventListener("click", () => resume({decision, ...(comment ? {comment: comment.value} : {})}));
      actions.append(button);
    });
    panel.append(actions);
  } else panel.append(node("p", "muted", "This interaction requires the host application."));
  $("interaction").append(panel);
}
function updateControls() {
  const run = state.run;
  const operations = run?.operations || [];
  const inFlight = state.busy || Boolean(state.operationId) || operations.length > 0;
  const pending = Boolean(run?.pending_interaction);
  const running = run && !terminal.has(run.status) && !pending;
  $("cancelRun").disabled = !run?.run_id || terminal.has(run?.status) || state.busy || operations.some((job) => job.kind === "cancel");
  if ($("messageInput")) {
    $("messageInput").disabled = inFlight || pending || Boolean(running);
    $("sendButton").disabled = $("messageInput").disabled || !$("messageInput").value.trim();
  }
  $("interaction")?.querySelectorAll("button, textarea").forEach((control) => { control.disabled = inFlight || control.dataset.blocked === "true"; });
  $("runtimeStatus").textContent = state.busy || state.operationId ? "Operation in progress" : run?.status || "Ready";
}
function renderTimeline(events) {
  replaceWhenChanged("timeline", events, (values) => {
    if (!values?.length) { $("eventDetail").replaceChildren(node("p", "muted", "No event selected")); return [node("p", "muted", "No recorded events")]; }
    state.eventIndex = Math.min(state.eventIndex, values.length - 1);
    return values.map((event, index) => {
      const button = node("button", "", `${index + 1}. ${event.type || "event"} / ${event.node || event.phase || event.status || ""}`);
      button.type = "button";
      button.setAttribute("aria-pressed", String(index === state.eventIndex));
      button.addEventListener("click", () => {
        state.eventIndex = index;
        $("timeline").querySelectorAll("button").forEach((item, i) => item.setAttribute("aria-pressed", String(i === index)));
        $("eventDetail").replaceChildren(jsonBlock(event));
      });
      if (index === state.eventIndex) $("eventDetail").replaceChildren(jsonBlock(event));
      return button;
    });
  });
}
function renderRun(run) {
  state.run = run;
  const offline = run?.offline ?? state.meta.offline;
  $("environmentLabel").textContent = offline === true ? "Offline / simulated model" : offline === false ? "Connected model" : "Model mode unknown";
  const model = run?.model_label ?? state.meta.model_label;
  if (model) $("environmentLabel").textContent += ` / ${model}`;
  $("modeLabel").textContent = `Execution: ${run?.execution_mode || "unknown"}`;
  $("emptyState").hidden = Boolean(run);
  if ($("diagnosticRun")) $("diagnosticRun").hidden = !run;
  const messages = run?.messages || [];
  replaceWhenChanged("messages", [messages, run?.user_request, Boolean(run)], ([values]) => {
    if (!run) return [];
    if (page === "chat" && !values.length && run.user_request) {
      const request = node("article", "message user");
      request.append(node("div", "message-role", "Request"), node("div", "message-content", run.user_request));
      return [request];
    }
    return messageNodes(values);
  });
  replaceWhenChanged("finalResponse", [run?.final_response, messages], ([value]) => {
    if (!value || messages.some((message) => message.role === "assistant" && message.content === value)) return [];
    const section = node("div", "message"); section.append(node("h3", "", "Run response"), node("div", "message-content", display(value))); return [section];
  });
  if (page === "chat") {
    const events = (run?.events || []).filter((event) => /tool|artifact/.test(event.type || ""));
    replaceWhenChanged("tools", [run?.tools, events], () => {
      const items = [...(Array.isArray(run?.tools) ? run.tools : []), ...events];
      return items.map((item) => details(`${item.tool_name || item.tool_call?.tool_name || item.type || "Tool"}${item.status ? ` / ${item.status}` : ""}`, item));
    });
    renderInteraction(run?.pending_interaction);
  } else if (run) {
    $("runIdentity").textContent = `Run: ${run.run_id} | Conversation: ${run.thread_id || "unknown"}`;
    renderTimeline(run.events);
    replaceWhenChanged("parameters", [run.parameter_sources, run.parameter_verification, run.pending_interaction?.actions], () => [details("Sources", run.parameter_sources), details("Verification", run.parameter_verification), details("Pending actions", run.pending_interaction?.actions)]);
    for (const field of ["observations", "context"]) replaceWhenChanged(field, run[field], (value) => [jsonBlock(value)]);
    replaceWhenChanged("artifacts", [run.artifacts, run.pending_interaction], () => {
      const elements = [jsonBlock(run.artifacts)];
      if (run.pending_interaction?.type === "artifact_review") elements.push(sampleTable(run.pending_interaction.sample), details("Pending review proof", run.pending_interaction.evidence));
      return elements;
    });
    replaceWhenChanged("resources", run.metrics, (metrics) => {
      const grid = node("dl", "metric-grid");
      for (const [key, label] of [["prompt_tokens", "Model input tokens"], ["completion_tokens", "Model output tokens"], ["total_tokens", "Model total tokens"], ["latency_ms", "Model latency (ms)"], ["tool_latency_ms", "Tool latency (ms)"], ["active_runtime_ms", "Active runtime (ms)"], ["wall_runtime_ms", "Wall runtime (ms)"], ["user_wait_ms", "User wait (ms)"]]) {
        const item = node("div"); item.append(node("dt", "", label), node("dd", "", display(metrics?.[key]))); grid.append(item);
      }
      return [grid];
    });
  }
  if (run?.error && (!Array.isArray(run.error) || run.error.length)) showError(typeof run.error === "string" && labels[run.error] ? labels[run.error] : display(run.error));
  updateControls();
  updatePeer();
}
async function poll() {
  const generation = state.generation;
  try {
    if (!state.meta.page) { state.meta = await api("/api/meta"); updatePeer(); }
    const listing = await api("/api/runs");
    if (generation !== state.generation) return;
    state.runs = listing.runs || [];
    $("historyNote").textContent = listing.truncated ? "Recent runs only" : "";
    if (state.operationId) {
      let operation;
      try { operation = await api(`/api/operations/${encodeURIComponent(state.operationId)}`); }
      catch (error) { if (generation === state.generation) { state.operationId = null; locationState(); } throw error; }
      if (generation !== state.generation) return;
      if (operation.run_id) state.runId = operation.run_id;
      if (operation.status !== "running") {
        state.operationId = null;
        if (operation.status === "failed") showError(labels[operation.error_type] || "Runtime operation failed.");
      }
      locationState();
    }
    let run = state.runId ? await api(`/api/runs/${encodeURIComponent(state.runId)}`) : null;
    if (generation !== state.generation) return;
    if (run?.thread_id) state.threadId = run.thread_id;
    renderRun(run);
    renderList();
  } catch (error) {
    if (generation === state.generation) {
      showError(error.message || "Local service unavailable.");
      $("runtimeStatus").textContent = "Connection or request error";
    }
  } finally { window.setTimeout(poll, document.hidden ? 2500 : 800); }
}

$("runFilter").addEventListener("input", renderList);
$("messageInput")?.addEventListener("input", updateControls);
$("newChat")?.addEventListener("click", () => {
  selectRun(null); state.threadId = crypto.randomUUID(); $("messageInput").value = ""; updateControls(); $("messageInput").focus();
});
$("composer")?.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("messageInput").value.trim();
  if (text && !$("sendButton").disabled) submitOperation("/api/runs", {user_request: text, thread_id: state.threadId}, {clearComposer: true});
});
$("cancelRun").addEventListener("click", () => {
  if (state.runId && !$("cancelRun").disabled) submitOperation(`/api/runs/${encodeURIComponent(state.runId)}/cancel`, {});
});
poll();
