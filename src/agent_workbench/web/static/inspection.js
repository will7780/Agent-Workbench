"use strict";

window.WorkbenchInspection = (() => {
  const el = (tag, text, className = "") => {
    const item = document.createElement(tag);
    if (text !== undefined) item.textContent = text;
    item.className = className;
    return item;
  };
  const print = (value) => el("pre", value == null ? "\u672a\u91c7\u96c6" : JSON.stringify(value, null, 2));
  const words = {
    completed: "\u8282\u70b9\u7ed3\u675f", running: "\u8fd0\u884c\u4e2d", success: "\u5de5\u5177\u8fd4\u56de\u6210\u529f",
    failed: "\u5931\u8d25", error: "\u9519\u8bef", blocked: "\u5df2\u963b\u65ad", recorded: "\u5df2\u8bb0\u5f55",
    waiting: "\u7b49\u5f85\u7528\u6237", completion_unrecorded: "\u7ed3\u675f\u72b6\u6001\u672a\u91c7\u96c6",
    clarification: "\u8ffd\u95ee", confirmation: "\u786e\u8ba4", artifact_review: "\u4ea7\u7269\u5ba1\u6838",
    model_decide_node: "\u6a21\u578b\u51b3\u7b56", tool_guard_node: "\u6743\u9650\u95e8\u7981",
    parameter_verify_node: "\u53c2\u6570\u6821\u9a8c", artifact_check_node: "\u4ea7\u7269\u68c0\u67e5",
    artifact_review_node: "\u4ea7\u7269\u5ba1\u6838", human_confirm_node: "\u4eba\u5de5\u786e\u8ba4",
    artifact_bind_node: "\u4ea7\u7269\u7248\u672c\u7ed1\u5b9a", tool_execute_node: "\u6267\u884c\u5de5\u5177",
    observe_node: "\u5904\u7406\u5de5\u5177\u53cd\u9988", context_prepare_node: "\u51c6\u5907\u4e0a\u4e0b\u6587",
    context_compact_node: "\u6574\u7406\u4e0a\u4e0b\u6587", finalize_node: "\u8f93\u51fa\u7ed3\u679c",
  };
  const dialog = el("dialog", undefined, "inspection-dialog");
  dialog.id = "inspectionDialog";
  dialog.setAttribute("aria-labelledby", "inspectionTitle");
  const header = el("header", undefined, "inspection-header");
  const title = el("h2"); title.id = "inspectionTitle";
  const close = el("button", "\u00d7", "close-inspection");
  close.type = "button"; close.title = "\u5173\u95ed"; close.setAttribute("aria-label", "\u5173\u95ed");
  close.addEventListener("click", () => dialog.close());
  header.append(title, close);
  const content = el("div", undefined, "inspection-content");
  dialog.append(header, content);
  document.body.append(dialog);
  let currentRun = null, view = null, generation = 0, selection = null, follow = true, flowSignature = "";
  let flowList, flowDetail, flowNote, followBox;
  const catalogueButton = document.getElementById("openCatalogue");
  const flowButton = document.getElementById("openFlow");

  async function get(path) {
    const response = await fetch(path, {cache: "no-store", credentials: "same-origin", signal: AbortSignal.timeout(10000)});
    if (!response.ok) throw new Error("inspection_unavailable");
    return response.json();
  }
  function open(kind, text) {
    generation++;
    view = kind;
    title.textContent = text;
    content.replaceChildren();
    if (!dialog.open) dialog.showModal();
  }
  dialog.addEventListener("close", () => { generation++; view = null; });
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !event.isComposing) { event.preventDefault(); dialog.close(); }
  });

  async function showCatalogue() {
    open("catalogue", "\u5de5\u5177\u76ee\u5f55");
    const request = generation;
    content.append(el("p", "\u6b63\u5728\u8bfb\u53d6\u5df2\u6ce8\u518c\u5de5\u5177\u2026", "muted"));
    try {
      const catalog = await get("/api/catalogue");
      if (request !== generation) return;
      content.replaceChildren();
      if (!catalog.available) {
        content.append(el("p", "\u5f53\u524d\u5bbf\u4e3b\u672a\u63d0\u4f9b\u5de5\u5177\u76ee\u5f55\u3002", "muted"));
        return;
      }
      const tools = Array.isArray(catalog.tools) ? catalog.tools : [];
      content.append(el("p", (catalog.total ?? "?") + " \u4e2a\u5de5\u5177 \u00b7 \u76ee\u5f55 " + (catalog.version ?? "?") + " \u00b7 \u6ce8\u518c\u58f0\u660e\uff0c\u975e\u5f53\u524d\u4efb\u52a1\u6388\u6743", "inspection-note"));
      if (catalog.truncated) content.append(el("p", "\u76ee\u5f55\u8fc7\u957f\uff0c\u4ec5\u5c55\u793a\u524d 200 \u9879\u3002", "muted"));
      const layout = el("div", undefined, "inspection-grid");
      const left = el("div", undefined, "catalogue-navigation");
      const search = el("input"); search.type = "search"; search.maxLength = 120;
      search.placeholder = "\u641c\u7d22\u5de5\u5177"; search.setAttribute("aria-label", "\u641c\u7d22\u5de5\u5177");
      const list = el("nav", undefined, "catalogue-list"); list.setAttribute("aria-label", "\u5de5\u5177\u5217\u8868");
      const detail = el("section", undefined, "inspection-detail"); detail.id = "catalogueDetail";
      left.append(search, list); layout.append(left, detail); content.append(layout);
      let chosen = null, detailRequest = 0;
      async function selectTool(tool) {
        chosen = tool.name;
        list.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.name === chosen)));
        const seq = ++detailRequest;
        detail.replaceChildren(el("h3", tool.label), el("code", tool.name), el("p", tool.description || "", "muted"),
          el("p", "Risk: " + (tool.risk ?? "Unknown") + " \u00b7 " + tool.group),
          el("h3", "\u5404\u6a21\u5f0f\u58f0\u660e"), print(tool.declared_modes),
          el("h3", "\u53c2\u6570\u5408\u540c"));
        const schema = el("div", "\u52a0\u8f7d\u4e2d\u2026"); detail.append(schema);
        try {
          const record = await get("/api/catalogue/" + encodeURIComponent(tool.name));
          if (request === generation && seq === detailRequest) schema.replaceChildren(print(record.parameter_contract));
        } catch {
          if (request === generation && seq === detailRequest) schema.textContent = "\u5408\u540c\u8bfb\u53d6\u5931\u8d25";
        }
      }
      function filter() {
        const text = search.value.trim().toLocaleLowerCase();
        const matches = tools.filter(t => [t.name, t.label, t.group, t.description].join(" ").toLocaleLowerCase().includes(text));
        list.replaceChildren(...matches.map(tool => {
          const button = el("button", undefined, "catalogue-item"); button.type = "button";
          button.dataset.name = tool.name; button.setAttribute("aria-pressed", String(chosen === tool.name));
          button.append(el("strong", tool.label), el("small", tool.name), el("small", tool.group + " \u00b7 " + (tool.risk ?? "Unknown")));
          button.addEventListener("click", () => selectTool(tool));
          return button;
        }));
        if (!matches.length) {
          list.append(el("p", "\u6ca1\u6709\u5339\u914d\u5de5\u5177", "muted"));
          chosen = null; detailRequest++; detail.replaceChildren();
        } else if (!matches.some(t => t.name === chosen)) selectTool(matches[0]);
      }
      search.addEventListener("input", filter);
      filter(); search.focus();
    } catch {
      if (request === generation) content.replaceChildren(el("p", "\u5de5\u5177\u76ee\u5f55\u8bfb\u53d6\u5931\u8d25\uff0c\u8bf7\u91cd\u65b0\u6253\u5f00\u3002", "danger"));
    }
  }
  function selectFlow(item) {
    selection = item.id;
    flowList.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.nodeId === selection)));
    flowDetail.replaceChildren(el("h3", words[item.label] || item.label), el("code", item.label),
      el("p", words[item.status] || item.status, "muted"));
    if (item.id === "pending") flowDetail.append(print(currentRun?.pending_interaction));
    for (const index of item.event_refs || []) {
      flowDetail.append(el("h3", "Event " + (index + 1)), print(currentRun?.events?.[index]));
    }
  }
  function renderFlow() {
    if (!dialog.open || view !== "flow") return;
    const flow = currentRun?.flow;
    const items = Array.isArray(flow?.nodes) ? flow.nodes : [];
    const signature = JSON.stringify([currentRun?.run_id, currentRun?.status, flow, currentRun?.events, currentRun?.pending_interaction]);
    if (signature === flowSignature) return;
    flowSignature = signature;
    flowNote.textContent = [
      currentRun ? "Run: " + currentRun.run_id + " \u00b7 " + currentRun.status : "\u672a\u9009\u62e9\u8fd0\u884c",
      "\u5b9e\u9645\u89c2\u6d4b\u987a\u5e8f \u00b7 \u8282\u70b9\u7ed3\u675f\u4e0d\u4ee3\u8868\u4e1a\u52a1\u6210\u529f",
      flow?.history_may_be_truncated ? "\u4ec5\u663e\u793a\u6709\u9650\u5386\u53f2\uff0c\u65e9\u671f\u4e8b\u4ef6\u53ef\u80fd\u7f3a\u5931" : "",
    ].filter(Boolean).join(" | ");
    const selected = follow ? items.at(-1) : items.find(n => n.id === selection) || items.at(-1);
    selection = selected?.id;
    flowList.replaceChildren(...items.map((item, i) => {
      const li = el("li", undefined, "flow-step");
      const button = el("button", undefined, "flow-node");
      button.type = "button"; button.dataset.nodeId = item.id;
      button.setAttribute("aria-pressed", String(selection === item.id));
      button.dataset.tone = ["failed", "error", "blocked"].includes(item.status) ? "bad"
        : item.status === "waiting" ? "waiting" : item.status === "running" ? "active" : "neutral";
      button.append(el("span", (i + 1) + ". " + (words[item.label] || item.label), "flow-title"),
        el("small", (words[item.status] || item.status) + (item.round == null ? "" : " \u00b7 Round " + item.round)));
      button.addEventListener("click", () => { follow = false; followBox.checked = false; selectFlow(item); });
      li.append(button); return li;
    }));
    if (selected) {
      selectFlow(selected);
      if (follow) flowList.lastElementChild?.scrollIntoView({block: "nearest"});
    } else {
      flowList.append(el("li", "\u672a\u91c7\u96c6\u5230\u6d41\u7a0b\u4e8b\u4ef6", "muted"));
      flowDetail.replaceChildren(el("p", "\u6ca1\u6709\u53ef\u6838\u5bf9\u7684\u6d41\u7a0b\u8bc1\u636e\u3002", "muted"));
    }
    if (flow && !flow.has_event_evidence && selected) flowDetail.append(el("p", "\u4ec5\u6709\u6682\u505c\u72b6\u6001\uff0c\u6d41\u7a0b\u4e8b\u4ef6\u672a\u91c7\u96c6\u3002", "muted"));
  }
  function showFlow() {
    open("flow", "\u8fd0\u884c\u6d41\u7a0b");
    flowSignature = ""; selection = null; follow = true;
    flowNote = el("p", undefined, "inspection-note");
    const controls = el("label", undefined, "follow-flow");
    followBox = el("input"); followBox.type = "checkbox"; followBox.checked = true;
    followBox.addEventListener("change", () => { follow = followBox.checked; flowSignature = ""; renderFlow(); });
    controls.append(followBox, el("span", "\u8ddf\u968f\u6700\u65b0\u8282\u70b9"));
    const grid = el("div", undefined, "inspection-grid");
    flowList = el("ol", undefined, "flow-list"); flowList.setAttribute("aria-label", "\u8fd0\u884c\u8282\u70b9");
    flowDetail = el("section", undefined, "inspection-detail"); flowDetail.id = "flowDetail";
    grid.append(flowList, flowDetail); content.append(flowNote, controls, grid);
    renderFlow();
  }
  catalogueButton?.addEventListener("click", showCatalogue);
  flowButton?.addEventListener("click", showFlow);
  return {update(run) {
    if (run?.run_id !== currentRun?.run_id) { selection = null; flowSignature = ""; }
    currentRun = run;
    if (flowButton) flowButton.disabled = !run;
    renderFlow();
  }};
})();
