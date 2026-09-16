"""Synthetic loopback browser QA for catalogue/flow; never invoke a model."""
import json
import os
import runpy
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


def main():
    os.environ["AGENT_WORKBENCH_DISABLE_CENTRAL_ENV"] = "1"
    from agent_workbench.web import server
    root = Path(__file__).resolve().parents[1]
    server.STATIC_DIR = root / "src" / "agent_workbench" / "web" / "static"
    fixture = runpy.run_path(str(root / "tests" / "test_web.py"))
    helpers = runpy.run_path(str(root / "tests" / "test_web_inspection.py"))
    runtime = fixture["FakeRuntime"]()
    runtime.services.registry = helpers["fixture_registry"]()
    runtime.add("run_review", kind="artifact_review")
    events = [
        {"type": "graph_node", "node": "context_prepare_node", "phase": "started", "round_idx": 1, "status": "running"},
        {"type": "graph_node", "node": "context_prepare_node", "phase": "completed", "round_idx": 1},
        {"type": "graph_node", "node": "model_decide_node", "phase": "started", "round_idx": 1, "status": "running"},
        {"type": "model_call", "tool_call_names": ["notes__read"]},
        {"type": "graph_node", "node": "model_decide_node", "phase": "completed", "round_idx": 1},
        {"type": "tool_result", "tool_call": {"tool_name": "notes__read"}, "observation": {"status": "success", "summary": "Synthetic notes"}},
        {"type": "graph_node", "node": "artifact_review_node", "phase": "started", "round_idx": 1, "status": "running"},
    ]
    runtime.runs["run_review"]["state"]["agent_trace"] = events
    runtime.add("run_empty")
    servers = server.create_servers(runtime, chat_port=0, diagnostics_port=0, offline=True)
    threads = [threading.Thread(target=s.serve_forever, kwargs={"poll_interval": .01}) for s in servers]
    for thread in threads:
        thread.start()
    ports = {s.server_address[1] for s in servers}
    output = Path(tempfile.mkdtemp(prefix="workbench-inspection-qa-"))
    errors, checks = [], []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000})

            def local_only(route):
                target = urlsplit(route.request.url)
                if target.hostname == "127.0.0.1" and target.port in ports:
                    route.continue_()
                else:
                    errors.append("unexpected_network")
                    route.abort()

            context.route("**/*", local_only)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            for server_index in (0, 1):
                url = "http://127.0.0.1:" + str(servers[server_index].server_address[1])
                for width, height, label in [(1440, 1000, "desktop"), (390, 844, "mobile")]:
                    page.set_viewport_size({"width": width, "height": height})
                    page.goto(url + "/?run_id=run_review")
                    expect(page.locator("#openFlow")).to_be_enabled()
                    page.locator("#openCatalogue").click()
                    expect(page.locator(".catalogue-item")).to_have_count(2)
                    expect(page.locator("#catalogueDetail")).to_contain_text('"required"')
                    search = page.locator(".catalogue-navigation input")
                    search.fill("archive")
                    expect(page.locator(".catalogue-item")).to_have_count(1)
                    expect(page.locator("#catalogueDetail")).to_contain_text("notes__archive")
                    search.fill("not_registered")
                    expect(page.locator(".catalogue-item")).to_have_count(0)
                    search.fill("read")
                    expect(page.locator("#catalogueDetail")).to_contain_text('"required"')
                    page.screenshot(path=str(output / ("catalogue-" + str(server_index) + "-" + label + ".png")))
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    page.keyboard.press("Escape")
                    expect(page.locator("#inspectionDialog")).not_to_be_visible()
                    expect(page.locator("#openCatalogue")).to_be_focused()
                    page.locator("#openFlow").click()
                    expect(page.locator(".flow-node")).to_have_count(6)
                    expect(page.locator("#flowDetail")).to_contain_text("int_fixture")
                    page.locator(".flow-node").first.click()
                    expect(page.locator("#flowDetail")).to_contain_text("Event 2")
                    expect(page.locator(".follow-flow input")).not_to_be_checked()
                    page.screenshot(path=str(output / ("flow-" + str(server_index) + "-" + label + ".png")))
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    boxes = page.locator(".inspection-grid > *").evaluate_all("(els) => els.map(e => { const b=e.getBoundingClientRect(); return {x:b.x,y:b.y,r:b.right,b:b.bottom}; })")
                    if width < 620:
                        assert boxes[0]["b"] <= boxes[1]["y"]
                    else:
                        assert boxes[0]["r"] <= boxes[1]["x"]
                    page.locator(".close-inspection").click()
                    checks.append(str(server_index) + "/" + label + ": search, schema, flow, evidence, focus, layout")
            # Polling updates the open flow without submitting an operation.
            page.locator("#openFlow").click()
            with runtime.lock:
                runtime.runs["run_review"]["state"]["pending_interaction"] = None
                runtime.runs["run_review"]["state"]["status"] = "failed"
                runtime.runs["run_review"]["state"]["agent_trace"].append(
                    {"type": "tool_result", "tool_call": {"tool_name": "notes__archive"}, "observation": {"status": "blocked"}})
            expect(page.locator("#flowDetail")).to_contain_text("notes__archive")
            expect(page.locator('.flow-node[data-tone="bad"]')).to_have_count(1)
            page.locator(".close-inspection").click()
            page.goto(url + "/?run_id=run_empty")
            expect(page.locator("#openFlow")).to_be_enabled()
            page.locator("#openFlow").click()
            expect(page.locator(".flow-node")).to_have_count(0)
            assert not runtime.calls
            assert not errors, errors
            checks.append("live updates, blocked evidence, missing history; zero execution calls")
            context.close()
            browser.close()
        result = {"status": "passed", "checks": checks, "screenshots": str(output), "model_calls": 0}
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        for item in servers:
            item.shutdown()
            item.server_close()
        for thread in threads:
            thread.join()
        servers[0].RequestHandlerClass.service.close()


if __name__ == "__main__":
    main()
