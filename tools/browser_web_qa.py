"""Optional offline Chromium QA; temporary HTTP ports, loopback traffic only.

Run from an installed/editable workbench checkout:
    python tools/browser_web_qa.py [--demo]
Verify a frozen wheel from its external environment without source imports:
    python -I -B tools/browser_web_qa.py --demo-only --installed \
        --wheel PATH --expected-wheel-sha256 SHA256
Screenshots go to a new temporary directory. Nothing is written to runtime data.
"""

from __future__ import annotations

import os
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import runpy
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect, sync_playwright

def verify_installation(wheel, expected_sha256):
    checkout = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec("agent_workbench")
    if spec is None or not spec.origin:
        raise RuntimeError("Installed agent_workbench package is unavailable")
    origin = Path(spec.origin).resolve()
    print(f"WORKBENCH_MODULE_ORIGIN={origin}", flush=True)
    if origin.is_relative_to(checkout):
        raise RuntimeError("Installed QA refuses checkout module origin")
    distribution = importlib.metadata.distribution("agent-workbench")
    package_root = Path(distribution.locate_file("agent_workbench")).resolve()
    if origin != package_root / "__init__.py":
        raise RuntimeError("Module origin does not match the installed distribution")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("Installed QA refuses editable distributions")
    wheel = wheel.resolve(strict=True)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) or digest != expected_sha256.lower():
        raise RuntimeError("Frozen wheel SHA-256 mismatch")
    verified_files = 0
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith("agent_workbench/"):
                continue
            installed = (package_root.parent / entry.filename).resolve()
            if not installed.is_relative_to(package_root):
                raise RuntimeError("Wheel package member escapes its package root")
            if not installed.is_file() or installed.read_bytes() != archive.read(entry):
                raise RuntimeError(f"Installed file differs from frozen wheel: {entry.filename}")
            verified_files += 1
    if not verified_files:
        raise RuntimeError("Wheel contains no agent_workbench package files")
    installation = {
        "mode": "installed-wheel", "python": sys.executable,
        "module_origin": str(origin), "package_root": str(package_root),
        "version": distribution.version, "wheel": str(wheel),
        "wheel_sha256": digest, "verified_package_files": verified_files,
    }
    print(json.dumps(installation, indent=2), flush=True)
    return installation


def audit_module_origins(installation):
    if installation is None:
        return
    root = Path(installation["package_root"])
    checkout = Path(__file__).resolve().parents[1]
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != "agent_workbench" and not name.startswith("agent_workbench."):
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            raise RuntimeError(f"Workbench module has no verifiable origin: {name}")
        origin = Path(filename).resolve()
        if origin.is_relative_to(checkout) or not origin.is_relative_to(root):
            raise RuntimeError(f"Installed QA refuses non-package module origin: {name}")
        origins[name] = str(origin)
    installation["loaded_module_origins"] = origins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Also verify the host-owned offline demo runtime")
    parser.add_argument("--demo-only", action="store_true", help="Verify only the real offline demo")
    parser.add_argument("--installed", action="store_true", help="Reject checkout imports and verify frozen wheel files; requires --demo-only")
    parser.add_argument("--wheel", type=Path, help="Frozen wheel to verify in installed mode")
    parser.add_argument("--expected-wheel-sha256", help="Expected SHA-256 of the frozen wheel")
    args = parser.parse_args()
    if args.installed and not (args.demo_only and args.wheel and args.expected_wheel_sha256):
        parser.error("--installed requires --demo-only, --wheel and --expected-wheel-sha256")
    if not args.installed and (args.wheel or args.expected_wheel_sha256):
        parser.error("Wheel verification options require --installed")
    os.environ["AGENT_WORKBENCH_DISABLE_CENTRAL_ENV"] = "1"
    installation = verify_installation(args.wheel, args.expected_wheel_sha256) if args.installed else None
    if args.demo_only:
        verify_demo(Path(tempfile.mkdtemp(prefix="workbench-web-qa-")), installation)
        return
    from agent_workbench.web.server import create_servers

    fixture = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tests" / "test_web.py"))
    runtime = fixture["FakeRuntime"]()
    runtime.add("run_artifact", kind="artifact_review", can_approve=False)
    runtime.add("run_clarification", kind="clarification")
    runtime.add("run_confirmation", kind="confirmation")
    runtime.runs["run_artifact"]["state"]["messages"][0]["content"] = '<img src=x onerror="window.xss=true"> Synthetic request'
    runtime.runs["run_artifact"]["state"]["pending_interaction"]["sample"][0]["title"] = "LongSyntheticTitle" * 12
    servers = create_servers(runtime, chat_port=0, diagnostics_port=0, offline=True, model_label="Synthetic QA model")
    service = servers[0].RequestHandlerClass.service
    service.event_sink({"run_id": "run_artifact", "type": "artifact.review", "content_hash": "synthetic_revision_hash"})
    threads = [threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}) for server in servers]
    for thread in threads:
        thread.start()
    output = Path(tempfile.mkdtemp(prefix="workbench-web-qa-"))
    errors = []
    urls = [f"http://127.0.0.1:{server.server_address[1]}" for server in servers]
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000})

            def route(request_route):
                target = urlsplit(request_route.request.url)
                if target.hostname != "127.0.0.1" or target.port not in [server.server_address[1] for server in servers]:
                    errors.append("non-loopback request")
                    request_route.abort()
                else:
                    request_route.continue_()

            context.route("**/*", route)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(urls[0] + "/?run_id=run_artifact")
            expect(page.get_by_label("Review comment", exact=True)).to_be_visible()
            expect(page.get_by_role("button", name="Approve", exact=True)).to_be_disabled()
            page.get_by_label("Review comment", exact=True).fill("Please revise the synthetic title")
            page.wait_for_timeout(1100)
            expect(page.get_by_label("Review comment", exact=True)).to_have_value("Please revise the synthetic title")
            assert page.locator("img").count() == 0
            page.screenshot(path=str(output / "chat-desktop.png"), full_page=True)
            page.get_by_role("button", name="Request changes", exact=True).click()
            expect(page.get_by_text("Revision: 2", exact=False)).to_be_visible()
            page.get_by_role("button", name="Reject", exact=True).click()
            expect(page.locator("#runtimeStatus")).to_have_text("rejected")

            page.goto(urls[0] + "/?run_id=run_clarification")
            page.get_by_label("Answer", exact=True).fill("Use a short synthetic title")
            page.get_by_role("button", name="Submit answer", exact=True).click()
            expect(page.locator("#runtimeStatus")).to_have_text("completed")
            page.goto(urls[0] + "/?run_id=run_confirmation")
            page.get_by_role("button", name="Approve", exact=True).click()
            expect(page.locator("#runtimeStatus")).to_have_text("completed")

            page.get_by_role("button", name="New conversation", exact=True).click()
            page.get_by_label("Message", exact=True).fill("hold")
            page.get_by_role("button", name="Send message", exact=True).click()
            expect(page.get_by_role("button", name="Cancel run", exact=True)).to_be_enabled()
            page.get_by_role("button", name="Cancel run", exact=True).click()
            expect(page.locator("#runtimeStatus")).to_have_text("cancelled")

            for width, height, label in ((1440, 1000, "desktop"), (390, 844, "mobile")):
                page.set_viewport_size({"width": width, "height": height})
                runtime.add("run_artifact", kind="artifact_review", can_approve=False)
                runtime.runs["run_artifact"]["state"]["pending_interaction"]["sample"][0]["title"] = "SyntheticLongTitle" * 16
                page.goto(urls[0] + "/?run_id=run_artifact")
                expect(page.get_by_label("Review comment", exact=True)).to_be_visible()
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(output / f"chat-{label}.png"), full_page=True)
                page.goto(urls[1] + "/?run_id=run_artifact")
                expect(page.get_by_role("heading", name="Timeline", exact=True)).to_be_visible()
                expect(page.locator(".message-role")).to_have_text(["user", "assistant", "tool"])
                expect(page.locator("#resources dd")).to_have_text(["0", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"])
                page.get_by_role("button", name="1. artifact.review", exact=False).click()
                expect(page.locator("#eventDetail")).to_contain_text("synthetic_revision_hash")
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(output / f"diagnostics-{label}.png"), full_page=True)
            assert not errors, errors
            browser.close()
    finally:
        runtime.release.set()
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()
        service.close()
    print(f"PASS: desktop/mobile forms, polling, revise/reject/approve, cancellation, roles, nulls, XSS, layout. Screenshots: {output}")
    if args.demo:
        verify_demo(output)


def verify_demo(output, installation=None):
    from agent_workbench.demo import DEMO_MODEL, DEMO_REQUEST, build_demo_runtime
    from agent_workbench.web.server import create_servers

    audit_module_origins(installation)

    with tempfile.TemporaryDirectory(prefix="workbench-web-demo-qa-") as directory:
        runtime = build_demo_runtime(directory)
        servers = create_servers(runtime, chat_port=0, diagnostics_port=0, offline=True, model_label=DEMO_MODEL)
        threads = [threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}) for server in servers]
        for thread in threads:
            thread.start()
        urls = [f"http://127.0.0.1:{server.server_address[1]}" for server in servers]
        report = {"runtime": "real offline AgentRuntime", "cases": [], "screenshots": [], "ports": [server.server_address[1] for server in servers]}
        if installation is not None:
            report["installation"] = installation
        service = servers[0].RequestHandlerClass.service
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
                errors = []

                def local_only(route):
                    if any(route.request.url.startswith(url + "/") for url in urls):
                        route.continue_()
                    else:
                        errors.append("Unexpected non-loopback request")
                        route.abort()

                context.route("**/*", local_only)
                page = context.new_page()
                page.on("pageerror", lambda error: errors.append(str(error)))

                def capture(name, target=None):
                    if target is not None:
                        target.scroll_into_view_if_needed()
                    problems = page.evaluate("""() => {
                      const problems = [];
                      if (document.documentElement.scrollWidth > innerWidth) problems.push('page overflow');
                      for (const item of document.querySelectorAll('button, .topbar h1, .environment span, td, th')) {
                        const rect = item.getBoundingClientRect();
                        if (!rect.width || !rect.height) continue;
                        if (item.scrollWidth > item.clientWidth + 2) problems.push('text overflow: ' + item.tagName);
                        if (item.tagName === 'BUTTON' && !(item.textContent.trim() || item.getAttribute('aria-label'))) problems.push('unlabelled button');
                      }
                      return problems;
                    }""")
                    assert not problems, problems
                    path = output / (name + ".png")
                    page.screenshot(path=str(path), full_page=True)
                    report["screenshots"].append(str(path))

                def wait_review(previous_id=None):
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        panel = page.locator("#interaction > section")
                        if panel.count() and page.locator("#reviewComment").count():
                            current_id = panel.get_attribute("data-interaction-id")
                            if current_id and current_id != previous_id:
                                expect(page.get_by_role("button", name="Approve", exact=True)).to_be_enabled(timeout=30000)
                                return current_id
                        if page.locator("#runtimeStatus").inner_text() in {"completed", "failed", "stopped"}:
                            capture("demo-review-not-reached")
                            raise AssertionError("Host demo terminated before a new artifact review; evidence was not substituted")
                        page.wait_for_timeout(100)
                    capture("demo-review-timeout")
                    raise AssertionError("Timed out waiting for a real pending artifact review")

                def diagnostic_drilldown(run_id, decision, label, detailed):
                    page.goto(urls[1] + "/?run_id=" + run_id)
                    expect(page.get_by_role("heading", name="Timeline", exact=True)).to_be_visible()
                    view = service.get_run(run_id)
                    expect(page.locator("#runtimeStatus")).to_have_text(view["status"])
                    roles = [message["role"] for message in view["messages"]]
                    assert "tool" in roles and "assistant" in roles and "user" in roles
                    expect(page.locator(".message-role")).to_have_text(roles)
                    review_button = page.get_by_role("button", name=re.compile(r"\d+\. artifact\.review\b")).last
                    review_button.click()
                    event = json.loads(page.locator("#eventDetail").inner_text())
                    assert event["type"] == "artifact.review" and event["decision"] == decision
                    assert event.get("content_hash"), "Review content hash unavailable"
                    capture(f"demo-{label}-diagnostic-review", page.locator("#timelineSection"))
                    if detailed:
                        page.get_by_role("link", name="Parameters", exact=True).click()
                        page.locator("#parameters summary").filter(has_text="Sources").click()
                        page.locator("#parameters summary").filter(has_text="Verification").click()
                        capture(f"demo-{label}-parameters", page.locator("#parametersSection"))
                        for section, title in (("observations", "Observations"), ("context", "Context"), ("artifacts", "Artifacts")):
                            page.get_by_role("link", name=title, exact=True).click()
                            expected = "Unknown" if view[section] is None else json.dumps(view[section], ensure_ascii=False, indent=2)
                            expect(page.locator(f"#{section} > pre").first).to_have_text(expected)
                            capture(f"demo-{label}-{section}", page.locator(f"#{section}Section"))
                    page.get_by_role("link", name="Resources", exact=True).click()
                    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "latency_ms", "tool_latency_ms", "active_runtime_ms", "wall_runtime_ms", "user_wait_ms")
                    expected_metrics = ["Unknown" if view["metrics"][key] is None else format(view["metrics"][key], ".15g") for key in keys]
                    expect(page.locator("#resources dd")).to_have_text(expected_metrics)
                    assert any(value == "Unknown" for value in expected_metrics), "Offline unknown usage was not retained"
                    capture(f"demo-{label}-resources", page.locator("#resourcesSection"))
                    return {"decision": event["decision"], "content_hash": event["content_hash"], "metrics": view["metrics"]}

                for width, height, viewport in ((1440, 1000, "desktop"), (390, 844, "mobile")):
                    page.set_viewport_size({"width": width, "height": height})
                    for scenario in ("approve", "reject", "request_changes"):
                        label = viewport + "-" + scenario
                        before = set(Path(directory).glob("offline-demo-*/archives/*.json"))
                        page.goto(urls[0])
                        expect(page.locator("#environmentLabel")).to_contain_text("Offline / simulated model")
                        page.get_by_label("Message", exact=True).fill(DEMO_REQUEST)
                        page.get_by_role("button", name="Send message", exact=True).click()
                        old_id = wait_review()
                        expect(page.locator("#historyNote")).to_have_text("")
                        run_id = parse_qs(urlsplit(page.url).query)["run_id"][0]
                        pending = runtime.get_run(run_id)["state"]["pending_interaction"]
                        assert pending["can_approve"] is True and pending.get("sample")
                        initial_hash = pending["evidence"]["content_hash"]
                        capture(f"demo-{label}-sample", page.locator("#interaction table"))
                        page.locator("#interaction summary").filter(has_text="Artifact proof").click()
                        capture(f"demo-{label}-proof", page.locator("#interaction details"))
                        capture(f"demo-{label}-controls", page.locator(".interaction-actions"))
                        if scenario == "request_changes":
                            comment = f"Emphasize reading-time variance in the {viewport} revision."
                            page.get_by_label("Review comment", exact=True).fill(comment)
                            page.wait_for_timeout(1100)
                            expect(page.get_by_label("Review comment", exact=True)).to_have_value(comment)
                            page.get_by_role("button", name="Request changes", exact=True).click()
                            new_id = wait_review(old_id)
                            pending = runtime.get_run(run_id)["state"]["pending_interaction"]
                            assert pending["interaction_id"] == new_id
                            assert pending["evidence"]["content_hash"] != initial_hash
                            assert comment in json.dumps(pending["sample"])
                            assert set(Path(directory).glob("offline-demo-*/archives/*.json")) == before
                            stale = context.request.post(urls[0] + f"/api/runs/{run_id}/resume", headers={"Origin": urls[0]}, data={"interaction_id": old_id, "response": {"type": "artifact_review", "decision": "approve"}})
                            assert stale.status == 409, "Stale review was not refused"
                            capture(f"demo-{label}-rereview", page.locator("#interaction table"))
                        final_decision = "reject" if scenario == "reject" else "approve"
                        page.get_by_role("button", name=final_decision.title(), exact=True).click()
                        expect(page.locator("#interaction")).to_be_empty(timeout=30000)
                        expect(page.locator("#runtimeStatus")).to_have_text(re.compile(r"^(completed|stopped|rejected)$"), timeout=30000)
                        after = set(Path(directory).glob("offline-demo-*/archives/*.json"))
                        assert len(after - before) == (0 if scenario == "reject" else 1)
                        if scenario == "request_changes":
                            archived = json.loads(next(iter(after - before)).read_text(encoding="utf-8"))
                            assert all(comment in row["note"] for row in archived)
                        proof = diagnostic_drilldown(run_id, final_decision, label, scenario == "request_changes")
                        report["cases"].append({"viewport": viewport, "scenario": scenario, "run_id": run_id,
                                                "archive_delta": len(after - before), "re_review": scenario == "request_changes", **proof})
                assert not errors, errors
                browser.close()
            audit_module_origins(installation)
            if installation is not None:
                for name in ("agent_workbench", "agent_workbench.demo", "agent_workbench.runtime", "agent_workbench.web.server", "agent_workbench.web.service"):
                    print(f"LOADED_MODULE {name}={installation['loaded_module_origins'][name]}", flush=True)
                print(f"Audited {len(installation['loaded_module_origins'])} installed module origins", flush=True)
            report["status"] = "passed"
        except Exception:
            report["status"] = "failed"
            try:
                if "page" in locals() and not page.is_closed():
                    page.screenshot(path=str(output / "demo-failure.png"), full_page=True)
            except Exception:
                pass
            raise
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()
            service.close()
            (output / "demo-qa-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Real demo QA artifacts: {output}", flush=True)
    print("PASS: real offline AgentRuntime; six desktop/mobile review paths; fresh revisions; stale approval rejected; actual archive counts; diagnostic drilldown.")


if __name__ == "__main__":
    main()
