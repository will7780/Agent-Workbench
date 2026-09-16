"""Offline chat-spinner checks; synthetic server, no credentials or model calls."""
import json
import os
import runpy
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


class SlowEvent(threading.Event):
    def wait(self, timeout=None):
        return super().wait(30)


def main():
    os.environ['AGENT_WORKBENCH_DISABLE_CENTRAL_ENV'] = '1'
    from agent_workbench.web import server
    root = Path(__file__).resolve().parents[1]
    server.STATIC_DIR = root / 'src' / 'agent_workbench' / 'web' / 'static'
    fixture = runpy.run_path(str(root / 'tests' / 'test_web.py'))
    runtime = fixture['FakeRuntime']()
    runtime.release = SlowEvent()
    runtime.add('run_clarification', kind='clarification')
    servers = server.create_servers(runtime, chat_port=0, diagnostics_port=0, offline=True)
    threads = [threading.Thread(target=s.serve_forever, kwargs={'poll_interval': .01}) for s in servers]
    for thread in threads:
        thread.start()
    ports = {s.server_address[1] for s in servers}
    url = f'http://127.0.0.1:{servers[0].server_address[1]}'
    output = Path(tempfile.mkdtemp(prefix='workbench-activity-qa-'))
    errors, checks = [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={'width': 1440, 'height': 1000})

            def local_only(route):
                target = urlsplit(route.request.url)
                if target.hostname == '127.0.0.1' and target.port in ports:
                    route.continue_()
                else:
                    errors.append('unexpected_network_request')
                    route.abort()

            context.route('**/*', local_only)
            page = context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            expect(page.locator('#runtimeStatus')).to_have_text('Ready')
            expect(page.locator('#assistantActivity')).to_be_hidden()

            for width, height, label in [(1440, 1000, 'desktop'), (390, 844, 'mobile')]:
                runtime.release.clear()
                page.set_viewport_size({'width': width, 'height': height})
                page.get_by_role('button', name='New conversation', exact=True).click()
                page.get_by_label('Message', exact=True).fill('hold')
                immediate = page.evaluate("""() => {
                    document.getElementById('sendButton').click();
                    return !document.getElementById('assistantActivity').hidden;
                }""")
                assert immediate, 'Spinner must appear before POST resolves'
                expect(page.locator('#assistantActivity')).to_be_visible()
                expect(page.locator('#composer')).to_have_attribute('aria-busy', 'true')
                expect(page.get_by_role('button', name='Cancel run', exact=True)).to_be_enabled()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                assert page.locator('.activity-spinner').evaluate("e => getComputedStyle(e).animationName") == 'activity-spin'
                page.screenshot(path=str(output / f'thinking-{label}.png'), full_page=True)
                page.reload()
                expect(page.locator('#assistantActivity')).to_be_visible()
                if label == 'desktop':
                    page.get_by_role('button', name='Cancel run', exact=True).click()
                    expect(page.locator('#runtimeStatus')).to_have_text('cancelled')
                else:
                    runtime.release.set()
                    expect(page.locator('#runtimeStatus')).to_have_text('completed')
                expect(page.locator('#assistantActivity')).to_be_hidden()
                checks.append(label + ': immediate, animation, refresh, terminal cleanup, no overflow')

            page.goto(url + '/?run_id=run_clarification')
            expect(page.get_by_label('Answer', exact=True)).to_be_visible()
            expect(page.locator('#assistantActivity')).to_be_hidden()
            page.get_by_label('Answer', exact=True).fill('Synthetic answer')
            assert page.evaluate("""() => {
                document.querySelector('#interaction button[type=submit]').click();
                return !document.getElementById('assistantActivity').hidden;
            }""")
            expect(page.locator('#runtimeStatus')).to_have_text('completed')
            expect(page.locator('#assistantActivity')).to_be_hidden()
            checks.append('pending hides spinner; resume immediately shows it')

            page.get_by_role('button', name='New conversation', exact=True).click()
            page.get_by_label('Message', exact=True).fill('fail')
            page.get_by_role('button', name='Send message', exact=True).click()
            expect(page.locator('#error')).to_be_visible()
            expect(page.locator('#assistantActivity')).to_be_hidden()
            checks.append('failed asynchronous operation stops spinner')

            page.get_by_role('button', name='New conversation', exact=True).click()
            page.route('**/api/runs', lambda route: route.fulfill(status=400,
                       content_type='application/json', body='{"error_type":"sensitive_input_rejected"}')
                       if route.request.method == 'POST' else route.continue_())
            page.get_by_label('Message', exact=True).fill('Rejected fixture request')
            page.get_by_role('button', name='Send message', exact=True).click()
            expect(page.locator('#error')).to_be_visible()
            expect(page.locator('#assistantActivity')).to_be_hidden()
            expect(page.get_by_label('Message', exact=True)).to_be_enabled()
            page.unroute('**/api/runs')
            checks.append('rejected POST stops spinner and preserves composer')

            page.get_by_role('button', name='New conversation', exact=True).click()
            runtime.release.clear()
            page.emulate_media(reduced_motion='reduce')
            page.get_by_label('Message', exact=True).fill('hold')
            page.get_by_role('button', name='Send message', exact=True).click()
            expect(page.locator('#assistantActivity')).to_be_visible()
            assert page.locator('.activity-spinner').evaluate("e => getComputedStyle(e).animationName") == 'none'
            page.get_by_role('button', name='New conversation', exact=True).click()
            expect(page.locator('#assistantActivity')).to_be_hidden()
            runtime.release.set()
            checks.append('reduced motion and run switching')
            assert not errors, errors
            context.close()
            browser.close()
        report = {'status': 'passed', 'checks': checks, 'screenshots': str(output), 'model_calls': 0}
        (output / 'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        runtime.release.set()
        for item in servers:
            item.shutdown()
            item.server_close()
        for thread in threads:
            thread.join()
        servers[0].RequestHandlerClass.service.close()


if __name__ == '__main__':
    main()
