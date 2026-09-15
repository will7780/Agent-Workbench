"""All default tests are offline and cannot load real central credentials."""
import os
import pytest

os.environ['AGENT_WORKBENCH_DISABLE_CENTRAL_ENV'] = '1'


@pytest.fixture(autouse=True)
def offline_only(monkeypatch, tmp_path):
    monkeypatch.setenv('AGENT_WORKBENCH_DISABLE_CENTRAL_ENV', '1')
    monkeypatch.setenv('AGENT_API_ENV_FILE', str(tmp_path / 'absent.env'))
    for name in list(os.environ):
        if any(part in name for part in ('API_KEY', 'API_TOKEN', 'AUTH_TOKEN')):
            monkeypatch.delenv(name, raising=False)
    import requests
    def no_network(*args, **kwargs):
        raise AssertionError('Real provider/network calls are forbidden in default tests')
    monkeypatch.setattr(requests.sessions.Session, 'request', no_network)
