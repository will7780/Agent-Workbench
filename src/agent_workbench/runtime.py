"""Application facade around the shared LangGraph implementation."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .configuration import ConfigProvider, StaticConfigProvider
from .langgraph_interactions import InteractionRuntimeStore
from .langgraph_state_graph import (run_react_state_graph_runtime,
                                   resume_react_state_graph_runtime)
from .redaction import redact_recursive
from .registry import ToolRegistry
from .runtime_errors import RuntimeCancelled
from .runtime_telemetry import RuntimeTelemetryCollector
from .tools.execution_adapters import ToolExecutor


@dataclass
class RuntimeServices:
    registry: ToolRegistry
    executor: ToolExecutor
    model_client: Optional[Callable] = None
    config_provider: ConfigProvider = field(default_factory=StaticConfigProvider)
    artifact_validator: Any = None
    event_sink: Optional[Callable] = None
    parameter_judge: Any = None
    allow_real_model: bool = False


def public_payload(value):
    if isinstance(value, dict):
        value = {k: public_payload(v) for k, v in value.items()
                 if str(k).lower() not in {'reasoning_content', 'chain_of_thought',
                                          'hidden_reasoning', 'thinking', 'analysis'}}
    elif isinstance(value, (list, tuple)):
        value = [public_payload(v) for v in value]
    safe, _ = redact_recursive(value)
    return safe


def conversation_history(messages):
    """Keep completed tool exchanges; cancellation can leave unresolved calls."""
    history, index = [], 0
    while index < len(messages):
        message = copy.deepcopy(messages[index])
        index += 1
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            responses = []
            while index < len(messages) and messages[index].get('role') == 'tool':
                responses.append(copy.deepcopy(messages[index]))
                index += 1
            answered = {response.get('tool_call_id') for response in responses}
            message['tool_calls'] = [call for call in message['tool_calls'] if call.get('id') in answered]
            if message['tool_calls']:
                allowed = {call['id'] for call in message['tool_calls']}
                history.append(message)
                history.extend(response for response in responses if response.get('tool_call_id') in allowed)
            elif message.get('content'):
                message.pop('tool_calls')
                history.append(message)
        elif message.get('role') in {'user', 'assistant'}:
            history.append(message)
    return history


class AgentRuntime:
    """Local, in-process paused runs; historical reports survive a restart."""

    def __init__(self, services: RuntimeServices, *, data_dir=None):
        self.services = services
        self.data_dir = Path(data_dir or Path.home() / '.agent-workbench').resolve()
        self._lock = threading.RLock()
        self._runs = {}
        self._cancel = {}
        self._active = set()
        self._history = {}
        self._scopes = {}
        self.interactions = InteractionRuntimeStore(max_entries=128)
        self._reports = self.data_dir / 'reports'
        self._reports.mkdir(parents=True, exist_ok=True)
        for path in sorted(self._reports.glob('run-*.json'))[-1000:]:
            try:
                result = json.loads(path.read_text(encoding='utf-8'))
                run_id = result['state']['run_id']
                if result['state'].get('pending_interaction'):
                    result['state']['execution_status'] = 'stopped'
                    result['report']['status'] = 'stopped'
                    result['state']['pending_interaction'] = None
                    result['report']['pending_interaction'] = None
                    result['report']['restart_required'] = True
                self._runs[run_id] = public_payload(result)
            except (ValueError, KeyError, TypeError, OSError):
                continue

    def _save(self, result):
        result = public_payload(result)
        state, report = result['state'], result['report']
        run_id = state['run_id']
        report['run_id'] = run_id
        with self._lock:
            self._runs[run_id] = copy.deepcopy(result)
            path = self._reports / ('run-' + hashlib.sha256(run_id.encode()).hexdigest() + '.json')
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding='utf-8')
            temp.replace(path)
        return copy.deepcopy(result)

    def _event(self, run_id, event):
        event = public_payload({'run_id': run_id, **event})
        with self._lock:
            result = self._runs.get(run_id)
            if result:
                result['state'].setdefault('live_events', []).append(event)
        if self.services.event_sink:
            try:
                self.services.event_sink(event)
            except Exception:
                pass

    def get_run(self, run_id):
        with self._lock:
            if run_id not in self._runs:
                raise KeyError('run_not_found')
            return copy.deepcopy(self._runs[run_id])

    def list_runs(self):
        with self._lock:
            return copy.deepcopy(list(reversed(list(self._runs.values()))))

    def _terminal_error(self, run_id, reason, state=None):
        state = dict(state or {})
        state.update(run_id=run_id, execution_status='stopped', pending_interaction=None)
        state.setdefault('errors', []).append({'type': reason, 'message': reason})
        self.interactions.complete(run_id)
        return {'state': state, 'report': {**state, 'status': 'stopped'}, 'report_text': reason}

    def _finish(self, run_id, result):
        with self._lock:
            self._active.discard(run_id)
            scope = self._scopes.get(run_id)
            if scope and not result['state'].get('pending_interaction') and result['state'].get('messages'):
                self._history[scope] = conversation_history(result['state']['messages'])
        return self._save(result)

    def start(self, request):
        if not isinstance(request, dict):
            raise ValueError('request_must_be_an_object')
        text = request.get('message', request.get('user_request', ''))
        if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
            raise ValueError('invalid_user_message')
        run_id = str(request.get('run_id') or uuid.uuid4().hex)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
            raise ValueError('invalid_run_id')
        config = copy.deepcopy(self.services.config_provider.snapshot())
        if config.get('company_profile_config'):
            from .skill_registry import prepare_company_skill_runtime
            config['company_skill_runtime'] = prepare_company_skill_runtime(
                config['company_profile_config'], registry=self.services.registry)
        thread_id = str(request.get('thread_id') or uuid.uuid4().hex)
        from .session_checkpoint import opaque_company_profile_binding_id
        binding = opaque_company_profile_binding_id(config.get('company_skill_runtime'))
        scope = (thread_id, hashlib.sha256(binding.encode()).hexdigest())
        with self._lock:
            if run_id in self._runs:
                raise ValueError('run_already_exists')
            if any(self._scopes.get(r) == scope and (r in self._active or self._runs[r]['state'].get('pending_interaction'))
                   for r in self._runs):
                raise ValueError('thread_has_active_run')
            self._scopes[run_id] = scope
            self._active.add(run_id)
            self._cancel[run_id] = threading.Event()
            self._runs[run_id] = {'state': {'run_id': run_id, 'thread_id': thread_id,
                                           'user_request': text, 'execution_status': 'running'},
                                  'report': {'run_id': run_id, 'status': 'running'}, 'report_text': ''}
            history = copy.deepcopy(self._history.get(scope, []))
        if self.services.model_client is None and not self.services.allow_real_model:
            return self._finish(run_id, self._terminal_error(run_id, 'model_not_configured'))
        profile = (config.get('company_skill_runtime') or {}).get('company_profile') or {}
        if profile.get('enabled') and binding == 'enabled':
            return self._finish(run_id, self._terminal_error(run_id, 'profile_snapshot_unavailable'))
        registry = copy.copy(self.services.registry)
        registry._modules = copy.deepcopy(registry._modules)
        registry.executor = self.services.executor
        registry.artifact_validator = self.services.artifact_validator or registry.artifact_validator
        registry.cancel_requested = self._cancel[run_id].is_set
        telemetry = RuntimeTelemetryCollector()
        try:
            result = run_react_state_graph_runtime(
                text, run_id=run_id, thread_id=thread_id, registry=registry,
                execution_mode=config.get('execution_mode', 'dry_run'),
                user_confirmed=False, module_config=config.get('module_config'),
                parameter_sources=config.get('parameter_sources'),
                llm_model=config.get('model'), llm_complete_fn=self.services.model_client,
                parameter_judge=self.services.parameter_judge,
                max_tool_rounds=int(config.get('max_tool_rounds', 12)),
                interactions_enabled=True, interaction_runtime_store=self.interactions,
                progress_callback=lambda event: self._event(run_id, event),
                company_skill_runtime=config.get('company_skill_runtime'),
                knowledge_config=config.get('knowledge_config'),
                memory_dir=(Path(config['memory_dir']) / scope[1]) if config.get('memory_dir') else None,
                context_spill_dir=self.data_dir / 'context' / run_id,
                runtime_telemetry_collector=telemetry, history_messages=history,
            )
        except RuntimeCancelled as exc:
            result = self._terminal_error(run_id, 'run_cancelled', exc.state)
        except Exception as exc:
            result = self._terminal_error(run_id, 'runtime_error:' + type(exc).__name__)
        result['state'].update(telemetry.snapshot())
        result['report'].update(telemetry.snapshot())
        return self._finish(run_id, result)

    def resume(self, run_id, interaction_id, response):
        with self._lock:
            current = self.get_run(run_id)
            if run_id in self._active:
                raise ValueError('run_is_busy')
            if self._cancel.get(run_id) and self._cancel[run_id].is_set():
                raise ValueError('run_cancelled')
            pending = current['state'].get('pending_interaction')
            if not pending or pending.get('interaction_id') != interaction_id:
                raise ValueError('interaction_mismatch')
            if not isinstance(response, dict):
                raise ValueError('invalid_interaction_response')
            payload = {'type': pending['type'], **response, 'interaction_id': interaction_id}
            from .langgraph_interactions import sanitize_interaction_response
            _, error = sanitize_interaction_response(payload, pending)
            if error:
                raise ValueError(error)
            self._active.add(run_id)
        result = resume_react_state_graph_runtime(run_id, payload, interaction_runtime_store=self.interactions)
        return self._finish(run_id, result)

    def cancel(self, run_id):
        with self._lock:
            current = self.get_run(run_id)
            if run_id not in self._active and not current['state'].get('pending_interaction'):
                return current
            event = self._cancel.get(run_id)
            if event:
                event.set()
            if run_id in self._active:
                current['report']['cancellation_requested'] = True
                return current
        return self._finish(run_id, self._terminal_error(run_id, 'run_cancelled', current['state']))
