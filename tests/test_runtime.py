import json
from agent_workbench.registry import ToolRegistry
from agent_workbench.configuration import StaticConfigProvider
from agent_workbench.llm_client import LLMChatWithToolsResult, LLMToolCall
from agent_workbench.runtime import AgentRuntime, RuntimeServices
from agent_workbench.tools.adapter_contracts import AdapterOutput
from agent_workbench.tools.execution_adapters import RegisteredToolExecutor


def make_runtime(tmp_path, *, required=False, model=None, config=None):
    registry = ToolRegistry(payload={'version': '1', 'modules': {'notes': {'actions': {
        'write': {'risk_level': 'L2', 'execution_adapter': 'write',
                  'dry_run_status': 'available', 'local_write_status': 'available', 'live_status': 'available',
                  'required_params': ['value'], 'parameter_contract': {
                      'type': 'object', 'properties': {'value': {'type': 'integer', 'minimum': 1}},
                      'required': ['value'], 'additionalProperties': False}}
    }}}})
    calls = []
    executor = RegisteredToolExecutor()
    def write(request):
        calls.append(dict(request.params))
        return AdapterOutput(status='success', summary='Saved note in isolated example')
    executor.register('write', write, modes=('dry_run', 'local_write', 'live'))
    services = RuntimeServices(registry, executor, model_client=model,
                               config_provider=StaticConfigProvider(config))
    return AgentRuntime(services, data_dir=tmp_path), calls


def sequence(*responses):
    iterator = iter(responses)
    return lambda config, messages, tools: next(iterator)


def call(value=1, **kwargs):
    return LLMChatWithToolsResult(tool_calls=[LLMToolCall(id='one', name='notes__write',
                                                        arguments={'value': value, **kwargs})])


def final(text='The note was saved.'):
    return LLMChatWithToolsResult(content=text)


def test_normal_execution_and_roles(tmp_path):
    runtime, calls = make_runtime(tmp_path, model=sequence(call(), final()))
    result = runtime.start({'message': 'Write note one', 'thread_id': 'test'})
    assert calls == [{'value': 1}], result
    assert result['state']['run_id']
    snapshots = result['report']['llm_context_snapshots']
    assert snapshots
    assert 'Write note one' in json.dumps(snapshots)


def test_invalid_parameters_never_execute(tmp_path):
    runtime, calls = make_runtime(tmp_path, model=sequence(call('wrong'), final('Invalid parameter.')))
    runtime.start({'message': 'Write a note'})
    assert not calls


def test_no_model_is_explicit(tmp_path):
    runtime, calls = make_runtime(tmp_path)
    result = runtime.start({'message': 'Hello'})
    assert result['state']['errors'][0]['type'] == 'model_not_configured'
    assert not calls


def test_explicit_host_configuration_context_hook(tmp_path):
    captured = []
    def model(config, messages, tools):
        captured.extend(messages)
        return final('Acknowledged')
    runtime, _ = make_runtime(tmp_path, model=model)
    runtime.services.registry.config_context_builder = lambda config: 'Host context: fixture-defaults-v1'
    runtime.start({'message': 'Read current configuration'})
    assert any(message.get('role') == 'system' and message.get('content') ==
               'Host context: fixture-defaults-v1' for message in captured)


def test_cancel_does_not_rewrite_completed_history(tmp_path):
    runtime, calls = make_runtime(tmp_path, model=sequence(final('Hello')))
    result = runtime.start({'message': 'Hello'})
    assert runtime.cancel(result['state']['run_id']) == result


def test_unknown_tool_never_executes(tmp_path):
    model = sequence(LLMChatWithToolsResult(tool_calls=[LLMToolCall(id='bad', name='unknown__tool', arguments={})]), final())
    runtime, calls = make_runtime(tmp_path, model=model)
    runtime.start({'message': 'Write a note'})
    assert not calls


def test_hidden_reasoning_not_persisted(tmp_path):
    model = sequence(LLMChatWithToolsResult(content='Hello', raw_message={
        'role': 'assistant', 'content': 'Hello', 'reasoning_content': 'hidden-test-value'}))
    runtime, calls = make_runtime(tmp_path, model=model)
    result = runtime.start({'message': 'Hello'})
    assert 'hidden-test-value' not in json.dumps(result)
    assert all('hidden-test-value' not in p.read_text(encoding='utf-8') for p in (tmp_path / 'reports').glob('*.json'))


def test_approve_once_no_replay(tmp_path):
    import pytest
    runtime, calls = make_runtime(tmp_path, model=sequence(call(), final()), config={'execution_mode': 'local_write'})
    result = runtime.start({'message': 'Write note one'})
    pending = result['state']['pending_interaction']
    assert pending['type'] == 'confirmation', result
    run_id = result['state']['run_id']
    assert not calls
    runtime.resume(run_id, pending['interaction_id'], {'decision': 'approve'})
    assert calls == [{'value': 1}]
    with pytest.raises(ValueError):
        runtime.resume(run_id, pending['interaction_id'], {'decision': 'approve'})
    assert len(calls) == 1


def test_old_approval_cannot_authorize_reused_model_call_id(tmp_path):
    import pytest
    runtime, calls = make_runtime(tmp_path, model=sequence(call(1), call(9), final()),
                                  config={'execution_mode': 'local_write'})
    first = runtime.start({'message': 'Write both notes'})
    run_id = first['state']['run_id']
    old_id = first['state']['pending_interaction']['interaction_id']
    second = runtime.resume(run_id, old_id, {'decision': 'approve'})
    new_id = second['state']['pending_interaction']['interaction_id']
    assert new_id != old_id
    assert calls == [{'value': 1}]
    with pytest.raises(ValueError, match='interaction_mismatch'):
        runtime.resume(run_id, old_id, {'decision': 'approve'})
    assert calls == [{'value': 1}]
    runtime.resume(run_id, new_id, {'decision': 'approve'})
    assert calls == [{'value': 1}, {'value': 9}]


def test_cancel_after_resume_keeps_executed_evidence(tmp_path):
    holder, rounds = {}, []
    def model(config, messages, tools):
        rounds.append(1)
        if len(rounds) == 1:
            return call()
        holder['runtime'].cancel('cancel-resumed')
        return final()
    runtime, calls = make_runtime(tmp_path, model=model, config={'execution_mode': 'local_write'})
    holder['runtime'] = runtime
    first = runtime.start({'message': 'Write note', 'run_id': 'cancel-resumed'})
    result = runtime.resume('cancel-resumed', first['state']['pending_interaction']['interaction_id'],
                            {'decision': 'approve'})
    assert calls == [{'value': 1}]
    assert result['state']['execution_status'] == 'stopped'
    assert result['state']['tool_calls']
    assert result['state']['observations']
    assert any(e['type'] == 'run_cancelled' for e in result['state']['errors'])
    assert not any(e['type'] == 'interaction_resume_failed' for e in result['state']['errors'])


def test_parameter_judge_nonfinite_scores_are_uncertain():
    from agent_workbench.eval_judge import CallableEvalJudge
    for score in (float('nan'), float('inf'), -float('inf'), True):
        result = CallableEvalJudge(lambda payload: {'decision': 'aligned', 'score': score}).judge_parameter_intent({})
        assert result.decision == 'uncertain'
        assert result.score is None


def test_cancelled_partial_tool_exchange_does_not_break_next_turn():
    from agent_workbench.runtime import conversation_history
    messages = [{'role': 'user', 'content': 'Create two notes'},
                {'role': 'assistant', 'tool_calls': [{'id': 'done'}, {'id': 'cancelled'}]},
                {'role': 'tool', 'tool_call_id': 'done', 'content': 'Saved'}]
    cleaned = conversation_history(messages)
    assert cleaned[1]['tool_calls'] == [{'id': 'done'}]
    assert messages[1]['tool_calls'] == [{'id': 'done'}, {'id': 'cancelled'}]
    assert conversation_history(messages[:2]) == messages[:1]


def test_rejection_and_tampering(tmp_path):
    import pytest
    runtime, calls = make_runtime(tmp_path, model=sequence(call(), final('Declined.')), config={'execution_mode': 'local_write'})
    result = runtime.start({'message': 'Write note one'})
    pending = result['state']['pending_interaction']
    run_id = result['state']['run_id']
    with pytest.raises(ValueError):
        runtime.resume(run_id, pending['interaction_id'], {'decision': 'approve', 'params': {'value': 9}})
    assert not calls
    runtime.resume(run_id, pending['interaction_id'], {'decision': 'reject'})
    assert not calls


def test_cancel_paused_and_restart_not_resumable(tmp_path):
    import pytest
    runtime, calls = make_runtime(tmp_path, model=sequence(call()), config={'execution_mode': 'local_write'})
    result = runtime.start({'message': 'Write a note'})
    run_id = result['state']['run_id']
    pending = result['state']['pending_interaction']
    second = AgentRuntime(runtime.services, data_dir=tmp_path)
    assert second.get_run(run_id)['report']['restart_required'] is True
    with pytest.raises(ValueError):
        second.resume(run_id, pending['interaction_id'], {'decision': 'approve'})
    runtime.cancel(run_id)
    with pytest.raises(ValueError):
        runtime.resume(run_id, pending['interaction_id'], {'decision': 'approve'})
    assert not calls


def test_multi_turn_context_without_replaying_tools(tmp_path):
    captured = []
    def model(config, messages, tools):
        captured.append(messages)
        return final('Acknowledged')
    runtime, calls = make_runtime(tmp_path, model=model)
    runtime.start({'message': 'Use the blue folder.', 'thread_id': 'same'})
    runtime.start({'message': 'Correction: use the red folder.', 'thread_id': 'same'})
    users = [m['content'] for m in captured[-1] if m['role'] == 'user']
    assert users == ['Use the blue folder.', 'Correction: use the red folder.']
    runtime.start({'message': 'Start fresh.', 'thread_id': 'other'})
    assert 'blue folder' not in json.dumps(captured[-1])
    assert not calls


def test_round_limit_and_cooperative_cancel(tmp_path):
    runtime, calls = make_runtime(tmp_path, model=lambda *args: call(), config={'max_tool_rounds': 2})
    result = runtime.start({'message': 'Write note'})
    assert len(calls) <= 2
    assert any(e['type'] == 'max_rounds' for e in result['state']['errors'])
    holder = {}
    def canceller(config, messages, tools):
        holder['runtime'].cancel('cancel-active')
        return call()
    runtime, calls = make_runtime(tmp_path / 'cancel', model=canceller)
    holder['runtime'] = runtime
    result = runtime.start({'message': 'Write note', 'run_id': 'cancel-active'})
    assert not calls
    assert any(e['type'] == 'run_cancelled' for e in result['state']['errors'])
