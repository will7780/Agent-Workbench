import json
from agent_workbench.langgraph_context_compact import (
    _compact_observation_payload, compact_messages_for_llm, ContextCompactOptions)


def test_review_feedback_survives_repeated_compaction():
    observation = {'summary': 'x' * 1000, 'artifacts': [{
        'type': 'artifact.review', 'decision': 'request_changes', 'comment': 'Use shorter sentences.',
        'version': 2, 'review_id': 'review-2', 'tool_call_id': 'call-2'}]}
    compact = _compact_observation_payload(observation, tool_call_id='call-2')
    twice = _compact_observation_payload(compact, tool_call_id='call-2')
    assert twice['artifact_refs'][0]['comment'] == 'Use shorter sentences.'
    assert twice['artifact_refs'][0]['decision'] == 'request_changes'


def test_spill_filename_not_controlled_by_model(tmp_path):
    call_id = '../escape/../../test'
    messages = [{'role': 'assistant', 'tool_calls': [{'id': call_id, 'type': 'function',
                 'function': {'name': 'notes__read', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': call_id,
                 'content': json.dumps({'summary': 'x' * 2000, 'status': 'success'})}]
    compact, budget = compact_messages_for_llm(messages, options=ContextCompactOptions(spill_dir=tmp_path))
    assert list(tmp_path.glob('round*.json'))
    assert all(p.parent == tmp_path for p in tmp_path.rglob('*') if p.is_file())


def test_long_trace_preserved_and_explicit_truncation():
    from agent_workbench.redaction import redact_recursive
    trace = [{'type': 'event', 'index': index} for index in range(120)]
    safe, _ = redact_recursive({'events': trace})
    assert len(safe['events']) == 120
    short, _ = redact_recursive(trace, max_list_items=2)
    assert short[-1] == {'_truncated': True, 'omitted_count': 118}
    safe, _ = redact_recursive({'reasoning_content': 'private reasoning', 'reasoning_tokens': 12})
    assert safe == {'reasoning_tokens': 12}
