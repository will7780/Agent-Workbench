"""Synthetic-only security regressions for detached output redaction."""

import copy
import json
import socket

import pytest

from agent_workbench.redaction import (
    REDACTED, contains_secret_blob, redact_recursive, redact_scalar, redact_text,
)


@pytest.fixture(autouse=True)
def synthetic_only(monkeypatch):
    from agent_workbench import central_env, llm_client

    def forbidden(*args, **kwargs):
        raise AssertionError("Credential loading and network access are forbidden")

    monkeypatch.setattr(central_env, "load_central_agent_env", forbidden)
    monkeypatch.setattr(llm_client, "load_central_agent_env", forbidden)
    for method in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, method, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for flag in ("MEMORY_ENABLED", "MEMORY_REVIEW_ENABLED", "SESSION_LEDGER_ENABLED"):
        monkeypatch.setenv("AGENT_WORKBENCH_" + flag, "0")


def test_json_strings_are_structured_redacted_and_detached():
    marker = "SYNTHETIC_JSON_VALUE"
    params = {"token": marker, "items": [{"password": marker, "count": 3}],
              "nested": json.dumps({"authorization": marker, "value": 7}),
              "total_tokens": 12, "reasoning_tokens": None}
    raw = json.dumps(params, indent=2)
    source = {"arguments": raw, "params": params}
    before = copy.deepcopy(source)
    safe, changed = redact_recursive(source)
    assert changed and source == before
    assert marker not in json.dumps(safe)
    decoded = json.loads(safe["arguments"])
    assert decoded["token"] == REDACTED
    assert decoded["items"] == [{"password": REDACTED, "count": 3}]
    assert json.loads(decoded["nested"]) == {"authorization": REDACTED, "value": 7}
    assert decoded["total_tokens"] == 12 and decoded["reasoning_tokens"] is None
    safe["params"]["items"][0]["count"] = 99
    assert source == before
    assert json.loads(redact_recursive(raw)[0]) == decoded
    assert json.loads(redact_scalar("arguments", raw)[0]) == decoded
    assert redact_text(raw) == redact_recursive(raw)
    assert redact_recursive(safe)[0] == safe


@pytest.mark.parametrize("raw", [' { "value" : 7, "label" : "\\u4e2d" } ',
                                 '[1, {"total_tokens": 3}]', '[history snipped]',
                                 '{not a JSON object', 'ordinary text'])
def test_non_sensitive_strings_are_preserved_exactly(raw):
    assert redact_recursive(raw) == (raw, False)


@pytest.mark.parametrize("key", ["token", "analysis"])
def test_duplicate_keys_do_not_preserve_discarded_sensitive_content(key):
    raw = '{"value": {' + json.dumps(key) + ': "SYNTHETIC_DUPLICATE_VALUE"}, "value": 7}'
    safe, changed = redact_recursive(raw)
    assert changed and json.loads(safe) == {"value": 7}
    assert "SYNTHETIC_DUPLICATE_VALUE" not in safe


def test_escaped_json_field_names_are_checked_after_decoding():
    raw = r'{"to\u006ben": "SYNTHETIC_ESCAPED_VALUE", "anal\u0079sis": "SYNTHETIC_ESCAPED_VALUE"}'
    safe, changed = redact_recursive(raw)
    assert changed and json.loads(safe) == {"token": REDACTED}


@pytest.mark.parametrize("text", [
    'api_key=SYNTHETIC_FIRST token=SYNTHETIC_SECOND',
    'password="SYNTHETIC_FIRST with spaces"; token=SYNTHETIC_SECOND',
    '[REDACTED] token = SYNTHETIC_SECOND; Authorization: Basic SYNTHETIC_FIRST',
    'TOKEN=SYNTHETIC_SECOND, api-key=SYNTHETIC_FIRST',
    '{"api_key": "SYNTHETIC_FIRST", "token": "SYNTHETIC_SECOND", "value": 7}',
])
def test_every_credential_in_text_is_redacted(text):
    safe, changed = redact_text(text)
    assert changed
    assert "SYNTHETIC_FIRST" not in safe and "SYNTHETIC_SECOND" not in safe
    assert redact_text(safe)[0] == safe
    assert contains_secret_blob(text)
    assert not contains_secret_blob(safe)
    if text.startswith("{"):
        assert json.loads(safe)["value"] == 7


@pytest.mark.parametrize("text", [
    'api_key=SYNTHETIC_FIRST token=SYNTHETIC_SECOND',
    json.dumps({"api_key": "SYNTHETIC_FIRST", "nested": [{"token": "SYNTHETIC_SECOND"}]}),
])
def test_display_and_export_boundaries_remove_all_markers(text):
    from agent_workbench.runtime import public_payload
    from agent_workbench.trace_export import export_trace
    from agent_workbench.web.service import public_data

    result = {"state": {"run_id": "synthetic-redaction", "final_response": text}}
    original = copy.deepcopy(result)
    for safe in (public_payload(result), public_data(result), export_trace(result)):
        encoded = json.dumps(safe)
        assert "SYNTHETIC_FIRST" not in encoded and "SYNTHETIC_SECOND" not in encoded
    assert result == original


@pytest.mark.parametrize("key", ["analysis", "reasoning", "scratchpad", "reasoning_content",
                                 "chain_of_thought", "hidden_reasoning", "thinking", "ANALYSIS"])
def test_hidden_fields_removed_before_context_spill(tmp_path, key):
    from agent_workbench.langgraph_context_compact import ContextCompactOptions, compact_messages_for_llm
    from agent_workbench.runtime import public_payload
    from agent_workbench.web.service import public_data

    marker = "SYNTHETIC_HIDDEN_VALUE"
    payload = {"summary": "x" * 2000, key: marker,
               "details": [{key: marker, "reasoning_tokens": 12}], "analysis_summary": "Public summary"}
    raw = json.dumps(payload)
    messages = [{"role": "tool", "tool_call_id": "synthetic-call", "content": raw}]
    before = copy.deepcopy(messages)
    for safe in (redact_recursive(payload)[0], redact_recursive(raw)[0],
                 public_payload(messages), public_data(messages)):
        assert marker not in json.dumps(safe)
    compact, budget = compact_messages_for_llm(messages, options=ContextCompactOptions(spill_dir=tmp_path))
    assert messages == before and budget["artifacts_spilled"] == 1
    assert marker not in json.dumps(compact)
    spilled = list(tmp_path.glob("*.json"))
    assert len(spilled) == 1
    stored = json.loads(spilled[0].read_text(encoding="utf-8"))
    assert marker not in json.dumps(stored)
    assert stored["details"] == [{"reasoning_tokens": 12}]
    assert stored["analysis_summary"] == "Public summary"


def test_serialized_containers_obey_depth_and_item_limits():
    marker = "SYNTHETIC_DEEP_VALUE"
    nested = {"token": marker}
    for _ in range(6):
        nested = {"payload": json.dumps(nested)}
    safe, _ = redact_recursive(nested, max_depth=3)
    assert marker not in json.dumps(safe)
    assert "_truncated" in json.dumps(safe) or REDACTED in json.dumps(safe)
    safe, _ = redact_recursive(json.dumps([{"value": 1}, {"token": marker}]), max_list_items=1)
    assert json.loads(safe) == [{"value": 1}, {"_truncated": True, "omitted_count": 1}]
    too_deep = "[" * 2000 + json.dumps({"token": marker}) + "]" * 2000
    safe, changed = redact_recursive(too_deep)
    assert changed and marker not in safe


def _json_encode(value, layers):
    for _ in range(layers):
        value = json.dumps(value)
    return value


@pytest.mark.parametrize("layers", [1, 2, 3, 6])
def test_multiple_json_encodings_preserve_shape_and_source(layers):
    marker = "SYNTHETIC_ENCODED_VALUE"
    payload = {"token": marker, "analysis": marker,
               "items": [{"password": marker, "value": 7}], "reasoning_tokens": 12}
    raw = _json_encode(payload, layers)
    source = {"arguments": raw, "original": payload}
    before = copy.deepcopy(source)
    safe, changed = redact_recursive(source)
    assert changed and source == before
    assert marker not in json.dumps(safe)
    decoded = safe["arguments"]
    for _ in range(layers):
        decoded = json.loads(decoded)
    assert decoded == {"token": REDACTED, "items": [{"password": REDACTED, "value": 7}],
                       "reasoning_tokens": 12}
    assert redact_recursive(safe)[0] == safe
    harmless = _json_encode({"value": 7}, layers)
    assert redact_recursive(harmless) == (harmless, False)
    assert redact_text(raw) == redact_recursive(raw)
    assert redact_text(harmless) == (harmless, False)


@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_encoded_string_layers_exhaust_shared_depth_budget(limit):
    raw = _json_encode({"token": "SYNTHETIC_LIMIT_VALUE"}, 6)
    source = {"arguments": raw}
    before = copy.deepcopy(source)
    for safe in (redact_recursive(raw, max_depth=limit)[0],
                 redact_recursive(source, max_depth=limit)[0],
                 redact_scalar("arguments", raw, max_depth=limit)[0]):
        assert "SYNTHETIC_LIMIT_VALUE" not in json.dumps(safe)
        assert REDACTED in json.dumps(safe) or "_truncated" in json.dumps(safe)
    assert source == before
    limited, changed = redact_recursive(_json_encode([1, 2], 3), max_list_items=1)
    assert changed
    for _ in range(3):
        limited = json.loads(limited)
    assert limited == [1, {"_truncated": True, "omitted_count": 1}]


@pytest.mark.parametrize("layers", [2, 3, 6])
def test_encoded_json_inside_report_prose_is_redacted(layers):
    raw = _json_encode({"token": "SYNTHETIC_PROSE_VALUE", "value": 7}, layers)
    text = "Report summary: " + raw + "\nPublic footer."
    safe, changed = redact_recursive(text)
    assert changed and "SYNTHETIC_PROSE_VALUE" not in safe
    assert safe.startswith("Report summary: ") and safe.endswith("\nPublic footer.")
    assert redact_recursive(text, max_depth=1)[0].endswith("\nPublic footer.")
    assert "SYNTHETIC_PROSE_VALUE" not in redact_recursive(text, max_depth=1)[0]
    assert redact_text(text) == redact_recursive(text)


@pytest.mark.parametrize("raw", [
    _json_encode({"token": "SYNTHETIC_INVALID_VALUE", "analysis": "private"}, 6)[:240],
    _json_encode({"analysis": "SYNTHETIC_INVALID_VALUE"}, 3)[:-1],
    '{"token": "SYNTHETIC_INVALID_VALUE", broken}',
    '[{"analysis": "SYNTHETIC_INVALID_VALUE"}',
    '"token": "SYNTHETIC_INVALID_VALUE"',
    '"analysis": "SYNTHETIC_INVALID_VALUE"',
    r'{"to\u006ben": "SYNTHETIC_INVALID_VALUE", broken}',
])
def test_invalid_or_truncated_sensitive_json_fails_closed(raw):
    assert "SYNTHETIC_INVALID_VALUE" in raw
    source = {"preview": raw}
    before = copy.deepcopy(source)
    assert redact_text(raw) == (REDACTED, True)
    assert redact_recursive(raw) == (REDACTED, True)
    assert redact_recursive(source) == ({"preview": REDACTED}, True)
    assert source == before


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("header", ["Cookie", "cOoKiE", "Set-Cookie"])
def test_whole_cookie_header_value_is_removed_without_touching_other_headers(newline, header):
    prefix = "Accept: application/json" + newline
    suffix = newline + "X-Trace-ID: public-fixture" + newline + "Content-Type: text/plain"
    text = prefix + "\t" + header + ': first="SYNTHETIC_COOKIE_A"; second=SYNTHETIC_COOKIE_B; Path=/' + suffix
    safe, changed = redact_text(text)
    assert changed and safe == prefix + "\t" + header + ": " + REDACTED + suffix
    assert contains_secret_blob(text) and not contains_secret_blob(safe)
    assert redact_text(safe)[0] == safe
    encoded = _json_encode(text, 3)
    assert "SYNTHETIC_COOKIE" not in redact_recursive(encoded)[0]
    assert redact_text(encoded) == redact_recursive(encoded)
    decoded = redact_text(encoded)[0]
    for _ in range(3):
        decoded = json.loads(decoded)
    assert decoded == safe


def test_empty_and_partially_redacted_cookie_headers():
    text = "Cookie:\r\nAccept: application/json\r\nCookie: [REDACTED]; second=SYNTHETIC_COOKIE_VALUE\r\nX-Public: yes"
    safe, changed = redact_text(text)
    assert changed
    assert safe == "Cookie:\r\nAccept: application/json\r\nCookie: [REDACTED]\r\nX-Public: yes"
    assert redact_text("Cookie:\nAccept: application/json") == ("Cookie:\nAccept: application/json", False)


@pytest.mark.parametrize("case", ["double_json", "multiple_json", "cookies", "encoded_cookies"])
def test_remaining_blockers_do_not_leak_in_runtime_reports_or_exports(tmp_path, case):
    from agent_workbench.llm_client import LLMChatWithToolsResult
    from agent_workbench.registry import ToolRegistry
    from agent_workbench.runtime import AgentRuntime, RuntimeServices
    from agent_workbench.tools.execution_adapters import RegisteredToolExecutor
    from agent_workbench.trace_export import export_trace
    from agent_workbench.web.service import project_result, public_data

    marker = "SYNTHETIC_FREEZE_VALUE"
    headers = "Accept: application/json\r\nCookie: first=fixture; second=" + marker + "\r\nX-Public: yes"
    text = {"double_json": _json_encode({"token": marker}, 2),
            "multiple_json": _json_encode({"token": marker, "analysis": marker}, 6),
            "cookies": headers, "encoded_cookies": _json_encode(headers, 3)}[case]
    model_result = LLMChatWithToolsResult(content=text)

    def model(config, messages, tools):
        assert config.api_key is None
        return model_result

    runtime = AgentRuntime(RuntimeServices(ToolRegistry(payload={"modules": {}}),
                                           RegisteredToolExecutor(), model_client=model), data_dir=tmp_path)
    result = runtime.start({"message": "Run the synthetic output fixture."})
    assert result["state"]["execution_status"] == "completed"
    assert not result["state"].get("tool_calls") and model_result.content == text
    for safe in (result, runtime.get_run(result["state"]["run_id"]), project_result(result), export_trace(result),
                 public_data({"state": {"run_id": "synthetic-direct", "final_response": text}})):
        assert marker not in json.dumps(safe)
    reports = list((tmp_path / "reports").glob("*.json"))
    assert reports and all(marker not in p.read_text(encoding="utf-8") for p in reports)


@pytest.mark.parametrize("raw_message", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_runtime_redacts_outputs_without_changing_model_or_execution_parameters(tmp_path, raw_message, valid):
    from agent_workbench.configuration import StaticConfigProvider
    from agent_workbench.llm_client import LLMChatWithToolsResult, LLMToolCall
    from agent_workbench.registry import ToolRegistry
    from agent_workbench.runtime import AgentRuntime, RuntimeServices
    from agent_workbench.tools.adapter_contracts import AdapterOutput
    from agent_workbench.tools.execution_adapters import RegisteredToolExecutor
    from agent_workbench.trace_export import export_trace
    from agent_workbench.web.service import project_result

    marker = "SYNTHETIC_EXECUTION_VALUE"
    params = {"value": 7 if valid else "invalid", "token": marker,
              "details": _json_encode({"password": marker}, 3), "analysis": marker,
              "headers": "Cookie: first=fixture; second=" + marker + "\r\nAccept: application/json"}
    original = copy.deepcopy(params)
    function = {"name": "notes__write", "arguments": json.dumps(params)}
    raw = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "synthetic-call", "type": "function", "function": function}]}
    responses = iter([
        LLMChatWithToolsResult(tool_calls=[LLMToolCall(id="synthetic-call", name="notes__write", arguments=params)],
                               raw_message=raw if raw_message else None),
        LLMChatWithToolsResult(content="Synthetic run finished."),
    ])
    model_inputs, executed, events = [], [], []

    def model(config, messages, tools):
        assert config.api_key is None
        declared = next(tool["function"]["parameters"] for tool in tools
                        if tool["function"]["name"] == "notes__write")
        assert declared["properties"]["token"]["type"] == "string"
        assert declared["properties"]["analysis"]["type"] == "string"
        model_inputs.append(copy.deepcopy(messages))
        return next(responses)

    def execute(request):
        executed.append(copy.deepcopy(request.params))
        return AdapterOutput(status="success", summary="Synthetic adapter finished.")

    registry = ToolRegistry(payload={"modules": {"notes": {"actions": {"write": {
        "risk_level": "L2", "execution_adapter": "synthetic-write", "local_write_status": "available",
        "parameter_contract": {"type": "object", "required": list(params), "additionalProperties": False,
                               "properties": {"value": {"type": "integer"}, "token": {"type": "string"},
                                              "details": {"type": "string"}, "analysis": {"type": "string"},
                                              "headers": {"type": "string"}}},
    }}}}})
    executor = RegisteredToolExecutor()
    executor.register("synthetic-write", execute, modes=("local_write",))
    runtime = AgentRuntime(RuntimeServices(
        registry, executor, model_client=model, event_sink=events.append,
        config_provider=StaticConfigProvider({"execution_mode": "local_write"}),
    ), data_dir=tmp_path)
    result = runtime.start({"message": "Perform the synthetic fixture operation."})
    if valid:
        assert not executed
        assert marker not in json.dumps(result)
        pending = result["state"]["pending_interaction"]
        result = runtime.resume(result["state"]["run_id"], pending["interaction_id"], {"decision": "approve"})
    assert executed == ([original] if valid else [])
    assert params == original and json.loads(function["arguments"]) == original
    actual_calls = [call for message in model_inputs[-1] for call in message.get("tool_calls", [])]
    assert len(actual_calls) == 1
    assert json.loads(actual_calls[0]["function"]["arguments"]) == original
    for safe in (result, events, project_result(result), export_trace(result)):
        assert marker not in json.dumps(safe)
    reports = list((tmp_path / "reports").glob("*.json"))
    assert reports and all(marker not in p.read_text(encoding="utf-8") for p in reports)
