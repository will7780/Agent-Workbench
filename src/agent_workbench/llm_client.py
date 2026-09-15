# -*- coding: utf-8 -*-
"""Explicitly configured compatible chat client with injectable offline transport."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from .central_env import load_central_agent_env
from .redaction import redact_recursive
from .runtime_telemetry import LLMUsage

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_CHAT_COMPLETIONS_PATH = "/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_PROVIDER = "deepseek"

CompleteFn = Callable[["LLMClientConfig", List[Dict[str, str]]], "LLMChatResult"]
ToolsCompleteFn = Callable[
    ["LLMClientConfig", List[Dict[str, Any]], Optional[List[Dict[str, Any]]]],
    "LLMChatWithToolsResult",
]


@dataclass
class LLMClientConfig:
    api_key: Optional[str]
    model: str
    provider: str
    base_url: str
    chat_completions_path: str


def build_chat_completions_url(base_url: str, path: str) -> str:
    """稳健拼接 chat completions URL（base 末尾 / 与 path 开头 / 均归一化）。"""
    base = (base_url or "").rstrip("/")
    normalized_path = "/" + (path or "").strip("/")
    return f"{base}{normalized_path}"


@dataclass
class LLMChatResult:
    content: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    prompt_summary: Optional[str] = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.error_type is None and bool(self.content)


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class LLMChatWithToolsResult:
    content: Optional[str] = None
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    model: Optional[str] = None
    provider: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    prompt_summary: Optional[str] = None
    finish_reason: Optional[str] = None
    raw_message: Optional[Dict[str, Any]] = None

    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: Optional[float] = None
    @property
    def ok(self) -> bool:
        return self.error_type is None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def parse_tool_call_arguments(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _parse_tool_calls_from_message(message: Dict[str, Any]) -> List[LLMToolCall]:
    calls: List[LLMToolCall] = []
    for item in message.get("tool_calls") or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        call_id = str(item.get("id") or f"call_{len(calls)}")
        arguments = parse_tool_call_arguments(fn.get("arguments"))
        calls.append(LLMToolCall(id=call_id, name=name, arguments=arguments))
    return calls


def load_llm_config(*, model_override: Optional[str] = None, load_credentials: bool = True) -> LLMClientConfig:
    if load_credentials:
        load_central_agent_env()
    return LLMClientConfig(
        api_key=os.environ.get("DEEPSEEK_API_KEY") if load_credentials else None,
        model=(
            model_override
            or os.environ.get("AGENT_WORKBENCH_LLM_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
            or DEFAULT_MODEL
        ),
        provider=os.environ.get("AGENT_WORKBENCH_LLM_PROVIDER", DEFAULT_PROVIDER),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        chat_completions_path=os.environ.get(
            "DEEPSEEK_CHAT_COMPLETIONS_PATH",
            DEFAULT_CHAT_COMPLETIONS_PATH,
        ),
    )


def _default_complete(config: LLMClientConfig, messages: List[Dict[str, str]]) -> LLMChatResult:
    url = build_chat_completions_url(config.base_url, config.chat_completions_path)
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90, allow_redirects=False)
    except requests.RequestException as exc:
        return LLMChatResult(
            model=config.model,
            provider=config.provider,
            error_type="llm_network_error",
            error_message="Provider request failed",
        )

    if response.status_code >= 300:
        return LLMChatResult(
            model=config.model,
            provider=config.provider,
            error_type="llm_api_error",
            error_message=f"Provider HTTP {response.status_code}",
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = LLMUsage.from_api_payload(data.get("usage"))
    except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
        return LLMChatResult(
            model=config.model,
            provider=config.provider,
            error_type="llm_parse_error",
            error_message=f"invalid API response: {exc}",
        )

    return LLMChatResult(
        content=content,
        model=config.model,
        provider=config.provider,
        usage=usage,
    )


def chat_completion_json(
    messages: List[Dict[str, str]],
    *,
    config: Optional[LLMClientConfig] = None,
    model_override: Optional[str] = None,
    complete_fn: Optional[CompleteFn] = None,
) -> LLMChatResult:
    """
    调用 DeepSeek chat completion，期望返回 JSON 文本。

    缺少 API key 时返回结构化错误，不抛异常。
    """
    resolved = config or load_llm_config(model_override=model_override, load_credentials=complete_fn is None)
    if not resolved.api_key and complete_fn is None:
        return LLMChatResult(
            model=resolved.model,
            provider=resolved.provider,
            error_type="llm_unavailable",
            error_message="DEEPSEEK_API_KEY is not set; LLM planner fallback is unavailable.",
        )

    user_preview = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_preview = str(redact_recursive(msg.get("content") or "")[0])[:120]
            break

    runner = complete_fn or _default_complete
    started = time.perf_counter()
    result = runner(resolved, messages)
    if result.latency_ms is None:
        result.latency_ms = round(max(0.0, time.perf_counter() - started) * 1000.0, 3)
    if result.prompt_summary is None:
        result.prompt_summary = user_preview
    if result.model is None:
        result.model = resolved.model
    if result.provider is None:
        result.provider = resolved.provider
    return result


def _default_complete_with_tools(
    config: LLMClientConfig,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> LLMChatWithToolsResult:
    url = build_chat_completions_url(config.base_url, config.chat_completions_path)
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90, allow_redirects=False)
    except requests.RequestException as exc:
        return LLMChatWithToolsResult(
            model=config.model,
            provider=config.provider,
            error_type="llm_network_error",
            error_message="Provider request failed",
        )

    if response.status_code >= 300:
        return LLMChatWithToolsResult(
            model=config.model,
            provider=config.provider,
            error_type="llm_api_error",
            error_message=f"Provider HTTP {response.status_code}",
        )

    try:
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason")
        tool_calls = _parse_tool_calls_from_message(message)
        usage = LLMUsage.from_api_payload(data.get("usage"))
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
    except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
        return LLMChatWithToolsResult(
            model=config.model,
            provider=config.provider,
            error_type="llm_parse_error",
            error_message=f"invalid API response: {exc}",
        )

    return LLMChatWithToolsResult(
        content=content,
        tool_calls=tool_calls,
        model=config.model,
        provider=config.provider,
        finish_reason=finish_reason,
        raw_message={key: message[key] for key in ("role", "content", "tool_calls") if key in message},
        usage=usage,
    )


def chat_completion_with_tools(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    *,
    config: Optional[LLMClientConfig] = None,
    model_override: Optional[str] = None,
    complete_fn: Optional[ToolsCompleteFn] = None,
) -> LLMChatWithToolsResult:
    """
    调用 OpenAI-compatible chat completion，支持 tools / tool_calls。

    缺少 API key 时返回结构化错误，不抛异常。不影响 chat_completion_json。
    """
    resolved = config or load_llm_config(model_override=model_override, load_credentials=complete_fn is None)
    if not resolved.api_key and not complete_fn:
        return LLMChatWithToolsResult(
            model=resolved.model,
            provider=resolved.provider,
            error_type="llm_unavailable",
            error_message="DEEPSEEK_API_KEY is not set; tool-calling agent is unavailable.",
        )

    user_preview = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_preview = str(redact_recursive(msg.get("content") or "")[0])[:120]
            break

    runner = complete_fn or _default_complete_with_tools
    started = time.perf_counter()
    result = runner(resolved, messages, tools)
    if result.latency_ms is None:
        result.latency_ms = round(max(0.0, time.perf_counter() - started) * 1000.0, 3)
    if result.prompt_summary is None:
        result.prompt_summary = user_preview
    if result.model is None:
        result.model = resolved.model
    if result.provider is None:
        result.provider = resolved.provider
    return result
