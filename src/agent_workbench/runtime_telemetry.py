# -*- coding: utf-8 -*-
"""Auditable runtime timing, token usage, and exact price-card accounting."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .redaction import redact_recursive


def _nullable_non_negative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


@dataclass
class LLMUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cache_hit_tokens: Optional[int] = None
    cache_miss_tokens: Optional[int] = None

    @classmethod
    def from_api_payload(cls, payload: Any) -> "LLMUsage":
        raw = payload if isinstance(payload, Mapping) else {}
        completion_details = raw.get("completion_tokens_details")
        completion_details = completion_details if isinstance(completion_details, Mapping) else {}
        prompt_details = raw.get("prompt_tokens_details")
        prompt_details = prompt_details if isinstance(prompt_details, Mapping) else {}
        prompt = _nullable_non_negative_int(raw.get("prompt_tokens"))
        completion = _nullable_non_negative_int(raw.get("completion_tokens"))
        total = _nullable_non_negative_int(raw.get("total_tokens"))
        if total is None and prompt is not None and completion is not None:
            total = prompt + completion
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=_nullable_non_negative_int(
                raw.get("reasoning_tokens", completion_details.get("reasoning_tokens"))
            ),
            cache_hit_tokens=_nullable_non_negative_int(
                raw.get("cache_hit_tokens", raw.get("prompt_cache_hit_tokens", prompt_details.get("cached_tokens")))
            ),
            cache_miss_tokens=_nullable_non_negative_int(
                raw.get("cache_miss_tokens", raw.get("prompt_cache_miss_tokens"))
            ),
        )

    def to_dict(self) -> Dict[str, Optional[int]]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
        }


def normalize_llm_usage(value: Any) -> LLMUsage:
    return value if isinstance(value, LLMUsage) else LLMUsage.from_api_payload(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_price_card_catalog(path: Optional[Path] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    configured = path
    if configured is None:
        raw = os.environ.get("AGENT_WORKBENCH_LLM_PRICE_CARDS_FILE", "").strip()
        configured = Path(raw) if raw else None
    if configured is None:
        return None, "price_card_not_configured"
    try:
        payload = json.loads(Path(configured).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None, "price_card_unavailable"
    if not isinstance(payload, dict) or not isinstance(payload.get("cards"), list):
        return None, "price_card_invalid"
    return payload, None


def _matching_price_card(
    catalog: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    occurred_at: str,
) -> Optional[Dict[str, Any]]:
    instant = _iso_datetime(occurred_at)
    if instant is None:
        return None
    matches: List[Dict[str, Any]] = []
    for item in catalog.get("cards") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("provider") or "") != provider or str(item.get("model") or "") != model:
            continue
        start = _iso_datetime(item.get("effective_from"))
        end = _iso_datetime(item.get("effective_to")) if item.get("effective_to") else None
        if start is None or instant < start or (end is not None and instant >= end):
            continue
        if isinstance(item.get("rates_per_million"), dict) and item["rates_per_million"]:
            matches.append(item)
    return matches[0] if len(matches) == 1 else None


def estimate_llm_cost(
    llm_calls: List[Dict[str, Any]],
    *,
    price_card_path: Optional[Path] = None,
    price_card_catalog: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    catalog: Optional[Mapping[str, Any]] = price_card_catalog
    error_type: Optional[str] = None
    if catalog is None:
        loaded, error_type = load_price_card_catalog(price_card_path)
        catalog = loaded
    if catalog is None:
        return {
            "estimated_cost": None,
            "currency": None,
            "price_card_version": None,
            "cost_status": error_type or "price_card_unavailable",
        }
    total_cost = 0.0
    currency: Optional[str] = None
    for call in llm_calls:
        card = _matching_price_card(
            catalog,
            provider=str(call.get("provider") or ""),
            model=str(call.get("model") or ""),
            occurred_at=str(call.get("occurred_at") or ""),
        )
        if card is None:
            return {
                "estimated_cost": None,
                "currency": None,
                "price_card_version": catalog.get("version"),
                "cost_status": "price_card_no_exact_match",
            }
        card_currency = str(card.get("currency") or "")
        if not card_currency or (currency and currency != card_currency):
            return {
                "estimated_cost": None,
                "currency": None,
                "price_card_version": catalog.get("version"),
                "cost_status": "price_card_invalid",
            }
        currency = card_currency
        usage = normalize_llm_usage(call.get("usage")).to_dict()
        for item_name, raw_rate in (card.get("rates_per_million") or {}).items():
            if item_name not in usage:
                return {
                    "estimated_cost": None,
                    "currency": currency,
                    "price_card_version": catalog.get("version"),
                    "cost_status": "price_card_billing_item_unsupported",
                }
            count = usage[item_name]
            if count is None:
                return {
                    "estimated_cost": None,
                    "currency": currency,
                    "price_card_version": catalog.get("version"),
                    "cost_status": "usage_incomplete",
                }
            try:
                rate = float(raw_rate)
            except (TypeError, ValueError):
                return {
                    "estimated_cost": None,
                    "currency": currency,
                    "price_card_version": catalog.get("version"),
                    "cost_status": "price_card_invalid",
                }
            total_cost += (count / 1_000_000.0) * rate
    if not llm_calls:
        return {
            "estimated_cost": 0.0,
            "currency": str(catalog.get("currency") or "") or None,
            "price_card_version": catalog.get("version"),
            "cost_status": "no_llm_calls",
        }
    return {
        "estimated_cost": round(total_cost, 10),
        "currency": currency,
        "price_card_version": catalog.get("version"),
        "cost_status": "estimated",
    }


class RuntimeTelemetryCollector:
    """Thread-safe collector that excludes explicit user wait from active runtime."""

    def __init__(
        self,
        *,
        monotonic_fn: Callable[[], float] = time.monotonic,
        wall_time_fn: Callable[[], float] = time.time,
        price_card_path: Optional[Path] = None,
        price_card_catalog: Optional[Mapping[str, Any]] = None,
    ):
        self._monotonic = monotonic_fn
        self._wall_time = wall_time_fn
        self._price_card_path = price_card_path
        self._price_card_catalog = price_card_catalog
        self._lock = threading.RLock()
        self._started = float(self._monotonic())
        self._started_wall = float(self._wall_time())
        self._paused_at: Optional[float] = None
        self._paused_total = 0.0
        self._sequence = 0
        self._open_spans: Dict[str, Dict[str, Any]] = {}
        self._spans: List[Dict[str, Any]] = []
        self._llm_calls: List[Dict[str, Any]] = []

    def start_span(self, category: str, name: str, **metadata: Any) -> str:
        with self._lock:
            self._sequence += 1
            span_id = f"span_{self._sequence}"
            safe, _ = redact_recursive(metadata)
            self._open_spans[span_id] = {
                "span_id": span_id,
                "category": str(category),
                "name": str(name),
                "started": float(self._monotonic()),
                "metadata": safe if isinstance(safe, dict) else {},
            }
            return span_id

    def end_span(self, span_id: str, *, status: str = "completed", error_type: Optional[str] = None) -> None:
        with self._lock:
            span = self._open_spans.pop(span_id, None)
            if span is None:
                return
            elapsed = max(0.0, float(self._monotonic()) - float(span.pop("started")))
            span.update(
                {
                    "status": str(status),
                    "duration_ms": round(elapsed * 1000.0, 3),
                    "error_type": str(error_type)[:120] if error_type else None,
                }
            )
            self._spans.append(span)

    def record_llm_result(self, kind: str, result: Any) -> None:
        usage = normalize_llm_usage(getattr(result, "usage", None))
        entry = {
            "kind": "judge" if str(kind) == "judge" else "agent",
            "provider": getattr(result, "provider", None),
            "model": getattr(result, "model", None),
            "usage": usage.to_dict(),
            "latency_ms": getattr(result, "latency_ms", None),
            "status": "failed" if getattr(result, "error_type", None) else "completed",
            "error_type": getattr(result, "error_type", None),
            "occurred_at": _utc_now(),
        }
        safe, _ = redact_recursive(entry)
        with self._lock:
            self._llm_calls.append(safe if isinstance(safe, dict) else entry)

    def record_tool_call(self, *, name: str, duration_ms: Optional[float], status: str) -> None:
        with self._lock:
            self._spans.append(
                {
                    "span_id": f"tool_{len(self._spans) + 1}",
                    "category": "tool",
                    "name": str(name or "")[:160],
                    "metadata": {},
                    "status": str(status or "unknown"),
                    "duration_ms": round(max(0.0, float(duration_ms or 0.0)), 3),
                    "error_type": None,
                }
            )

    def pause(self) -> None:
        with self._lock:
            if self._paused_at is None:
                self._paused_at = float(self._monotonic())

    def resume(self) -> None:
        with self._lock:
            if self._paused_at is not None:
                self._paused_total += max(0.0, float(self._monotonic()) - self._paused_at)
                self._paused_at = None

    @staticmethod
    def _aggregate_token(calls: List[Dict[str, Any]], kind: str, field: str) -> Optional[int]:
        values = [
            normalize_llm_usage(item.get("usage")).to_dict().get(field)
            for item in calls
            if item.get("kind") == kind
        ]
        if not values or any(value is None for value in values):
            return None
        return sum(int(value) for value in values)

    @staticmethod
    def _aggregate_latency(calls: List[Dict[str, Any]], kind: str) -> Optional[float]:
        values = [item.get("latency_ms") for item in calls if item.get("kind") == kind]
        if not values or any(value is None for value in values):
            return None
        try:
            return round(sum(max(0.0, float(value)) for value in values), 3)
        except (TypeError, ValueError):
            return None


    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = float(self._monotonic())
            paused = self._paused_total
            if self._paused_at is not None:
                paused += max(0.0, now - self._paused_at)
            wall_ms = max(0.0, now - self._started) * 1000.0
            active_ms = max(0.0, wall_ms - paused * 1000.0)
            spans = list(self._spans)
            calls = list(self._llm_calls)
        node_ms = sum(float(item.get("duration_ms") or 0.0) for item in spans if item.get("category") == "node")
        tool_ms = sum(float(item.get("duration_ms") or 0.0) for item in spans if item.get("category") == "tool")
        agent_calls = [item for item in calls if item.get("kind") == "agent"]
        judge_calls = [item for item in calls if item.get("kind") == "judge"]
        usage: Dict[str, Any] = {
            "agent_llm_call_count": len(agent_calls),
            "judge_llm_call_count": len(judge_calls),
            "tool_call_count": sum(1 for item in spans if item.get("category") == "tool"),
            "agent_llm_latency_ms": self._aggregate_latency(calls, "agent"),
            "judge_llm_latency_ms": self._aggregate_latency(calls, "judge"),
            "tool_latency_ms": round(tool_ms, 3),
            "node_latency_ms": round(node_ms, 3),
            "active_runtime_ms": round(active_ms, 3),
            "wall_runtime_ms": round(wall_ms, 3),
            "user_wait_ms": round(paused * 1000.0, 3),
        }
        for kind in ("agent", "judge"):
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
            ):
                usage[f"{kind}_{field}"] = self._aggregate_token(calls, kind, field)
        usage["usage_complete"] = all(
            normalize_llm_usage(item.get("usage")).total_tokens is not None for item in calls
        )
        usage.update(
            estimate_llm_cost(
                calls,
                price_card_path=self._price_card_path,
                price_card_catalog=self._price_card_catalog,
            )
        )
        telemetry = {
            "started_at": datetime.fromtimestamp(self._started_wall, tz=timezone.utc).isoformat(),
            "active_runtime_ms": usage["active_runtime_ms"],
            "wall_runtime_ms": usage["wall_runtime_ms"],
            "user_wait_ms": usage["user_wait_ms"],
            "spans": spans,
            "llm_calls": calls,
        }
        safe_telemetry, _ = redact_recursive(telemetry)
        safe_usage, _ = redact_recursive(usage)
        return {
            "runtime_telemetry": safe_telemetry if isinstance(safe_telemetry, dict) else {},
            "resource_usage": safe_usage if isinstance(safe_usage, dict) else {},
        }


def empty_runtime_evidence() -> Dict[str, Any]:
    return {
        "runtime_telemetry": {
            "started_at": None,
            "active_runtime_ms": None,
            "wall_runtime_ms": None,
            "user_wait_ms": None,
            "spans": [],
            "llm_calls": [],
        },
        "resource_usage": {
            "agent_llm_call_count": 0,
            "judge_llm_call_count": 0,
            "tool_call_count": 0,
            "agent_prompt_tokens": None,
            "agent_completion_tokens": None,
            "agent_total_tokens": None,
            "judge_prompt_tokens": None,
            "judge_completion_tokens": None,
            "judge_total_tokens": None,
            "estimated_cost": None,
            "currency": None,
            "price_card_version": None,
            "cost_status": "telemetry_unavailable",
        },
    }

