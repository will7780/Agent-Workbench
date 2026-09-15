# -*- coding: utf-8 -*-
"""按 tenant_id + provider_id 隔离的知识 Provider 熔断器。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

from .local_knowledge_provider import sanitize_tenant_id

CIRCUIT_CLOSED = "closed"
CIRCUIT_OPEN = "open"
CIRCUIT_HALF_OPEN = "half_open"

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 30.0


@dataclass
class CircuitRecord:
    state: str = CIRCUIT_CLOSED
    failures: int = 0
    opened_at: Optional[float] = None
    half_open_inflight: bool = False


class CircuitBreakerStore:
    """
    进程级 / Registry 生命周期级熔断状态。
    按 (tenant_id, provider_id) 隔离；状态访问加锁，half-open 只允许一个探针。
    不记录查询正文、用户身份明文或凭据。
    """

    def __init__(self, records: Optional[Dict[Tuple[str, str], CircuitRecord]] = None) -> None:
        self._lock = threading.Lock()
        self._records: Dict[Tuple[str, str], CircuitRecord] = records if records is not None else {}

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    def snapshot(self, key: Tuple[str, str]) -> CircuitRecord:
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return CircuitRecord()
            return CircuitRecord(
                state=rec.state,
                failures=rec.failures,
                opened_at=rec.opened_at,
                half_open_inflight=rec.half_open_inflight,
            )

    def mutate(self, key: Tuple[str, str], mutator: Callable[[CircuitRecord], Any]) -> Any:
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                rec = CircuitRecord()
                self._records[key] = rec
            return mutator(rec)


_PROCESS_CIRCUIT_STORE = CircuitBreakerStore()


def process_circuit_store() -> CircuitBreakerStore:
    return _PROCESS_CIRCUIT_STORE


def reset_circuit_store() -> None:
    _PROCESS_CIRCUIT_STORE.reset()


def _as_store(store: Optional[Union[CircuitBreakerStore, Dict[Tuple[str, str], CircuitRecord]]]) -> CircuitBreakerStore:
    if store is None:
        return CircuitBreakerStore()
    if isinstance(store, CircuitBreakerStore):
        return store
    return CircuitBreakerStore(records=store)


class KnowledgeCircuitBreaker:
    """
    不记录查询正文、用户身份明文或凭据。
    clock/state store 可注入，便于测试。
    """

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Optional[Callable[[], float]] = None,
        store: Optional[Union[CircuitBreakerStore, Dict[Tuple[str, str], CircuitRecord]]] = None,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold or DEFAULT_FAILURE_THRESHOLD))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds or 0.0))
        self.clock = clock or time.monotonic
        self.store = _as_store(store)

    def _key(self, tenant_id: str, provider_id: str) -> Tuple[str, str]:
        return (sanitize_tenant_id(tenant_id), str(provider_id or "").strip() or "unknown")

    def _refresh_locked(self, rec: CircuitRecord) -> str:
        now = self.clock()
        if rec.state == CIRCUIT_OPEN and rec.opened_at is not None:
            if now - rec.opened_at >= self.cooldown_seconds:
                rec.state = CIRCUIT_HALF_OPEN
                rec.half_open_inflight = False
        return rec.state

    def state(self, tenant_id: str, provider_id: str) -> str:
        key = self._key(tenant_id, provider_id)

        def _read(rec: CircuitRecord) -> str:
            return self._refresh_locked(rec)

        return str(self.store.mutate(key, _read))

    def allow(self, tenant_id: str, provider_id: str) -> Tuple[bool, str]:
        key = self._key(tenant_id, provider_id)

        def _allow(rec: CircuitRecord) -> Tuple[bool, str]:
            current = self._refresh_locked(rec)
            if current == CIRCUIT_OPEN:
                return False, CIRCUIT_OPEN
            if current == CIRCUIT_HALF_OPEN:
                if rec.half_open_inflight:
                    return False, CIRCUIT_HALF_OPEN
                rec.half_open_inflight = True
                return True, CIRCUIT_HALF_OPEN
            return True, CIRCUIT_CLOSED

        return self.store.mutate(key, _allow)

    def record_success(self, tenant_id: str, provider_id: str) -> str:
        key = self._key(tenant_id, provider_id)

        def _ok(rec: CircuitRecord) -> str:
            rec.state = CIRCUIT_CLOSED
            rec.failures = 0
            rec.opened_at = None
            rec.half_open_inflight = False
            return rec.state

        return str(self.store.mutate(key, _ok))

    def record_failure(self, tenant_id: str, provider_id: str) -> str:
        key = self._key(tenant_id, provider_id)

        def _fail(rec: CircuitRecord) -> str:
            rec.failures += 1
            rec.half_open_inflight = False
            if rec.state == CIRCUIT_HALF_OPEN or rec.failures >= self.failure_threshold:
                rec.state = CIRCUIT_OPEN
                rec.opened_at = self.clock()
            return rec.state

        return str(self.store.mutate(key, _fail))

    def snapshot(self, tenant_id: str, provider_id: str) -> Dict[str, Any]:
        rec = self.store.snapshot(self._key(tenant_id, provider_id))
        return {
            "state": self.state(tenant_id, provider_id),
            "failures": int(rec.failures),
        }

    def reset(self) -> None:
        self.store.reset()
