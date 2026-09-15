# -*- coding: utf-8 -*-
"""EmbeddingProvider 协议与测试用 fake；不绑定具体模型厂商。"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Protocol, Sequence

from .knowledge_provider import ensure_knowledge_deadline


class EmbeddingUnavailable(Exception):
    """向量服务不可用，调用方应降级为 BM25。"""


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str], *, deadline: Optional[float] = None) -> List[List[float]]: ...


class FakeEmbeddingProvider:
    """确定性哈希向量，仅用于测试与验收，不访问网络。"""

    def __init__(self, *, dim: int = 8, unavailable: bool = False, delay_s: float = 0.0) -> None:
        self.dim = max(2, int(dim))
        self.unavailable = unavailable
        self.delay_s = max(0.0, float(delay_s or 0.0))
        self.calls: List[List[str]] = []

    def embed(self, texts: Sequence[str], *, deadline: Optional[float] = None) -> List[List[float]]:
        import time

        if self.delay_s:
            started = time.monotonic()
            while time.monotonic() - started < self.delay_s:
                ensure_knowledge_deadline(deadline)
                time.sleep(0.01)
        ensure_knowledge_deadline(deadline)
        self.calls.append([str(t) for t in texts])
        if self.unavailable:
            raise EmbeddingUnavailable("vector_unavailable")
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        values: List[float] = []
        for idx in range(self.dim):
            raw = digest[idx % len(digest)]
            values.append((raw / 255.0) * 2.0 - 1.0)
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = sum(float(left[i]) * float(right[i]) for i in range(size))
    n1 = math.sqrt(sum(float(left[i]) ** 2 for i in range(size))) or 1.0
    n2 = math.sqrt(sum(float(right[i]) ** 2 for i in range(size))) or 1.0
    return dot / (n1 * n2)
