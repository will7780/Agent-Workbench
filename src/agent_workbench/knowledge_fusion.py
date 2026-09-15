# -*- coding: utf-8 -*-
"""BM25 与向量结果归一化、去重与稳定 fusion（RRF）。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FUSION_K = 60


def hit_key(hit: Dict[str, Any]) -> Tuple[str, str]:
    return (str(hit.get("document_id") or ""), str(hit.get("chunk_id") or ""))


def min_max_normalize(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [1.0 for _ in scores]
    return [(value - lo) / (hi - lo) for value in scores]


def _clone(hit: Dict[str, Any]) -> Dict[str, Any]:
    return dict(hit)


def reciprocal_rank_fusion(
    bm25_hits: Iterable[Dict[str, Any]],
    vector_hits: Iterable[Dict[str, Any]],
    *,
    k: int = FUSION_K,
) -> List[Dict[str, Any]]:
    """
    稳定 fusion：RRF 分 + BM25 名次作为并列打破键，再按 document_id/chunk_id。
    同一 (document_id, chunk_id) 只保留一条。
    """
    bm25_list = [dict(hit) for hit in bm25_hits or []]
    vector_list = [dict(hit) for hit in vector_hits or []]
    bm25_scores = min_max_normalize([float(hit.get("score") or 0.0) for hit in bm25_list])
    vector_scores = min_max_normalize([float(hit.get("score") or 0.0) for hit in vector_list])
    for hit, norm in zip(bm25_list, bm25_scores):
        hit["normalized_score"] = norm
        hit.setdefault("retrieval_reason", "bm25")
    for hit, norm in zip(vector_list, vector_scores):
        hit["normalized_score"] = norm
        hit.setdefault("retrieval_reason", "vector")

    ranks: Dict[Tuple[str, str], Dict[str, Any]] = {}
    fused: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add(hits: List[Dict[str, Any]], channel: str) -> None:
        for rank, hit in enumerate(hits, start=1):
            key = hit_key(hit)
            if not key[0] or not key[1]:
                continue
            record = fused.get(key)
            if record is None:
                record = _clone(hit)
                record["fusion_score"] = 0.0
                record["channels"] = []
                fused[key] = record
                ranks[key] = {"bm25_rank": 10**9, "vector_rank": 10**9}
            else:
                if float(hit.get("score") or 0.0) > float(record.get("score") or 0.0):
                    record["score"] = hit.get("score")
                if hit.get("snippet") and not record.get("snippet"):
                    record["snippet"] = hit.get("snippet")
            record["fusion_score"] = float(record.get("fusion_score") or 0.0) + (1.0 / (k + rank))
            channels = list(record.get("channels") or [])
            if channel not in channels:
                channels.append(channel)
            record["channels"] = channels
            ranks[key][f"{channel}_rank"] = rank
            reasons = []
            if "bm25" in channels:
                reasons.append("bm25")
            if "vector" in channels:
                reasons.append("vector")
            record["retrieval_reason"] = "+".join(reasons) if len(reasons) > 1 else (reasons[0] if reasons else channel)

    add(bm25_list, "bm25")
    add(vector_list, "vector")

    ordered = list(fused.values())
    ordered.sort(
        key=lambda hit: (
            -float(hit.get("fusion_score") or 0.0),
            ranks[hit_key(hit)]["bm25_rank"],
            ranks[hit_key(hit)]["vector_rank"],
            str(hit.get("document_id") or ""),
            str(hit.get("chunk_id") or ""),
        )
    )
    return ordered
