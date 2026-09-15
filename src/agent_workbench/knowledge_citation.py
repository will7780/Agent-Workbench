# -*- coding: utf-8 -*-
"""企业知识 citation 完整性、来源一致性与 lexical groundedness。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol, Set

from .knowledge_provider import format_citation
from .redaction import redact_text

KB_CITATION_RE = re.compile(r"\[KB:([^/\]]+)/([^#\]]+)#([^\]]+)\]")
_STRONG_FACT_MARKERS = (
    "规定",
    "必须",
    "禁止",
    "制度",
    "政策",
    "SOP",
    "根据知识",
    "根据文档",
    "标准作业",
    "官方说明",
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s*")
_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class GroundednessJudge(Protocol):
    def score(self, claim: str, evidence: str) -> Dict[str, Any]: ...


def parse_kb_citations(text: str) -> List[str]:
    found: List[str] = []
    seen = set()
    for match in KB_CITATION_RE.finditer(text or ""):
        citation = format_citation(match.group(1), match.group(2), match.group(3))
        if citation not in seen:
            seen.add(citation)
            found.append(citation)
    return found


def _included_hits(knowledge_search: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for hit in (knowledge_search or {}).get("hits") or []:
        if isinstance(hit, dict) and hit.get("included_in_llm"):
            hits.append(hit)
    return hits


def _included_citations(knowledge_search: Optional[Dict[str, Any]]) -> Set[str]:
    allowed: Set[str] = set()
    for hit in _included_hits(knowledge_search):
        citation = str(hit.get("citation") or "").strip()
        if citation:
            allowed.add(citation)
    return allowed


def _looks_like_strong_enterprise_claim(text: str) -> bool:
    body = str(text or "").strip()
    if not body:
        return False
    return any(marker in body for marker in _STRONG_FACT_MARKERS)


def _tokens(text: str) -> Set[str]:
    values = set()
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0).lower()
        if len(token) >= 2:
            values.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) >= 2:
            for idx in range(len(token) - 1):
                values.add(token[idx : idx + 2])
    return values


def _lexical_score(claim: str, evidence: str) -> float:
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(evidence)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    overlap = claim_tokens & evidence_tokens
    return round(len(overlap) / max(1, len(claim_tokens)), 4)


def _split_claims(text: str) -> List[str]:
    body = str(text or "").strip()
    if not body:
        return []
    parts = [item.strip() for item in _SENTENCE_SPLIT_RE.split(body) if item.strip()]
    return parts or [body]


def evaluate_citation_groundedness(
    final_response: Optional[str],
    knowledge_search: Optional[Dict[str, Any]] = None,
    *,
    judge: Optional[GroundednessJudge] = None,
) -> Dict[str, Any]:
    """
    基于本次 included_in_llm chunk 的确定性 lexical 校验。
    可注入 judge，但默认不调用真实 LLM，不保存 chain-of-thought。
    """
    search = knowledge_search or {}
    groundedness_method = "judge" if judge is not None else "lexical"
    response, _ = redact_text(str(final_response or ""))
    included = _included_hits(search)
    allowed = _included_citations(search)
    claims_out: List[Dict[str, Any]] = []
    unsupported = 0
    invalid_remote = 0
    cited_supported = 0
    evaluated = 0

    for sentence in _split_claims(response):
        citations = parse_kb_citations(sentence)
        strong = _looks_like_strong_enterprise_claim(sentence)
        if not citations and not strong:
            continue
        evaluated += 1
        claim_text, _ = redact_text(sentence)
        if citations:
            for citation in citations:
                if citation not in allowed:
                    invalid_remote += 1
                    unsupported += 1
                    claims_out.append(
                        {
                            "claim": claim_text[:240],
                            "citation": citation,
                            "score": 0.0,
                            "reason": "citation_not_in_included_chunks",
                            "evidence_refs": [],
                        }
                    )
                    continue
                evidence_hits = [hit for hit in included if str(hit.get("citation") or "") == citation]
                evidence = " ".join(str(hit.get("snippet") or "") for hit in evidence_hits)
                if not evidence.strip():
                    cited_supported += 1
                    claims_out.append(
                        {
                            "claim": claim_text[:240],
                            "citation": citation,
                            "score": 1.0,
                            "reason": "citation_included",
                            "evidence_refs": [citation],
                        }
                    )
                    continue
                if judge is not None:
                    judged = judge.score(claim_text, evidence) or {}
                    score = float(judged.get("score") or 0.0)
                    reason = str(judged.get("reason") or "judge")
                else:
                    score = _lexical_score(claim_text, evidence)
                    reason = "lexical_overlap"
                supported = score >= 0.2 and bool(evidence.strip())
                if supported:
                    cited_supported += 1
                else:
                    unsupported += 1
                claims_out.append(
                    {
                        "claim": claim_text[:240],
                        "citation": citation,
                        "score": score,
                        "reason": reason if supported else "unsupported_by_chunk",
                        "evidence_refs": [str(hit.get("citation") or "") for hit in evidence_hits],
                    }
                )
        elif strong and included:
            unsupported += 1
            claims_out.append(
                {
                    "claim": claim_text[:240],
                    "citation": None,
                    "score": 0.0,
                    "reason": "missing_citation",
                    "evidence_refs": [],
                }
            )

    coverage = round(cited_supported / evaluated, 4) if evaluated else None
    if evaluated == 0:
        groundedness = None
    elif unsupported == 0:
        groundedness = 1.0
    else:
        groundedness = round(max(0.0, cited_supported / evaluated), 4)

    return {
        "citation_groundedness": groundedness,
        "citation_groundedness_method": groundedness_method,
        "citation_coverage": coverage,
        "unsupported_claim_count": unsupported,
        "invalid_remote_citation_count": invalid_remote,
        "claims": claims_out,
        "semantic_fact_check": groundedness_method == "judge",
        "note": (
            "injected judge groundedness against included_in_llm chunks; no chain-of-thought stored"
            if groundedness_method == "judge"
            else "lexical groundedness against included_in_llm chunks; no chain-of-thought stored"
        ),
    }


def verify_knowledge_citations(
    final_response: Optional[str],
    knowledge_search: Optional[Dict[str, Any]] = None,
    *,
    judge: Optional[GroundednessJudge] = None,
) -> Dict[str, Any]:
    """
    citation 完整性/来源一致性 + lexical groundedness。
    不保存模型隐藏推理。
    """
    search = knowledge_search or {}
    response = str(final_response or "")
    citations = parse_kb_citations(response)
    allowed = _included_citations(search)
    injected = bool(search.get("knowledge_context_injected")) or bool(_included_hits(search))
    issues: List[Dict[str, Any]] = []
    for citation in citations:
        if citation not in allowed:
            issues.append(
                {
                    "type": "citation_invalid",
                    "citation": citation,
                    "detail": "最终回答引用了未进入本次 LLM 上下文的知识片段（ACL 过滤、预算省略或未知来源）",
                }
            )
    if injected and not citations and _looks_like_strong_enterprise_claim(response):
        issues.append(
            {
                "type": "citation_missing",
                "detail": "使用企业知识给出强事实结论但未提供 citation",
            }
        )

    grounded = evaluate_citation_groundedness(final_response, search, judge=judge)
    if int(grounded.get("unsupported_claim_count") or 0) > 0 and not any(
        item.get("type") == "citation_invalid" for item in issues
    ):
        if any(item.get("reason") == "unsupported_by_chunk" for item in grounded.get("claims") or []):
            issues.append(
                {
                    "type": "groundedness_failed",
                    "detail": "企业事实结论未能被本次 included_in_llm chunk 支持",
                }
            )

    if issues:
        if any(item.get("type") == "citation_invalid" for item in issues):
            status = "citation_invalid"
        elif any(item.get("type") == "groundedness_failed" for item in issues):
            status = "groundedness_failed"
        else:
            status = "citation_missing"
    elif citations:
        status = "citation_valid"
    else:
        status = "not_applicable"

    payload = {
        "status": status,
        "citations_in_answer": citations,
        "allowed_citations": sorted(allowed),
        "issues": issues,
        "note": grounded.get("note"),
    }
    payload.update(grounded)
    return payload
