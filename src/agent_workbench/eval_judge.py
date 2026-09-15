from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol
from .llm_client import CompleteFn, chat_completion_json
from .redaction import redact_recursive, redact_text
from .runtime_telemetry import LLMUsage, RuntimeTelemetryCollector

@dataclass
class EvalJudgeResult:
    decision: str
    score: Optional[float] = None
    issue_codes: List[str] = field(default_factory=list)
    summary: str = ''
    evidence_refs: List[str] = field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_ms: Optional[float] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        safe_summary, _ = redact_text(self.summary)
        return {'decision': self.decision, 'score': self.score, 'issue_codes': list(self.issue_codes), 'summary': safe_summary[:500], 'evidence_refs': list(self.evidence_refs), 'provider': self.provider, 'model': self.model, 'usage': self.usage.to_dict(), 'latency_ms': self.latency_ms, 'error_type': self.error_type}

class EvalJudge(Protocol):

    def judge_parameter_intent(self, payload: Dict[str, Any]) -> EvalJudgeResult:
        ...

class CallableEvalJudge:
    _DECISIONS = {'parameter_intent': {'aligned', 'misaligned', 'uncertain'}}

    def __init__(self, fn: Callable[[Dict[str, Any]], Any], *, telemetry: Optional[RuntimeTelemetryCollector]=None):
        self.fn = fn
        self.telemetry = telemetry

    def _call(self, task: str, payload: Dict[str, Any]) -> EvalJudgeResult:
        try:
            raw = self.fn({'judge_task': task, **dict(payload)})
        except Exception:
            result = EvalJudgeResult(decision='uncertain', issue_codes=['eval_judge_failed'], error_type='eval_judge_failed')
            if self.telemetry is not None:
                self.telemetry.record_llm_result('judge', result)
            return result
        if isinstance(raw, EvalJudgeResult):
            result = raw
        elif isinstance(raw, dict):
            result = EvalJudgeResult(decision=str(raw.get('decision') or 'uncertain').strip().lower(), score=raw.get('score') if isinstance(raw.get('score'), (int, float)) else None, issue_codes=[str(item)[:100] for item in raw.get('issue_codes') or []][:20], summary=str(raw.get('summary') or '')[:500], evidence_refs=[str(item)[:160] for item in raw.get('evidence_refs') or []][:20], provider=str(raw.get('provider')) if raw.get('provider') else None, model=str(raw.get('model')) if raw.get('model') else None, usage=LLMUsage.from_api_payload(raw.get('usage')), latency_ms=raw.get('latency_ms') if isinstance(raw.get('latency_ms'), (int, float)) else None, error_type=str(raw.get('error_type'))[:120] if raw.get('error_type') else None)
        else:
            result = EvalJudgeResult(decision='uncertain', issue_codes=['parameter_judge_invalid_response'], error_type='parameter_judge_invalid_response')
        if result.decision not in self._DECISIONS[task]:
            result.decision = 'uncertain'
            result.error_type = result.error_type or 'parameter_judge_invalid_response'
        if result.score is not None:
            if isinstance(result.score, bool) or not isinstance(result.score, (int, float)) or not math.isfinite(result.score):
                result.score = None
                result.decision = 'uncertain'
                result.error_type = 'parameter_judge_invalid_score'
            else:
                result.score = max(0.0, min(1.0, float(result.score)))
        if self.telemetry is not None:
            self.telemetry.record_llm_result('judge', result)
        return result

    def judge_parameter_intent(self, payload: Dict[str, Any]) -> EvalJudgeResult:
        return self._call('parameter_intent', payload)

class NativeEvalJudge:
    """Use the existing JSON LLM client and retain only bounded structured fields."""
    _DECISIONS = {'parameter_intent': {'aligned', 'misaligned', 'uncertain'}}

    def __init__(self, *, model: Optional[str]=None, complete_fn: Optional[CompleteFn]=None, telemetry: Optional[RuntimeTelemetryCollector]=None):
        self.model = model or os.environ.get('AGENT_WORKBENCH_PARAMETER_JUDGE_MODEL') or None
        self.complete_fn = complete_fn
        self.telemetry = telemetry

    def _run(self, task: str, payload: Dict[str, Any]) -> EvalJudgeResult:
        safe_payload, _ = redact_recursive(payload)
        messages = [{'role': 'system', 'content': 'You are a parameter alignment reviewer. Return one JSON object only. Do not provide chain-of-thought or hidden reasoning. Return keys: decision, score, issue_codes, summary, evidence_refs. score must be between 0 and 1. Use only supplied evidence.'}, {'role': 'user', 'content': json.dumps({'task': task, 'evidence': safe_payload}, ensure_ascii=False, sort_keys=True, default=str)}]
        result = chat_completion_json(messages, model_override=self.model, complete_fn=self.complete_fn)
        if self.telemetry is not None:
            self.telemetry.record_llm_result('judge', result)
        if not result.ok:
            return EvalJudgeResult(decision='uncertain', provider=result.provider, model=result.model, usage=result.usage, latency_ms=result.latency_ms, error_type=result.error_type or 'eval_judge_unavailable', issue_codes=['eval_judge_unavailable'])
        try:
            raw = json.loads(result.content or '')
        except (json.JSONDecodeError, TypeError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        decision = str(raw.get('decision') or 'uncertain').strip().lower()
        if decision not in self._DECISIONS[task]:
            decision = 'uncertain'
        try:
            score = float(raw.get('score'))
        except (TypeError, ValueError):
            score = None
        if score is not None:
            if isinstance(raw.get('score'), bool) or not math.isfinite(score):
                score, decision = None, 'uncertain'
            else:
                score = max(0.0, min(1.0, score))
        issue_codes = [str(item)[:100] for item in raw.get('issue_codes') or [] if isinstance(item, str)][:20]
        evidence_refs = [str(item)[:160] for item in raw.get('evidence_refs') or [] if isinstance(item, str)][:20]
        summary, _ = redact_text(str(raw.get('summary') or ''))
        return EvalJudgeResult(decision=decision, score=score, issue_codes=issue_codes, summary=summary[:500], evidence_refs=evidence_refs, provider=result.provider, model=result.model, usage=result.usage, latency_ms=result.latency_ms, error_type='eval_judge_uncertain' if decision == 'uncertain' else None)

    def judge_parameter_intent(self, payload: Dict[str, Any]) -> EvalJudgeResult:
        return self._run('parameter_intent', payload)
