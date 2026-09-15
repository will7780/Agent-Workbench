from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
RUNNING_SUMMARY_MAX_CHARS = 2000
_ARTIFACT_REF_KEYS = ('type', 'path', 'step', 'site', 'status', 'artifact_count')
_SENSITIVE_MARKERS = ('password', 'token', 'secret', 'api_key', 'cookie', 'connection')

def compact_running_summary(text: str, *, max_chars: int=RUNNING_SUMMARY_MAX_CHARS) -> str:
    """Plan-level running summary 上限压缩，保留较新的尾部内容。"""
    cleaned = (text or '').strip()
    if len(cleaned) <= max_chars:
        return cleaned
    prefix = '...[truncated]\n'
    budget = max(0, max_chars - len(prefix))
    return f'{prefix}{cleaned[-budget:]}'

def _sanitize_summary(text: Any, *, max_chars: int=400) -> str:
    raw = str(text or '').strip()
    lowered = raw.lower()
    if any((marker in lowered for marker in _SENSITIVE_MARKERS)):
        return '(summary omitted)'
    if len(raw) > max_chars:
        return raw[:max_chars - 3] + '...'
    return raw

def _compact_artifact_refs(artifacts: List[Dict[str, Any]], *, limit: int=20) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for art in artifacts[:limit]:
        if not isinstance(art, dict):
            continue
        ref = {k: art[k] for k in _ARTIFACT_REF_KEYS if k in art}
        if ref:
            refs.append(ref)
    return refs

def _compact_last_observation(last_observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not last_observation:
        return None
    return {'step_id': last_observation.get('step_id'), 'status': last_observation.get('status'), 'summary': _sanitize_summary(last_observation.get('summary')), 'error_type': (last_observation.get('error') or {}).get('type')}
