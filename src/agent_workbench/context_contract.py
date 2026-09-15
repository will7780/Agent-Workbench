from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
from .redaction import REDACTED, redact_plan_dict, redact_recursive, redact_text
MAX_MODULE_CONFIG_KEYS = 24

def estimate_chars(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))

def estimate_tokens(chars: int) -> int:
    return max(1, chars // 4)

def _safe_module_config_summary(module_config: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    if not module_config:
        return ({}, False, 'no_module_config')
    keys = sorted(module_config.keys())[:MAX_MODULE_CONFIG_KEYS]
    subset = {k: module_config[k] for k in keys}
    redacted, was_redacted = redact_recursive(subset)
    omission = None
    if len(module_config) > MAX_MODULE_CONFIG_KEYS:
        omission = 'module_config_truncated'
    return (redacted, was_redacted, omission)

def _manifest_entry(section: str, *, source: str, included: bool, payload: Any=None, truncated: bool=False, redacted: bool=False, omission_reason: Optional[str]=None, sent_to_llm: bool=False) -> Dict[str, Any]:
    chars = estimate_chars(payload) if included and payload is not None else 0
    return {'section': section, 'source': source, 'included': included, 'estimated_chars': chars, 'estimated_tokens': estimate_tokens(chars), 'truncated': truncated, 'redacted': redacted, 'omission_reason': omission_reason, 'sent_to_llm': sent_to_llm, 'payload': payload}
