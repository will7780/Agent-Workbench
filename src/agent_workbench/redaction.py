# -*- coding: utf-8 -*-
"""共享递归脱敏 — 用于 context、session、report、UI 输出。"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|token|cookie|password|authorization|secret|credential|connection)",
    re.IGNORECASE,
)

SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"^Bearer\s+\S+", re.IGNORECASE),
    re.compile(r"^sk-[A-Za-z0-9_-]+$"),
    re.compile(r"mysql(\+pymysql)?://\S+", re.IGNORECASE),
)

EMBEDDED_SECRET_PATTERNS = (
    re.compile(r"\b(sk-[A-Za-z0-9_-]{8,})\b"),
    re.compile(r"Bearer\s+[^\s\"',;}\]]+", re.IGNORECASE),
    re.compile(r"mysql(\+pymysql)?://[^\s\"']+", re.IGNORECASE),
    re.compile(r'''(?i)\b(?:api[_-]?key|password)[ \t]+(?![:=])'''
               r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;"'}\]]+)'''),
)

_CREDENTIAL_ASSIGNMENT = re.compile(
    r'''(?P<prefix>(?<![\w-])(?:[A-Za-z0-9]+[_-])*'''
    r'''(?:api[_-]?key|token|password|authorization|cookie|secret|credential)["']?[ \t]*[:=][ \t]*)'''
    r'''(?P<value>\[REDACTED\]|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|'''
    r'''(?:Bearer|Basic)[ \t]+[^\s,;"'}\]]+|[^\s,;"'{}\[\]]+)''',
    re.IGNORECASE,
)

_COOKIE_HEADER = re.compile(
    r"(?P<prefix>(?<![^\r\n])[ \t]*(?:set-)?cookie[ \t]*:[ \t]*)(?P<value>[^\r\n]*)",
    re.IGNORECASE,
)
_JSON_START = re.compile(r'[\[{\"]')

HIDDEN_REASONING_KEYS = frozenset(
    {"reasoning_content", "chain_of_thought", "hidden_reasoning", "thinking",
     "analysis", "reasoning", "scratchpad"}
)

REDACTED = "[REDACTED]"


def redact_text(text: str) -> Tuple[str, bool]:
    """Return the same JSON-first output projection as recursive redaction."""
    return _redact_string(text, max_depth=32, max_list_items=10000)


def _redact_plain_text(text: str) -> Tuple[str, bool]:
    """脱敏嵌入在普通文本中的疑似凭据片段；保留业务语义。"""
    if not text:
        return text, False
    redacted_any = False

    def mask_cookie_header(match):
        nonlocal redacted_any
        if match.group("value").strip() in {"", REDACTED}:
            return match.group(0)
        redacted_any = True
        return match.group("prefix") + REDACTED

    def mask_assignment(match):
        nonlocal redacted_any
        value = match.group("value")
        if value.strip("\"'") == REDACTED:
            return match.group(0)
        redacted_any = True
        quote = value[0] if value[0] in "\"'" else ""
        return match.group("prefix") + quote + REDACTED + quote

    # Cookie separators and quoted values still belong to the same header line.
    out = _COOKIE_HEADER.sub(mask_cookie_header, text)
    # Preserve quotes and separators so text boundaries do not corrupt JSON.
    out = _CREDENTIAL_ASSIGNMENT.sub(mask_assignment, out)
    for pattern in EMBEDDED_SECRET_PATTERNS:
        new_out, count = pattern.subn(REDACTED, out)
        if count:
            redacted_any = True
            out = new_out
    return out, redacted_any


def is_sensitive_key(key: str) -> bool:
    return bool(SENSITIVE_KEY_PATTERN.search(str(key)))


_TOKEN_COUNT_KEY_PATTERN = re.compile(
    r"^(?:(?:agent|judge)_)?(?:prompt|completion|total|reasoning|cache_hit|cache_miss)_tokens$|^(?:estimated|max_total)_tokens$"
)


def is_safe_token_count(key: str, value: Any) -> bool:
    """Allow only nullable non-negative numeric usage counters, never credential values."""
    if not _TOKEN_COUNT_KEY_PATTERN.fullmatch(str(key or "")):
        return False
    return value is None or (
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )

_SAFE_SENSITIVE_BOOLEAN_KEYS = frozenset(
    {
        "no_secret_leak_in_memory",
        "total_tokens_budget_pass",
    }
)


def is_safe_sensitive_value(key: str, value: Any) -> bool:
    return is_safe_token_count(key, value) or (key in _SAFE_SENSITIVE_BOOLEAN_KEYS and isinstance(value, bool))


def _redact_string(text: str, *, max_depth: int, max_list_items: int,
                   whole_scalar: bool = False) -> Tuple[str, bool]:
    original = text
    redacted_any = False
    duplicate_keys = False

    def object_pairs(pairs):
        nonlocal duplicate_keys
        result = {}
        for key, value in pairs:
            duplicate_keys = duplicate_keys or key in result
            result[key] = value
        return result

    decoder = json.JSONDecoder(object_pairs_hook=object_pairs)
    parts = []
    cursor = scan = 0
    found_json = False
    plain_patterns = (_COOKIE_HEADER, _CREDENTIAL_ASSIGNMENT, *EMBEDDED_SECRET_PATTERNS)

    def append_plain(end):
        nonlocal redacted_any
        safe, was = _redact_plain_text(text[cursor:end])
        parts.append(safe)
        redacted_any = redacted_any or was

    # Decode intact JSON before any text replacement can damage its escapes.
    # Plain credentials/header values take precedence over quotes inside them.
    while True:
        match = _JSON_START.search(text, scan)
        plain_matches = [candidate for pattern in plain_patterns
                         if (candidate := pattern.search(text, scan)) is not None]
        plain_match = min(plain_matches, key=lambda candidate: candidate.start(), default=None)
        if plain_match is not None and (match is None or plain_match.start() < match.start()):
            append_plain(plain_match.end())
            cursor = scan = plain_match.end()
            continue
        if match is None:
            break
        start = match.start()
        duplicate_keys = False
        try:
            decoded, end = decoder.raw_decode(text, start)
        except RecursionError:
            return REDACTED, True
        except ValueError:
            fragment = text[start + 1:]
            # A truncated/invalid encoded preview cannot safely be partly decoded.
            if text[start] in "[{" and fragment.lstrip().startswith(('"', "[", "{")):
                return REDACTED, True
            if text[start] == '"' and ("\\" in fragment or fragment.lstrip().startswith(("{", "["))):
                return REDACTED, True
            scan = start + 1
            continue
        if (isinstance(decoded, str) and text[end:].lstrip().startswith(":")
                and (is_sensitive_key(decoded) or decoded.lower() in HIDDEN_REASONING_KEYS)):
            return REDACTED, True
        found_json = True
        if max_depth <= 0:
            safe, was = REDACTED, True
        else:
            # Containers consume depth in redact_recursive; string wrappers do so here.
            remaining = max_depth - int(isinstance(decoded, str))
            safe, was = redact_recursive(decoded, max_depth=remaining, max_list_items=max_list_items)
        append_plain(start)
        if was or duplicate_keys or safe != decoded:
            parts.append(json.dumps(safe, ensure_ascii=False))
            redacted_any = True
        else:
            parts.append(text[start:end])
        cursor = scan = end
    if whole_scalar and not found_json and any(p.search(original) for p in SENSITIVE_VALUE_PATTERNS):
        return REDACTED, True
    append_plain(len(text))
    return "".join(parts), redacted_any


def redact_scalar(key: str, value: Any, *, max_depth: int = 32,
                  max_list_items: int = 10000) -> Tuple[Any, bool]:
    if is_sensitive_key(key) and not is_safe_sensitive_value(key, value):
        return REDACTED, True
    if isinstance(value, str):
        return _redact_string(value, max_depth=max_depth, max_list_items=max_list_items,
                              whole_scalar=True)
    return value, False


def redact_recursive(
    data: Any,
    *,
    max_depth: int = 32,
    max_list_items: int = 10000,
) -> Tuple[Any, bool]:
    """Return a detached boundary projection; never mutate execution inputs."""
    if data is None:
        return None, False
    if isinstance(data, str):
        return _redact_string(data, max_depth=max_depth, max_list_items=max_list_items)
    if isinstance(data, list):
        if max_depth <= 0:
            return [{"_truncated": True, "omitted_count": len(data)}], False
        redacted_any = False
        out: List[Any] = []
        for item in data[:max_list_items]:
            safe, was = redact_recursive(item, max_depth=max_depth - 1, max_list_items=max_list_items)
            out.append(safe)
            redacted_any = redacted_any or was
        if len(data) > max_list_items:
            out.append({"_truncated": True, "omitted_count": len(data) - max_list_items})
        return out, redacted_any
    if not isinstance(data, dict):
        return data, False
    if max_depth <= 0:
        return {"_truncated": True}, False
    redacted_any = False
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if str(key).lower() in HIDDEN_REASONING_KEYS:
            redacted_any = True
            continue
        if is_sensitive_key(key) and not is_safe_sensitive_value(str(key), value):
            out[key] = REDACTED
            redacted_any = True
            continue
        if isinstance(value, dict):
            nested, nested_redacted = redact_recursive(
                value, max_depth=max_depth - 1, max_list_items=max_list_items
            )
            out[key] = nested
            redacted_any = redacted_any or nested_redacted
        elif isinstance(value, list):
            nested, nested_redacted = redact_recursive(
                value, max_depth=max_depth - 1, max_list_items=max_list_items
            )
            out[key] = nested
            redacted_any = redacted_any or nested_redacted
        else:
            safe, was = redact_scalar(str(key), value, max_depth=max_depth - 1,
                                      max_list_items=max_list_items)
            out[key] = safe
            redacted_any = redacted_any or was
    return out, redacted_any


def redact_plan_dict(plan_dict: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not plan_dict:
        return None, False
    redacted, was = redact_recursive(copy.deepcopy(plan_dict))
    return redacted, was


def contains_secret_blob(text: str) -> bool:
    """检测文本是否仍含疑似明文凭据（测试用）。"""
    if not text:
        return False
    if any(match.group("value").strip() not in {"", REDACTED}
           for match in _COOKIE_HEADER.finditer(text)):
        return True
    if any(match.group("value").strip("\"'") != REDACTED
           for match in _CREDENTIAL_ASSIGNMENT.finditer(text)):
        return True
    for pattern in EMBEDDED_SECRET_PATTERNS:
        if pattern.search(text):
            return True
    for pattern in SENSITIVE_VALUE_PATTERNS:
        if pattern.search(text):
            return True
    return False
