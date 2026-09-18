from __future__ import annotations

import re

_CONTROL_RE = re.compile(r"[\r\n\x00-\x1f\x7f]+")
_PASSWORD_RE = re.compile(r"(?i)\b(password|passcode|token)\b[:= ]+\S+")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_CODE_RE = re.compile(r"(?i)\b(?:otp|mfa|2fa|code|pin)[:= ]+\d{4,8}\b")
_LONG_SECRET_RE = re.compile(r"\b[A-Za-z0-9_-]{24,}\b")


def sanitize_text(value: str, *, max_length: int = 600) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = _CONTROL_RE.sub(" ", value).strip()
    redacted = _CODE_RE.sub("[redacted code]", cleaned)
    redacted = _PASSWORD_RE.sub("[redacted secret]", redacted)
    redacted = _CARD_RE.sub("[redacted card-like value]", redacted)
    redacted = _LONG_SECRET_RE.sub(_preserve_long_safe_word, redacted)
    if len(redacted) <= max_length:
        return redacted
    return redacted[: max_length - 14].rstrip() + "...[truncated]"


def sanitize_list(values: list[str], *, max_items: int = 12, max_length: int = 600) -> list[str]:
    sanitized: list[str] = []
    for value in values[:max_items]:
        cleaned = sanitize_text(value, max_length=max_length)
        if cleaned:
            sanitized.append(cleaned)
    return sanitized


def _preserve_long_safe_word(match: re.Match[str]) -> str:
    value = match.group(0)
    if value.isalpha() and len(value) <= 32:
        return value
    return "[redacted token-like value]"
