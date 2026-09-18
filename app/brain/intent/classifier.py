from __future__ import annotations

from app.brain.intent.models import ConfidenceCategory, Intent, IntentClassification
from app.brain.intent.rules import classify_obvious


def classify(text: str) -> IntentClassification:
    """Return metadata only; this function never executes a tool."""
    result = classify_obvious(text)
    if result is not None:
        return result
    if not text or not text.strip():
        return IntentClassification(Intent.UNSUPPORTED, ConfidenceCategory.LOW, False, False, text)
    return IntentClassification(Intent.UNSUPPORTED, ConfidenceCategory.LOW, False, False, text)

from app.brain.intent.models import ConfidenceCategory, IntentCategory, IntentResult
from app.brain.intent.rules import classify_with_rules


class IntentClassifier:
    """Classify a user request into a fixed intent category."""

    def __init__(self) -> None:
        self._rules = classify_with_rules

    def classify(self, raw_input: str) -> IntentResult:
        if raw_input is None:
            raise ValueError("input must not be empty")
        text = raw_input.strip()
        if not text:
            raise ValueError("input must not be empty")
        if len(text) > 400:
            raise ValueError("input too long")
        if any(ord(ch) < 32 for ch in text):
            raise ValueError("input contains control characters")
        normalized = " ".join(text.split())
        rule_result = self._rules(normalized)
        if rule_result is None:
            return IntentResult(
                intent=IntentCategory.UNSUPPORTED,
                confidence_category=ConfidenceCategory.LOW,
                tools_required=False,
                multi_step_required=False,
                original_text=text,
                normalized_text=normalized,
            )
        intent, confidence, tools_required, multi_step_required = rule_result
        return IntentResult(
            intent=intent,
            confidence_category=confidence,
            tools_required=tools_required,
            multi_step_required=multi_step_required,
            original_text=text,
            normalized_text=normalized,
        )
