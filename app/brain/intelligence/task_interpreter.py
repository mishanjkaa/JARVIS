from __future__ import annotations

import re
from pathlib import Path

from app.brain.filesystem.state import get_filesystem_state
from app.brain.intent.models import ConfidenceCategory
from app.brain.intelligence.models import ClarificationRequest, NaturalLanguageTask, TaskConstraint, TaskIntent

_ACTION_PATTERN = re.compile(
    r"\b(create|make|write|add|append|calculate|check|run|inspect|summari[sz]e|install|delete|remove|read|show|review|verify|prepare|look at|open|navigate|save|click|scroll|reload|refresh|switch|close)\b",
    re.IGNORECASE,
)
_RUSSIAN_ACTION_PATTERN = re.compile(
    r"(\u0441\u043e\u0437\u0434\u0430\u0439|\u0441\u0434\u0435\u043b\u0430\u0439|\u0437\u0430\u043f\u0443\u0441\u0442\u0438|\u043f\u0440\u043e\u0432\u0435\u0440\u044c|\u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0438|\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438|\u0441\u043a\u0430\u0436\u0438|\u0443\u0441\u0442\u0430\u043d\u043e\u0432\u0438|\u0443\u0434\u0430\u043b\u0438|\u0434\u043e\u0431\u0430\u0432\u044c|\u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0439|\u043f\u043e\u043a\u0430\u0436\u0438|\u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u044c|\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0443|\u043d\u0430\u0436\u043c\u0438|\u043f\u0440\u043e\u043a\u0440\u0443\u0442\u0438|\u043f\u0435\u0440\u0435\u043a\u043b\u044e\u0447\u0438|\u043e\u0431\u043d\u043e\u0432\u0438|\u0437\u0430\u043a\u0440\u043e\u0439)",
    re.IGNORECASE,
)
_CONVERSATION_PATTERN = re.compile(
    r"^(what is|what's|who is|why|how does|explain|\u0447\u0442\u043e \u0442\u0430\u043a\u043e\u0435|\u043e\u0431\u044a\u044f\u0441\u043d\u0438)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_PATTERN = re.compile(
    r"\b(research|internet research|voice recording|download from the web)\b",
    re.IGNORECASE,
)
_PATH_PATTERN = re.compile(r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+|[A-Za-z0-9_.-]+\.(?:py|txt|md|json|log|png|jpg|jpeg|webp))")
_URL_PATTERN = re.compile(r"(https?://[^\s)]+)", re.IGNORECASE)
_DOMAIN_PATTERN = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s]*)?\b")
_DIRECT_COMMAND_PATTERN = re.compile(
    r"^(?:"
    r"(?:python|py|git|pip)\b"
    r"|show git status\b"
    r"|run (?:the )?(?:project )?unit tests\b"
    r"|install (?:python )?package\b"
    r"|create (?:file|folder|directory)\b(?!\s+(?:named|called|for)\b)"
    r"|read\b\s+\S+"
    r"|write\b.+\bto\b.+"
    r"|append\b.+\bto\b.+"
    r"|rename\b\s+\S+(?:\bto\b.+)?"
    r"|copy\b\s+\S+(?:\bto\b.+)?"
    r"|move\b\s+\S+(?:\bto\b.+)?"
    r"|delete\b\s+\S+"
    r"|list (?:directory|folder)\b"
    r"|calculate\b"
    r"|calc\b"
    r"|work out\s*[-+*/().\d]"
    r"|make a note\b"
    r")",
    re.IGNORECASE,
)
_AMBIGUOUS_DELETION_PATTERN = re.compile(
    r"^(?:"
    r"delete old files|remove the old stuff|remove old stuff|remove the old files|"
    r"\u0443\u0434\u0430\u043b\u0438 \u0441\u0442\u0430\u0440\u044b\u0435 \u0444\u0430\u0439\u043b\u044b|"
    r"\u043e\u0447\u0438\u0441\u0442\u0438 \u043d\u0435\u043d\u0443\u0436\u043d\u043e\u0435|"
    r"\u0443\u0434\u0430\u043b\u0438 \u043b\u0438\u0448\u043d\u0435\u0435 \u0438\u0437 \u043f\u0440\u043e\u0435\u043a\u0442\u0430"
    r")$",
    re.IGNORECASE,
)
_VAGUE_PROJECT_ACTION_PATTERN = re.compile(
    r"^(?:"
    r"do something useful with the project|do something helpful with the project|"
    r"make something useful with the project|"
    r"\u0441\u0434\u0435\u043b\u0430\u0439 \u0447\u0442\u043e-\u043d\u0438\u0431\u0443\u0434\u044c "
    r"\u043f\u043e\u043b\u0435\u0437\u043d\u043e\u0435 \u0441 \u043f\u0440\u043e\u0435\u043a\u0442\u043e\u043c"
    r")\.?$",
    re.IGNORECASE,
)
_EXPLICIT_PROJECT_DELETION_PATTERN = re.compile(
    r"^(?:"
    r"delete (?:the )?(?:entire|whole) project(?: without confirmation)?|"
    r"remove (?:the )?(?:entire|whole) project(?: without confirmation)?|"
    r"\u0443\u0434\u0430\u043b\u0438 (?:\u0432\u0435\u0441\u044c|\u0446\u0435\u043b\u0438\u043a\u043e\u043c) \u043f\u0440\u043e\u0435\u043a\u0442(?: \u0431\u0435\u0437 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u0438\u044f)?"
    r")\.?$",
    re.IGNORECASE,
)
_READ_ONLY_PATTERN = re.compile(
    r"\b(check|inspect|look at|show|read|summari[sz]e|tell me|review|explain|status|what changed|untracked)\b",
    re.IGNORECASE,
)
_RUSSIAN_READ_ONLY_PATTERN = re.compile(
    r"(\u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0438|\u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0439|\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438|\u0441\u043a\u0430\u0436\u0438|\u043f\u043e\u043a\u0430\u0436\u0438|\u043f\u0440\u043e\u0432\u0435\u0440\u044c|\u0441\u0442\u0430\u0442\u0443\u0441|\u043d\u0435 \u043e\u0442\u0441\u043b\u0435\u0436\u0438\u0432)",
    re.IGNORECASE,
)
_WRITE_PATTERN = re.compile(r"\b(create|make|write|append|add|prepare)\b", re.IGNORECASE)
_RUSSIAN_WRITE_PATTERN = re.compile(
    r"(\u0441\u043e\u0437\u0434\u0430\u0439|\u0441\u0434\u0435\u043b\u0430\u0439|\u0437\u0430\u043f\u0438\u0448\u0438|\u0434\u043e\u0431\u0430\u0432\u044c|\u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u044c)",
    re.IGNORECASE,
)
_BROWSER_UNSUPPORTED_PATTERN = re.compile(
    r"\b(log ?in|sign ?in|authenticate|upload|download|cookie banner|cookies|captcha|payment|purchase|checkout|register|sign up|create account)\b",
    re.IGNORECASE,
)
_BROWSER_AUTH_PATTERN = re.compile(
    r"\b(log ?in(?:to)?|sign ?in|authenticate|password|passcode|otp|mfa|2fa|verification code|account)\b",
    re.IGNORECASE,
)
_BROWSER_FORM_PATTERN = re.compile(
    r"\b(form|search field|search box|text field|input field|textarea|enter|type|fill|clear|submit)\b",
    re.IGNORECASE,
)
_BROWSER_UNSUPPORTED_FORM_PATTERN = re.compile(
    r"\b(upload|download|cookie banner|cookies|captcha|payment|purchase|checkout|register|sign up|create account)\b",
    re.IGNORECASE,
)
_BROWSER_NO_SUBMIT_PATTERN = re.compile(
    r"(without submitting|do not submit|don't submit|fill only|enter only|leave it filled|do not send|"
    r"\u0431\u0435\u0437 \u043e\u0442\u043f\u0440\u0430\u0432\u043a\u0438|"
    r"\u043d\u0435 \u043e\u0442\u043f\u0440\u0430\u0432\u043b\u044f\u0439|"
    r"\u0442\u043e\u043b\u044c\u043a\u043e \u0437\u0430\u043f\u043e\u043b\u043d\u0438|"
    r"\u0442\u043e\u043b\u044c\u043a\u043e \u0432\u0432\u0435\u0434\u0438|"
    r"\u043d\u0435 \u043d\u0430\u0436\u0438\u043c\u0430\u0439 \u043e\u0442\u043f\u0440\u0430\u0432\u0438\u0442\u044c)",
    re.IGNORECASE,
)
_BROWSER_FILL_PATTERN = re.compile(
    r"\b(enter|type|fill)\b|"
    r"(\u0432\u0432\u0435\u0434\u0438|\u043d\u0430\u0431\u0435\u0440\u0438|\u0437\u0430\u043f\u043e\u043b\u043d\u0438)",
    re.IGNORECASE,
)
_BROWSER_SUBMIT_PATTERN = re.compile(
    r"\b(submit|send)\b|(\u043e\u0442\u043f\u0440\u0430\u0432)",
    re.IGNORECASE,
)
_BROWSER_SENSITIVE_VALUE_PATTERNS = (
    re.compile(
        r"(?:enter|type|fill)\s+(?P<value>.+?)\s+into\s+(?:the\s+)?(?:password|passcode|token|otp|mfa|2fa|verification code)[^,.]*",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:enter|type|fill)\s+(?:the\s+)?(?:password|passcode|token|otp|mfa|2fa|verification code)[^,.]*?\s+(?:with|as)\s+(?P<value>.+?)(?:[,.]| and\b|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\u0432\u0432\u0435\u0434\u0438|\u043d\u0430\u0431\u0435\u0440\u0438|\u0437\u0430\u043f\u043e\u043b\u043d\u0438)\s+(?P<value>.+?)\s+"
        r"(?:\u0432|\u0432 \u043f\u043e\u043b\u0435)\s+(?:\u043f\u0430\u0440\u043e\u043b|\u043f\u0430\u0441\u0441\u043a\u043e\u0434|\u0442\u043e\u043a\u0435\u043d|\u043a\u043e\u0434 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d)[^,.]*",
        re.IGNORECASE,
    ),
)
_VISION_DESCRIBE_PATTERN = re.compile(
    r"\b(describe|analy[sz]e|what is in|what's in|look at)\b.*\b(image|picture|photo)\b|"
    r"(\u043e\u043f\u0438\u0448\u0438|\u043f\u0440\u043e\u0430\u043d\u0430\u043b\u0438\u0437\u0438\u0440\u0443\u0439)\b.*(\u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d|\u043a\u0430\u0440\u0442\u0438\u043d|\u0444\u043e\u0442\u043e)",
    re.IGNORECASE,
)
_VISION_OCR_PATTERN = re.compile(
    r"\b(read|extract)\b.*\b(text|ocr)\b|\btext in\b.*\.(?:png|jpg|jpeg|webp)\b|"
    r"(\u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0439|\u0438\u0437\u0432\u043b\u0435\u043a\u0438)\b.*(\u0442\u0435\u043a\u0441\u0442|ocr)",
    re.IGNORECASE,
)
_VISION_FIND_PATTERN = re.compile(
    r"\b(find|locate)\b.*\b(image|picture|photo|\.png|\.jpg|\.jpeg|\.webp)\b|"
    r"(\u043d\u0430\u0439\u0434\u0438|\u043e\u0442\u044b\u0449\u0438)\b.*(\u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d|\u043a\u0430\u0440\u0442\u0438\u043d|\u0444\u043e\u0442\u043e)",
    re.IGNORECASE,
)
_VISION_IMAGE_HINT_PATTERN = re.compile(
    r"\b(image|picture|photo|ocr|visual|vision)\b|"
    r"(\u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d|\u043a\u0430\u0440\u0442\u0438\u043d|\u0444\u043e\u0442\u043e|\u0432\u0438\u0437\u0443\u0430\u043b|\u0442\u0435\u043a\u0441\u0442 \u043d\u0430 \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d)",
    re.IGNORECASE,
)
# NOTE (2026-09-19, real-hardware bug): this used to also block "desktop"/"screen
# capture"/"window capture" and Russian "\u044d\u043a\u0440\u0430\u043d"/"\u043e\u043a\u043d\u043e", written back when RFC-007A was
# vision's only capability (describe/OCR/find on a trusted local image FILE only -- no
# screen access at all). RFC-007C later added real desktop screenshot support
# (desktop.capture_screen/capture_window + vision.describe_desktop_capture and friends),
# but this blocklist was never updated, so a live request like "\u0434\u0436\u0430\u0440\u0432\u0438\u0441 \u0447\u0442\u043e \u0442\u044b \u0432\u0438\u0434\u0438\u0448\u044c \u043d\u0430
# \u044d\u043a\u0440\u0430\u043d\u0435" was still hard-rejected with "not implemented in RFC-007A" before it ever reached
# the part of the system that actually now supports it -- see
# _requested_desktop_visual_operation() below, which is checked first and gives screen/
# desktop requests their own supported classification instead of falling into this
# catch-all. Camera/webcam/face-recognition/speaker-verification/live-microphone/captcha/QR
# genuinely still have no implementation anywhere in this project, so those stay blocked.
_VISION_UNSUPPORTED_PATTERN = re.compile(
    r"\b(camera|webcam|face|recognition|speaker|voice|microphone|captcha|qr)\b|"
    r"(\u043a\u0430\u043c\u0435\u0440|\u0432\u0435\u0431\u043a\u0430\u043c|\u043b\u0438\u0446|\u0440\u0430\u0441\u043f\u043e\u0437\u043d\u0430\u0432|\u0433\u043e\u043b\u043e\u0441|\u043c\u0438\u043a\u0440\u043e\u0444\u043e\u043d|\u043a\u0430\u043f\u0447|\u043a\u0443\u0430\u0440)",
    re.IGNORECASE,
)
# Owner-requested (2026-09-19 "screen understanding" discussion): a deterministic
# classification for "what's on my screen" style requests, mirroring how
# _requested_browser_visual_operation() narrows the LLM planner's tool catalog down to
# just the relevant browser+vision tools instead of leaving it to choose from the entire
# catalog. Deliberately keyed on "screen"/"desktop"/"monitor" (and Russian equivalents) so
# it never overlaps with the browser-visual patterns above, which are keyed on "page"/
# "tab"/"website"/"site" instead.
_DESKTOP_VISUAL_HINT_PATTERN = re.compile(
    r"\b(screen|desktop)\b|"
    r"\b(my|the) monitor\b|"
    r"(\u044d\u043a\u0440\u0430\u043d|\u0440\u0430\u0431\u043e\u0447\u0435\u043c \u0441\u0442\u043e\u043b\u0435|\u043c\u043e\u043d\u0438\u0442\u043e\u0440)",
    re.IGNORECASE,
)
_DESKTOP_VISUAL_FIND_PATTERN = re.compile(
    r"\b(find|locate)\b.*\b(on (my |the )?screen|on (my |the )?desktop)\b|"
    r"(\u043d\u0430\u0439\u0434\u0438|\u043e\u0442\u044b\u0449\u0438).*(\u043d\u0430 \u044d\u043a\u0440\u0430\u043d\u0435|\u043d\u0430 \u0440\u0430\u0431\u043e\u0447\u0435\u043c \u0441\u0442\u043e\u043b\u0435)",
    re.IGNORECASE,
)
_DESKTOP_VISUAL_TEXT_PATTERN = re.compile(
    r"\b(read|extract)\b.*\b(screen|desktop)\b|"
    r"(\u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0439|\u0438\u0437\u0432\u043b\u0435\u043a\u0438).*(\u044d\u043a\u0440\u0430\u043d)",
    re.IGNORECASE,
)
_DESKTOP_VISUAL_DESCRIBE_PATTERN = re.compile(
    r"\b(what'?s on|what is on|describe|look at|tell me what'?s on|capture)\b.*\b(my screen|the screen|screen|desktop)\b|"
    r"\b(what do you see|what can you see)\b",
    re.IGNORECASE,
)
# Real owner-reported failure (2026-09-19): a spoken "what's on my screen" request,
# transcribed as Russian, fell through to the plain conversational AI (which then
# hallucinated a "I can't see anything, I'm virtual" reply, occasionally with mixed-in
# Chinese characters) because the exact phrase spoken didn't match the rigid, fixed-word-
# order Russian alternatives above ("\u0447\u0442\u043e \u043d\u0430 \u044d\u043a\u0440\u0430\u043d\u0435", "\u0447\u0442\u043e \u0442\u044b \u0432\u0438\u0434\u0438\u0448\u044c", "\u0447\u0442\u043e \u0432\u0438\u0434\u043d\u043e" -- each
# locked to that exact sequence). Natural spoken Russian doesn't reliably keep that order
# or stick to one verb ("\u0433\u043b\u044f\u043d\u044c \u0447\u0442\u043e \u0442\u0430\u043c \u043d\u0430 \u044d\u043a\u0440\u0430\u043d\u0435", "\u0441\u043a\u0430\u0436\u0438, \u0447\u0442\u043e \u0441\u0435\u0439\u0447\u0430\u0441 \u043f\u0440\u043e\u0438\u0441\u0445\u043e\u0434\u0438\u0442 \u043d\u0430
# \u044d\u043a\u0440\u0430\u043d\u0435", "\u0447\u0442\u043e \u0432\u044b \u0432\u0438\u0434\u0438\u0442\u0435"), so this uses the same word-level keyword-set approach already
# proven for _ru_open_intent_command() in command_normalizer.py instead of phrase-locked
# regex alternation.
_RU_DESKTOP_DESCRIBE_VERBS = {
    "\u0432\u0438\u0434\u0438\u0448\u044c",  # \u0432\u0438\u0434\u0438\u0448\u044c
    "\u0432\u0438\u0434\u0435\u0448\u044c",  # \u0432\u0438\u0434\u0435\u0448\u044c (common misspelling/mishearing)
    "\u0432\u0438\u0434\u0438\u0442\u0435",  # \u0432\u0438\u0434\u0438\u0442\u0435
    "\u0432\u0438\u0436\u0443",  # \u0432\u0438\u0436\u0443
    "\u0432\u0438\u0434\u043d\u043e",  # \u0432\u0438\u0434\u043d\u043e
    "\u043f\u043e\u043a\u0430\u0436\u0438",  # \u043f\u043e\u043a\u0430\u0436\u0438
    "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438",  # \u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438
    "\u0441\u043a\u0430\u0436\u0438",  # \u0441\u043a\u0430\u0436\u0438
    "\u043e\u0431\u044a\u044f\u0441\u043d\u0438",  # \u043e\u0431\u044a\u044f\u0441\u043d\u0438
    "\u0433\u043b\u044f\u043d\u044c",  # \u0433\u043b\u044f\u043d\u044c
    "\u0432\u0437\u0433\u043b\u044f\u043d\u0438",  # \u0432\u0437\u0433\u043b\u044f\u043d\u0438
    "\u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0438",  # \u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0438
    "\u0441\u043c\u043e\u0442\u0440\u0438",  # \u0441\u043c\u043e\u0442\u0440\u0438
    "\u043e\u043f\u0438\u0448\u0438",  # \u043e\u043f\u0438\u0448\u0438
    "\u043f\u0440\u043e\u0438\u0441\u0445\u043e\u0434\u0438\u0442",  # \u043f\u0440\u043e\u0438\u0441\u0445\u043e\u0434\u0438\u0442
}


def _ru_desktop_describe_intent(normalized: str) -> bool:
    lowered_words = [word.strip(",.!?\u2014-\u00ab\u00bb") for word in normalized.lower().split()]
    if any(word in _RU_DESKTOP_DESCRIBE_VERBS for word in lowered_words):
        return True
    # No-verb fallback: a bare "\u0447\u0442\u043e ... \u043d\u0430 \u044d\u043a\u0440\u0430\u043d\u0435" (what's [there] on [my] screen) with
    # no explicit verb at all still counts -- the caller already confirmed a screen/
    # desktop/monitor mention is present somewhere in the sentence.
    return "\u0447\u0442\u043e" in lowered_words  # \u0447\u0442\u043e (what)
_BROWSER_VISUAL_DESCRIBE_PATTERN = re.compile(
    r"\b(visually describe|describe .*visible|what is visible|what's visible|look at .*visible)\b|"
    r"(\u0432\u0438\u0437\u0443\u0430\u043b\u044c\u043d\u043e \u043e\u043f\u0438\u0448\u0438|"
    r"\u0447\u0442\u043e \u0432\u0438\u0434\u043d\u043e|"
    r"\u0447\u0442\u043e \u0432\u0438\u0434\u0438\u043c\u043e)",
    re.IGNORECASE,
)
_BROWSER_VISUAL_TEXT_PATTERN = re.compile(
    r"\b(read|extract)\b.*\b(visible text|text on the current browser page|text on the current page|text on the browser page)\b|"
    r"(\u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0439|\u0438\u0437\u0432\u043b\u0435\u043a\u0438).*(\u0432\u0438\u0434\u0438\u043c\u044b\u0439 \u0442\u0435\u043a\u0441\u0442)",
    re.IGNORECASE,
)
_BROWSER_VISUAL_FIND_PATTERN = re.compile(
    r"\b(visually find|find .*visible|locate .*visible)\b|"
    r"(\u0432\u0438\u0437\u0443\u0430\u043b\u044c\u043d\u043e \u043d\u0430\u0439\u0434\u0438|\u043d\u0430\u0439\u0434\u0438 .* \u043d\u0430 \u0441\u0442\u0440\u0430\u043d\u0438\u0446)",
    re.IGNORECASE,
)
_BROWSER_VISUAL_PAGE_HINT_PATTERN = re.compile(
    r"\b(page|browser page|current page|tab|website|site)\b|"
    r"(\u0441\u0442\u0440\u0430\u043d\u0438\u0446|\u0432\u043a\u043b\u0430\u0434\u043a|\u0441\u0430\u0439\u0442)",
    re.IGNORECASE,
)


def interpret_task(raw_input: str, *, clarification_request: ClarificationRequest | None = None) -> NaturalLanguageTask:
    text = raw_input.strip()
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    language = "ru" if any("\u0400" <= char <= "\u04ff" for char in normalized) else "en"
    referenced_paths = [match.group(1).rstrip(".,)") for match in _PATH_PATTERN.finditer(normalized)]
    referenced_urls = _extract_urls(normalized)
    artifacts = _extract_artifacts(normalized, referenced_paths)
    contents = _extract_contents(normalized)
    output_texts = _extract_output_texts(normalized)
    requires_execution = _requires_execution(normalized)
    requires_tests = _requires_tests(normalized)
    requires_summary = _requires_summary(normalized)
    read_only_task = _is_read_only_task(normalized)
    destructive_scope_unclear = _AMBIGUOUS_DELETION_PATTERN.match(lowered) is not None
    vague_project_action = _VAGUE_PROJECT_ACTION_PATTERN.match(lowered) is not None
    explicit_project_deletion = _EXPLICIT_PROJECT_DELETION_PATTERN.match(lowered) is not None
    requested_operation = _requested_operation(normalized, artifacts, read_only_task, requires_tests, requires_execution)
    browser_operation = _requested_browser_operation(normalized, artifacts, referenced_urls)
    browser_visual_operation = _requested_browser_visual_operation(normalized, referenced_urls)
    desktop_visual_operation = _requested_desktop_visual_operation(normalized)
    vision_operation = _requested_vision_operation(normalized, referenced_paths)
    requires_code_write = _requires_written_artifact(normalized)

    if clarification_request is not None and not explicit_project_deletion:
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.CLARIFICATION_RESPONSE,
            confidence=ConfidenceCategory.HIGH,
            goal=f"{clarification_request.original_goal} Clarification: {normalized}",
            expected_result=clarification_request.original_goal,
            referenced_paths=referenced_paths,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary,
            read_only_task=read_only_task,
            requires_execution=requires_execution,
            requires_verification=requires_execution or requires_tests,
            requires_stdout_match=bool(output_texts and requires_execution),
            requires_tests=requires_tests,
            requires_code_write=requires_code_write,
            requested_operation=requested_operation,
        )

    if destructive_scope_unclear:
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.AMBIGUOUS_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            ambiguity_level="high",
            referenced_paths=referenced_paths,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            read_only_task=False,
            destructive_scope_unclear=True,
            requested_operation="ambiguous_delete",
        )

    if vague_project_action:
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.AMBIGUOUS_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            ambiguity_level="high",
            referenced_paths=referenced_paths,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary,
            read_only_task=False,
            destructive_scope_unclear=False,
            requested_operation="ambiguous_project_request",
        )

    if explicit_project_deletion:
        constraints: list[TaskConstraint] = [TaskConstraint("target_scope", "project_root")]
        if _requests_without_confirmation(normalized):
            constraints.append(TaskConstraint("approval_bypass_requested", "true"))
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.ACTIONABLE_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            expected_result="project root deletion request",
            referenced_paths=["."],
            execution_requested=True,
            ambiguity_level="low",
            language=language,
            constraints=constraints,
            requested_artifacts=["project_root"],
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=False,
            read_only_task=False,
            destructive_scope_unclear=False,
            requires_execution=True,
            requires_verification=False,
            requires_stdout_match=False,
            requires_tests=False,
            requires_code_write=False,
            requested_operation="delete_project_root",
        )

    effective_browser_operation = browser_visual_operation or browser_operation

    if effective_browser_operation.startswith("browser_unsupported_"):
        sensitive_value = _extract_sensitive_browser_value(normalized)
        safe_text = _redact_sensitive_browser_text(text, sensitive_value)
        safe_normalized = _redact_sensitive_browser_text(normalized, sensitive_value)
        safe_contents = _sanitize_text_list(contents, sensitive_value)
        safe_outputs = _sanitize_text_list(output_texts, sensitive_value)
        constraints: list[TaskConstraint] = []
        if sensitive_value:
            constraints.extend(
                [
                    TaskConstraint("sensitive_field", "password"),
                    TaskConstraint("sensitive_value_length", str(len(sensitive_value))),
                ]
            )
        return NaturalLanguageTask(
            raw_input=safe_text,
            normalized_input=safe_normalized,
            intent=TaskIntent.UNSUPPORTED_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=safe_normalized,
            referenced_paths=referenced_paths,
            language=language,
            constraints=constraints,
            requested_artifacts=artifacts,
            requested_contents=safe_contents,
            requested_output_texts=safe_outputs,
            requested_summary=requires_summary,
            read_only_task=False,
            requires_execution=False,
            requires_verification=False,
            requires_tests=False,
            requires_code_write=False,
            requested_operation=effective_browser_operation,
        )

    if desktop_visual_operation and not browser_visual_operation:
        constraints: list[TaskConstraint] = []
        query = _extract_vision_query(normalized)
        if query:
            constraints.append(TaskConstraint("vision_query", query))
        detail_level = _extract_vision_detail_level(normalized)
        if detail_level:
            constraints.append(TaskConstraint("vision_detail_level", detail_level))
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.ACTIONABLE_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            expected_result=_expected_result(normalized, artifacts, output_texts, desktop_visual_operation),
            referenced_paths=referenced_paths,
            constraints=constraints,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary or desktop_visual_operation == "desktop_visual_describe",
            read_only_task=True,
            destructive_scope_unclear=False,
            requires_execution=False,
            requires_verification=True,
            requires_stdout_match=False,
            requires_tests=False,
            requires_code_write=False,
            requested_operation=desktop_visual_operation,
        )

    if vision_operation == "vision_unsupported":
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.UNSUPPORTED_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            referenced_paths=referenced_paths,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary,
            read_only_task=True,
            requires_execution=False,
            requires_verification=False,
            requires_tests=False,
            requires_code_write=False,
            requested_operation=vision_operation,
        )

    if vision_operation == "vision_missing_path" and not browser_visual_operation:
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.AMBIGUOUS_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            ambiguity_level="high",
            referenced_paths=[],
            language=language,
            requested_artifacts=[],
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary,
            read_only_task=True,
            destructive_scope_unclear=False,
            requires_execution=False,
            requires_verification=False,
            requires_tests=False,
            requires_code_write=False,
            requested_operation=vision_operation,
        )

    if vision_operation and not browser_visual_operation:
        constraints: list[TaskConstraint] = []
        query = _extract_vision_query(normalized)
        if query:
            constraints.append(TaskConstraint("vision_query", query))
        detail_level = _extract_vision_detail_level(normalized)
        if detail_level:
            constraints.append(TaskConstraint("vision_detail_level", detail_level))
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.ACTIONABLE_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            expected_result=_expected_result(normalized, artifacts, output_texts, vision_operation),
            referenced_paths=referenced_paths,
            constraints=constraints,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary or vision_operation == "vision_describe_image",
            read_only_task=True,
            destructive_scope_unclear=False,
            requires_execution=False,
            requires_verification=True,
            requires_stdout_match=False,
            requires_tests=False,
            requires_code_write=False,
            requested_operation=vision_operation,
        )

    if _looks_like_direct_command(normalized) and not effective_browser_operation.startswith("browser_"):
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.DIRECT_COMMAND,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            referenced_paths=referenced_paths,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary,
            read_only_task=read_only_task,
            requires_execution=requires_execution,
            requires_verification=requires_execution or requires_tests,
            requires_stdout_match=bool(output_texts and requires_execution),
            requires_tests=requires_tests,
            requires_code_write=requires_code_write,
            requested_operation=requested_operation,
        )

    if _UNSUPPORTED_PATTERN.search(normalized):
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.UNSUPPORTED_TASK,
            confidence=ConfidenceCategory.HIGH,
            goal=normalized,
            referenced_paths=referenced_paths,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary,
            read_only_task=read_only_task,
            requires_execution=requires_execution,
            requires_verification=requires_execution or requires_tests,
            requires_stdout_match=bool(output_texts and requires_execution),
            requires_tests=requires_tests,
            requires_code_write=requires_code_write,
            requested_operation=requested_operation,
        )

    if _CONVERSATION_PATTERN.search(normalized) and not (_ACTION_PATTERN.search(normalized) or _RUSSIAN_ACTION_PATTERN.search(normalized)):
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.CONVERSATION,
            confidence=ConfidenceCategory.MEDIUM,
            goal=normalized,
            execution_requested=False,
            referenced_paths=referenced_paths,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary,
            read_only_task=read_only_task,
            requires_execution=False,
            requires_verification=False,
            requires_tests=False,
            requested_operation=requested_operation,
        )

    if _ACTION_PATTERN.search(normalized) or _RUSSIAN_ACTION_PATTERN.search(normalized) or referenced_paths:
        constraints: list[TaskConstraint] = []
        if "json" in lowered:
            constraints.append(TaskConstraint("format", "json"))
        if "small" in lowered or "simple" in lowered or "\u043d\u0435\u0431\u043e\u043b\u044c\u0448" in lowered or "\u043c\u0430\u043b\u0435\u043d\u044c\u043a" in lowered:
            constraints.append(TaskConstraint("size", "small"))
        if effective_browser_operation == "browser_form_fill" and _browser_submission_is_prohibited(normalized):
            constraints.append(TaskConstraint("must_not_submit", "true"))
        browser_visual_query = _extract_browser_visual_query(normalized)
        if browser_visual_query:
            constraints.append(TaskConstraint("browser_visual_query", browser_visual_query))
        return NaturalLanguageTask(
            raw_input=text,
            normalized_input=normalized,
            intent=TaskIntent.ACTIONABLE_TASK,
            confidence=ConfidenceCategory.MEDIUM,
            goal=normalized,
            expected_result=_expected_result(normalized, artifacts, output_texts, effective_browser_operation or requested_operation),
            referenced_paths=referenced_paths,
            constraints=constraints,
            language=language,
            requested_artifacts=artifacts,
            requested_contents=contents,
            requested_output_texts=output_texts,
            requested_summary=requires_summary,
            read_only_task=read_only_task or bool(browser_visual_operation),
            destructive_scope_unclear=False,
            requires_execution=requires_execution,
            requires_verification=requires_execution or requires_tests or requires_summary,
            requires_stdout_match=bool(output_texts and requires_execution),
            requires_tests=requires_tests,
            requires_code_write=requires_code_write,
            requested_operation=effective_browser_operation or requested_operation,
        )

    return NaturalLanguageTask(
        raw_input=text,
        normalized_input=normalized,
        intent=TaskIntent.CONVERSATION,
        confidence=ConfidenceCategory.LOW,
        goal=normalized,
        execution_requested=False,
        referenced_paths=referenced_paths,
        language=language,
        requested_artifacts=artifacts,
        requested_contents=contents,
        requested_output_texts=output_texts,
        requested_summary=requires_summary,
        read_only_task=read_only_task,
        requires_execution=False,
        requires_verification=False,
        requires_tests=False,
        requested_operation=requested_operation,
    )


def clarification_for(task: NaturalLanguageTask) -> ClarificationRequest:
    goal = task.goal
    if task.language == "ru":
        if task.destructive_scope_unclear:
            question = "\u041a\u0430\u043a\u0438\u0435 \u0438\u043c\u0435\u043d\u043d\u043e \u0444\u0430\u0439\u043b\u044b \u0438\u043b\u0438 \u043f\u0430\u043f\u043a\u0438 \u0432\u044b \u0441\u0447\u0438\u0442\u0430\u0435\u0442\u0435 \u0441\u0442\u0430\u0440\u044b\u043c\u0438?"
        elif task.requested_operation == "vision_missing_path":
            question = "\u0423\u043a\u0430\u0436\u0438\u0442\u0435 \u0442\u043e\u0447\u043d\u044b\u0439 \u043f\u0443\u0442\u044c \u043a \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044e."
        else:
            question = "\u041a\u0430\u043a\u043e\u0439 \u0438\u043c\u0435\u043d\u043d\u043e \u0444\u0430\u0439\u043b, \u043f\u0443\u0442\u044c \u0438\u043b\u0438 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u043d\u0443\u0436\u043d\u043e \u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u044c?"
    else:
        if task.destructive_scope_unclear:
            question = "Which specific files or folders do you consider old?"
        elif task.requested_operation == "vision_missing_path":
            question = "What exact image path do you want me to analyze?"
        else:
            question = "What exact file, path, or output do you want me to use?"
    return ClarificationRequest(question=question, reason="important details are missing", original_goal=goal)


def _extract_artifacts(normalized: str, referenced_paths: list[str]) -> list[str]:
    artifacts = list(referenced_paths)
    pattern = re.compile(r"(?:called|named|file|script|module)\s+([A-Za-z0-9_.-]+\.(?:py|txt|md|json|log))", re.IGNORECASE)
    for match in pattern.finditer(normalized):
        artifacts.append(match.group(1))
    unique: list[str] = []
    seen: set[str] = set()
    for artifact in artifacts:
        lowered = artifact.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(artifact)
    return unique


def _extract_urls(normalized: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in _URL_PATTERN.finditer(normalized):
        value = match.group(1).rstrip(".,)")
        lowered = value.lower()
        if lowered not in seen:
            seen.add(lowered)
            urls.append(value)
    for match in _DOMAIN_PATTERN.finditer(normalized):
        value = match.group(0).rstrip(".,)")
        lowered = value.lower()
        if lowered in seen:
            continue
        if value.lower().endswith((".py", ".txt", ".md", ".json", ".log", ".png", ".jpg", ".jpeg")):
            continue
        seen.add(lowered)
        urls.append(f"https://{value}")
    return urls


def _extract_output_texts(normalized: str) -> list[str]:
    patterns = (
        re.compile(r"(?:prints?|outputs?|says?)\s+([A-Za-z0-9 _.-]+?)(?:[,.]| then| and|$)", re.IGNORECASE),
        re.compile(r"(?:\u043d\u0430\u043f\u0435\u0447\u0430\u0442\u0430\u0435\u0442|\u0432\u044b\u0432\u043e\u0434\u0438\u0442)\s+([A-Za-z0-9 _.-]+?)(?:[,.]| \u0438| \u0437\u0430\u0442\u0435\u043c|$)", re.IGNORECASE),
    )
    outputs: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            candidate = match.group(1).strip().strip("\"'")
            if candidate:
                outputs.append(candidate)
    return outputs


def _extract_contents(normalized: str) -> list[str]:
    patterns = (
        re.compile(r"(?:containing|with text)\s+([A-Za-z0-9 _.,'\"-]+?)(?:[,.]| then| and|$)", re.IGNORECASE),
        re.compile(r"(?:\u0441 \u0442\u0435\u043a\u0441\u0442\u043e\u043c)\s+([A-Za-z0-9 _.,'\"-]+?)(?:[,.]| \u0438| \u0437\u0430\u0442\u0435\u043c|$)", re.IGNORECASE),
    )
    contents: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(normalized):
            candidate = match.group(1).strip().strip("\"'")
            if candidate:
                contents.append(candidate)
    return contents


def _requires_execution(normalized: str) -> bool:
    lowered = normalized.lower()
    return any(
        token in lowered
        for token in (
            "run it",
            "run the file",
            "run it.",
            "verify it by running it",
            "execute it",
            "check it by running it",
            "\u0437\u0430\u043f\u0443\u0441\u0442\u0438",
            "\u043f\u0440\u043e\u0432\u0435\u0440\u044c \u0437\u0430\u043f\u0443\u0441\u043a\u043e\u043c",
        )
    )


def _requires_tests(normalized: str) -> bool:
    lowered = normalized.lower()
    return any(
        token in lowered
        for token in (
            "add tests",
            "add a test",
            "with tests",
            "unit test",
            "test it",
            "\u0434\u043e\u0431\u0430\u0432\u044c \u0442\u0435\u0441\u0442",
            "\u0434\u043e\u0431\u0430\u0432\u044c \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0443",
            "\u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0443",
            "\u0442\u0435\u0441\u0442",
        )
    )


def _requires_summary(normalized: str) -> bool:
    lowered = normalized.lower()
    return any(
        token in lowered
        for token in (
            "tell me",
            "summarize",
            "summary",
            "which files",
            "what changed",
            "page title",
            "what is on this page",
            "what this page is about",
            "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438",
            "\u0441\u043a\u0430\u0436\u0438",
            "\u043a\u0430\u043a\u0438\u0435 \u0444\u0430\u0439\u043b\u044b",
            "\u0437\u0430\u0433\u043e\u043b\u043e\u0432\u043e\u043a",
            "\u043e \u0447\u0451\u043c \u044d\u0442\u0430 \u0441\u0442\u0440\u0430\u043d\u0438\u0446\u0430",
        )
    )


def _is_read_only_task(normalized: str) -> bool:
    lowered = normalized.lower()
    has_read = bool(_READ_ONLY_PATTERN.search(normalized) or _RUSSIAN_READ_ONLY_PATTERN.search(normalized))
    has_write = bool(_WRITE_PATTERN.search(normalized) or _RUSSIAN_WRITE_PATTERN.search(normalized))
    has_delete = any(token in lowered for token in ("delete", "remove", "\u0443\u0434\u0430\u043b\u0438", "\u043e\u0447\u0438\u0441\u0442\u0438"))
    if ("git" in lowered or "\u0440\u0435\u043f\u043e\u0437\u0438\u0442\u043e\u0440" in lowered) and has_read and not has_write and not has_delete:
        return True
    if has_read and not has_write and not has_delete:
        return True
    return False


def _requested_operation(normalized: str, artifacts: list[str], read_only_task: bool, requires_tests: bool, requires_execution: bool) -> str:
    lowered = normalized.lower()
    if ("git" in lowered or "\u0440\u0435\u043f\u043e\u0437\u0438\u0442\u043e\u0440" in lowered) and read_only_task:
        return "git_status"
    if artifacts and any(artifact.lower().endswith(".py") for artifact in artifacts):
        if requires_tests:
            return "python_module_with_tests"
        if requires_execution:
            return "python_script"
    if requires_tests:
        return "module_with_tests"
    if read_only_task:
        return "read_only_summary"
    if artifacts:
        return "file_task"
    return ""


def _requested_browser_operation(normalized: str, artifacts: list[str], referenced_urls: list[str]) -> str:
    lowered = normalized.lower()
    browser_tokens = (
        "website",
        "web page",
        "page title",
        "current page",
        "current tab",
        "new tab",
        "open tabs",
        "list tabs",
        "switch tab",
        "previous tab",
        "next tab",
        "reload",
        "refresh",
        "scroll",
        "click",
        "link",
        "button",
        "search field",
        "search box",
        "input field",
        "text field",
        "textarea",
        "form",
        "enter",
        "type",
        "fill",
        "clear",
        "submit",
        "\u0441\u0442\u0440\u0430\u043d\u0438\u0446",
        "\u0437\u0430\u0433\u043e\u043b\u043e\u0432\u043e\u043a",
        "\u0432\u043a\u043b\u0430\u0434\u043a",
        "\u043f\u0440\u043e\u043a\u0440\u0443\u0442",
        "\u043d\u0430\u0436\u043c\u0438",
        "\u043e\u0431\u043d\u043e\u0432",
        "\u043f\u0435\u0440\u0435\u043a\u043b\u044e\u0447",
        "\u0441\u0441\u044b\u043b\u043a",
        "\u043a\u043d\u043e\u043f\u043a",
        "\u0444\u043e\u0440\u043c",
        "\u043f\u043e\u0438\u0441\u043a\u043e\u0432",
        "\u043f\u043e\u043b\u0435",
        "\u0432\u0432\u0435\u0434\u0438",
        "\u043d\u0430\u0431\u0435\u0440\u0438",
        "\u0437\u0430\u043f\u043e\u043b\u043d\u0438",
        "\u043e\u0447\u0438\u0441\u0442\u0438",
        "\u043e\u0442\u043f\u0440\u0430\u0432",
    )
    if not referenced_urls and not any(token in lowered for token in browser_tokens):
        return ""
    if _BROWSER_AUTH_PATTERN.search(normalized) or any(
        token in lowered
        for token in (
            "\u043f\u0430\u0440\u043e\u043b",
            "\u0432\u043e\u0439\u0434\u0438",
            "\u0432\u043e\u0439\u0442\u0438",
            "\u0430\u0432\u0442\u043e\u0440\u0438\u0437",
            "\u0430\u043a\u043a\u0430\u0443\u043d\u0442",
            "\u043a\u043e\u0434 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436",
        )
    ):
        return "browser_unsupported_auth"
    if _BROWSER_UNSUPPORTED_FORM_PATTERN.search(normalized) or any(
        token in lowered
        for token in (
            "\u0437\u0430\u0433\u0440\u0443\u0437",
            "\u0441\u043a\u0430\u0447\u0430\u0439",
            "\u043a\u0430\u043f\u0447",
            "\u043a\u0430\u043f\u0447\u0430",
            "\u043e\u043f\u043b\u0430\u0442",
            "\u043f\u043e\u043a\u0443\u043f",
            "\u0440\u0435\u0433\u0438\u0441\u0442\u0440",
        )
    ):
        return "browser_unsupported_interaction"
    if _BROWSER_UNSUPPORTED_PATTERN.search(normalized):
        return "browser_unsupported_interaction"
    if _BROWSER_FORM_PATTERN.search(normalized) or any(
        token in lowered
        for token in (
            "\u0444\u043e\u0440\u043c",
            "\u043f\u043e\u043b\u0435",
            "\u043f\u043e\u0438\u0441\u043a\u043e\u0432",
            "\u0432\u0432\u0435\u0434\u0438",
            "\u043d\u0430\u0431\u0435\u0440\u0438",
            "\u0437\u0430\u043f\u043e\u043b\u043d\u0438",
            "\u043e\u0447\u0438\u0441\u0442\u0438",
            "\u043e\u0442\u043f\u0440\u0430\u0432",
        )
    ):
        if any(token in lowered for token in ("clear", "\u043e\u0447\u0438\u0441\u0442")):
            return "browser_form_clear"
        if _browser_submission_is_prohibited(normalized):
            if _BROWSER_FILL_PATTERN.search(normalized):
                return "browser_form_fill"
            return "browser_form_inspection"
        if _BROWSER_SUBMIT_PATTERN.search(normalized):
            if _BROWSER_FILL_PATTERN.search(normalized):
                return "browser_form_fill_submit"
            return "browser_form_submit"
        if _BROWSER_FILL_PATTERN.search(normalized):
            return "browser_form_fill"
        return "browser_form_inspection"
    if any(token in lowered for token in ("inspect clickable", "clickable elements", "\u043a\u043b\u0438\u043a\u0430\u0431\u0435\u043b", "\u043d\u0430\u0436\u0438\u043c\u0430\u0435\u043c")):
        return "browser_clickable_inspection"
    if any(token in lowered for token in ("new tab", "another tab", "separate tab", "\u043d\u043e\u0432\u043e\u0439 \u0432\u043a\u043b\u0430\u0434", "\u043d\u043e\u0432\u0443\u044e \u0432\u043a\u043b\u0430\u0434")):
        return "browser_new_tab"
    if any(token in lowered for token in ("list open tabs", "list tabs", "show tabs", "open tabs", "\u0441\u043f\u0438\u0441\u043e\u043a \u0432\u043a\u043b\u0430\u0434", "\u043f\u043e\u043a\u0430\u0436\u0438 \u0432\u043a\u043b\u0430\u0434")):
        return "browser_list_tabs"
    if any(token in lowered for token in ("switch back", "switch tab", "previous tab", "next tab", "\u043f\u0435\u0440\u0435\u043a\u043b\u044e\u0447", "\u043f\u0440\u0435\u0434\u044b\u0434\u0443\u0449")):
        return "browser_switch_tab"
    if any(token in lowered for token in ("close current tab", "close the current tab", "close tab", "\u0437\u0430\u043a\u0440\u043e\u0439 \u0432\u043a\u043b\u0430\u0434")):
        return "browser_close_tab"
    if any(token in lowered for token in ("reload", "refresh", "\u043e\u0431\u043d\u043e\u0432")):
        return "browser_reload"
    if any(token in lowered for token in ("scroll to", "\u043f\u0440\u043e\u043a\u0440\u0443\u0442\u0438 \u043a")):
        return "browser_scroll_to_element"
    if any(token in lowered for token in ("scroll", "\u043f\u0440\u043e\u043a\u0440\u0443\u0442")):
        return "browser_scroll"
    if any(token in lowered for token in ("click", "\u043d\u0430\u0436\u043c\u0438")):
        return "browser_click"
    if any(token in lowered for token in ("screenshot", "\u0441\u043d\u0438\u043c\u043e\u043a", "\u0441\u043a\u0440\u0438\u043d\u0448\u043e\u0442")):
        return "browser_screenshot"
    if any(token in lowered for token in ("page title", "title", "\u0437\u0430\u0433\u043e\u043b\u043e\u0432\u043e\u043a")):
        return "browser_title"
    if any(token in lowered for token in ("summarize", "summary", "tell me", "\u043a\u0440\u0430\u0442\u043a\u043e", "\u0440\u0430\u0441\u0441\u043a\u0430\u0436\u0438", "\u043e \u0447\u0451\u043c")):
        return "browser_summary"
    return "browser_navigation"


def _requested_browser_visual_operation(normalized: str, referenced_urls: list[str]) -> str:
    lowered = normalized.lower()
    if not referenced_urls and not _BROWSER_VISUAL_PAGE_HINT_PATTERN.search(normalized):
        return ""
    if _BROWSER_VISUAL_FIND_PATTERN.search(normalized):
        return "browser_visual_find_element"
    if _BROWSER_VISUAL_TEXT_PATTERN.search(normalized):
        return "browser_visual_extract_text"
    if _BROWSER_VISUAL_DESCRIBE_PATTERN.search(normalized):
        return "browser_visual_describe"
    if any(
        token in lowered
        for token in (
            "visually describe the page",
            "visible page text",
            "visible on the current page",
            "text on the current browser page",
            "text on the current page",
            "text on the browser page",
            "\u0432\u0438\u0437\u0443\u0430\u043b\u044c\u043d\u043e \u043e\u043f\u0438\u0448\u0438",
            "\u0432\u0438\u0434\u0438\u043c\u044b\u0439 \u0442\u0435\u043a\u0441\u0442",
        )
    ):
        if "text" in lowered or "\u0442\u0435\u043a\u0441\u0442" in lowered:
            return "browser_visual_extract_text"
        return "browser_visual_describe"
    return ""


def _looks_like_code_request(normalized: str) -> bool:
    lowered = normalized.lower()
    return any(
        token in lowered
        for token in (
            ".py",
            "python file",
            "python script",
            "module",
            "script",
            "\u043c\u043e\u0434\u0443\u043b",
            "\u0441\u043a\u0440\u0438\u043f\u0442",
            "upper case",
            "\u0432 \u0432\u0435\u0440\u0445\u043d\u0438\u0439 \u0440\u0435\u0433\u0438\u0441\u0442\u0440",
        )
    )


def _requires_written_artifact(normalized: str) -> bool:
    lowered = normalized.lower()
    if _looks_like_code_request(normalized):
        return True
    return any(
        token in lowered
        for token in (
            "create",
            "write",
            "append",
            "prepare",
            "\u0441\u043e\u0437\u0434\u0430\u0439",
            "\u0437\u0430\u043f\u0438\u0448\u0438",
            "\u0434\u043e\u0431\u0430\u0432\u044c",
            "\u043f\u043e\u0434\u0433\u043e\u0442\u043e\u0432\u044c",
        )
    )


def _requests_without_confirmation(normalized: str) -> bool:
    lowered = normalized.lower()
    return "without confirmation" in lowered or "\u0431\u0435\u0437 \u043f\u043e\u0434\u0442\u0432\u0435\u0440\u0436\u0434\u0435\u043d\u0438\u044f" in lowered


def _expected_result(normalized: str, artifacts: list[str], output_texts: list[str], requested_operation: str) -> str:
    if requested_operation == "git_status":
        return "git summary"
    if requested_operation == "browser_title":
        return "page title"
    if requested_operation == "browser_summary":
        return "page summary"
    if requested_operation == "browser_screenshot":
        return "saved screenshot"
    if requested_operation == "vision_describe_image":
        return "image description"
    if requested_operation == "vision_extract_text":
        return "image text"
    if requested_operation == "vision_find_visual_element":
        return "matching visual element"
    if requested_operation == "desktop_visual_describe":
        return "screen description"
    if requested_operation == "desktop_visual_extract_text":
        return "screen text"
    if requested_operation == "desktop_visual_find_element":
        return "matching element on screen"
    if requested_operation == "browser_visual_describe":
        return "grounded browser visual description"
    if requested_operation == "browser_visual_extract_text":
        return "grounded visible browser text"
    if requested_operation == "browser_visual_find_element":
        return "grounded browser visual search outcome"
    if requested_operation == "browser_navigation":
        return "page opened successfully"
    if artifacts:
        return ", ".join(Path(path).name for path in artifacts)
    if output_texts:
        return output_texts[0]
    lowered = normalized.lower()
    if "tests" in lowered or "\u0442\u0435\u0441\u0442" in lowered:
        return "test result"
    return "requested task completed"


def _looks_like_direct_command(normalized: str) -> bool:
    if _DIRECT_COMMAND_PATTERN.search(normalized):
        return True
    lowered = normalized.lower()
    return get_filesystem_state().last_touched_file is not None and lowered.startswith(("write ", "append "))


def _browser_submission_is_prohibited(normalized: str) -> bool:
    return _BROWSER_NO_SUBMIT_PATTERN.search(normalized) is not None


def _requested_desktop_visual_operation(normalized: str) -> str:
    hint_present = _DESKTOP_VISUAL_HINT_PATTERN.search(normalized) is not None
    if hint_present:
        if _DESKTOP_VISUAL_FIND_PATTERN.search(normalized):
            return "desktop_visual_find_element"
        if _DESKTOP_VISUAL_TEXT_PATTERN.search(normalized):
            return "desktop_visual_extract_text"
        if _DESKTOP_VISUAL_DESCRIBE_PATTERN.search(normalized) or _ru_desktop_describe_intent(normalized):
            return "desktop_visual_describe"
    # Real owner-reported failure (2026-09-19): plain "what do you see?" / "что ты
    # видишь?" style questions never mention "screen"/"desktop"/"экран" at all -- the
    # user doesn't need to name what JARVIS is looking at, since screen capture is
    # JARVIS's only visual sense. Those phrases were written into
    # _DESKTOP_VISUAL_DESCRIBE_PATTERN from the start, but living inside the
    # hint_present-gated block above meant they could never actually match (the hint
    # gate always failed first) -- effectively dead code since RFC-007C. This check
    # runs unconditionally so a bare "what/что do you see" question is recognized on
    # its own, without requiring a screen/desktop mention.
    if _bare_what_do_you_see(normalized):
        return "desktop_visual_describe"
    return ""


_WHAT_DO_YOU_SEE_PATTERN = re.compile(r"\b(what do you see|what can you see)\b", re.IGNORECASE)


def _bare_what_do_you_see(normalized: str) -> bool:
    if _WHAT_DO_YOU_SEE_PATTERN.search(normalized):
        return True
    lowered_words = [word.strip(",.!?—-«»") for word in normalized.lower().split()]
    has_question_word = "что" in lowered_words  # что (what)
    has_seeing_verb = any(
        word in {
            "видишь",  # видишь
            "видешь",  # видешь
            "видите",  # видите
            "видно",  # видно
            "наблюдаешь",  # наблюдаешь
        }
        for word in lowered_words
    )
    return has_question_word and has_seeing_verb


def _requested_vision_operation(normalized: str, referenced_paths: list[str]) -> str:
    lowered = normalized.lower()
    image_paths = [path for path in referenced_paths if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    has_vision_intent = bool(
        _VISION_DESCRIBE_PATTERN.search(normalized)
        or _VISION_OCR_PATTERN.search(normalized)
        or _VISION_FIND_PATTERN.search(normalized)
        or _VISION_IMAGE_HINT_PATTERN.search(normalized)
    )
    if _VISION_UNSUPPORTED_PATTERN.search(normalized):
        return "vision_unsupported"
    if not image_paths and not has_vision_intent:
        return ""
    if image_paths and not has_vision_intent:
        if re.search(r"^\s*(describe|read|extract|find|locate)\s+\S+\.(?:png|jpg|jpeg|webp)\b", normalized, re.IGNORECASE):
            has_vision_intent = True
        elif re.search(
            r"^\s*(\u043e\u043f\u0438\u0448\u0438|\u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0439|\u0438\u0437\u0432\u043b\u0435\u043a\u0438|\u043d\u0430\u0439\u0434\u0438|\u043e\u0442\u044b\u0449\u0438)\s+\S+\.(?:png|jpg|jpeg|webp)\b",
            normalized,
            re.IGNORECASE,
        ):
            has_vision_intent = True
        elif "ocr" in lowered or "\u0442\u0435\u043a\u0441\u0442" in lowered:
            has_vision_intent = True
    if not has_vision_intent:
        return ""
    if not image_paths:
        return "vision_missing_path"
    if _VISION_OCR_PATTERN.search(normalized):
        return "vision_extract_text"
    if _VISION_FIND_PATTERN.search(normalized):
        return "vision_find_visual_element"
    if _VISION_DESCRIBE_PATTERN.search(normalized):
        return "vision_describe_image"
    if "text" in lowered or "ocr" in lowered or "\u0442\u0435\u043a\u0441\u0442" in lowered:
        return "vision_extract_text"
    return "vision_describe_image"


def _extract_vision_query(normalized: str) -> str:
    patterns = (
        re.compile(r"(?:find|locate)\s+(.+?)\s+(?:in|inside|within)\s+[A-Za-z0-9_./-]+\.(?:png|jpg|jpeg|webp)", re.IGNORECASE),
        re.compile(r"(?:\u043d\u0430\u0439\u0434\u0438|\u043e\u0442\u044b\u0449\u0438)\s+(.+?)\s+\u043d\u0430\s+[A-Za-z0-9_./-]+\.(?:png|jpg|jpeg|webp)", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(normalized)
        if match is not None:
            candidate = match.group(1).strip().strip("\"'")
            if candidate:
                return candidate[:120]
    return ""


def _extract_browser_visual_query(normalized: str) -> str:
    patterns = (
        re.compile(r"(?:visually find|find|locate)\s+(.+?)\s+(?:on|in|within)\s+(?:the\s+)?(?:page|browser page|current page|site|website)", re.IGNORECASE),
        re.compile(r"(?:\u0432\u0438\u0437\u0443\u0430\u043b\u044c\u043d\u043e \u043d\u0430\u0439\u0434\u0438|\u043d\u0430\u0439\u0434\u0438)\s+(.+?)\s+\u043d\u0430\s+(?:\u0441\u0442\u0440\u0430\u043d\u0438\u0446|\u0441\u0430\u0439\u0442)", re.IGNORECASE),
        re.compile(r"(?:visually find|find visually|visually locate|visually identify)\s+(.+?)(?:[.?!]|$)", re.IGNORECASE),
        re.compile(r"(?:\u0432\u0438\u0437\u0443\u0430\u043b\u044c\u043d\u043e \u043d\u0430\u0439\u0434\u0438|\u0432\u0438\u0437\u0443\u0430\u043b\u044c\u043d\u043e \u043e\u0442\u044b\u0449\u0438)\s+(.+?)(?:[.?!]|$)", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(normalized)
        if match is None:
            continue
        candidate = _strip_visual_query_leading_article(match.group(1).strip().strip("\"'"))
        if candidate:
            return candidate[:120]
    return ""


def _strip_visual_query_leading_article(value: str) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        return ""
    candidate = re.sub(r"^(?:a|an|the)\s+", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def _extract_vision_detail_level(normalized: str) -> str:
    lowered = normalized.lower()
    if any(token in lowered for token in ("brief", "short", "\u043a\u0440\u0430\u0442\u043a")):
        return "brief"
    if any(token in lowered for token in ("detailed", "detail", "\u043f\u043e\u0434\u0440\u043e\u0431")):
        return "detailed"
    return "normal"


def _extract_sensitive_browser_value(normalized: str) -> str:
    for pattern in _BROWSER_SENSITIVE_VALUE_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        value = str(match.group("value") or "").strip().strip("\"'")
        if value:
            return value
    return ""


def _redacted_text_marker(length: int) -> str:
    safe_length = max(0, int(length))
    return f"[redacted text length={safe_length}]"


def _redact_sensitive_browser_text(value: str, sensitive_value: str) -> str:
    if not sensitive_value or not isinstance(value, str):
        return value
    return value.replace(sensitive_value, _redacted_text_marker(len(sensitive_value)))


def _sanitize_text_list(values: list[str], sensitive_value: str) -> list[str]:
    if not sensitive_value:
        return list(values)
    return [_redact_sensitive_browser_text(value, sensitive_value) for value in values]
