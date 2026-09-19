import logging
import webbrowser
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Short names a voice/typed command can use instead of a full address -- e.g. "open site
# google" or (via the Russian aliases in command_normalizer.py) "открой сайт гугл". Kept
# small and deliberately unambiguous; anything else is treated as a literal domain/URL by
# open_website() below rather than failing outright.
_KNOWN_SITE_ALIASES = {
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "youtube": "https://www.youtube.com",
    "github": "https://github.com",
    "yandex": "https://yandex.com",
    "vk": "https://vk.com",
    "wikipedia": "https://www.wikipedia.org",
}


def open_google() -> str:
    try:
        webbrowser.open(_KNOWN_SITE_ALIASES["google"])
        return "Opened Google."
    except Exception:
        logger.exception("Failed to open Google")
        return "Could not open Google."


def open_website(target: str) -> str:
    """Open an arbitrary site by name or address in the owner's default (visible) browser.

    This exists because the owner's own voice requests to open a specific site (e.g. "open
    site job.pt") were previously falling through to the AI/agent runtime, which -- on the
    small local model this project uses -- sometimes tried to reach the URL through the
    sandboxed terminal tool (rejected by terminal policy as "not a trusted local path",
    since a URL looks path-like) and sometimes just hallucinated a conversational "it's
    open now" reply without opening anything at all. A specific site to open is exactly the
    kind of well-defined, deterministic action the rest of this router already handles
    without involving the AI at all (see open_browser/open_youtube/open_github above), so
    this follows that same pattern instead of relying on the model to pick the right tool.
    """

    cleaned = target.strip().strip("\"'").rstrip(".,!?")
    if not cleaned:
        return "Please specify which site to open, for example 'open site github.com'."
    lowered = cleaned.lower()
    url = _KNOWN_SITE_ALIASES.get(lowered)
    if url is None:
        if "://" in cleaned:
            url = cleaned
        elif "." in cleaned and " " not in cleaned:
            url = f"https://{cleaned}"
        else:
            return (
                f"I don't recognize '{cleaned}' as a website. Try a full address instead, "
                "e.g. 'open site example.com'."
            )
    try:
        webbrowser.open(url)
        return f"Opened {cleaned}."
    except Exception:
        logger.exception("Failed to open website: %s", cleaned)
        return f"Could not open {cleaned}."


def open_youtube() -> str:
    try:
        webbrowser.open("https://www.youtube.com")
        return "Opened YouTube."
    except Exception:
        logger.exception("Failed to open YouTube")
        return "Could not open YouTube."


def open_github() -> str:
    try:
        webbrowser.open("https://github.com")
        return "Opened GitHub."
    except Exception:
        logger.exception("Failed to open GitHub")
        return "Could not open GitHub."


def search_web(query: str) -> str:
    cleaned_query = query.strip()
    if not cleaned_query:
        return "Please provide a search query."

    try:
        encoded_query = quote_plus(cleaned_query)
        url = f"https://www.google.com/search?q={encoded_query}"
        webbrowser.open(url)
        logger.info("Web search requested")
        return f"Opened search results for: {cleaned_query}"
    except Exception:
        logger.exception("Failed to open web search")
        return "Could not open the search results."
