import logging
import webbrowser
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


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
