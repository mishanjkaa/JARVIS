import re


def normalize_command(command: str) -> str:
    if not command:
        return ""

    normalized = re.sub(r"\s+", " ", command.strip())
    words = normalized.split(" ")

    if not words:
        return ""

    lowered = " ".join(words).lower()

    alias_map = {
        "hi": "hello",
        "hey": "hello",
        "hello jarvis": "hello",
        "what time is it": "time",
        "tell me the time": "time",
        "current time": "time",
        "what is the date": "date",
        "what's the date": "date",
        "tell me the date": "date",
        "today's date": "date",
        "launch notepad": "open notepad",
        "start notepad": "open notepad",
        "launch calculator": "open calculator",
        "start calculator": "open calculator",
        "open the browser": "open browser",
        "launch browser": "open browser",
        "show commands": "help",
        "commands": "help",
        "show system info": "system info",
        "what system am i using": "system info",
        "show disk space": "disk space",
        "storage info": "disk space",
        "what is my computer name": "computer name",
        "open my desktop": "open desktop",
        "open my downloads": "open downloads",
        "open my documents": "open documents",
        "open my pictures": "open pictures",
        "open my music": "open music",
        "open my videos": "open videos",
        "lock pc": "lock computer",
        "turn off computer": "shutdown computer",
        "shut down computer": "shutdown computer",
        "reboot computer": "restart computer",
        "restart pc": "restart computer",
        "never mind": "cancel action",
        "show my memories": "memory list",
        "what do you remember": "memory list",
        "list memories": "memory list",
        "show history": "history",
        "clear my history": "clear history",
        "what was my last calculation": "what was my last calculation",
        "what was my last calculation?": "what was my last calculation",
        "what did i search": "what did i search",
        "what did i search?": "what did i search",
        "voice status": "voice status",
        "voice on": "voice on",
        "voice off": "voice off",
        "ai status": "ai status",
        "ai on": "ai on",
        "ai off": "ai off",
        "ai model": "ai model",
        "ai diagnostics": "ai diagnostics",
        "quit": "exit",
        "shutdown jarvis": "exit",
        "goodbye": "exit",
    }

    if lowered in alias_map:
        return alias_map[lowered]

    if lowered.startswith("remember that "):
        rest = normalized[len("remember that "):]
        if " = " in rest:
            key, value = rest.split(" = ", 1)
            return f"remember {key.lower()} = {value}"

    if lowered.startswith("what is my "):
        key = lowered[len("what is my "):].strip()
        return f"recall {key}"

    if lowered.startswith("forget my "):
        key = lowered[len("forget my "):].strip()
        return f"forget {key}"

    if lowered.startswith("search "):
        query = normalized[len("search "):].strip()
        if query:
            return f"search {query.lower()}"
        return "search"

    if len(words) == 1:
        return words[0].lower()

    return normalized.lower()
