import logging
import re

from app.brain.configuration.config_commands import config_get, config_reload, config_reset, config_show, config_set
from app.brain.history.timeline import clear_timeline, format_timeline
from app.brain.intent.classifier import classify
from app.brain.plugins.trusted_plugins import TRUSTED_PLUGINS
from app.brain.browser.controller import get_browser_controller
from app.brain.vision.controller import get_vision_controller
from app.brain.ai.orchestrator import AIOrchestrator
from app.brain.ai.ollama_provider import OllamaProvider
from app.brain.ai.state import get_ai_state, reset_ai_state
from app.brain.command_normalizer import normalize_command
from app.brain.computer.app_launcher import open_browser, open_calculator, open_notepad
from app.brain.computer.folder_actions import open_known_folder
from app.brain.computer.power_actions import execute_power_action
from app.brain.computer.system_info import get_computer_name, get_disk_space, get_system_info
from app.brain.context.history import clear_history, get_history, record_safe_command
from app.brain.context.state import get_context, update_context
from app.brain.internet.web_actions import open_github, open_youtube, search_web
from app.brain.memory.store import (
    CATEGORY_LEARNED_PATTERN,
    SOURCE_AI_PROPOSED,
    find_similar_keys,
    forget_memory,
    list_memories,
    recall_memory,
    save_memory,
)
from app.brain.planner.notes import add_note, delete_note, list_notes
from app.brain.planner.approval import has_active_pending_approval
from app.brain.planner.tasks import add_task, complete_task, delete_task, list_tasks
from app.brain.security.confirmation import (
    cancel_pending_action,
    confirm_pending_action,
    request_confirmation,
)
from app.brain.skills.calculator import calculate_expression
from app.brain.skills.self_check import run_self_check
from app.brain.skills.system_info import get_local_date, get_local_time
from app.brain.voice.voice_controller import voice_controller, voice_off, voice_on, voice_status
from app.brain.runtime.conversation_runtime import ConversationRuntime
from app.brain.configuration.runtime_config import get_effective_runtime_config, set_runtime_config_value
from app.brain.agent.controller import get_agent_controller
from app.brain.intelligence.controller import get_intelligence_controller
from app.brain.terminal.controller import get_terminal_controller

logger = logging.getLogger(__name__)

_CALCULATION_REQUEST_PATTERN = re.compile(r"^\s*(calculate|calc|work out)\b", re.IGNORECASE)
_FOLLOW_UP_SEPARATOR_PATTERN = re.compile(r"\b(and|then)\b", re.IGNORECASE)
_CALCULATION_FOLLOW_UP_PATTERN = re.compile(
    r"\b(save|record|note|make a note|write (it|that|the result|the answer) down)\b",
    re.IGNORECASE,
)

_PROVIDER = OllamaProvider()
_ORCHESTRATOR = AIOrchestrator(provider=_PROVIDER)
_CONVERSATION_RUNTIME = ConversationRuntime(provider=_PROVIDER)


def _build_ai_response(raw_input: str) -> str:
    state = get_ai_state()
    state.enabled = bool(get_effective_runtime_config().get("ai_enabled", False))
    return _CONVERSATION_RUNTIME.handle(raw_input)


def _build_tool_plan_response(raw_input: str) -> str:
    from app.brain.planner.approval import store_pending_plan
    from app.brain.planner.plan_validator import validate_plan
    from app.brain.planner.planner_v2 import create_plan_from_request

    plan = create_plan_from_request(raw_input)
    validation = validate_plan(plan)
    if not validation.valid:
        return "Plan rejected safely."
    return store_pending_plan(plan)


def _is_explicit_multi_step(raw_input: str) -> bool:
    value = raw_input.lower()
    if _CALCULATION_REQUEST_PATTERN.search(value) and _FOLLOW_UP_SEPARATOR_PATTERN.search(value) and _CALCULATION_FOLLOW_UP_PATTERN.search(value):
        return True
    return " and " in value and (("calculate" in value and "note" in value) or ("open" in value and ("downloads" in value or "calculator" in value)) or ("tasks" in value and "note" in value))


def _has_pending_plan() -> bool:
    return has_active_pending_approval()


def route_command(command: str) -> str:
    raw_command = command.strip()
    normalized_command = normalize_command(raw_command)

    try:
        if normalized_command == "hello":
            logger.info("Recognized command: hello")
            return "Hello Misha. I am ready."

        if normalized_command == "status":
            logger.info("Recognized command: status")
            return "All systems are operational."

        if normalized_command == "version":
            return "2.5.0"

        if normalized_command == "about":
            return "JARVIS is a deterministic-first assistant with optional local AI planning and guarded tools."

        if normalized_command == "time":
            logger.info("Recognized command: time")
            return f"Current time is {get_local_time()}."

        if normalized_command == "date":
            logger.info("Recognized command: date")
            return f"Today is {get_local_date()}."

        if normalized_command == "help":
            logger.info("Recognized command: help")
            return """Available commands:

AI
- ai status
- ai on
- ai off
- ai model
- ai diagnostics

Agent planning
- show pending plan
- approve plan
- cancel plan
- reject plan
- intelligence status
- intelligence provider status
- intelligence provider check
- planning status
- show last plan
- show last interpretation
- cancel clarification
- agent status
- agent task
- agent cancel
- emergency stop
- terminal status
- terminal history
- terminal cancel
- vision status
- vision browser status
- vision captures
- vision clear captures
- desktop captures
- desktop clear captures
- vision provider status
- vision provider check

Conversation and timeline
- conversation on
- conversation off
- conversation status
- developer mode on
- developer mode off
- developer mode status
- conversation summary
- clear conversation
- timeline
- clear timeline

Configuration
- config show
- config set <key> <value>

Plugins
- plugins list
- plugin status <plugin_id>
- plugin enable <plugin_id>
- plugin disable <plugin_id>

General
- hello
- status
- time
- date
- help

Computer
- system info
- disk space
- computer name
- open notepad
- open calculator
- open browser
- open desktop
- open downloads
- open documents
- open pictures
- open music
- open videos

Internet
- open youtube
- open github
- search <query>

Calculator
- calculate <expression>
- calc <expression>

Memory
- remember <key> = <value>
- recall <key>
- forget <key>
- memory list
- memory list learned

Notes
- note <text>
- notes list
- note delete <number>

Tasks
- task add <text>
- tasks list
- task done <number>
- task delete <number>

Context and history
- what was my last calculation
- what did i search
- what folder did you open
- what application did you open
- history
- clear history

Safety
- lock computer
- shutdown computer
- restart computer
- confirm lock
- confirm shutdown
- confirm restart
- cancel action

Diagnostics
- self check

Voice
- voice status
- voice on
- voice off

Exit
- exit"""

        if _is_explicit_multi_step(raw_command):
            if _has_pending_plan():
                return 'A plan is already pending. Use "approve plan" or "cancel plan" before starting another task.'
            return _build_tool_plan_response(raw_command)

        if normalized_command.startswith("remember "):
            body = raw_command[9:].strip() if raw_command.lower().startswith("remember ") else raw_command
            if body.lower().startswith("that "):
                body = body[5:].strip()
            if "=" not in body:
                logger.info("Recognized malformed remember command")
                return "Please use the format: remember <key> = <value>."
            key, value = body.split("=", 1)
            key = key.strip()
            value = value.strip()
            normalized_key = key.lower().strip()
            logger.info("Recognized memory command: remember key=%s", normalized_key)
            max_entries = int(get_effective_runtime_config().get("memory_max_entries", 500))
            return save_memory(normalized_key, value, max_entries=max_entries)

        if normalized_command.startswith("recall "):
            key = raw_command[7:].strip() if raw_command.lower().startswith("recall ") else raw_command
            normalized_key = key.lower().strip()
            logger.info("Recognized memory command: recall key=%s", normalized_key)
            result = recall_memory(normalized_key)
            if result == f"No memory found for '{normalized_key}'.":
                suggestions = find_similar_keys(normalized_key)
                if suggestions:
                    result += " Did you mean: " + ", ".join(suggestions) + "?"
            return result

        if normalized_command.startswith("forget "):
            key = raw_command[7:].strip() if raw_command.lower().startswith("forget ") else raw_command
            normalized_key = key.lower().strip()
            logger.info("Recognized memory command: forget key=%s", normalized_key)
            return forget_memory(normalized_key)

        if normalized_command == "memory list learned":
            logger.info("Recognized memory command: list learned")
            return list_memories(category=CATEGORY_LEARNED_PATTERN, source=SOURCE_AI_PROPOSED)

        if normalized_command == "memory list":
            logger.info("Recognized memory command: list")
            return list_memories()

        if normalized_command == "system info":
            logger.info("Recognized command: system info")
            return get_system_info()

        if normalized_command == "disk space":
            logger.info("Recognized command: disk space")
            return get_disk_space()

        if normalized_command == "computer name":
            logger.info("Recognized command: computer name")
            return get_computer_name()

        if normalized_command.startswith("open "):
            folder_name = normalized_command[len("open "):].strip()
            if folder_name in {"desktop", "downloads", "documents", "pictures", "music", "videos"}:
                logger.info("Recognized command: open folder")
                result = open_known_folder(folder_name)
                if result.startswith("Opened "):
                    update_context(last_opened_folder=folder_name.capitalize())
                    record_safe_command("folder opened")
                return result

        if normalized_command.startswith("calculate "):
            expression = normalized_command[len("calculate "):].strip()
            logger.info("Recognized command: calculate")
            result = calculate_expression(expression)
            if not result.startswith("Please") and not result.startswith("Division") and not result.startswith("That"):
                update_context(last_calculator_result=result)
            record_safe_command("calculation")
            return result

        if normalized_command.startswith("calc "):
            expression = normalized_command[len("calc "):].strip()
            logger.info("Recognized command: calc")
            result = calculate_expression(expression)
            if not result.startswith("Please") and not result.startswith("Division") and not result.startswith("That"):
                update_context(last_calculator_result=result)
            record_safe_command("calculation")
            return result

        if normalized_command.startswith("note "):
            text = raw_command[len("note "):].strip()
            logger.info("Recognized command: note")
            return f"Added note {add_note(text)}."

        if normalized_command == "notes list":
            logger.info("Recognized command: notes list")
            return list_notes()

        if normalized_command.startswith("note delete "):
            note_number = normalized_command[len("note delete "):].strip()
            try:
                note_id = int(note_number)
            except ValueError:
                return "Please provide a numeric note id."
            logger.info("Recognized command: note delete")
            return delete_note(note_id)

        if normalized_command.startswith("task add "):
            text = raw_command[len("task add "):].strip()
            logger.info("Recognized command: task add")
            return f"Added task {add_task(text)}."

        if normalized_command == "tasks list":
            logger.info("Recognized command: tasks list")
            return list_tasks()

        if normalized_command.startswith("task done "):
            task_number = normalized_command[len("task done "):].strip()
            try:
                task_id = int(task_number)
            except ValueError:
                return "Please provide a numeric task id."
            logger.info("Recognized command: task done")
            return complete_task(task_id)

        if normalized_command.startswith("task delete "):
            task_number = normalized_command[len("task delete "):].strip()
            try:
                task_id = int(task_number)
            except ValueError:
                return "Please provide a numeric task id."
            logger.info("Recognized command: task delete")
            return delete_task(task_id)

        if normalized_command == "self check":
            logger.info("Recognized command: self check")
            return run_self_check()

        if normalized_command == "ai status":
            enabled = bool(get_effective_runtime_config().get("ai_enabled", False))
            get_ai_state().enabled = enabled
            return f"AI is {'enabled' if enabled else 'disabled'}."

        if normalized_command == "ai on":
            set_runtime_config_value("ai_enabled", True)
            get_ai_state().enabled = True
            return "AI enabled for this session."

        if normalized_command == "ai off":
            set_runtime_config_value("ai_enabled", False)
            get_ai_state().enabled = False
            return "AI disabled for this session."

        if normalized_command == "ai model":
            return f"Configured model: {get_effective_runtime_config().get('ollama_model', 'default') or 'default'}"

        if normalized_command == "ai diagnostics":
            state = get_ai_state()
            return f"AI diagnostics: enabled={state.enabled} provider={state.provider_name} status={state.last_provider_status_category}"

        if normalized_command == "conversation on":
            set_runtime_config_value("ai_allow_conversation", True)
            return "Conversation enabled."

        if normalized_command == "conversation off":
            set_runtime_config_value("ai_allow_conversation", False)
            return "Conversation disabled."

        if normalized_command == "conversation status":
            enabled = bool(get_effective_runtime_config().get("ai_allow_conversation", True))
            return f"Conversation is {'enabled' if enabled else 'disabled'}."

        if normalized_command == "developer mode on":
            set_runtime_config_value("developer_mode", True)
            return "Developer mode enabled."

        if normalized_command == "developer mode off":
            set_runtime_config_value("developer_mode", False)
            return "Developer mode disabled."

        if normalized_command == "developer mode status":
            enabled = bool(get_effective_runtime_config().get("developer_mode", False))
            return f"Developer mode is {'enabled' if enabled else 'disabled'}."

        if normalized_command == "intelligence status":
            return get_intelligence_controller().status_message()

        if normalized_command == "intelligence provider status":
            return get_intelligence_controller().provider_status_message()

        if normalized_command == "intelligence provider check":
            return get_intelligence_controller().provider_check_message()

        if normalized_command == "planning status":
            return get_intelligence_controller().planning_status_message()

        if normalized_command == "show last plan":
            return get_intelligence_controller().show_last_plan()

        if normalized_command == "show last interpretation":
            return get_intelligence_controller().show_last_interpretation()

        if normalized_command == "cancel clarification":
            return get_intelligence_controller().cancel_clarification()

        if normalized_command == "show pending plan":
            from app.brain.planner.approval import show_pending_plan
            return show_pending_plan()

        if normalized_command == "approve plan":
            from app.brain.planner.approval import approve_pending_plan
            return approve_pending_plan()

        if normalized_command in {"cancel plan", "reject plan", "deny plan", "decline plan"}:
            from app.brain.planner.approval import cancel_pending_plan
            return cancel_pending_plan()

        if normalized_command == "agent status":
            return get_agent_controller().get_status_message()

        if normalized_command == "agent task":
            return get_agent_controller().get_task_message()

        if normalized_command == "agent cancel":
            return get_agent_controller().cancel_pending_or_running_task()

        if normalized_command in {"terminal status", "command status"}:
            return get_terminal_controller().status_message()

        if normalized_command in {"terminal history", "command history"}:
            return get_terminal_controller().history_message()

        if normalized_command in {"terminal cancel", "cancel command"}:
            cancelled = get_terminal_controller().cancel_active_execution()
            if cancelled == "No terminal command is running.":
                return cancelled
            get_agent_controller().cancel_pending_or_running_task()
            return cancelled

        if normalized_command == "browser status":
            return get_browser_controller().status_message()

        if normalized_command == "browser sessions":
            return get_browser_controller().sessions_message()

        if normalized_command == "browser tabs":
            return get_browser_controller().tabs_message()

        if normalized_command == "browser close all":
            return get_browser_controller().close_all_sessions()

        if normalized_command == "vision status":
            return get_vision_controller().status_message()

        if normalized_command == "vision browser status":
            return get_vision_controller().browser_status_message()

        if normalized_command == "vision captures":
            return get_vision_controller().captures_message()

        if normalized_command == "vision clear captures":
            return get_vision_controller().clear_browser_captures()

        if normalized_command == "desktop captures":
            return get_vision_controller().desktop_captures_message()

        if normalized_command == "desktop clear captures":
            return get_vision_controller().clear_desktop_captures()

        if normalized_command == "vision provider status":
            return get_vision_controller().provider_status_message()

        if normalized_command == "vision provider check":
            return get_vision_controller().provider_check_message()

        if normalized_command == "emergency stop":
            return get_agent_controller().engage_emergency_stop()

        if normalized_command == "conversation summary":
            from app.brain.context.conversation import get_summaries
            summaries = get_summaries()
            return "No conversation entries yet." if not summaries else "\n".join(summaries)

        if normalized_command == "clear conversation":
            from app.brain.context.conversation import clear_conversation
            clear_conversation()
            return "Conversation summary cleared."

        if normalized_command == "timeline":
            return format_timeline()

        if normalized_command == "clear timeline":
            clear_timeline()
            return "Timeline cleared."

        if normalized_command == "config show":
            return config_show()

        if normalized_command.startswith("config get "):
            return config_get(normalized_command[len("config get "):].strip())

        if normalized_command.startswith("config reset "):
            return config_reset(normalized_command[len("config reset "):].strip())

        if normalized_command == "config reload":
            return config_reload()

        if normalized_command.startswith("config set "):
            parts = raw_command.split(maxsplit=3)
            if len(parts) != 4:
                return "Please use: config set <key> <value>."
            value: object = parts[3]
            if value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            return config_set(parts[2], value)

        if normalized_command == "plugins list":
            return "Trusted plugins: " + ", ".join(sorted(TRUSTED_PLUGINS))

        if normalized_command.startswith("plugin status "):
            plugin_id = normalized_command[len("plugin status "):].strip()
            return "Plugin is trusted." if plugin_id in TRUSTED_PLUGINS else "Unknown plugin."

        if normalized_command.startswith("plugin enable ") or normalized_command.startswith("plugin disable "):
            plugin_id = normalized_command.split()[-1]
            if plugin_id not in TRUSTED_PLUGINS:
                return "Unknown plugin."
            from app.brain.plugins.registry import get_plugin_registry
            registry = get_plugin_registry()
            if normalized_command.startswith("plugin enable "):
                try:
                    return "Plugin enabled." if registry.enable(plugin_id) else "Plugin enable rejected."
                except ValueError:
                    return "Plugin enable failed safely."
            registry.disable(plugin_id)
            return "Plugin disabled."

        if normalized_command == "lock computer":
            logger.info("Recognized command: lock computer")
            return request_confirmation("lock")

        if normalized_command == "shutdown computer":
            logger.info("Recognized command: shutdown computer")
            return request_confirmation("shutdown")

        if normalized_command == "restart computer":
            logger.info("Recognized command: restart computer")
            return request_confirmation("restart")

        if normalized_command == "confirm lock":
            logger.info("Recognized command: confirm lock")
            return "Lock request confirmed." if confirm_pending_action("lock") else "No pending lock action."

        if normalized_command == "confirm shutdown":
            logger.info("Recognized command: confirm shutdown")
            if confirm_pending_action("shutdown"):
                return execute_power_action("shutdown")
            return "No pending shutdown action."

        if normalized_command == "confirm restart":
            logger.info("Recognized command: confirm restart")
            if confirm_pending_action("restart"):
                return execute_power_action("restart")
            return "No pending restart action."

        if normalized_command == "cancel action":
            logger.info("Recognized command: cancel action")
            cancel_pending_action()
            return "Cancelled the pending action."

        if normalized_command == "open notepad":
            logger.info("Recognized command: open notepad")
            result = open_notepad()
            if result.startswith("Opened"):
                update_context(last_opened_application="Notepad")
                record_safe_command("application opened")
            return result

        if normalized_command == "open calculator":
            logger.info("Recognized command: open calculator")
            result = open_calculator()
            if result.startswith("Opened"):
                update_context(last_opened_application="Calculator")
                record_safe_command("application opened")
            return result

        if normalized_command == "open browser":
            logger.info("Recognized command: open browser")
            result = open_browser()
            if result.startswith("Opened"):
                update_context(last_opened_application="Browser")
                record_safe_command("application opened")
            return result

        if normalized_command == "open youtube":
            logger.info("Recognized command: open youtube")
            result = open_youtube()
            if result.startswith("Opened"):
                update_context(last_opened_application="YouTube")
                record_safe_command("application opened")
            return result

        if normalized_command == "open github":
            logger.info("Recognized command: open github")
            result = open_github()
            if result.startswith("Opened"):
                update_context(last_opened_application="GitHub")
                record_safe_command("application opened")
            return result

        if normalized_command == "search" or normalized_command.startswith("search "):
            query = raw_command[len("search "):].strip() if raw_command.lower().startswith("search ") else ""
            if not query:
                return "Please provide a search query."
            result = search_web(query)
            if result.startswith("Opened search results for:"):
                update_context(last_web_search=query)
                record_safe_command("web search")
            return result

        if normalized_command == "history":
            history_entries = get_history()
            if not history_entries:
                return "No history yet."
            return " | ".join(history_entries)

        if normalized_command == "clear history":
            clear_history()
            return "History cleared."

        if normalized_command.startswith("what was my last calculation"):
            previous_result = get_context().last_calculator_result
            if previous_result is None:
                return "I don't have a previous calculation yet."
            return previous_result

        if normalized_command.startswith("what did i search"):
            previous_search = get_context().last_web_search
            if previous_search is None:
                return "I have not searched anything yet."
            return previous_search

        if normalized_command.startswith("what folder did you open"):
            previous_folder = get_context().last_opened_folder
            if previous_folder is None:
                return "I haven't opened a folder recently."
            return previous_folder

        if normalized_command.startswith("what application did you open"):
            previous_application = get_context().last_opened_application
            if previous_application is None:
                return "I haven't opened an application recently."
            return previous_application

        if normalized_command == "voice status":
            return voice_status()

        if normalized_command == "voice on":
            return voice_on()

        if normalized_command == "voice off":
            return voice_off()

        if normalized_command == "exit":
            logger.info("Recognized command: exit")
            return "shutdown"

        logger.info("Input forwarded to intelligence runtime")
        intelligence_response = get_intelligence_controller().handle(raw_command)
        if intelligence_response is not None:
            return intelligence_response
        classification = classify(raw_command)
        if classification.tools_required:
            return _build_tool_plan_response(raw_command)
        return _build_ai_response(raw_command)
    except Exception:
        logger.exception("An error occurred while processing a command")
        return "Sorry, something went wrong while processing your command."
