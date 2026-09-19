import unittest
from unittest.mock import patch

from app.brain.agent.state import reset_agent_runtime_state
from app.brain.audit.audit_log import reset_audit_log
from app.brain.configuration.runtime_config import reset_runtime_config, set_runtime_config_value
from app.brain.context.history import clear_history, get_history
from app.brain.context.state import get_context, reset_context
from app.brain.filesystem.state import reset_filesystem_state
from app.brain.memory.store import CATEGORY_LEARNED_PATTERN, SOURCE_AI_PROPOSED
from app.brain.planner.state import reset_planner_state
from app.brain.router import route_command


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_context()
        clear_history()
        reset_planner_state()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_runtime_config()
        reset_filesystem_state()

    def test_hello_command(self) -> None:
        self.assertEqual(route_command("hello"), "Hello Misha. I am ready.")

    def test_status_command(self) -> None:
        self.assertEqual(route_command("status"), "All systems are operational.")

    def test_help_command(self) -> None:
        result = route_command("help")
        self.assertIn("hello", result)
        self.assertIn("remember <key> = <value>", result)
        self.assertIn("memory list", result)

    def test_unknown_command(self) -> None:
        # ai_enabled now defaults to True (owner's explicit request: JARVIS should be
        # usable immediately at startup) -- this test is specifically about the
        # disabled-state fallback message, so it turns AI off itself.
        set_runtime_config_value("ai_enabled", False)
        self.assertEqual(route_command("fly"), "AI is disabled right now.")

    @patch("app.brain.router.get_system_info")
    def test_system_info_command(self, get_system_info_mock) -> None:
        get_system_info_mock.return_value = "OS: Windows"
        self.assertEqual(route_command("system info"), "OS: Windows")
        get_system_info_mock.assert_called_once_with()

    @patch("app.brain.router.get_disk_space")
    def test_disk_space_command(self, get_disk_space_mock) -> None:
        get_disk_space_mock.return_value = "Total: 1.00 GB"
        self.assertEqual(route_command("disk space"), "Total: 1.00 GB")
        get_disk_space_mock.assert_called_once_with()

    @patch("app.brain.router.get_computer_name")
    def test_computer_name_command(self, get_computer_name_mock) -> None:
        get_computer_name_mock.return_value = "Computer name: DESKTOP"
        self.assertEqual(route_command("computer name"), "Computer name: DESKTOP")
        get_computer_name_mock.assert_called_once_with()

    @patch("app.brain.router.open_known_folder")
    def test_known_folder_command(self, open_known_folder_mock) -> None:
        open_known_folder_mock.return_value = "Opened Downloads."
        self.assertEqual(route_command("open downloads"), "Opened Downloads.")
        open_known_folder_mock.assert_called_once_with("downloads")

    def test_calculator_command(self) -> None:
        self.assertEqual(route_command("calculate 25 * 4"), "Result: 100")

    def test_combined_calculation_bypasses_deterministic_calculator(self) -> None:
        with patch("app.brain.router.calculate_expression") as calculate_expression_mock:
            response = route_command("calculate 10 + 4 and make a note with the result")
        self.assertIn("This plan is MEDIUM risk.", response)
        calculate_expression_mock.assert_not_called()

    @patch("app.brain.router.add_note")
    @patch("app.brain.router.add_task")
    @patch("app.brain.router.list_notes")
    @patch("app.brain.router.list_tasks")
    def test_note_and_task_commands(self, list_tasks_mock, list_notes_mock, add_task_mock, add_note_mock) -> None:
        add_note_mock.return_value = 1
        add_task_mock.return_value = 1
        list_notes_mock.return_value = "1. hello"
        list_tasks_mock.return_value = "1. review [pending]"

        self.assertEqual(route_command("note hello"), "Added note 1.")
        self.assertEqual(route_command("notes list"), "1. hello")
        self.assertEqual(route_command("task add review"), "Added task 1.")
        self.assertEqual(route_command("tasks list"), "1. review [pending]")

    def test_self_check_command(self) -> None:
        self.assertIn("PASS", route_command("self check"))

    def test_exit_command(self) -> None:
        self.assertEqual(route_command("exit"), "shutdown")

    @patch("app.brain.router.open_notepad")
    def test_open_notepad_command(self, open_notepad_mock) -> None:
        open_notepad_mock.return_value = "Opened Notepad."
        self.assertEqual(route_command("open notepad"), "Opened Notepad.")

    @patch("app.brain.router.open_calculator")
    def test_open_calculator_command(self, open_calculator_mock) -> None:
        open_calculator_mock.return_value = "Opened Calculator."
        self.assertEqual(route_command("open calculator"), "Opened Calculator.")

    @patch("app.brain.router.open_browser")
    def test_open_browser_command(self, open_browser_mock) -> None:
        open_browser_mock.return_value = "Opened the default web browser."
        self.assertEqual(route_command("open browser"), "Opened the default web browser.")

    @patch("app.brain.router.save_memory")
    def test_remember_command_parses_value(self, save_memory_mock) -> None:
        save_memory_mock.return_value = "Saved memory for 'favorite_ide'."
        self.assertEqual(route_command("remember favorite_ide = VS Code"), "Saved memory for 'favorite_ide'.")
        save_memory_mock.assert_called_once_with("favorite_ide", "VS Code", max_entries=500)

    @patch("app.brain.router.recall_memory")
    def test_recall_command_parses_key(self, recall_memory_mock) -> None:
        recall_memory_mock.return_value = "VS Code"
        self.assertEqual(route_command("recall favorite_ide"), "VS Code")
        recall_memory_mock.assert_called_once_with("favorite_ide")

    @patch("app.brain.router.find_similar_keys")
    @patch("app.brain.router.recall_memory")
    def test_recall_command_suggests_similar_keys_on_miss(self, recall_memory_mock, find_similar_keys_mock) -> None:
        recall_memory_mock.return_value = "No memory found for 'favorite_id'."
        find_similar_keys_mock.return_value = ["favorite_ide", "favorite_idea"]
        result = route_command("recall favorite_id")
        self.assertEqual(result, "No memory found for 'favorite_id'. Did you mean: favorite_ide, favorite_idea?")
        find_similar_keys_mock.assert_called_once_with("favorite_id")

    @patch("app.brain.router.find_similar_keys")
    @patch("app.brain.router.recall_memory")
    def test_recall_command_has_no_hint_when_no_similar_keys(self, recall_memory_mock, find_similar_keys_mock) -> None:
        recall_memory_mock.return_value = "No memory found for 'zzz'."
        find_similar_keys_mock.return_value = []
        self.assertEqual(route_command("recall zzz"), "No memory found for 'zzz'.")

    @patch("app.brain.router.find_similar_keys")
    @patch("app.brain.router.recall_memory")
    def test_recall_command_has_no_hint_on_a_hit(self, recall_memory_mock, find_similar_keys_mock) -> None:
        recall_memory_mock.return_value = "VS Code"
        self.assertEqual(route_command("recall favorite_ide"), "VS Code")
        find_similar_keys_mock.assert_not_called()

    @patch("app.brain.router.forget_memory")
    def test_forget_command_parses_key(self, forget_memory_mock) -> None:
        forget_memory_mock.return_value = "Forgot memory for 'favorite_ide'."
        self.assertEqual(route_command("forget favorite_ide"), "Forgot memory for 'favorite_ide'.")
        forget_memory_mock.assert_called_once_with("favorite_ide")

    @patch("app.brain.router.list_memories")
    def test_memory_list_command(self, list_memories_mock) -> None:
        list_memories_mock.return_value = "favorite_ide: VS Code"
        self.assertEqual(route_command("memory list"), "favorite_ide: VS Code")
        list_memories_mock.assert_called_once_with()

    @patch("app.brain.router.list_memories")
    def test_memory_list_learned_command_filters_to_learned_ai_proposed(self, list_memories_mock) -> None:
        list_memories_mock.return_value = "pattern_x: value [learned_pattern/ai_proposed]"
        self.assertEqual(route_command("memory list learned"), "pattern_x: value [learned_pattern/ai_proposed]")
        list_memories_mock.assert_called_once_with(category=CATEGORY_LEARNED_PATTERN, source=SOURCE_AI_PROPOSED)

    def test_aliases_route_correctly(self) -> None:
        self.assertEqual(route_command("hi"), "Hello Misha. I am ready.")
        self.assertIn("Current time is", route_command("what time is it"))
        self.assertEqual(route_command("launch notepad"), "Opened Notepad.")
        result = route_command("show commands")
        self.assertIn("Available commands:", result)
        self.assertIn("system info", result)
        self.assertIn("disk space", result)
        self.assertIn("computer name", result)
        self.assertIn("open downloads", result)
        self.assertIn("memory list", result)
        self.assertEqual(route_command("quit"), "shutdown")

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_open_youtube(self, open_mock) -> None:
        open_mock.return_value = True
        self.assertEqual(route_command("open youtube"), "Opened YouTube.")

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_open_github(self, open_mock) -> None:
        open_mock.return_value = True
        self.assertEqual(route_command("open github"), "Opened GitHub.")

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_open_google(self, open_mock) -> None:
        open_mock.return_value = True
        self.assertEqual(route_command("open google"), "Opened Google.")
        open_mock.assert_called_once_with("https://www.google.com")

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_open_site_with_a_domain(self, open_mock) -> None:
        open_mock.return_value = True
        self.assertEqual(route_command("open site job.pt"), "Opened job.pt.")
        open_mock.assert_called_once_with("https://job.pt")

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_open_site_with_a_known_short_name(self, open_mock) -> None:
        open_mock.return_value = True
        self.assertEqual(route_command("open site youtube"), "Opened youtube.")
        open_mock.assert_called_once_with("https://www.youtube.com")

    def test_open_site_without_a_target_asks_for_one_instead_of_guessing(self) -> None:
        self.assertEqual(
            route_command("open site"),
            "Please specify which site to open, for example 'open site github.com'.",
        )

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_open_site_with_an_unrecognized_bare_word_does_not_guess_a_url(self, open_mock) -> None:
        self.assertEqual(
            route_command("open site somethingmadeup"),
            "I don't recognize 'somethingmadeup' as a website. Try a full address instead, "
            "e.g. 'open site example.com'.",
        )
        open_mock.assert_not_called()

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_open_site_russian_voice_phrasing(self, open_mock) -> None:
        # This is the exact real-world failure this was added to fix: the owner's spoken
        # "открой сайт <адрес>" was previously falling through to the AI/agent runtime,
        # which sometimes tried (and failed) to reach the URL through the sandboxed
        # terminal tool, and sometimes just hallucinated a conversational reply without
        # opening anything.
        open_mock.return_value = True
        self.assertEqual(route_command("открой сайт job.pt"), "Opened job.pt.")
        open_mock.assert_called_once_with("https://job.pt")

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_search_query_is_encoded(self, open_mock) -> None:
        open_mock.return_value = True
        self.assertEqual(route_command("search OpenAI GPT-5"), "Opened search results for: OpenAI GPT-5")
        open_mock.assert_called_once_with("https://www.google.com/search?q=OpenAI+GPT-5")

    @patch("app.brain.internet.web_actions.webbrowser.open")
    def test_empty_search_is_rejected(self, open_mock) -> None:
        self.assertEqual(route_command("search   "), "Please provide a search query.")
        open_mock.assert_not_called()

    @patch("app.brain.router.search_web")
    def test_context_and_history_commands(self, search_web_mock) -> None:
        search_web_mock.return_value = "Opened search results for: OpenAI"

        self.assertEqual(route_command("calculate 2 + 3"), "Result: 5")
        self.assertEqual(get_context().last_calculator_result, "Result: 5")
        self.assertEqual(route_command("what was my last calculation?"), "Result: 5")

        self.assertEqual(route_command("search OpenAI"), "Opened search results for: OpenAI")
        self.assertEqual(get_context().last_web_search, "OpenAI")
        self.assertEqual(route_command("what did I search?"), "OpenAI")

        history_result = route_command("history")
        self.assertIn("calculation", history_result)
        self.assertIn("web search", history_result)

        self.assertEqual(route_command("clear history"), "History cleared.")
        self.assertEqual(route_command("history"), "No history yet.")

    @patch("app.brain.router.open_known_folder")
    def test_context_questions_for_folder_and_application(self, open_known_folder_mock) -> None:
        open_known_folder_mock.return_value = "Opened Downloads."
        self.assertEqual(route_command("open downloads"), "Opened Downloads.")
        self.assertEqual(route_command("what folder did you open?"), "Downloads")

        with patch("app.brain.router.open_calculator", return_value="Opened Calculator.") as open_calculator_mock:
            self.assertEqual(route_command("open calculator"), "Opened Calculator.")
            self.assertEqual(route_command("what application did you open"), "Calculator")
            open_calculator_mock.assert_called_once_with()

    @patch("app.brain.router.open_known_folder")
    def test_failed_folder_opening_does_not_update_context(self, open_known_folder_mock) -> None:
        open_known_folder_mock.return_value = "The Desktop folder is not available."
        self.assertEqual(route_command("open desktop"), "The Desktop folder is not available.")
        self.assertIsNone(get_context().last_opened_folder)

    def test_voice_commands(self) -> None:
        # voice_status() now reports enabled/mic-active/enrolled state (RFC-009), mirroring
        # vision status/location status, rather than the old fixed "Voice is enabled/disabled."
        # voice_enabled now defaults to True (owner's explicit request), so this test turns
        # it off itself first to exercise the "no" -> "yes" -> "no" transition it's after.
        set_runtime_config_value("voice_enabled", False)
        self.assertIn("Voice enabled: no", route_command("voice status"))
        self.assertEqual(route_command("voice on"), "Voice enabled for this session.")
        self.assertIn("Voice enabled: yes", route_command("voice status"))
        self.assertEqual(route_command("voice off"), "Voice disabled for this session.")
        self.assertIn("Voice enabled: no", route_command("voice status"))

    def test_developer_mode_commands(self) -> None:
        self.assertEqual(route_command("developer mode status"), "Developer mode is disabled.")
        self.assertEqual(route_command("developer mode on"), "Developer mode enabled.")
        self.assertEqual(route_command("developer mode status"), "Developer mode is enabled.")
        self.assertEqual(route_command("developer mode off"), "Developer mode disabled.")
        self.assertEqual(route_command("developer mode status"), "Developer mode is disabled.")

    def test_help_groups_are_readable(self) -> None:
        result = route_command("help")
        self.assertIn("General", result)
        self.assertIn("Context and history", result)
        self.assertIn("Voice", result)
        self.assertIn("self check", result)


if __name__ == "__main__":
    unittest.main()
