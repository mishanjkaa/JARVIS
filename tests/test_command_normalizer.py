import unittest

from app.brain.command_normalizer import normalize_command


class CommandNormalizerTests(unittest.TestCase):
    def test_collapses_repeated_whitespace(self) -> None:
        self.assertEqual(normalize_command("  hello   jarvis  "), "hello")

    def test_preserves_unicode(self) -> None:
        self.assertEqual(normalize_command("  café  "), "café")

    def test_preserves_memory_value_case(self) -> None:
        self.assertEqual(normalize_command("remember that nickname = MiSha"), "remember nickname = MiSha")

    def test_preserves_search_query_case(self) -> None:
        self.assertEqual(normalize_command("search OpenAI GPT-5"), "search openai gpt-5")

    def test_aliases_route_correctly(self) -> None:
        self.assertEqual(normalize_command("hi"), "hello")
        self.assertEqual(normalize_command("what time is it"), "time")
        self.assertEqual(normalize_command("launch notepad"), "open notepad")
        self.assertEqual(normalize_command("show commands"), "help")
        self.assertEqual(normalize_command("show system info"), "system info")
        self.assertEqual(normalize_command("show disk space"), "disk space")
        self.assertEqual(normalize_command("what is my computer name"), "computer name")
        self.assertEqual(normalize_command("open my downloads"), "open downloads")
        self.assertEqual(normalize_command("lock pc"), "lock computer")
        self.assertEqual(normalize_command("shutdown computer"), "shutdown computer")
        self.assertEqual(normalize_command("never mind"), "cancel action")
        self.assertEqual(normalize_command("quit"), "exit")

    def test_memory_aliases(self) -> None:
        self.assertEqual(normalize_command("what is my favorite_ide"), "recall favorite_ide")
        self.assertEqual(normalize_command("forget my favorite_ide"), "forget favorite_ide")

    def test_russian_voice_aliases_route_to_the_same_deterministic_commands(self) -> None:
        # RFC-009 follow-up: the owner's real voice requests, transcribed as Russian text,
        # were falling through to the AI/agent runtime and being handled unreliably (see
        # open_website()'s docstring). These map the same well-known actions the English
        # aliases above already cover.
        self.assertEqual(normalize_command("привет"), "hello")
        self.assertEqual(normalize_command("который час"), "time")
        self.assertEqual(normalize_command("какая сегодня дата"), "date")
        self.assertEqual(normalize_command("открой браузер"), "open browser")
        self.assertEqual(normalize_command("открой блокнот"), "open notepad")
        self.assertEqual(normalize_command("открой ютуб"), "open youtube")
        self.assertEqual(normalize_command("открой гугл"), "open google")
        self.assertEqual(normalize_command("выход"), "exit")

    def test_open_site_russian_alias_carries_the_target_through(self) -> None:
        self.assertEqual(normalize_command("открой сайт job.pt"), "open site job.pt")
        self.assertEqual(normalize_command("Открой Сайт GitHub.com"), "open site github.com")
        # No target given -- distinct from a missing command entirely, so the router can
        # give a clear "please specify which site" answer instead of an "unrecognized
        # command" one.
        self.assertEqual(normalize_command("открой сайт"), "open site")

    def test_russian_open_intent_tolerates_filler_words_and_other_verbs(self) -> None:
        # Real owner-reported failure: "открой сайт википедия" spoken with natural filler
        # ("пожалуйста") or a different verb ("зайди" -- go to, rather than "открой" --
        # open) missed the exact alias_map entries/prefix check and fell through to the AI
        # runtime instead of opening anything. This is order-and-filler tolerant instead.
        self.assertEqual(normalize_command("Открой, пожалуйста, сайт Википедия"), "open site википедия")
        self.assertEqual(normalize_command("зайди на сайт вконтакте"), "open site вконтакте")
        self.assertEqual(normalize_command("можешь открыть браузер"), "open browser")
        self.assertEqual(normalize_command("пожалуйста открой ютуб"), "open youtube")
        self.assertEqual(normalize_command("запусти калькулятор"), "open calculator")
