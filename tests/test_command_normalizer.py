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
