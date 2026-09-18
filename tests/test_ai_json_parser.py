import unittest

from app.brain.ai.json_parser import extract_json_object


class AIJsonParserTests(unittest.TestCase):
    def test_parses_markdown_json(self) -> None:
        text = '```json\n{"category": "conversation", "message": "hi"}\n```'
        payload = extract_json_object(text)
        self.assertEqual(payload["category"], "conversation")

    def test_rejects_trailing_content(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object('{"category": "conversation"} trailing')

    def test_rejects_oversized_payload(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object('{"category": "conversation"}' + ('x' * 20000))
