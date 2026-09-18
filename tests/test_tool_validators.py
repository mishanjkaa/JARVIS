import unittest

from app.brain.tools.validators import validate_arguments, validate_app_name, validate_folder_name, validate_path_text, validate_query, validate_text


class ToolValidatorsTests(unittest.TestCase):
    def test_text_validation(self) -> None:
        self.assertEqual(validate_text("hello", field_name="text"), "hello")

    def test_rejects_control_characters(self) -> None:
        with self.assertRaises(ValueError):
            validate_text("bad\nvalue", field_name="text")

    def test_validates_app_and_folder_names(self) -> None:
        self.assertEqual(validate_app_name("notepad"), "notepad")
        self.assertEqual(validate_folder_name("desktop"), "desktop")

    def test_validate_arguments_rejects_extra(self) -> None:
        schema = {"query": {"type": "query"}}
        with self.assertRaises(ValueError):
            validate_arguments(schema, {"query": "hi", "extra": 1})

    def test_validates_path_and_optional_arguments(self) -> None:
        self.assertEqual(validate_path_text("notes.txt"), "notes.txt")
        schema = {"path": {"type": "path", "required": False}, "recursive": {"type": "bool", "required": False}}
        self.assertEqual(validate_arguments(schema, {"recursive": False}), {"recursive": False})
