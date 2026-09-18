import unittest

from app.brain.history.timeline import add_timeline_entry, clear_timeline, get_timeline, reset_timeline


class TimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_timeline()

    def test_timeline_is_newest_first(self) -> None:
        add_timeline_entry("Calculation completed", success=True, tool_name="calculator.calculate")
        add_timeline_entry("Note created", success=True, tool_name="notes.create")
        entries = get_timeline()
        self.assertEqual(entries[0].display_label, "Note created")
        self.assertEqual(entries[1].display_label, "Calculation completed")

    def test_clear_timeline_is_isolated(self) -> None:
        add_timeline_entry("Calculation completed", success=True, tool_name="calculator.calculate")
        clear_timeline()
        self.assertEqual(get_timeline(), [])
