import unittest
from unittest.mock import patch

from app.brain.security.confirmation import (
    cancel_pending_action,
    confirm_pending_action,
    get_pending_action,
    request_confirmation,
)


class ConfirmationTests(unittest.TestCase):
    def test_request_and_confirm(self) -> None:
        request_confirmation("shutdown")
        self.assertEqual(get_pending_action(), "shutdown")
        self.assertTrue(confirm_pending_action("shutdown"))

    def test_cancel_clears_pending_action(self) -> None:
        request_confirmation("restart")
        cancel_pending_action()
        self.assertIsNone(get_pending_action())

    def test_confirmation_mismatch_fails(self) -> None:
        request_confirmation("lock")
        self.assertFalse(confirm_pending_action("shutdown"))
