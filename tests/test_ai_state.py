import unittest

from app.brain.ai.state import get_ai_state, reset_ai_state
from app.brain.ai.models import ProviderStatus, ProviderStatusCategory


class AIStateTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_ai_state()

    def test_state_can_be_reset(self) -> None:
        state = get_ai_state()
        state.enabled = True
        state.model_name = "foo"
        reset_ai_state()
        self.assertFalse(get_ai_state().enabled)
        self.assertEqual(get_ai_state().model_name, "")

    def test_provider_status_updates(self) -> None:
        state = get_ai_state()
        state.update_provider_status(ProviderStatus(category=ProviderStatusCategory.READY))
        self.assertEqual(state.last_provider_status_category, ProviderStatusCategory.READY.value)
