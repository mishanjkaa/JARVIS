import unittest

from app.brain.ai.models import AIResponse, ProviderStatus, ProviderStatusCategory
from app.brain.ai.provider import Provider


class AIProviderTests(unittest.TestCase):
    def test_provider_protocol_is_present(self) -> None:
        self.assertTrue(hasattr(Provider, 'status'))
        self.assertTrue(hasattr(Provider, 'generate_text'))
        self.assertTrue(hasattr(Provider, 'generate_structured'))

    def test_provider_status_is_structured(self) -> None:
        status = ProviderStatus(category=ProviderStatusCategory.READY, detail="ok")
        self.assertEqual(status.category.value, "ready")
        self.assertEqual(status.detail, "ok")
