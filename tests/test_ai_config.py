import unittest

from config.config_loader import load_config
from config.settings import DEFAULT_SETTINGS


class AIConfigTests(unittest.TestCase):
    def test_load_config_merges_defaults(self) -> None:
        config = load_config()
        self.assertEqual(config["ai_enabled"], DEFAULT_SETTINGS["ai_enabled"])
        self.assertEqual(config["ai_provider"], DEFAULT_SETTINGS["ai_provider"])
