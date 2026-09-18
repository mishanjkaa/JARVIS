import unittest

from app.brain.intent.classifier import IntentClassifier
from app.brain.intent.models import IntentCategory


class IntentClassifierTests(unittest.TestCase):
    def test_classifies_calculation_intent(self) -> None:
        classifier = IntentClassifier()
        result = classifier.classify("work out 25 times 4")
        self.assertEqual(result.intent, IntentCategory.CALCULATION)
        self.assertTrue(result.tools_required)
        self.assertFalse(result.multi_step_required)

    def test_classifies_memory_write(self) -> None:
        classifier = IntentClassifier()
        result = classifier.classify("remember that this project uses Python")
        self.assertEqual(result.intent, IntentCategory.MEMORY_WRITE)

    def test_classifies_filesystem_request(self) -> None:
        classifier = IntentClassifier()
        result = classifier.classify("Create file notes.txt")
        self.assertEqual(result.intent, IntentCategory.FILESYSTEM)

    def test_rejects_empty_input(self) -> None:
        classifier = IntentClassifier()
        with self.assertRaises(ValueError):
            classifier.classify("   ")
