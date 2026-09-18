import unittest

from app.brain.skills.calculator import calculate_expression


class CalculatorTests(unittest.TestCase):
    def test_handles_basic_math(self) -> None:
        self.assertEqual(calculate_expression("25 * 4"), "Result: 100")
        self.assertEqual(calculate_expression("(10 + 5) / 3"), "Result: 5.0")

    def test_rejects_names(self) -> None:
        self.assertEqual(calculate_expression("x + 1"), "Please use a simple arithmetic expression.")

    def test_rejects_calls(self) -> None:
        self.assertEqual(calculate_expression("pow(2, 3)"), "Please use a simple arithmetic expression.")

    def test_rejects_division_by_zero(self) -> None:
        self.assertEqual(calculate_expression("8 / 0"), "Division by zero is not allowed.")

    def test_rejects_large_power(self) -> None:
        self.assertEqual(calculate_expression("2 ** 1000"), "That expression is too large.")
