import unittest

from app.brain.tools.models import ToolDefinition, ToolResult
from app.brain.tools.executor import ToolExecutor
from app.brain.tools.registry import ToolRegistry


class ToolExecutorTests(unittest.TestCase):
    def test_executes_valid_plan(self) -> None:
        executor = ToolExecutor(ToolRegistry())
        result = executor.execute_plan([
            {"tool_name": "system.get_time", "arguments": {}}
        ], max_steps=5)
        self.assertTrue(result["success"])

    def test_rejects_invalid_step(self) -> None:
        executor = ToolExecutor(ToolRegistry())
        result = executor.execute_plan([{"tool_name": "unknown", "arguments": {}}], max_steps=5)
        self.assertFalse(result["success"])

    def test_preserves_browser_visual_grounding_diagnostics_beyond_default_nested_dict_limit(self) -> None:
        diagnostics = {f"extra_{index}": index for index in range(35)}
        diagnostics.update(
            {
                "dom_grounding_attempted": True,
                "locator_outcome": "found",
                "final_grounding_state": "candidate_unverified",
                "final_verification_type": "",
                "required_target_properties": ["color:red", "shape:circle"],
            }
        )

        def _handler(_: dict[str, object]) -> ToolResult:
            return ToolResult(
                True,
                "success",
                "ok",
                reference_fields={
                    "vision_operation": "find_visual_element_in_browser_capture",
                    "grounding_diagnostics": diagnostics,
                },
            )

        tool = ToolDefinition(
            name="vision.find_visual_element_in_browser_capture",
            description="test tool",
            argument_schema={},
            risk_level="low",
            requires_confirmation=False,
            handler=_handler,
            formatter=lambda result: result.message,
        )
        executor = ToolExecutor(ToolRegistry([tool]))
        result = executor.execute_step(
            {"id": 1, "tool_name": "vision.find_visual_element_in_browser_capture", "arguments": {}},
            prior_results={},
            result_size_limit=1200,
        )
        self.assertTrue(result.success)
        grounded = result.reference_fields.get("grounding_diagnostics")
        self.assertIsInstance(grounded, dict)
        self.assertEqual(grounded.get("final_grounding_state"), "candidate_unverified")
        self.assertEqual(grounded.get("locator_outcome"), "found")
        self.assertEqual(grounded.get("required_target_properties"), ["color:red", "shape:circle"])
