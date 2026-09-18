from __future__ import annotations

from app.brain.context.history import record_safe_command
from app.brain.context.state import get_context, update_context
from app.brain.tools.models import ToolDefinition, ToolResult
from app.brain.planner.result_references import resolve_reference
from app.brain.tools.registry import ToolRegistry
from app.brain.tools.validators import validate_arguments


class ToolExecutor:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def execute_step(
        self,
        step: dict,
        *,
        prior_results: dict[int, dict],
        result_size_limit: int = 200,
    ) -> ToolResult:
        if not isinstance(step, dict):
            return ToolResult(False, "failed", "Plan step invalid.")
        tool_name = step.get("tool_name")
        arguments = step.get("arguments", {})
        step_id = step.get("id")
        if not isinstance(tool_name, str) or not isinstance(arguments, dict):
            return ToolResult(False, "failed", "Plan step invalid.")
        if not isinstance(step_id, int):
            return ToolResult(False, "failed", "Plan step invalid.")

        resolved_arguments = dict(arguments)
        for name, value in resolved_arguments.items():
            if isinstance(value, dict):
                try:
                    resolved_arguments[name] = resolve_reference(value, prior_results, step_id)
                except (TypeError, ValueError):
                    return ToolResult(False, "failed", "Invalid result reference.")
        try:
            definition = self.registry.get(tool_name)
        except KeyError:
            return ToolResult(False, "failed", "Unknown tool.")
        try:
            validated = validate_arguments(definition.argument_schema, resolved_arguments)
        except Exception:
            return ToolResult(False, "failed", "Invalid tool arguments.")
        try:
            result = definition.handler(validated)
        except Exception:
            result = ToolResult(False, "failed", "Tool execution failed.")
        if tool_name == "terminal.execute" or tool_name.startswith("browser.") or tool_name.startswith("vision."):
            result = self._bounded_result(result, result_size_limit, tool_name=tool_name)
            if not result.success and result.message == "Tool result too large.":
                return result
        if not result.success:
            return result
        bounded_result = result if tool_name == "terminal.execute" or tool_name.startswith("browser.") or tool_name.startswith("vision.") else self._bounded_result(result, result_size_limit, tool_name=tool_name)
        if not bounded_result.success:
            return bounded_result
        record_safe_command(self._safe_history_category(tool_name))
        self._update_context(tool_name, bounded_result)
        return bounded_result

    def execute_plan(self, plan: list[dict], *, max_steps: int = 5) -> dict:
        if not isinstance(plan, list):
            return {"success": False, "message": "Plan invalid.", "status_category": "failed"}
        if len(plan) > max_steps:
            return {"success": False, "message": "Plan too long.", "status_category": "failed"}

        results: list[ToolResult] = []
        for index, raw_step in enumerate(plan, 1):
            step = dict(raw_step)
            step.setdefault("id", index)
            result = self.execute_step(
                step,
                prior_results={step_index + 1: item.to_dict() for step_index, item in enumerate(results)},
                result_size_limit=200,
            )
            if not result.success:
                return {"success": False, "message": result.message, "status_category": result.status_category}
            results.append(result)
        return {"success": True, "message": "Plan executed.", "status_category": "success", "results": [result.to_dict() for result in results]}

    def _safe_history_category(self, tool_name: str) -> str:
        if tool_name == "calculator.calculate":
            return "calculation"
        if tool_name == "internet.search":
            return "web search"
        if tool_name == "memory.remember":
            return "memory command"
        if tool_name == "notes.create":
            return "note created"
        if tool_name == "tasks.create":
            return "task created"
        if tool_name == "computer.open_known_folder":
            return "folder opened"
        if tool_name == "computer.open_application":
            return "application opened"
        return "ai plan"

    def _update_context(self, tool_name: str, result: ToolResult) -> None:
        context = get_context()
        if tool_name == "calculator.calculate" and result.success:
            update_context(last_calculator_result=result.display_value or result.message)
        elif tool_name == "computer.open_known_folder" and result.success:
            update_context(last_opened_folder="Folder")
        elif tool_name == "computer.open_application" and result.success:
            update_context(last_opened_application="Application")

    def _bounded_result(self, result: ToolResult, result_size_limit: int, *, tool_name: str) -> ToolResult:
        if not isinstance(result_size_limit, int) or result_size_limit <= 0:
            return ToolResult(False, "failed", "Tool result too large.")
        if tool_name == "terminal.execute":
            effective_limit = max(result_size_limit, 700)
        elif tool_name == "browser.extract_visible_text":
            effective_limit = max(result_size_limit, 4000)
        elif tool_name in {"browser.inspect_elements", "browser.inspect_form_controls"}:
            effective_limit = max(result_size_limit, 2000)
        elif tool_name.startswith("browser."):
            effective_limit = max(result_size_limit, 700)
        elif tool_name == "vision.extract_text":
            effective_limit = max(result_size_limit, 4000)
        elif tool_name == "vision.describe_image":
            effective_limit = max(result_size_limit, 1200)
        elif tool_name == "vision.find_visual_element":
            effective_limit = max(result_size_limit, 1200)
        elif tool_name.startswith("vision."):
            effective_limit = max(result_size_limit, 1200)
        else:
            effective_limit = result_size_limit
        safe_message = self._bounded_text(result.message, effective_limit)
        safe_display_value = self._bounded_text(result.display_value, effective_limit)
        safe_reference_fields: dict[str, object] = {}
        for key, value in result.reference_fields.items():
            if not isinstance(key, str):
                continue
            sanitized = self._sanitize_reference_value(value, effective_limit)
            if sanitized is not None:
                safe_reference_fields[key] = sanitized
        bounded = ToolResult(
            success=result.success,
            status_category=result.status_category,
            message=safe_message,
            display_value=safe_display_value,
            reference_fields=safe_reference_fields,
        )
        if tool_name == "terminal.execute" or tool_name.startswith("browser.") or tool_name.startswith("vision."):
            return bounded
        content_size = len(bounded.message) + len(bounded.display_value)
        for key, value in bounded.reference_fields.items():
            content_size += len(key)
            content_size += len(str(value))
        if content_size > result_size_limit:
            return ToolResult(False, "failed", "Tool result too large.")
        return bounded

    def _bounded_text(self, value: str, result_size_limit: int) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()[:result_size_limit]

    def _sanitize_reference_value(self, value: object, result_size_limit: int) -> object | None:
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return self._bounded_text(value, result_size_limit)
        if isinstance(value, list):
            sanitized_items: list[object] = []
            for item in value[:50]:
                sanitized = self._sanitize_reference_value(item, result_size_limit)
                if sanitized is not None:
                    sanitized_items.append(sanitized)
            return sanitized_items
        if isinstance(value, dict):
            if self._looks_like_grounding_diagnostics(value):
                return self._sanitize_grounding_diagnostics(value, result_size_limit)
            sanitized_dict: dict[str, object] = {}
            for inner_key, inner_value in list(value.items())[:30]:
                if not isinstance(inner_key, str):
                    continue
                sanitized = self._sanitize_reference_value(inner_value, result_size_limit)
                if sanitized is not None:
                    sanitized_dict[str(inner_key)[:40]] = sanitized
            return sanitized_dict
        return None

    def _looks_like_grounding_diagnostics(self, value: dict[object, object]) -> bool:
        return any(
            isinstance(key, str)
            and key in {
                "dom_grounding_attempted",
                "locator_outcome",
                "final_grounding_state",
                "final_verification_type",
            }
            for key in value
        )

    def _sanitize_grounding_diagnostics(self, value: dict[object, object], result_size_limit: int) -> dict[str, object]:
        priority_keys = [
            "dom_grounding_attempted",
            "dom_grounding_result",
            "locator_outcome",
            "locator_candidates",
            "locator_coordinate_frame",
            "candidate_coordinate_space",
            "candidate_normalized_box",
            "candidate_pixel_box",
            "candidate_css_pixel_box",
            "viewport_dimensions",
            "viewport_css_dimensions",
            "visual_viewport_dimensions",
            "visual_viewport_offsets",
            "page_scroll",
            "device_pixel_ratio",
            "device_scale_factor",
            "decoded_screenshot_dimensions",
            "screenshot_scale_option",
            "viewport_only",
            "scale_x",
            "scale_y",
            "geometry_validation_result",
            "geometry_rejection_reason",
            "crop_verification_attempted",
            "crop_pixel_dimensions",
            "crop_context_padding_pixels",
            "verification_context_normalized_box",
            "verification_context_pixel_box",
            "actual_crop_dimensions",
            "source_capture_identity_match",
            "crop_red_pixel_ratio",
            "crop_white_pixel_ratio",
            "crop_non_background_ratio",
            "crop_observer_outcome",
            "full_frame_verification_attempted",
            "full_frame_observer_outcome",
            "full_frame_red_pixel_ratio",
            "full_frame_white_pixel_ratio",
            "full_frame_non_background_ratio",
            "background_only",
            "object_fully_visible",
            "canonical_dominant_colors",
            "canonical_observed_shapes",
            "canonical_object_categories",
            "required_target_properties",
            "matched_target_properties",
            "missing_target_properties",
            "final_grounding_state",
            "final_verification_type",
            "retry_count",
        ]
        sanitized_dict: dict[str, object] = {}
        remaining: list[tuple[str, object]] = []
        for inner_key, inner_value in value.items():
            if not isinstance(inner_key, str):
                continue
            remaining.append((inner_key, inner_value))
        for key in priority_keys:
            if key not in value or not isinstance(key, str):
                continue
            sanitized = self._sanitize_reference_value(value[key], result_size_limit)
            if sanitized is not None:
                sanitized_dict[key[:40]] = sanitized
        for inner_key, inner_value in remaining:
            if inner_key in sanitized_dict:
                continue
            if len(sanitized_dict) >= 60:
                break
            sanitized = self._sanitize_reference_value(inner_value, result_size_limit)
            if sanitized is not None:
                sanitized_dict[inner_key[:40]] = sanitized
        return sanitized_dict
