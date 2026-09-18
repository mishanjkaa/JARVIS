from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from app.brain.filesystem.path_policy import get_trusted_roots
from app.brain.intelligence.errors import DynamicPlanError, MalformedModelOutputError
from app.brain.intelligence.models import DynamicPlan, DynamicPlanStep, NaturalLanguageTask, ToolCatalogEntry
from app.brain.intelligence.vision_query import canonicalize_vision_find_query_text, looks_like_json_container_text, normalize_vision_query
from app.brain.planner.result_references import is_result_reference
from app.brain.terminal.policy import default_terminal_working_directory

_ALLOWED_PLAN_FIELDS = {"goal", "success_criteria", "steps"}
_ALLOWED_STEP_FIELDS = {"tool", "arguments", "description", "depends_on", "expected_result"}
_BROWSER_SESSION_FIELDS = {"session_id", "display_value"}
_BROWSER_SESSION_PRODUCERS = {"browser.start_session", "browser.get_active_session"}
_BROWSER_FORM_SUBMIT_OPERATIONS = {"browser_form_submit", "browser_form_fill_submit"}
_BROWSER_TEMPORARY_WORKFLOW_OPERATIONS = {
    "browser_navigation",
    "browser_title",
    "browser_summary",
    "browser_screenshot",
    "browser_scroll",
    "browser_form_inspection",
    "browser_form_fill_submit",
    "browser_visual_describe",
    "browser_visual_extract_text",
    "browser_visual_find_element",
}
_BROWSER_VISUAL_ANALYSIS_TOOLS = {
    "browser_visual_describe": "vision.describe_browser_capture",
    "browser_visual_extract_text": "vision.extract_text_from_browser_capture",
    "browser_visual_find_element": "vision.find_visual_element_in_browser_capture",
}
_BROWSER_FORM_CONTROL_TYPE_ALIASES = {
    "text": "text_input",
    "text_input": "text_input",
    "search": "text_input",
    "search-field": "text_input",
    "search_field": "text_input",
    "textbox": "text_input",
    "text field": "text_input",
    "visible_element_kind:text": "text_input",
    "visible_element_kind:text_input": "text_input",
    "textarea": "textarea",
}
_BROWSER_WAIT_UNTIL_ALIASES = {
    "domcontentloaded": "domcontentloaded",
    "dom_content_loaded": "domcontentloaded",
    "dom-content-loaded": "domcontentloaded",
    "load": "load",
    "networkidle": "networkidle",
    "network_idle": "networkidle",
    "network-idle": "networkidle",
}
_VISION_DETAIL_LEVEL_ALIASES = {
    "brief": "brief",
    "short": "brief",
    "normal": "normal",
    "medium": "normal",
    "standard": "normal",
    "default": "normal",
    "detailed": "detailed",
    "detail": "detailed",
    "full": "detailed",
}
_BROWSER_DEFAULT_ARGUMENTS: dict[str, dict[str, Any]] = {
    "browser.open_url": {"wait_until": "domcontentloaded", "timeout_seconds": 30},
    "browser.open_new_tab": {"wait_until": "domcontentloaded", "timeout_seconds": 30},
}
_DEBUG_MAX_STRING_LENGTH = 180
_DEBUG_MAX_ITEMS = 12


class DynamicPlanner:
    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider
        self.last_trace: dict[str, Any] = {}

    def create_plan(
        self,
        task: NaturalLanguageTask,
        *,
        tool_catalog: list[ToolCatalogEntry],
        context: dict[str, Any],
        max_steps: int,
    ) -> DynamicPlan:
        self.last_trace = {}
        if self.provider is not None and hasattr(self.provider, "create_plan"):
            payload = self.provider.create_plan(task, tool_catalog=tool_catalog, context=context, max_steps=max_steps)
        else:
            raise DynamicPlanError("Natural-language planning is unavailable right now.")
        self.last_trace["raw_provider_payload"] = _snapshot_debug_value(payload)
        return self._normalize_plan_payload(
            payload,
            original_request=task.raw_input,
            max_steps=max_steps,
            task=task,
            tool_catalog=tool_catalog,
        )

    def _normalize_plan_payload(
        self,
        payload: Any,
        *,
        original_request: str,
        max_steps: int,
        task: NaturalLanguageTask,
        tool_catalog: list[ToolCatalogEntry],
    ) -> DynamicPlan:
        if isinstance(payload, DynamicPlan):
            plan = payload
        else:
            if not isinstance(payload, dict):
                raise MalformedModelOutputError("structured plan is invalid")
            extra = set(payload) - _ALLOWED_PLAN_FIELDS
            if extra:
                raise MalformedModelOutputError("unexpected fields in plan output")
            goal = str(payload.get("goal") or "").strip()
            success_criteria = payload.get("success_criteria")
            raw_steps = payload.get("steps")
            if not goal or not isinstance(success_criteria, list) or not isinstance(raw_steps, list):
                raise MalformedModelOutputError("structured plan is invalid")
            steps: list[DynamicPlanStep] = []
            for raw_step in raw_steps:
                if not isinstance(raw_step, dict):
                    raise MalformedModelOutputError("structured step is invalid")
                extra_step = set(raw_step) - _ALLOWED_STEP_FIELDS
                if extra_step:
                    raise MalformedModelOutputError("unexpected fields in plan step")
                tool = str(raw_step.get("tool") or "").strip()
                arguments = raw_step.get("arguments")
                if not tool or not isinstance(arguments, dict):
                    raise MalformedModelOutputError("structured step is invalid")
                depends_on = raw_step.get("depends_on", [])
                if depends_on is None:
                    depends_on = []
                if not isinstance(depends_on, list) or not all(isinstance(item, int) for item in depends_on):
                    raise MalformedModelOutputError("step dependencies are invalid")
                steps.append(
                    DynamicPlanStep(
                        tool=tool,
                        arguments=dict(arguments),
                        description=str(raw_step.get("description") or "").strip(),
                        depends_on=list(depends_on),
                        expected_result=str(raw_step.get("expected_result") or "").strip(),
                    )
                )
            plan = DynamicPlan(
                goal=goal,
                success_criteria=[str(item).strip() for item in success_criteria if str(item).strip()],
                steps=steps,
                original_request=original_request,
            )
        self.last_trace["parsed_dynamic_plan"] = _snapshot_plan(plan)
        plan = self._canonicalize_vision_find_visual_element_plan(plan, task=task)
        plan = self._canonicalize_browser_visual_session_acquisition(plan, task=task)
        self.last_trace["browser_visual_session_canonicalized_plan"] = _snapshot_plan(plan)
        plan = self._finalize_browser_session_acquisition(plan, task=task, max_steps=max_steps)
        self.last_trace["browser_session_finalized_plan"] = _snapshot_plan(plan)
        plan = self._finalize_browser_visual_navigation_and_analysis(plan, task=task, max_steps=max_steps)
        self.last_trace["browser_visual_navigation_finalized_plan"] = _snapshot_plan(plan)
        plan = self._finalize_browser_visual_capture(plan, task=task, max_steps=max_steps)
        self.last_trace["browser_visual_finalized_plan"] = _snapshot_plan(plan)
        plan = self._canonicalize_browser_plan(plan, task=task, tool_catalog=tool_catalog)
        self.last_trace["browser_canonicalized_plan"] = _snapshot_plan(plan)
        plan = self._canonicalize_browser_visual_plan(plan, task=task)
        self.last_trace["browser_visual_canonicalized_plan"] = _snapshot_plan(plan)
        plan = self._finalize_browser_form_evidence(plan, task=task, max_steps=max_steps)
        self.last_trace["browser_form_finalized_plan"] = _snapshot_plan(plan)
        plan = self._canonicalize_browser_plan(plan, task=task, tool_catalog=tool_catalog)
        self.last_trace["browser_recanonicalized_plan"] = _snapshot_plan(plan)
        plan = self._finalize_browser_lifecycle(plan, task=task, max_steps=max_steps, tool_catalog=tool_catalog)
        self.last_trace["browser_lifecycle_finalized_plan"] = _snapshot_plan(plan)
        plan = self._canonicalize_browser_plan(plan, task=task, tool_catalog=tool_catalog)
        self.last_trace["browser_post_lifecycle_canonicalized_plan"] = _snapshot_plan(plan)
        plan = self._canonicalize_browser_visual_success_criteria(plan, task=task)
        self.last_trace["browser_visual_success_criteria_canonicalized_plan"] = _snapshot_plan(plan)
        if len(plan.steps) > max_steps:
            raise DynamicPlanError("plan exceeds the maximum number of steps")
        self.last_trace["pre_validation_plan"] = _snapshot_plan(plan)
        return plan

    def _canonicalize_vision_find_visual_element_plan(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
    ) -> DynamicPlan:
        if task.requested_operation != "vision_find_visual_element":
            return plan
        if not plan.steps:
            return plan
        if any(not step.tool.startswith("vision.") for step in plan.steps):
            return plan
        if any(step.tool != "vision.find_visual_element" for step in plan.steps):
            return plan
        requested_paths = {
            _normalize_local_path_token(path)
            for path in task.requested_artifacts
            if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        }
        raw_expected_query = _task_constraint_value(task, "vision_query")
        expected_query = normalize_vision_query(raw_expected_query)
        valid_steps: list[DynamicPlanStep] = []
        normalized_pairs: set[tuple[str, str]] = set()
        for step in plan.steps:
            normalized_path, normalized_query = _literal_vision_find_step_signature(step, expected_query=raw_expected_query)
            if not normalized_path or not normalized_query:
                continue
            if requested_paths and normalized_path not in requested_paths:
                return plan
            if expected_query and expected_query not in normalized_query:
                return plan
            valid_steps.append(step)
            normalized_pairs.add((normalized_path, normalized_query))
        if not valid_steps:
            return plan
        if len(normalized_pairs) != 1:
            return plan
        canonical_step = valid_steps[0]
        plan.steps = [
            DynamicPlanStep(
                tool="vision.find_visual_element",
                arguments={
                    "path": str(canonical_step.arguments.get("path") or "").strip(),
                    "query": str(canonical_step.arguments.get("query") or "").strip(),
                    **({"max_results": canonical_step.arguments["max_results"]} if isinstance(canonical_step.arguments.get("max_results"), int) and not isinstance(canonical_step.arguments.get("max_results"), bool) else {}),
                },
                description=canonical_step.description,
                depends_on=[],
                expected_result=canonical_step.expected_result,
            )
        ]
        return plan

    def _canonicalize_browser_visual_session_acquisition(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
    ) -> DynamicPlan:
        if not task.requested_operation.startswith("browser_visual_"):
            return plan
        if not _first_explicit_browser_url(task):
            return plan
        producer_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool in _BROWSER_SESSION_PRODUCERS]
        if not producer_indices:
            return plan
        start_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.start_session"]
        active_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.get_active_session"]
        if not start_indices and len(active_indices) == 1:
            active_step = plan.steps[active_indices[0] - 1]
            if active_step.arguments:
                return plan
            active_step.tool = "browser.start_session"
            active_step.arguments = {"headless": False}
            active_step.depends_on = []
            active_step.expected_result = active_step.expected_result or "session started successfully"
            return plan
        if len(start_indices) == 1 and active_indices:
            if any(plan.steps[index - 1].arguments for index in active_indices):
                return plan
            plan.steps = _remove_step_ids(
                plan.steps,
                remove_step_ids=set(active_indices),
                redirect_step_ids={index: start_indices[0] for index in active_indices},
            )
        return plan

    def _finalize_browser_session_acquisition(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
        max_steps: int,
    ) -> DynamicPlan:
        if not _browser_task_requires_temporary_session(task, plan):
            if not task.requested_operation.startswith("browser_visual_"):
                return plan
            producer_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool in _BROWSER_SESSION_PRODUCERS]
            if producer_indices:
                return plan
            if len(plan.steps) >= max_steps:
                raise DynamicPlanError("browser session acquisition could not be prepended within the plan step limit")
            shifted_steps = [_shift_step_numbering(step, delta=1, preserve_browser_session_step_one=True) for step in plan.steps]
            plan.steps = [
                DynamicPlanStep(
                    tool="browser.get_active_session",
                    arguments={},
                    description="Reuse the active browser session.",
                    depends_on=[],
                    expected_result="active browser session available",
                ),
                *shifted_steps,
            ]
            return plan
        producer_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool in _BROWSER_SESSION_PRODUCERS]
        if producer_indices:
            return plan
        if len(plan.steps) >= max_steps:
            raise DynamicPlanError("browser session acquisition could not be prepended within the plan step limit")
        shifted_steps = [_shift_step_numbering(step, delta=1, preserve_browser_session_step_one=True) for step in plan.steps]
        plan.steps = [
            DynamicPlanStep(
                tool="browser.start_session",
                arguments={"headless": False},
                description="Start a browser session.",
                depends_on=[],
                expected_result="session started successfully",
            ),
            *shifted_steps,
        ]
        return plan

    def _canonicalize_browser_plan(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
        tool_catalog: list[ToolCatalogEntry],
    ) -> DynamicPlan:
        if not task.requested_operation.startswith("browser_") or task.requested_operation.startswith("browser_unsupported_"):
            return plan
        shared_form_hints = _shared_browser_form_hints(plan)
        browser_navigation_origins = _browser_navigation_origins(plan)
        browser_argument_names = {
            entry.name: set(entry.argument_schema)
            for entry in tool_catalog
            if entry.enabled and entry.name.startswith("browser.")
        }
        all_browser_argument_names = set().union(*browser_argument_names.values()) if browser_argument_names else set()
        producer_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool in _BROWSER_SESSION_PRODUCERS]
        if len(producer_indices) != 1:
            return plan
        producer_step_id = producer_indices[0]
        session_tool_names = {
            entry.name
            for entry in tool_catalog
            if entry.enabled and entry.name.startswith("browser.") and "session_id" in entry.argument_schema
        }
        if not session_tool_names:
            return plan
        session_user_indices = [
            index
            for index, step in enumerate(plan.steps, 1)
            if step.tool in session_tool_names and step.tool != "browser.close_session" and index > producer_step_id
        ]
        final_session_user_step_id = session_user_indices[-1] if session_user_indices else None
        for index, step in enumerate(plan.steps, 1):
            allowed_arguments = browser_argument_names.get(step.tool, set())
            if step.tool.startswith("browser.") and allowed_arguments:
                for argument_name in list(step.arguments):
                    if argument_name in allowed_arguments:
                        continue
                    if argument_name in all_browser_argument_names:
                        del step.arguments[argument_name]
                defaults = _BROWSER_DEFAULT_ARGUMENTS.get(step.tool, {})
                for argument_name, default_value in defaults.items():
                    if argument_name not in step.arguments:
                        step.arguments[argument_name] = default_value
                _drop_empty_browser_optional_arguments(step.arguments)
            if step.tool in {"browser.input_text", "browser.clear_input"}:
                control_type = str(step.arguments.get("control_type") or "").strip().lower()
                if control_type in _BROWSER_FORM_CONTROL_TYPE_ALIASES:
                    step.arguments["control_type"] = _BROWSER_FORM_CONTROL_TYPE_ALIASES[control_type]
                _populate_missing_form_target_hints(step.arguments, shared_form_hints)
                _populate_task_form_target_hints(step.arguments, task)
                _populate_task_form_input_text(step.arguments, task)
            if step.tool == "browser.submit_form":
                submit_origin = _nearest_browser_origin(browser_navigation_origins, index)
                if submit_origin and not str(step.arguments.get("allowed_destination_origin") or "").strip():
                    step.arguments["allowed_destination_origin"] = submit_origin
            if "wait_until" in step.arguments:
                normalized_wait = _canonical_browser_wait_until(step.arguments.get("wait_until"))
                if normalized_wait is not None:
                    step.arguments["wait_until"] = normalized_wait
            if step.tool not in session_tool_names or index <= producer_step_id:
                continue
            session_value = step.arguments.get("session_id")
            if not _is_canonical_browser_session_reference(session_value, producer_step_id):
                step.arguments["session_id"] = {"from_step": producer_step_id, "field": "session_id"}
            _drop_invalid_optional_browser_references(step.arguments, current_step_id=index)
            dependencies = [
                dependency
                for dependency in step.depends_on
                if isinstance(dependency, int) and 0 < dependency < index
            ]
            for from_step in _result_reference_dependencies(step.arguments, current_step_id=index):
                if from_step not in dependencies:
                    dependencies.append(from_step)
            if producer_step_id not in dependencies:
                dependencies.append(producer_step_id)
            if step.tool == "browser.close_session" and final_session_user_step_id is not None and final_session_user_step_id not in dependencies:
                dependencies.append(final_session_user_step_id)
            step.depends_on = _dedupe_preserving_order(dependencies)
        return plan

    def _finalize_browser_form_evidence(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
        max_steps: int,
    ) -> DynamicPlan:
        if task.requested_operation not in _BROWSER_FORM_SUBMIT_OPERATIONS:
            return plan
        submit_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.submit_form"]
        producer_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool in _BROWSER_SESSION_PRODUCERS]
        if len(submit_indices) != 1 or len(producer_indices) != 1:
            return plan
        submit_step_id = submit_indices[0]
        producer_step_id = producer_indices[0]
        evidence_tools = {"browser.get_page_info", "browser.extract_visible_text"}
        if any(index > submit_step_id and step.tool in evidence_tools for index, step in enumerate(plan.steps, 1)):
            return plan
        if len(plan.steps) >= max_steps:
            raise DynamicPlanError("browser form evidence could not be appended within the plan step limit")
        close_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.close_session"]
        insert_at = close_indices[0] - 1 if close_indices else len(plan.steps)
        plan.steps.insert(
            insert_at,
            DynamicPlanStep(
                tool="browser.get_page_info",
                arguments={"session_id": {"from_step": producer_step_id, "field": "session_id"}},
                description="Capture grounded post-submit page evidence.",
                depends_on=[producer_step_id, submit_step_id],
                expected_result="page info captured after submission",
            ),
        )
        return plan

    def _finalize_browser_lifecycle(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
        max_steps: int,
        tool_catalog: list[ToolCatalogEntry],
    ) -> DynamicPlan:
        if task.requested_operation.startswith("browser_unsupported_"):
            return plan
        if not any(entry.name == "browser.close_session" and entry.enabled for entry in tool_catalog):
            return plan
        start_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.start_session"]
        close_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.close_session"]
        if close_indices or len(start_indices) != 1:
            return plan
        session_user_indices = [
            index
            for index, step in enumerate(plan.steps, 1)
            if step.tool.startswith("browser.") and step.tool not in {"browser.start_session", "browser.close_session"}
        ]
        if not session_user_indices:
            return plan
        if len(plan.steps) >= max_steps:
            raise DynamicPlanError("browser lifecycle cleanup could not be appended within the plan step limit")
        start_step_id = start_indices[0]
        last_session_user_step_id = session_user_indices[-1]
        depends_on = [start_step_id]
        if last_session_user_step_id != start_step_id:
            depends_on.append(last_session_user_step_id)
        plan.steps.append(
            DynamicPlanStep(
                tool="browser.close_session",
                arguments={"session_id": {"from_step": start_step_id, "field": "session_id"}},
                description="Close the browser session.",
                depends_on=depends_on,
                expected_result="browser session closed",
            )
        )
        return plan

    def _finalize_browser_visual_capture(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
        max_steps: int,
    ) -> DynamicPlan:
        expected_tool = _BROWSER_VISUAL_ANALYSIS_TOOLS.get(task.requested_operation)
        if not expected_tool:
            return plan
        producer_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool in _BROWSER_SESSION_PRODUCERS]
        if len(producer_indices) != 1:
            return plan
        capture_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.capture_view"]
        analysis_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == expected_tool]
        if not analysis_indices:
            return plan
        if capture_indices:
            return plan
        if len(plan.steps) >= max_steps:
            raise DynamicPlanError("browser visual capture could not be inserted within the plan step limit")
        insert_at = analysis_indices[0] - 1
        producer_step_id = producer_indices[0]
        dependency_source = producer_step_id
        if insert_at > 0:
            prior_step_id = insert_at
            if prior_step_id > producer_step_id:
                dependency_source = prior_step_id
        capture_step = DynamicPlanStep(
            tool="browser.capture_view",
            arguments={"session_id": {"from_step": producer_step_id, "field": "session_id"}},
            description="Capture the current visible browser viewport.",
            depends_on=sorted({producer_step_id, dependency_source}),
            expected_result="opaque browser capture created",
        )
        plan.steps.insert(insert_at, capture_step)
        return plan

    def _finalize_browser_visual_navigation_and_analysis(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
        max_steps: int,
    ) -> DynamicPlan:
        expected_tool = _BROWSER_VISUAL_ANALYSIS_TOOLS.get(task.requested_operation)
        if not expected_tool:
            return plan
        producer_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool in _BROWSER_SESSION_PRODUCERS]
        if len(producer_indices) != 1:
            return plan
        producer_step_id = producer_indices[0]
        explicit_url = _first_explicit_browser_url(task)
        open_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.open_url"]
        if explicit_url and not open_indices:
            if len(plan.steps) >= max_steps:
                raise DynamicPlanError("browser visual navigation could not be inserted within the plan step limit")
            insert_at = producer_step_id
            plan.steps.insert(
                insert_at,
                DynamicPlanStep(
                    tool="browser.open_url",
                    arguments={
                        "session_id": {"from_step": producer_step_id, "field": "session_id"},
                        "url": explicit_url,
                        "wait_until": "domcontentloaded",
                        "timeout_seconds": 30,
                    },
                    description="Open the requested page.",
                    depends_on=[producer_step_id],
                    expected_result="page loaded successfully",
                ),
            )
        analysis_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == expected_tool]
        if analysis_indices:
            return plan
        if len(plan.steps) >= max_steps:
            raise DynamicPlanError("browser visual analysis could not be inserted within the plan step limit")
        close_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.close_session"]
        insert_at = close_indices[0] - 1 if close_indices else len(plan.steps)
        dependency_step_id = producer_step_id
        if explicit_url:
            refreshed_open_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.open_url"]
            if refreshed_open_indices:
                dependency_step_id = refreshed_open_indices[-1]
        analysis_arguments: dict[str, Any] = {}
        expected_result = "grounded browser visual evidence captured"
        if expected_tool == "vision.describe_browser_capture":
            analysis_arguments = {"detail_level": "normal"}
            expected_result = "grounded browser visual description captured"
        elif expected_tool == "vision.extract_text_from_browser_capture":
            analysis_arguments = {"max_characters": 4000}
            expected_result = "grounded visible browser text captured"
        elif expected_tool == "vision.find_visual_element_in_browser_capture":
            query = _task_constraint_value(task, "browser_visual_query").strip()
            if not query:
                return plan
            analysis_arguments = {"query": query}
            expected_result = "grounded browser visual search completed"
        plan.steps.insert(
            insert_at,
            DynamicPlanStep(
                tool=expected_tool,
                arguments=analysis_arguments,
                description="Analyze the current browser viewport visually.",
                depends_on=[dependency_step_id],
                expected_result=expected_result,
            ),
        )
        return plan

    def _canonicalize_browser_visual_success_criteria(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
    ) -> DynamicPlan:
        if task.requested_operation != "browser_visual_find_element":
            return plan
        plan.success_criteria = [
            "browser viewport captured",
            "grounded visual search completed",
            "visual search outcome reported",
        ]
        return plan

    def _canonicalize_browser_visual_plan(
        self,
        plan: DynamicPlan,
        *,
        task: NaturalLanguageTask,
    ) -> DynamicPlan:
        expected_tool = _BROWSER_VISUAL_ANALYSIS_TOOLS.get(task.requested_operation)
        if not expected_tool:
            return plan
        capture_indices = [index for index, step in enumerate(plan.steps, 1) if step.tool == "browser.capture_view"]
        if len(capture_indices) != 1:
            return plan
        capture_step_id = capture_indices[0]
        expected_query = _task_constraint_value(task, "browser_visual_query")
        for index, step in enumerate(plan.steps, 1):
            if step.tool == "browser.capture_view":
                tab_id = step.arguments.get("tab_id")
                if tab_id is not None and not is_result_reference(tab_id):
                    step.arguments.pop("tab_id", None)
            if step.tool != expected_tool:
                continue
            capture_value = step.arguments.get("capture_id")
            if not (
                is_result_reference(capture_value)
                and capture_value.get("from_step") == capture_step_id
                and capture_value.get("field") == "capture_id"
            ):
                step.arguments["capture_id"] = {"from_step": capture_step_id, "field": "capture_id"}
            dependencies = [
                dependency
                for dependency in step.depends_on
                if isinstance(dependency, int) and 0 < dependency < index
            ]
            if capture_step_id not in dependencies:
                dependencies.append(capture_step_id)
            step.depends_on = _dedupe_preserving_order(dependencies)
            if step.tool == "vision.find_visual_element_in_browser_capture":
                query = step.arguments.get("query")
                if isinstance(query, str):
                    canonical_query = canonicalize_vision_find_query_text(query, expected_query=expected_query)
                    if canonical_query is not None:
                        if expected_query and normalize_vision_query(expected_query) in normalize_vision_query(canonical_query):
                            step.arguments["query"] = expected_query
                        else:
                            step.arguments["query"] = canonical_query
            if step.tool == "vision.describe_browser_capture":
                detail_level = str(step.arguments.get("detail_level") or "").strip().lower()
                step.arguments["detail_level"] = _VISION_DETAIL_LEVEL_ALIASES.get(detail_level, "normal")
        return plan


def _is_canonical_browser_session_reference(value: Any, start_step_id: int) -> bool:
    return (
        is_result_reference(value)
        and value.get("from_step") == start_step_id
        and value.get("field") in _BROWSER_SESSION_FIELDS
    )


def _dedupe_preserving_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _remove_step_ids(
    steps: list[DynamicPlanStep],
    *,
    remove_step_ids: set[int],
    redirect_step_ids: dict[int, int] | None = None,
) -> list[DynamicPlanStep]:
    if not remove_step_ids:
        return list(steps)
    redirect_map: dict[int, int] = {
        int(source_id): int(target_id)
        for source_id, target_id in (redirect_step_ids or {}).items()
        if source_id in remove_step_ids
    }
    retained_old_ids = [index for index in range(1, len(steps) + 1) if index not in remove_step_ids]
    new_id_by_old = {old_id: new_index for new_index, old_id in enumerate(retained_old_ids, 1)}
    rebuilt: list[DynamicPlanStep] = []
    for old_index, step in enumerate(steps, 1):
        if old_index in remove_step_ids:
            continue
        arguments = _remap_step_references(step.arguments, new_id_by_old, redirect_map)
        dependencies: list[int] = []
        for dependency in step.depends_on:
            target = dependency
            if dependency in remove_step_ids:
                target = redirect_map.get(dependency)
            if target is None:
                continue
            mapped = new_id_by_old.get(target)
            if mapped is None:
                continue
            dependencies.append(mapped)
        rebuilt.append(
            DynamicPlanStep(
                tool=step.tool,
                arguments=arguments,
                description=step.description,
                depends_on=_dedupe_preserving_order(dependencies),
                expected_result=step.expected_result,
            )
        )
    return rebuilt


def _remap_step_references(
    arguments: dict[str, Any],
    new_id_by_old: dict[int, int],
    redirect_map: dict[int, int],
) -> dict[str, Any]:
    remapped: dict[str, Any] = {}
    for key, value in arguments.items():
        if not is_result_reference(value):
            remapped[key] = value
            continue
        from_step = value.get("from_step")
        if not isinstance(from_step, int):
            remapped[key] = value
            continue
        target_old = redirect_map.get(from_step, from_step)
        target_new = new_id_by_old.get(target_old)
        if target_new is None:
            remapped[key] = value
            continue
        remapped[key] = {"from_step": target_new, "field": value.get("field")}
    return remapped


def _browser_task_requires_temporary_session(task: NaturalLanguageTask, plan: DynamicPlan) -> bool:
    if task.requested_operation.startswith("browser_unsupported_"):
        return False
    has_navigation = any(step.tool in {"browser.open_url", "browser.open_new_tab"} for step in plan.steps)
    if task.requested_operation in _BROWSER_TEMPORARY_WORKFLOW_OPERATIONS:
        return has_navigation
    if task.requested_operation in {"browser_form_fill", "browser_form_clear", "browser_form_submit"}:
        return has_navigation
    return False


def _shift_step_numbering(
    step: DynamicPlanStep,
    *,
    delta: int,
    preserve_browser_session_step_one: bool,
) -> DynamicPlanStep:
    shifted_arguments = {
        key: _shift_reference_value(
            value,
            step_tool=step.tool,
            argument_name=key,
            delta=delta,
            preserve_browser_session_step_one=preserve_browser_session_step_one,
        )
        for key, value in step.arguments.items()
    }
    return DynamicPlanStep(
        tool=step.tool,
        arguments=shifted_arguments,
        description=step.description,
        depends_on=[dependency + delta for dependency in step.depends_on],
        expected_result=step.expected_result,
    )


def _shift_reference_value(
    value: Any,
    *,
    step_tool: str,
    argument_name: str,
    delta: int,
    preserve_browser_session_step_one: bool,
) -> Any:
    if is_result_reference(value):
        from_step = value.get("from_step")
        field = value.get("field")
        if (
            preserve_browser_session_step_one
            and step_tool.startswith("browser.")
            and argument_name == "session_id"
            and from_step == 1
            and field in _BROWSER_SESSION_FIELDS
        ):
            return dict(value)
        shifted = dict(value)
        if isinstance(from_step, int):
            shifted["from_step"] = from_step + delta
        return shifted
    if isinstance(value, list):
        return [
            _shift_reference_value(
                item,
                step_tool=step_tool,
                argument_name=argument_name,
                delta=delta,
                preserve_browser_session_step_one=preserve_browser_session_step_one,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _shift_reference_value(
                item,
                step_tool=step_tool,
                argument_name=argument_name,
                delta=delta,
                preserve_browser_session_step_one=preserve_browser_session_step_one,
            )
            for key, item in value.items()
        }
    return value


def _shared_browser_form_hints(plan: DynamicPlan) -> dict[str, str]:
    for step in plan.steps:
        if step.tool != "browser.submit_form":
            continue
        hints: dict[str, str] = {}
        for field in ("label_hint", "placeholder_hint", "name_hint", "form_text_hint"):
            value = str(step.arguments.get(field) or "").strip()
            if value:
                hints[field] = value
        if hints:
            return hints
    return {}


def _populate_missing_form_target_hints(arguments: dict[str, Any], shared_hints: dict[str, str]) -> None:
    if not shared_hints:
        return
    if str(arguments.get("label_hint") or "").strip():
        return
    if str(arguments.get("placeholder_hint") or "").strip():
        return
    if str(arguments.get("name_hint") or "").strip():
        return
    ordinal = arguments.get("ordinal", 0)
    if isinstance(ordinal, int) and ordinal > 0:
        return
    if shared_hints.get("label_hint"):
        arguments["label_hint"] = shared_hints["label_hint"]
        return
    if shared_hints.get("placeholder_hint"):
        arguments["placeholder_hint"] = shared_hints["placeholder_hint"]
        return
    if shared_hints.get("name_hint"):
        arguments["name_hint"] = shared_hints["name_hint"]
        return
    if shared_hints.get("form_text_hint"):
        arguments["label_hint"] = shared_hints["form_text_hint"]


def _populate_task_form_target_hints(arguments: dict[str, Any], task: NaturalLanguageTask) -> None:
    if any(str(arguments.get(field) or "").strip() for field in ("label_hint", "placeholder_hint", "name_hint")):
        return
    ordinal = arguments.get("ordinal", 0)
    if isinstance(ordinal, int) and ordinal > 0:
        return
    lowered = task.raw_input.lower()
    if any(token in lowered for token in ("search field", "search box", "search form", "\u043f\u043e\u0438\u0441\u043a\u043e\u0432\u043e\u0435 \u043f\u043e\u043b\u0435", "\u043f\u043e\u0438\u0441\u043a\u043e\u0432\u043e\u043c \u043f\u043e\u043b\u0435")):
        arguments["label_hint"] = "Search"


def _populate_task_form_input_text(arguments: dict[str, Any], task: NaturalLanguageTask) -> None:
    text = arguments.get("text")
    if isinstance(text, str) and text.strip():
        return
    extracted = _extract_requested_form_text(task.raw_input)
    if extracted:
        arguments["text"] = extracted


def _extract_requested_form_text(raw_input: str) -> str:
    patterns = (
        re.compile(
            r"(?:enter|type|fill)\s+(?P<value>.+?)\s+into\s+(?:the\s+)?(?:search field|search box|text field|input field|textarea|field)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:\u0432\u0432\u0435\u0434\u0438|\u043d\u0430\u0431\u0435\u0440\u0438|\u0437\u0430\u043f\u043e\u043b\u043d\u0438)\s+(?P<value>.+?)\s+\u0432\s+(?:\u043f\u043e\u0438\u0441\u043a\u043e\u0432\u043e\u0435\s+)?(?:\u043f\u043e\u043b\u0435|\u0444\u043e\u0440\u043c\u0443)",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(raw_input)
        if match is None:
            continue
        value = str(match.group("value") or "").strip().strip("\"'")
        if value:
            return value
    return ""


def _drop_empty_browser_optional_arguments(arguments: dict[str, Any]) -> None:
    for field in (
        "label_hint",
        "placeholder_hint",
        "name_hint",
        "form_text_hint",
        "submit_text_hint",
        "text_hint",
        "href_hint",
        "page_context",
        "allowed_destination_origin",
    ):
        value = arguments.get(field)
        if isinstance(value, str) and not value.strip():
            arguments.pop(field, None)


def _drop_invalid_optional_browser_references(arguments: dict[str, Any], *, current_step_id: int) -> None:
    for field in ("page_version", "tab_id"):
        value = arguments.get(field)
        if not is_result_reference(value):
            continue
        from_step = value.get("from_step")
        ref_field = str(value.get("field") or "").strip()
        if not isinstance(from_step, int) or from_step <= 0 or from_step >= current_step_id or not ref_field:
            arguments.pop(field, None)


def _result_reference_dependencies(arguments: dict[str, Any], *, current_step_id: int) -> list[int]:
    dependencies: list[int] = []
    for value in arguments.values():
        if not is_result_reference(value):
            continue
        from_step = value.get("from_step")
        if isinstance(from_step, int) and 0 < from_step < current_step_id and from_step not in dependencies:
            dependencies.append(from_step)
    return dependencies


def _browser_navigation_origins(plan: DynamicPlan) -> dict[int, str]:
    origins: dict[int, str] = {}
    for index, step in enumerate(plan.steps, 1):
        if step.tool not in {"browser.open_url", "browser.open_new_tab"}:
            continue
        url = str(step.arguments.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            origins[index] = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
    return origins


def _nearest_browser_origin(origins: dict[int, str], step_index: int) -> str:
    for index in sorted(origins.keys(), reverse=True):
        if index < step_index:
            return origins[index]
    return ""


def _canonical_browser_wait_until(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in _BROWSER_WAIT_UNTIL_ALIASES:
        return _BROWSER_WAIT_UNTIL_ALIASES[normalized]
    collapsed = normalized.replace(" ", "").replace("-", "").replace("_", "")
    return _BROWSER_WAIT_UNTIL_ALIASES.get(collapsed)


def _normalize_local_path_token(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    parts: list[str] = []
    for segment in normalized.split("/"):
        token = segment.strip()
        if not token or token == ".":
            continue
        parts.append(token)
    return "/".join(parts).lower()


def _task_constraint_value(task: NaturalLanguageTask, name: str) -> str:
    for constraint in task.constraints:
        if constraint.name == name:
            return constraint.value
    return ""


def _first_explicit_browser_url(task: NaturalLanguageTask) -> str:
    for token in re.findall(r"https?://[^\s\"'>)]+", task.raw_input, re.IGNORECASE):
        candidate = str(token).strip().rstrip(".,)")
        parsed = urlparse(candidate)
        if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
            return candidate
    return ""


def _literal_vision_find_step_signature(step: DynamicPlanStep, *, expected_query: str) -> tuple[str, str]:
    path = step.arguments.get("path")
    query = step.arguments.get("query")
    if not isinstance(path, str) or not isinstance(query, str):
        return "", ""
    if any(is_result_reference(value) for value in step.arguments.values()):
        return "", ""
    canonical_query = canonicalize_vision_find_query_text(query, expected_query=expected_query)
    if canonical_query is None:
        if looks_like_json_container_text(query):
            return "", ""
        canonical_query = query.strip()
    normalized_path = _normalize_local_path_token(path)
    normalized_query = normalize_vision_query(canonical_query)
    if not normalized_path or not normalized_query:
        return "", ""
    step.arguments["query"] = canonical_query
    return normalized_path, normalized_query


def _snapshot_plan(plan: DynamicPlan) -> dict[str, Any]:
    return {
        "goal": _snapshot_debug_value(plan.goal),
        "success_criteria": _snapshot_debug_value(plan.success_criteria),
        "steps": [
            {
                "tool": _snapshot_debug_value(step.tool),
                "arguments": _snapshot_debug_value(step.arguments),
                "description": _snapshot_debug_value(step.description),
                "depends_on": list(step.depends_on),
                "expected_result": _snapshot_debug_value(step.expected_result),
            }
            for step in plan.steps[:_DEBUG_MAX_ITEMS]
        ],
    }


def _snapshot_debug_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "<truncated>"
    if isinstance(value, str):
        if len(value) <= _DEBUG_MAX_STRING_LENGTH:
            return value
        return value[:_DEBUG_MAX_STRING_LENGTH] + "...[truncated]"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_snapshot_debug_value(item, depth=depth + 1) for item in value[:_DEBUG_MAX_ITEMS]]
    if isinstance(value, DynamicPlan):
        return _snapshot_plan(value)
    if isinstance(value, dict):
        snapshot: dict[str, Any] = {}
        for key in list(value)[:_DEBUG_MAX_ITEMS]:
            snapshot[str(key)[:40]] = _snapshot_debug_value(value[key], depth=depth + 1)
        return snapshot
    return str(value)[:_DEBUG_MAX_STRING_LENGTH]


class HeuristicIntelligenceProvider:
    name = "heuristic_test"
    model = "heuristic-test-v1"

    def status(self) -> str:
        return "ready"

    def create_plan(
        self,
        task: NaturalLanguageTask,
        *,
        tool_catalog: list[ToolCatalogEntry],
        context: dict[str, Any],
        max_steps: int,
    ) -> dict[str, Any]:
        value = task.normalized_input
        lowered = value.lower()
        working_directory = default_terminal_working_directory()
        trusted_root = get_trusted_roots()[0].as_posix()

        if ("powershell" in lowered or "get-childitem" in lowered) and not (
            "create" in lowered or "file" in lowered or "containing" in lowered or "\u0441\u043e\u0437\u0434\u0430\u0439" in lowered
        ):
            return {
                "goal": task.goal,
                "success_criteria": ["forbidden command rejected"],
                "steps": [
                    {
                        "tool": "terminal.execute",
                        "arguments": {
                            "executable": "powershell",
                            "arguments": ["-Command", "Get-ChildItem"],
                            "working_directory": working_directory,
                            "timeout_seconds": 30,
                            "operation_type": "unsupported",
                            "raw_command": "powershell -Command Get-ChildItem",
                        },
                        "description": "Run a forbidden shell command.",
                    }
                ],
            }

        if "install" in lowered or "\u0443\u0441\u0442\u0430\u043d\u043e\u0432" in lowered:
            package = _extract_package_name(value) or "requests"
            return {
                "goal": task.goal,
                "success_criteria": [f"{package} is installed"],
                "steps": [
                    {
                        "tool": "terminal.execute",
                        "arguments": {
                            "executable": "python",
                            "arguments": ["-m", "pip", "install", package],
                            "working_directory": working_directory,
                            "timeout_seconds": 30,
                            "operation_type": "package_install",
                            "raw_command": f"python -m pip install {package}",
                        },
                        "description": "Install a Python package.",
                    }
                ],
            }

        if ("git" in lowered or "repository" in lowered or "\u0440\u0435\u043f\u043e\u0437\u0438\u0442\u043e\u0440" in lowered) and (
            "status" in lowered or "change" in lowered or "\u0438\u0437\u043c\u0435\u043d" in lowered or "untracked" in lowered
        ):
            return {
                "goal": task.goal,
                "success_criteria": ["git status inspected"],
                "steps": [
                    {
                        "tool": "terminal.execute",
                        "arguments": {
                            "executable": "git",
                            "arguments": ["status"],
                            "working_directory": working_directory,
                            "timeout_seconds": 30,
                            "operation_type": "git_read_only",
                            "raw_command": "git status",
                        },
                        "description": "Inspect git status.",
                    }
                ],
            }

        if "test" in lowered or "\u0442\u0435\u0441\u0442" in lowered:
            return {
                "goal": task.goal,
                "success_criteria": ["tests executed", "exit code 0 if passing"],
                "steps": [
                    {
                        "tool": "terminal.execute",
                        "arguments": {
                            "executable": "python",
                            "arguments": ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
                            "working_directory": trusted_root,
                            "timeout_seconds": 30,
                            "operation_type": "python",
                            "raw_command": 'python -m unittest discover -s tests -p "test_*.py"',
                        },
                        "description": "Run the project unit tests.",
                    }
                ],
            }

        file_path = _extract_path(value)
        if not file_path:
            raise DynamicPlanError("heuristic test mode could not create a complete plan")
        content = _extract_content(value)
        steps: list[dict[str, Any]] = [
            {
                "tool": "filesystem.create_text_file",
                "arguments": {"path": file_path},
                "description": f"Create {file_path}.",
            }
        ]
        success_criteria = [f"{file_path} exists"]
        if content:
            steps.append(
                {
                    "tool": "filesystem.write_text_file",
                    "arguments": {"path": file_path, "text": content},
                    "description": f"Write content to {file_path}.",
                }
            )
            success_criteria.append(f"{file_path} contains the requested content")
        if _should_run_python_file(value, file_path):
            steps.append(
                {
                    "tool": "terminal.execute",
                    "arguments": {
                        "executable": "python",
                        "arguments": [file_path],
                        "working_directory": trusted_root,
                        "timeout_seconds": 30,
                        "operation_type": "python",
                        "raw_command": f"python {file_path}",
                    },
                    "description": f"Run {file_path}.",
                }
            )
            success_criteria.append(f"{file_path} ran successfully")
        return {
            "goal": task.goal,
            "success_criteria": success_criteria,
            "steps": steps[:max_steps],
        }


def _extract_package_name(value: str) -> str | None:
    match = re.search(r"(?:package|\u043f\u0430\u043a\u0435\u0442)\s+([A-Za-z0-9_.-]+)", value, re.IGNORECASE)
    if match:
        return match.group(1)
    pip_match = re.search(r"install\s+([A-Za-z0-9_.-]+)$", value, re.IGNORECASE)
    if pip_match:
        return pip_match.group(1)
    return None


def _extract_path(value: str) -> str | None:
    match = re.search(r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:py|txt|md|json|log))", value)
    if match:
        return match.group(1)
    quoted = re.search(r"(?:named|file)\s+([A-Za-z0-9_.-]+\.(?:py|txt|md|json|log))", value, re.IGNORECASE)
    if quoted:
        return quoted.group(1)
    return None


def _extract_content(value: str) -> str:
    quoted = re.search(r'"([^"]+)"', value)
    if quoted:
        return quoted.group(1)
    if "hello" in value.lower() and ".py" in value.lower():
        return "print('Hello')"
    containing = re.search(r"(?:containing|with|\u0441\u043e\u0434\u0435\u0440\u0436\u0430\u0449\u0438\u0439|\u0441 \u0442\u0435\u043a\u0441\u0442\u043e\u043c)\s+(.+)$", value, re.IGNORECASE)
    if containing:
        text = containing.group(1).strip().rstrip(".")
        return text[:200]
    return ""


def _should_run_python_file(value: str, file_path: str) -> bool:
    lowered = value.lower()
    return file_path.endswith(".py") and any(
        token in lowered
        for token in (
            "run it",
            "\u0437\u0430\u043f\u0443\u0441\u0442\u0438",
            "run the file",
            "execute it",
            "\u0437\u0430\u0442\u0435\u043c \u0437\u0430\u043f\u0443\u0441\u0442\u0438",
        )
    )
