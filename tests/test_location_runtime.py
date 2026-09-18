from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.brain.agent.controller import AgentController
from app.brain.agent.models import AgentLifecycleState
from app.brain.agent.state import get_agent_runtime_state, reset_agent_runtime_state
from app.brain.audit.audit_log import reset_audit_log
from app.brain.configuration.runtime_config import get_effective_runtime_config, replace_runtime_config, reset_runtime_config, set_runtime_config_value
from app.brain.location import places
from app.brain.location.controller import get_location_controller, reset_location_controller
from app.brain.location.errors import LocationPlaceNotFoundError, LocationSessionNotFoundError, LocationUnavailableError
from app.brain.location.geocoding import Geocoder
from app.brain.location.models import NavigationSession
from app.brain.location.routing import RouteResult, RouteStep, RoutingProvider
from app.brain.location.server import handle_overland_request, is_bind_host_present, validate_bind_host
from app.brain.location.state import get_location_state, reset_location_state
from app.brain.planner.approval import store_pending_plan
from app.brain.planner.plan_models import AgentPlan, AgentStep
from app.brain.planner.state import reset_planner_state, reset_planner_now_provider, set_planner_now_provider
from app.brain.risk.analyzer import analyze_plan
from app.brain.risk.models import RiskLevel
from app.brain.router import route_command
from app.brain.tools.executor import ToolExecutor
from app.brain.tools.models import ToolDefinition, ToolResult
from app.brain.tools.registry import ToolRegistry
from tests.test_intelligence_runtime import _FakePlannerClock


def _tool(name: str, handler, *, risk_level: str = "read_only") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=name,
        argument_schema={},
        risk_level=risk_level,
        requires_confirmation=False,
        handler=handler,
        formatter=lambda result: result.display_value or result.message,
    )


def _fake_route(*, distance_meters: float = 1200.0, duration_seconds: float = 300.0) -> RouteResult:
    return RouteResult(
        distance_meters=distance_meters,
        duration_seconds=duration_seconds,
        steps=[RouteStep(instruction="Turn right onto Main Street.", distance_meters=distance_meters)],
    )


class _FakeGeocoder(Geocoder):
    def __init__(self, *, display_name: str = "Riga Old Town", search_result: tuple[float, float] = (56.95, 24.10)) -> None:
        self.display_name = display_name
        self.search_result = search_result

    def reverse(self, latitude: float, longitude: float) -> str:
        return self.display_name

    def search(self, query: str) -> tuple[float, float]:
        return self.search_result


class _FakeRoutingProvider(RoutingProvider):
    def __init__(self, *, route_result: RouteResult | None = None) -> None:
        self.route_result = route_result or _fake_route()
        self.calls: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def route(self, origin: tuple[float, float], destination: tuple[float, float]) -> RouteResult:
        self.calls.append((origin, destination))
        return self.route_result


def _overland_body(*, latitude: float, longitude: float, device_id: str = "iphone-1", accuracy: float = 8.0, timestamp: str | None = None) -> bytes:
    payload = {
        "locations": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                "properties": {
                    "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                    "horizontal_accuracy": accuracy,
                    "device_id": device_id,
                },
            }
        ]
    }
    return json.dumps(payload).encode("utf-8")


class LocationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path.cwd() / ".tmp-tests"
        self.temp_root.mkdir(exist_ok=True)
        self.places_file = (self.temp_root / f"places-{uuid4().hex}.json").absolute()
        self.places_patcher = patch("app.brain.location.places.get_places_file_path", return_value=self.places_file)
        self.places_patcher.start()

        reset_runtime_config()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_planner_state()
        reset_location_state()
        set_runtime_config_value("location_enabled", True)
        # location_shared_secret is deliberately excluded from set_runtime_config_value's
        # allowlist (it's a credential, set only by editing config.json) -- tests install it
        # directly on the effective-config snapshot instead, simulating that manual edit.
        config = get_effective_runtime_config()
        config["location_shared_secret"] = "test-secret"
        replace_runtime_config(config)
        self.fake_geocoder = _FakeGeocoder()
        self.fake_routing_provider = _FakeRoutingProvider()
        reset_location_controller(geocoder=self.fake_geocoder, routing_provider=self.fake_routing_provider)

    def tearDown(self) -> None:
        self.places_patcher.stop()
        reset_runtime_config()
        reset_agent_runtime_state()
        reset_audit_log()
        reset_planner_state()
        reset_location_state()
        reset_location_controller()
        if self.places_file.exists():
            self.places_file.unlink()

    # --- /location/overland ingestion --------------------------------------

    def test_unauthenticated_post_is_rejected_and_state_is_untouched(self) -> None:
        status, body = handle_overland_request(headers={}, raw_body=_overland_body(latitude=56.95, longitude=24.10))
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["result"], "unauthorized")
        self.assertEqual(get_location_state().points, {})

    def test_wrong_secret_is_rejected_and_state_is_untouched(self) -> None:
        status, _ = handle_overland_request(
            headers={"Authorization": "Bearer wrong-secret"},
            raw_body=_overland_body(latitude=56.95, longitude=24.10),
        )
        self.assertEqual(status, 401)
        self.assertEqual(get_location_state().points, {})

    def test_authenticated_post_stores_the_point(self) -> None:
        status, body = handle_overland_request(
            headers={"Authorization": "Bearer test-secret"},
            raw_body=_overland_body(latitude=56.95, longitude=24.10, device_id="iphone-1"),
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["result"], "ok")
        point = get_location_state().points["iphone-1"]
        self.assertAlmostEqual(point.latitude, 56.95)
        self.assertAlmostEqual(point.longitude, 24.10)

    def test_second_post_replaces_the_freshest_point_instead_of_accumulating(self) -> None:
        handle_overland_request(
            headers={"Authorization": "Bearer test-secret"},
            raw_body=_overland_body(latitude=56.95, longitude=24.10, device_id="iphone-1"),
        )
        handle_overland_request(
            headers={"Authorization": "Bearer test-secret"},
            raw_body=_overland_body(latitude=57.00, longitude=24.20, device_id="iphone-1"),
        )
        points = get_location_state().points
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points["iphone-1"].latitude, 57.00)
        self.assertAlmostEqual(points["iphone-1"].longitude, 24.20)

    def test_malformed_body_is_rejected_safely(self) -> None:
        status, _ = handle_overland_request(headers={"Authorization": "Bearer test-secret"}, raw_body=b"not json")
        self.assertEqual(status, 400)
        self.assertEqual(get_location_state().points, {})

    # --- bind-host validation -------------------------------------------------

    def test_wildcard_bind_host_is_rejected(self) -> None:
        self.assertFalse(is_bind_host_present("0.0.0.0"))
        with self.assertRaises(Exception):
            validate_bind_host("0.0.0.0")

    def test_empty_bind_host_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            validate_bind_host("")

    def test_loopback_bind_host_is_present_on_this_machine(self) -> None:
        self.assertTrue(is_bind_host_present("127.0.0.1"))
        validate_bind_host("127.0.0.1")  # does not raise

    # --- location.*/navigation.* behavior --------------------------------------

    def test_where_am_i_reports_freshest_point(self) -> None:
        get_location_controller().ingest_overland_point(
            device_id="iphone-1", latitude=56.95, longitude=24.10, accuracy_meters=8.0,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        message = get_location_controller().where_am_i()
        self.assertIn("Riga Old Town", message)
        self.assertNotIn("old.", message)

    def test_where_am_i_flags_stale_point(self) -> None:
        stale_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        get_location_controller().ingest_overland_point(
            device_id="iphone-1", latitude=56.95, longitude=24.10, accuracy_meters=8.0,
            captured_at=stale_time.isoformat(),
        )
        message = get_location_controller().where_am_i()
        self.assertIn("old", message)

    def test_where_am_i_without_any_point_raises(self) -> None:
        with self.assertRaises(LocationUnavailableError):
            get_location_controller().where_am_i()

    def test_distance_to_unknown_place_raises(self) -> None:
        get_location_controller().ingest_overland_point(
            device_id="iphone-1", latitude=56.95, longitude=24.10, accuracy_meters=8.0,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        with self.assertRaises(LocationPlaceNotFoundError):
            get_location_controller().distance_to("home")

    def test_save_place_then_distance_to_computes_haversine_distance(self) -> None:
        get_location_controller().save_place("home", 56.9496, 24.1052)
        get_location_controller().ingest_overland_point(
            device_id="iphone-1", latitude=56.9496, longitude=24.1052, accuracy_meters=8.0,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        message = get_location_controller().distance_to("home")
        self.assertIn("home", message)
        self.assertIn("0 meters", message)

    def test_save_place_normalizes_name_case(self) -> None:
        get_location_controller().save_place("Home", 56.9496, 24.1052)
        self.assertIsNotNone(places.get_place("HOME"))

    def test_navigation_start_then_get_next_instruction_then_stop(self) -> None:
        get_location_controller().ingest_overland_point(
            device_id="iphone-1", latitude=56.9496, longitude=24.1052, accuracy_meters=8.0,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        start_message = get_location_controller().start_navigation("56.95,24.10")
        self.assertIn("Navigating to", start_message)
        session_id = list(get_location_state().sessions.keys())[0]

        next_message = get_location_controller().get_next_instruction(session_id)
        self.assertIn("Turn right onto Main Street", next_message)

        stop_message = get_location_controller().stop_navigation(session_id)
        self.assertIn("ended", stop_message)
        self.assertNotIn(session_id, get_location_state().sessions)

    def test_navigation_get_next_instruction_reports_arrival(self) -> None:
        get_location_controller().ingest_overland_point(
            device_id="iphone-1", latitude=56.9496, longitude=24.1052, accuracy_meters=8.0,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )
        self.fake_routing_provider.route_result = _fake_route(distance_meters=5.0, duration_seconds=5.0)
        get_location_controller().start_navigation("56.95,24.10")
        session_id = list(get_location_state().sessions.keys())[0]
        message = get_location_controller().get_next_instruction(session_id)
        self.assertIn("arrived", message)

    def test_stop_unknown_session_raises(self) -> None:
        with self.assertRaises(LocationSessionNotFoundError):
            get_location_controller().stop_navigation("nav-does-not-exist")

    # --- risk classification -------------------------------------------------

    def test_read_only_location_and_navigation_tools_are_low_risk_and_auto_execute(self) -> None:
        for tool_name, arguments in [
            ("location.where_am_i", {}),
            ("location.distance_to", {"place": "home"}),
            ("navigation.start", {"destination": "home"}),
            ("navigation.get_next_instruction", {"session_id": "nav-1"}),
            ("navigation.stop", {"session_id": "nav-1"}),
        ]:
            plan = AgentPlan(steps=[AgentStep(step_id=1, tool_name=tool_name, arguments=arguments, risk_level="read_only", user_visible_description=tool_name)])
            assessment = analyze_plan(plan)
            self.assertEqual(assessment.level, RiskLevel.LOW, tool_name)
            self.assertTrue(assessment.auto_execute, tool_name)

    def test_save_place_is_medium_persistent_write_risk_and_never_auto_executes(self) -> None:
        plan = AgentPlan(steps=[AgentStep(step_id=1, tool_name="location.save_place", arguments={"name": "home", "latitude": 1.0, "longitude": 2.0}, risk_level="persistent_write", user_visible_description="Save home.")])
        set_runtime_config_value("developer_mode", True)
        assessment = analyze_plan(plan)
        self.assertEqual(assessment.level, RiskLevel.MEDIUM)
        self.assertFalse(assessment.auto_execute)
        self.assertTrue(assessment.requires_approval)

    # --- navigation-session cleanup: all four termination paths ---------------

    def _plan_with_fake_tool(self, name: str, handler) -> tuple[AgentController, AgentPlan]:
        controller = AgentController(ToolExecutor(ToolRegistry([_tool(name, handler)])))
        plan = AgentPlan(steps=[AgentStep(1, name, {})], original_request="do the thing")
        return controller, plan

    def _seed_session_for_task(self, task_id: int) -> str:
        session = NavigationSession(
            session_id="nav-cleanup-test",
            owner_request_id=None,
            owner_agent_task_id=task_id,
            destination_name="home",
            destination_latitude=1.0,
            destination_longitude=2.0,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        state = get_location_state()
        with state.lock:
            state.sessions[session.session_id] = session
        return session.session_id

    def test_session_is_cleaned_up_on_task_completion(self) -> None:
        controller, plan = self._plan_with_fake_tool("system.get_time", lambda args: ToolResult(True, "success", "ok", "ok"))
        task = controller.create_pending_task(plan)
        session_id = self._seed_session_for_task(task.task_id)

        controller.approve_and_execute_current_task()

        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.COMPLETED)
        self.assertNotIn(session_id, get_location_state().sessions)

    def test_session_is_cleaned_up_on_step_failure(self) -> None:
        controller, plan = self._plan_with_fake_tool("step.fails", lambda args: ToolResult(False, "failed", "boom"))
        task = controller.create_pending_task(plan)
        session_id = self._seed_session_for_task(task.task_id)

        controller.approve_and_execute_current_task()

        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.FAILED)
        self.assertNotIn(session_id, get_location_state().sessions)

    def test_session_is_cleaned_up_on_cancellation(self) -> None:
        controller, plan = self._plan_with_fake_tool("system.get_time", lambda args: ToolResult(True, "success", "ok", "ok"))
        task = controller.create_pending_task(plan)
        session_id = self._seed_session_for_task(task.task_id)

        controller.cancel_pending_or_running_task()

        self.assertEqual(get_agent_runtime_state().current_task.state, AgentLifecycleState.CANCELLED)
        self.assertNotIn(session_id, get_location_state().sessions)

    def test_session_is_cleaned_up_on_approval_expiration(self) -> None:
        # This is the path that bit RFC-007B and RFC-007C's own capture cleanup twice before:
        # a pending plan simply timing out is a separate code path
        # (app.brain.planner.approval._synchronize_terminal_pending_state_locked) from
        # AgentController.cancel_pending_or_running_task, covered by the three tests above.
        clock = _FakePlannerClock()
        set_planner_now_provider(clock.now)
        try:
            plan = AgentPlan(steps=[
                AgentStep(
                    step_id=1,
                    tool_name="location.save_place",
                    arguments={"name": "home", "latitude": 1.0, "longitude": 2.0},
                    risk_level="persistent_write",
                    user_visible_description="Save home.",
                ),
            ])
            summary = store_pending_plan(plan)
            self.assertIn("Pending plan:", summary)
            task = get_agent_runtime_state().current_task
            self.assertIsNotNone(task)
            self.assertEqual(task.state.value, "pending_approval")
            session_id = self._seed_session_for_task(task.task_id)

            clock.advance(seconds=61)
            expired = route_command("approve plan")
            self.assertEqual(expired, "The pending plan expired.")

            self.assertNotIn(session_id, get_location_state().sessions)
        finally:
            reset_planner_now_provider()
            reset_planner_state()
