from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.brain.intent.models import ConfidenceCategory
from app.brain.planner.plan_models import AgentPlan
from app.brain.risk.models import RiskLevel


class TaskIntent(str, Enum):
    CONVERSATION = "conversation"
    DIRECT_COMMAND = "direct_command"
    ACTIONABLE_TASK = "actionable_task"
    CLARIFICATION_RESPONSE = "clarification_response"
    UNSUPPORTED_TASK = "unsupported_task"
    AMBIGUOUS_TASK = "ambiguous_task"


class GoalEvaluationStatus(str, Enum):
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLARIFICATION_REQUIRED = "clarification_required"


@dataclass
class TaskConstraint:
    name: str
    value: str


@dataclass
class NaturalLanguageTask:
    raw_input: str
    normalized_input: str
    intent: TaskIntent
    confidence: ConfidenceCategory
    goal: str
    expected_result: str = ""
    referenced_paths: list[str] = field(default_factory=list)
    execution_requested: bool = True
    ambiguity_level: str = "low"
    language: str = "en"
    constraints: list[TaskConstraint] = field(default_factory=list)
    requested_artifacts: list[str] = field(default_factory=list)
    requested_contents: list[str] = field(default_factory=list)
    requested_output_texts: list[str] = field(default_factory=list)
    requested_summary: bool = False
    read_only_task: bool = False
    destructive_scope_unclear: bool = False
    requires_execution: bool = False
    requires_verification: bool = False
    requires_stdout_match: bool = False
    requires_tests: bool = False
    requires_code_write: bool = False
    requested_operation: str = ""


@dataclass
class ToolCatalogEntry:
    name: str
    description: str
    argument_schema: dict[str, Any]
    required_arguments: list[str]
    optional_arguments: list[str]
    risk_hint: str
    enabled: bool
    restrictions: list[str] = field(default_factory=list)


@dataclass
class DynamicPlanStep:
    tool: str
    arguments: dict[str, Any]
    description: str = ""
    depends_on: list[int] = field(default_factory=list)
    expected_result: str = ""


@dataclass
class DynamicPlan:
    goal: str
    success_criteria: list[str]
    steps: list[DynamicPlanStep]
    original_request: str


@dataclass
class PlanValidationResult:
    valid: bool
    reason: str = ""
    normalized_plan: AgentPlan | None = None
    semantic_incomplete: bool = False
    policy_rejected: bool = False


@dataclass
class GoalEvaluation:
    status: GoalEvaluationStatus
    summary: str
    evidence: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestPlanRecord:
    request_id: int
    raw_input: str
    status: str = "received"
    task: NaturalLanguageTask | None = None
    dynamic_plan: DynamicPlan | None = None
    agent_plan: AgentPlan | None = None
    risk_summary: str = ""
    evaluation: GoalEvaluation | None = None
    clarification_question: str = ""
    message: str = ""
    agent_task_id: int | None = None
    planning_trace: dict[str, Any] = field(default_factory=dict)
    validation_reason: str = ""


@dataclass
class ClarificationRequest:
    question: str
    reason: str
    original_goal: str


@dataclass
class IntelligenceResult:
    message: str
    task: NaturalLanguageTask | None = None
    dynamic_plan: DynamicPlan | None = None
    plan: AgentPlan | None = None
    risk_level: RiskLevel | None = None
    evaluation: GoalEvaluation | None = None
