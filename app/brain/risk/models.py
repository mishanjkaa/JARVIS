from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PolicyOutcome(str, Enum):
    ALLOWED = "allowed"
    APPROVAL_REQUIRED = "approval_required"
    REJECTED = "rejected"


@dataclass
class PlanRiskAssessment:
    level: RiskLevel
    auto_execute: bool
    requires_approval: bool
    project_scoped: bool
    policy_outcome: PolicyOutcome = PolicyOutcome.ALLOWED
    reasons: list[str] = field(default_factory=list)
