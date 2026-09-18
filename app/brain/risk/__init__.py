from app.brain.risk.models import PlanRiskAssessment, RiskLevel


def analyze_plan(*args, **kwargs):
    from app.brain.risk.analyzer import analyze_plan as _analyze_plan

    return _analyze_plan(*args, **kwargs)


__all__ = ["analyze_plan", "PlanRiskAssessment", "RiskLevel"]
