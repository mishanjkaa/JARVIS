__all__ = ["get_intelligence_controller", "reset_intelligence_controller"]


def get_intelligence_controller():
    from app.brain.intelligence.controller import get_intelligence_controller as _get

    return _get()


def reset_intelligence_controller():
    from app.brain.intelligence.controller import reset_intelligence_controller as _reset

    return _reset()
