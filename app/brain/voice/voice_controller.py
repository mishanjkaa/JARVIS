from app.brain.configuration.runtime_config import get_effective_runtime_config, set_runtime_config_value
from app.brain.voice.controller import get_voice_controller


class VoiceController:
    def __init__(self) -> None:
        self.enabled = False


voice_controller = VoiceController()


def get_voice_state() -> bool:
    """Kept for app.brain.skills.self_check's existing import. Reflects the persisted
    voice_enabled setting (previously this only reflected an in-memory flag that voice_on/
    voice_off never actually kept in sync with runtime config, unlike ai_enabled)."""
    return bool(get_effective_runtime_config().get("voice_enabled", False))


def voice_status() -> str:
    return get_voice_controller().status_message()


def voice_on() -> str:
    set_runtime_config_value("voice_enabled", True)
    voice_controller.enabled = True
    return "Voice enabled for this session."


def voice_off() -> str:
    set_runtime_config_value("voice_enabled", False)
    voice_controller.enabled = False
    return "Voice disabled for this session."


def voice_enroll() -> str:
    return get_voice_controller().enroll_sample()


def voice_cancel_enrollment() -> str:
    return get_voice_controller().cancel_enrollment()


def voice_forget_me() -> str:
    return get_voice_controller().forget_me()
