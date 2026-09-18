from app.brain.voice.speech_to_text import transcribe_speech
from app.brain.voice.text_to_speech import speak_text


class VoiceController:
    def __init__(self) -> None:
        self.enabled = False


voice_controller = VoiceController()


def get_voice_state() -> bool:
    return voice_controller.enabled


def voice_status() -> str:
    return "Voice is enabled." if voice_controller.enabled else "Voice is disabled."


def voice_on() -> str:
    voice_controller.enabled = True
    return "Voice enabled for this session."


def voice_off() -> str:
    voice_controller.enabled = False
    return "Voice disabled for this session."


def handle_voice_output(text: str) -> str:
    return speak_text(text)
