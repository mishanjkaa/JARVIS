import unittest
from unittest.mock import patch

from app.brain.ai.models import AIIntent, AIResponse, ProviderStatus, ProviderStatusCategory
from app.brain.configuration.runtime_config import reset_runtime_config, set_runtime_config_value
from app.brain.context.conversation import clear_conversation, get_conversation_summaries
from app.brain.router import _CONVERSATION_RUNTIME, route_command


class _FakeProvider:
    def __init__(self, response: AIResponse | None = None, side_effect: Exception | None = None) -> None:
        self.response = response
        self.side_effect = side_effect
        self.prompts: list[str] = []
        self.model = "stub-model"
        self.timeout = 1.0
        self.base_url = "http://localhost"

    def generate_text(self, prompt: str) -> AIResponse:
        self.prompts.append(prompt)
        if self.side_effect is not None:
            raise self.side_effect
        return self.response or AIResponse(
            category="conversation",
            message="stub",
            intent=AIIntent.CONVERSATION,
            status=ProviderStatus(category=ProviderStatusCategory.READY, provider="ollama"),
        )


class ConversationRuntimeEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_config()
        clear_conversation()

    def tearDown(self) -> None:
        clear_conversation()
        reset_runtime_config()

    def test_normal_question_routes_to_mocked_ollama(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        provider = _FakeProvider(
            AIResponse(
                category="conversation",
                message="Paris is the capital of France.",
                intent=AIIntent.CONVERSATION,
                status=ProviderStatus(category=ProviderStatusCategory.READY, provider="ollama"),
            )
        )
        with patch.object(_CONVERSATION_RUNTIME, "provider", provider):
            self.assertEqual(route_command("What is the capital of France?"), "Paris is the capital of France.")
        self.assertEqual(len(provider.prompts), 1)

    def test_prompt_instructs_the_model_to_reply_in_the_user_s_language(self) -> None:
        # Real-hardware bug: with a stronger local model swapped in (qwen2.5:7b-instruct),
        # a reply to Russian input came back with stray Chinese characters mixed into an
        # otherwise-Russian sentence. The system prompt sent to the model was entirely in
        # English with no instruction about matching the user's language or avoiding
        # code-switching, which is exactly the kind of prompt shape that invites this on a
        # multilingual model. This isn't a guarantee (a local model can still misbehave),
        # but the instruction needs to actually be there.
        set_runtime_config_value("ai_enabled", True)
        provider = _FakeProvider()
        with patch.object(_CONVERSATION_RUNTIME, "provider", provider):
            route_command("Привет, как дела?")
        prompt = provider.prompts[-1]
        self.assertIn("same natural language", prompt)
        self.assertIn("never mix languages or scripts", prompt)

    def test_deterministic_command_does_not_call_ollama(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        provider = _FakeProvider()
        with patch.object(_CONVERSATION_RUNTIME, "provider", provider):
            self.assertIn("Current time is", route_command("time"))
        self.assertEqual(provider.prompts, [])

    def test_tool_request_bypasses_conversation(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        provider = _FakeProvider()
        with patch.object(_CONVERSATION_RUNTIME, "provider", provider):
            response = route_command("work out 25 times 4")
        self.assertEqual(response, "Result: 100")
        self.assertEqual(provider.prompts, [])

    def test_provider_unavailable_returns_safe_response(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        provider = _FakeProvider(side_effect=OSError("offline"))
        with patch.object(_CONVERSATION_RUNTIME, "provider", provider):
            self.assertEqual(route_command("Tell me something interesting."), "Conversation is unavailable right now.")

    def test_ai_disabled_returns_safe_response(self) -> None:
        # ai_enabled now defaults to True (owner's explicit request: JARVIS should be
        # usable immediately at startup) -- this test is specifically about the
        # disabled-state fallback message, so it turns AI off itself.
        set_runtime_config_value("ai_enabled", False)
        self.assertEqual(route_command("Tell me something interesting."), "AI is disabled right now.")

    def test_conversation_disabled_returns_safe_response(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        set_runtime_config_value("ai_allow_conversation", False)
        self.assertEqual(route_command("Tell me something interesting."), "Conversation mode is disabled.")

    def test_timeout_returns_safe_response(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        provider = _FakeProvider(side_effect=TimeoutError())
        with patch.object(_CONVERSATION_RUNTIME, "provider", provider):
            self.assertEqual(route_command("Tell me something interesting."), "Conversation timed out. Please try again.")

    def test_bounded_history_limits_prompt_context(self) -> None:
        set_runtime_config_value("ai_enabled", True)
        provider = _FakeProvider(
            AIResponse(
                category="conversation",
                message="short answer",
                intent=AIIntent.CONVERSATION,
                status=ProviderStatus(category=ProviderStatusCategory.READY, provider="ollama"),
            )
        )
        with patch.object(_CONVERSATION_RUNTIME, "provider", provider):
            for index in range(4):
                set_runtime_config_value("ai_enabled", True)
                route_command(f"Question number {index}?")
            set_runtime_config_value("ai_enabled", True)
            set_runtime_config_value("ai_allow_conversation", True)
            with patch("app.brain.runtime.conversation_runtime.get_runtime_config", return_value={
                "assistant_name": "JARVIS",
                "language": "en",
                "voice_enabled": False,
                "ai_enabled": True,
                "ollama_model": "llama3.2",
                "ai_timeout_seconds": 20,
                "ai_max_plan_steps": 5,
                "ai_allow_conversation": True,
                "show_plan_preview": True,
            }), patch("app.brain.runtime.conversation_runtime.load_config", return_value={
                "ai_enabled": True,
                "ai_allow_conversation": True,
                "ollama_model": "llama3.2",
                "ai_timeout_seconds": 20,
                "ai_max_response_chars": 12000,
                "max_conversation_entries": 4,
                "ollama_base_url": "http://127.0.0.1:11434",
            }):
                route_command("Newest question?")
        prompt = provider.prompts[-1]
        self.assertIn("Question number 3?", prompt)
        self.assertNotIn("Question number 0?", prompt)
        self.assertLessEqual(len(get_conversation_summaries()), 4)
