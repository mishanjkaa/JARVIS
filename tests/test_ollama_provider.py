import io
import json
import socket
import unittest
import urllib.error
from unittest.mock import patch

from app.brain.ai.json_parser import extract_json_object
from app.brain.ai.models import ProviderStatusCategory
from app.brain.ai.ollama_provider import OllamaProvider
from app.brain.intelligence.errors import IntelligenceProviderRequestError
from app.brain.intelligence.models import ToolCatalogEntry
from app.brain.intelligence.structured_output import TASK_INTERPRETATION_SCHEMA, build_plan_output_schema


def _response_bytes(content: str) -> bytes:
    return json.dumps({"response": content}).encode("utf-8")


def _http_error(status: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://127.0.0.1:11434/api/generate",
        code=status,
        msg="error",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _tool_catalog() -> list[ToolCatalogEntry]:
    return [
        ToolCatalogEntry(
            name="browser.start_session",
            description="Start an isolated browser session.",
            argument_schema={"headless": {"type": "bool", "required": False}},
            required_arguments=[],
            optional_arguments=["headless"],
            risk_hint="local_safe",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="browser.open_url",
            description="Open a policy-approved URL in a browser session.",
            argument_schema={
                "session_id": {"type": "text", "max_length": 80},
                "url": {"type": "text", "max_length": 400},
                "wait_until": {"type": "text", "max_length": 40},
                "timeout_seconds": {"type": "integer", "required": False},
            },
            required_arguments=["session_id", "url", "wait_until"],
            optional_arguments=["timeout_seconds"],
            risk_hint="local_safe",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="browser.get_page_info",
            description="Read the current page URL and title.",
            argument_schema={"session_id": {"type": "text", "max_length": 80}},
            required_arguments=["session_id"],
            optional_arguments=[],
            risk_hint="read_only",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="browser.close_session",
            description="Close an isolated browser session.",
            argument_schema={"session_id": {"type": "text", "max_length": 80}},
            required_arguments=["session_id"],
            optional_arguments=[],
            risk_hint="local_safe",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="browser.capture_view",
            description="Capture the current browser viewport into an opaque temporary vision capture.",
            argument_schema={
                "session_id": {"type": "text", "max_length": 80},
                "tab_id": {"type": "text", "max_length": 80, "required": False},
            },
            required_arguments=["session_id"],
            optional_arguments=["tab_id"],
            risk_hint="read_only",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="browser.scroll_page",
            description="Scroll vertically within the current page.",
            argument_schema={
                "session_id": {"type": "text", "max_length": 80},
                "direction": {"type": "text", "max_length": 20, "required": False},
                "amount": {"type": "integer", "required": False},
            },
            required_arguments=["session_id"],
            optional_arguments=["direction", "amount"],
            risk_hint="local_safe",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="browser.take_screenshot",
            description="Save a screenshot inside trusted roots.",
            argument_schema={
                "session_id": {"type": "text", "max_length": 80},
                "path": {"type": "path"},
                "full_page": {"type": "bool", "required": False},
            },
            required_arguments=["session_id", "path"],
            optional_arguments=["full_page"],
            risk_hint="local_safe",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="filesystem.create_text_file",
            description="Create a UTF-8 text file.",
            argument_schema={"path": {"type": "path"}},
            required_arguments=["path"],
            optional_arguments=[],
            risk_hint="persistent_write",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="vision.describe_browser_capture",
            description="Describe a temporary browser viewport capture.",
            argument_schema={
                "capture_id": {"type": "text", "max_length": 80},
                "detail_level": {"type": "text", "max_length": 20, "required": False},
            },
            required_arguments=["capture_id"],
            optional_arguments=["detail_level"],
            risk_hint="read_only",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="vision.extract_text_from_browser_capture",
            description="Extract visible text from a temporary browser viewport capture.",
            argument_schema={
                "capture_id": {"type": "text", "max_length": 80},
                "language_hint": {"type": "text", "max_length": 20, "required": False},
                "max_characters": {"type": "integer", "required": False},
            },
            required_arguments=["capture_id"],
            optional_arguments=["language_hint", "max_characters"],
            risk_hint="read_only",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="vision.find_visual_element_in_browser_capture",
            description="Find a visually described element inside a temporary browser viewport capture.",
            argument_schema={
                "capture_id": {"type": "text", "max_length": 80},
                "query": {"type": "text", "max_length": 200},
                "max_results": {"type": "integer", "required": False},
            },
            required_arguments=["capture_id", "query"],
            optional_arguments=["max_results"],
            risk_hint="read_only",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="terminal.execute",
            description="Execute a validated terminal command.",
            argument_schema={
                "executable": {"type": "text", "max_length": 120},
                "arguments": {"type": "string_list", "required": False, "max_items": 20, "item_max_length": 200},
                "working_directory": {"type": "path", "required": False},
                "timeout_seconds": {"type": "integer", "required": False},
                "operation_type": {"type": "text", "max_length": 40},
                "raw_command": {"type": "text", "max_length": 200, "required": False},
            },
            required_arguments=["executable", "operation_type"],
            optional_arguments=["arguments", "working_directory", "timeout_seconds", "raw_command"],
            risk_hint="local_safe",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="vision.describe_image",
            description="Describe a trusted local image.",
            argument_schema={
                "path": {"type": "path"},
                "detail_level": {"type": "text", "required": False, "max_length": 20},
            },
            required_arguments=["path"],
            optional_arguments=["detail_level"],
            risk_hint="read_only",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="vision.extract_text",
            description="Extract visible text from a trusted local image.",
            argument_schema={
                "path": {"type": "path"},
                "language_hint": {"type": "text", "required": False, "max_length": 20},
                "max_characters": {"type": "integer", "required": False},
            },
            required_arguments=["path"],
            optional_arguments=["language_hint", "max_characters"],
            risk_hint="read_only",
            enabled=True,
        ),
        ToolCatalogEntry(
            name="vision.find_visual_element",
            description="Find a described visual element in a trusted local image.",
            argument_schema={
                "path": {"type": "path"},
                "query": {"type": "query", "max_length": 200},
                "max_results": {"type": "integer", "required": False},
            },
            required_arguments=["path", "query"],
            optional_arguments=["max_results"],
            risk_hint="read_only",
            enabled=True,
        ),
    ]


class JsonParserTests(unittest.TestCase):
    def test_extract_json_object_accepts_direct_object(self) -> None:
        self.assertEqual(extract_json_object('{"goal":"demo","steps":[],"success_criteria":["ok"]}')["goal"], "demo")

    def test_extract_json_object_accepts_markdown_json_fence(self) -> None:
        payload = extract_json_object('```json\n{"goal":"demo","steps":[],"success_criteria":["ok"]}\n```')
        self.assertEqual(payload["goal"], "demo")

    def test_extract_json_object_accepts_surrounding_whitespace(self) -> None:
        payload = extract_json_object('  \n {"goal":"demo","steps":[],"success_criteria":["ok"]}\n ')
        self.assertEqual(payload["goal"], "demo")

    def test_extract_json_object_rejects_invalid_json_syntax(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured output JSON decode failed"):
            extract_json_object('{"goal": "demo",}')

    def test_extract_json_object_rejects_multiple_json_objects(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured output contained multiple JSON values"):
            extract_json_object('{"goal":"demo"} {"goal":"second"}')

    def test_extract_json_object_rejects_text_before_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured output contained text before JSON"):
            extract_json_object('Here is the plan: {"goal":"demo"}')

    def test_extract_json_object_rejects_text_after_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured output contained text after JSON"):
            extract_json_object('{"goal":"demo"} done')

    def test_extract_json_object_rejects_truncated_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "structured output was truncated"):
            extract_json_object('{"goal":"demo"')


class OllamaProviderTests(unittest.TestCase):
    def test_plan_schema_uses_exact_tool_specific_argument_contracts(self) -> None:
        schema = build_plan_output_schema(_tool_catalog(), 5)
        step_variants = schema["properties"]["steps"]["items"]["oneOf"]
        tool_variants = {variant["properties"]["tool"]["const"]: variant for variant in step_variants}
        open_url_args = tool_variants["browser.open_url"]["properties"]["arguments"]["properties"]
        self.assertEqual(set(open_url_args), {"session_id", "url", "wait_until", "timeout_seconds"})
        self.assertIn("oneOf", open_url_args["session_id"])
        scroll_args = tool_variants["browser.scroll_page"]["properties"]["arguments"]["properties"]
        self.assertEqual(set(scroll_args), {"session_id", "direction", "amount"})
        screenshot_args = tool_variants["browser.take_screenshot"]["properties"]["arguments"]["properties"]
        self.assertEqual(set(screenshot_args), {"session_id", "path", "full_page"})
        vision_find_args = tool_variants["vision.find_visual_element"]["properties"]["arguments"]["properties"]
        self.assertEqual(set(vision_find_args), {"path", "query", "max_results"})
        self.assertNotIn("oneOf", vision_find_args["path"])
        self.assertNotIn("oneOf", vision_find_args["query"])

    def test_status_is_safe(self) -> None:
        provider = OllamaProvider()
        self.assertEqual(provider.status().provider, "ollama")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_generate_structured_handles_json(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes('{"category":"conversation","message":"hi"}')
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        response = provider.generate_structured("prompt")
        self.assertEqual(response.category, "conversation")
        self.assertEqual(response.message, "hi")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_accepts_direct_object_payload(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = b'{"goal": "demo", "success_criteria": ["ok"], "steps": []}'
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        payload = provider.request_json("prompt")
        self.assertEqual(payload["goal"], "demo")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_uses_schema_format_when_provided(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes('{"intent":"clarification_response","confidence":"high","goal":"clarify","expected_result":"","referenced_paths":[],"execution_requested":false,"ambiguity_level":"high","language":"en","constraints":[],"requested_artifacts":[],"requested_contents":[],"requested_output_texts":[],"requested_summary":false,"read_only_task":true,"destructive_scope_unclear":true,"requires_execution":false,"requires_verification":false,"requires_stdout_match":false,"requires_tests":false,"requires_code_write":false,"requested_operation":"clarify"}')
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        payload = provider.request_json("prompt", schema=TASK_INTERPRETATION_SCHEMA)
        request = open_mock.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))
        self.assertIsInstance(request_payload["format"], dict)
        self.assertEqual(payload["intent"], "clarification_response")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_missing_required_fields(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes('{"goal":"demo","steps":[]}')
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, r"\$\.success_criteria: missing required field"):
            provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_wrong_field_type(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes('{"goal":"demo","success_criteria":"ok","steps":[]}')
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, r"\$\.success_criteria: expected array"):
            provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_invalid_tool_identifier(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"demo","success_criteria":["ok"],"steps":[{"tool":"python.execute","arguments":{},"description":"run","depends_on":[],"expected_result":"ok"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, r"\$\.steps\[0\]\.tool: invalid enum value"):
            provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_invalid_enum_value(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"intent":"clarification_response","confidence":"critical","goal":"clarify","expected_result":"","referenced_paths":[],"execution_requested":false,"ambiguity_level":"high","language":"en","constraints":[],"requested_artifacts":[],"requested_contents":[],"requested_output_texts":[],"requested_summary":false,"read_only_task":true,"destructive_scope_unclear":true,"requires_execution":false,"requires_verification":false,"requires_stdout_match":false,"requires_tests":false,"requires_code_write":false,"requested_operation":"clarify"}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, r"\$\.confidence: invalid enum value"):
            provider.request_json("prompt", schema=TASK_INTERPRETATION_SCHEMA)

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_markdown_wrapped_text_with_extra_prose(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes('Here is JSON:\n```json\n{"goal":"demo","success_criteria":["ok"],"steps":[]}\n```')
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, "structured output contained text before JSON"):
            provider.request_json("prompt")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_multiple_json_values(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes('{"goal":"demo"}{"goal":"second"}')
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, "structured output contained multiple JSON values"):
            provider.request_json("prompt")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_truncated_json(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes('{"goal":"demo"')
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, "structured output was truncated"):
            provider.request_json("prompt")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_accepts_clarification_response(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"intent":"clarification_response","confidence":"high","goal":"clarify file targets","expected_result":"","referenced_paths":[],"execution_requested":false,"ambiguity_level":"high","language":"ru","constraints":[],"requested_artifacts":[],"requested_contents":[],"requested_output_texts":[],"requested_summary":false,"read_only_task":true,"destructive_scope_unclear":true,"requires_execution":false,"requires_verification":false,"requires_stdout_match":false,"requires_tests":false,"requires_code_write":false,"requested_operation":"clarify"}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        payload = provider.request_json("prompt", schema=TASK_INTERPRETATION_SCHEMA)
        self.assertEqual(payload["intent"], "clarification_response")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_accepts_valid_git_plan(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"Inspect git status","success_criteria":["git status captured"],"steps":[{"tool":"terminal.execute","arguments":{"executable":"git","arguments":["status"],"working_directory":".","timeout_seconds":30,"operation_type":"git_read_only","raw_command":"git status"},"description":"Inspect git status","depends_on":[],"expected_result":"git status output captured"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        payload = provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))
        self.assertEqual(payload["steps"][0]["tool"], "terminal.execute")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_accepts_valid_file_and_python_plan(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"Prepare sample_message.py","success_criteria":["sample_message.py exists","stdout contains Testing intelligence"],"steps":[{"tool":"filesystem.create_text_file","arguments":{"path":"sample_message.py"},"description":"Create the file","depends_on":[],"expected_result":"file exists"},{"tool":"terminal.execute","arguments":{"executable":"python","arguments":["sample_message.py"],"working_directory":".","timeout_seconds":30,"operation_type":"python","raw_command":"python sample_message.py"},"description":"Run the script","depends_on":[1],"expected_result":"stdout contains Testing intelligence"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        payload = provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))
        self.assertEqual(len(payload["steps"]), 2)

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_accepts_valid_browser_reference_plan(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"Open the page and capture the title","success_criteria":["page title captured"],"steps":[{"tool":"browser.start_session","arguments":{"headless":true},"description":"Start browser","depends_on":[],"expected_result":"session ready"},{"tool":"browser.open_url","arguments":{"session_id":{"from_step":1,"field":"session_id"},"url":"https://example.com","wait_until":"domcontentloaded","timeout_seconds":30},"description":"Open the page","depends_on":[1],"expected_result":"page loaded"},{"tool":"browser.get_page_info","arguments":{"session_id":{"from_step":1,"field":"session_id"}},"description":"Read page info","depends_on":[1],"expected_result":"title captured"},{"tool":"browser.close_session","arguments":{"session_id":{"from_step":1,"field":"session_id"}},"description":"Close browser","depends_on":[1],"expected_result":"session closed"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        payload = provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))
        self.assertEqual(payload["steps"][1]["arguments"]["session_id"]["field"], "session_id")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_cross_tool_args_on_browser_open_url(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"Scroll page","success_criteria":["page scrolled"],"steps":[{"tool":"browser.start_session","arguments":{"headless":false},"description":"Start browser","depends_on":[],"expected_result":"session ready"},{"tool":"browser.open_url","arguments":{"session_id":{"from_step":1,"field":"session_id"},"url":"https://en.wikipedia.org/wiki/Main_Page","wait_until":"domcontentloaded","timeout_seconds":30,"path":"../logs/example-page.png","full_page":false},"description":"Open page","depends_on":[1],"expected_result":"page opened"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, r"\$\.steps\[1\]\.arguments\.(?:path|full_page): unexpected field"):
            provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_cross_tool_args_on_browser_scroll_page(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"Scroll page","success_criteria":["page scrolled"],"steps":[{"tool":"browser.start_session","arguments":{"headless":false},"description":"Start browser","depends_on":[],"expected_result":"session ready"},{"tool":"browser.scroll_page","arguments":{"session_id":{"from_step":1,"field":"session_id"},"amount":100,"path":"../logs/example-page.png","full_page":false},"description":"Scroll","depends_on":[1],"expected_result":"page scrolled"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, r"\$\.steps\[1\]\.arguments\.(?:path|full_page): unexpected field"):
            provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_accepts_screenshot_specific_arguments_only_for_screenshot_tool(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"Capture screenshot","success_criteria":["screenshot saved"],"steps":[{"tool":"browser.start_session","arguments":{"headless":true},"description":"Start browser","depends_on":[],"expected_result":"session ready"},{"tool":"browser.take_screenshot","arguments":{"session_id":{"from_step":1,"field":"session_id"},"path":"logs/example-page.png","full_page":true},"description":"Take screenshot","depends_on":[1],"expected_result":"screenshot saved"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        payload = provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))
        self.assertEqual(payload["steps"][1]["arguments"]["path"], "logs/example-page.png")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_accepts_valid_browser_visual_capture_plan(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"Open the page and visually describe it","success_criteria":["browser viewport captured","grounded browser visual evidence captured"],"steps":[{"tool":"browser.start_session","arguments":{"headless":true},"description":"Start browser","depends_on":[],"expected_result":"session ready"},{"tool":"browser.open_url","arguments":{"session_id":{"from_step":1,"field":"session_id"},"url":"https://example.com","wait_until":"domcontentloaded","timeout_seconds":30},"description":"Open the page","depends_on":[1],"expected_result":"page loaded"},{"tool":"browser.capture_view","arguments":{"session_id":{"from_step":1,"field":"session_id"}},"description":"Capture viewport","depends_on":[1,2],"expected_result":"capture ready"},{"tool":"vision.describe_browser_capture","arguments":{"capture_id":{"from_step":3,"field":"capture_id"},"detail_level":"brief"},"description":"Describe capture","depends_on":[3],"expected_result":"description captured"},{"tool":"browser.close_session","arguments":{"session_id":{"from_step":1,"field":"session_id"}},"description":"Close browser","depends_on":[1,4],"expected_result":"session closed"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        payload = provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 8))
        self.assertEqual(payload["steps"][2]["tool"], "browser.capture_view")
        self.assertEqual(payload["steps"][3]["arguments"]["capture_id"]["field"], "capture_id")

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen")
    def test_request_json_rejects_cross_tool_args_on_browser_capture_view(self, open_mock) -> None:
        open_mock.return_value.__enter__.return_value.read.return_value = _response_bytes(
            '{"goal":"Capture browser viewport","success_criteria":["capture ready"],"steps":[{"tool":"browser.start_session","arguments":{"headless":true},"description":"Start browser","depends_on":[],"expected_result":"session ready"},{"tool":"browser.capture_view","arguments":{"session_id":{"from_step":1,"field":"session_id"},"path":"logs/capture.png","full_page":true},"description":"Capture viewport","depends_on":[1],"expected_result":"capture ready"}]}'
        )
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaisesRegex(ValueError, r"\$\.steps\[1\]\.arguments\.(?:path|full_page): unexpected field"):
            provider.request_json("prompt", schema=build_plan_output_schema(_tool_catalog(), 5))

    @patch("app.brain.ai.ollama_provider.urllib.request.urlopen", side_effect=urllib.error.URLError("offline"))
    def test_generate_structured_returns_fallback_on_request_error(self, open_mock) -> None:
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        response = provider.generate_structured("prompt")
        self.assertEqual(response.category, "fallback")
        self.assertEqual(response.message, "I am offline right now.")
        self.assertEqual(response.status.category, ProviderStatusCategory.UNAVAILABLE)

    @patch(
        "app.brain.ai.ollama_provider.urllib.request.urlopen",
        side_effect=_http_error(
            500,
            '{"error":"llama-server reported out-of-memory during startup: cudaMalloc failed: out of memory"}',
        ),
    )
    def test_request_json_preserves_http_error_category_and_sanitized_detail(self, open_mock) -> None:
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaises(IntelligenceProviderRequestError) as context:
            provider.request_json("sensitive prompt text", schema=TASK_INTERPRETATION_SCHEMA)
        self.assertEqual(context.exception.category, "provider_http_error")
        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("HTTP 500 from Ollama generate endpoint", context.exception.safe_detail)
        self.assertIn("out of memory", context.exception.safe_detail)
        self.assertNotIn("sensitive prompt text", context.exception.safe_detail)

    @patch(
        "app.brain.ai.ollama_provider.urllib.request.urlopen",
        side_effect=urllib.error.URLError(socket.timeout("timed out")),
    )
    def test_request_json_distinguishes_timeout_from_http_rejection(self, open_mock) -> None:
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        with self.assertRaises(IntelligenceProviderRequestError) as context:
            provider.request_json("prompt", schema=TASK_INTERPRETATION_SCHEMA)
        self.assertEqual(context.exception.category, "provider_timeout")
        self.assertIn("timed out", context.exception.safe_detail)

    @patch(
        "app.brain.ai.ollama_provider.urllib.request.urlopen",
        side_effect=_http_error(500, '{"error":"server exploded"}'),
    )
    def test_generate_structured_treats_http_error_as_provider_error_not_unavailable(self, open_mock) -> None:
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        response = provider.generate_structured("prompt")
        self.assertEqual(response.category, "fallback")
        self.assertEqual(response.status.category, ProviderStatusCategory.ERROR)

    @patch(
        "app.brain.ai.ollama_provider.urllib.request.urlopen",
        side_effect=_http_error(500, '{"error":"llama-server reported out-of-memory during startup"}'),
    )
    def test_generation_readiness_distinguishes_health_from_generate_failures(self, open_mock) -> None:
        provider = OllamaProvider(base_url="http://127.0.0.1:11434", model="test-model", timeout=5.0)
        ready, detail = provider.check_generation_ready()
        self.assertFalse(ready)
        self.assertIn("HTTP 500", detail)
