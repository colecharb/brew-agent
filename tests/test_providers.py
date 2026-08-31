"""The shim, both directions, on both wires. No network.

Translation is the whole risk of running these arms on a second vendor: an arm
reads a `ModelResponse` and cannot tell what produced it, so a mistranslation
does not raise — it returns an answer that is quietly wrong. Every test here
asserts on the *native* payload that went out or the normalised object that
came back, never on the neutral form in between, because the neutral form is
the part that cannot be wrong.
"""

import json
from dataclasses import dataclass, field

import anthropic
import httpx
import pytest

from brew_agent import providers
from brew_agent.baselines import CLASSIFY_TASTE, extract_recommendation
from brew_agent.config import ModelConfig
from brew_agent.providers import (
    AnthropicProvider,
    CohereProvider,
    ModelResponse,
    TextBlock,
    ToolUseBlock,
    assistant_message,
    cohere_schema,
    tool_results_message,
    user_message,
)
from brew_agent.tools import SUBMIT_RECOMMENDATION, SUBMIT_TOOL


def config(provider="anthropic", effort=None, model=None, max_tokens=16000):
    return ModelConfig(
        provider=provider,
        api_key="test",
        model=model or ("claude-opus-5" if provider == "anthropic" else "command-r7b-12-2024"),
        effort=effort,
        max_iterations=6,
        max_tokens=max_tokens,
    )


@pytest.fixture(autouse=True)
def _reset_warnings():
    providers._warned.clear()


# --- Anthropic -------------------------------------------------------------


@dataclass
class AnthropicBlock:
    type: str
    text: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)
    id: str = "toolu_1"


@dataclass
class AnthropicUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class AnthropicResponse:
    content: list
    stop_reason: str = "tool_use"
    usage: AnthropicUsage = field(default_factory=AnthropicUsage)


class RecordingMessages:
    def __init__(self, response=None, error=None):
        self._response = response or AnthropicResponse([])
        self._error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error and len(self.calls) == 1:
            raise self._error
        return self._response


class RecordingClient:
    def __init__(self, response=None, error=None):
        self.messages = RecordingMessages(response, error)


def bad_request(message: str) -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


class TestAnthropicRequests:
    def test_a_forced_tool_is_named(self):
        """Anthropic can pin one tool by name; the Cohere path cannot."""
        client = RecordingClient()
        AnthropicProvider(client).complete(
            config(), system="s", messages=[], tools=[], force_tool=SUBMIT_TOOL
        )
        sent = client.messages.calls[0]
        assert sent["tool_choice"] == {"type": "tool", "name": SUBMIT_TOOL}

    def test_max_tokens_comes_from_the_config(self):
        client = RecordingClient()
        AnthropicProvider(client).complete(
            config(max_tokens=1234), system="s", messages=[], tools=[]
        )
        assert client.messages.calls[0]["max_tokens"] == 1234

    def test_sampling_parameters_are_never_sent(self):
        """Claude Opus 5 rejects temperature/top_p/top_k with a 400."""
        client = RecordingClient()
        AnthropicProvider(client).complete(config(), system="s", messages=[], tools=[])
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in client.messages.calls[0]

    def test_an_assistant_turn_is_echoed_verbatim(self):
        """Reconstructing it would drop thinking-block signatures.

        Anthropic rejects a thinking block whose signature it did not write, so
        the round trip has to hand back the vendor's own content object rather
        than a rebuild of it from the normalised form.
        """
        raw = AnthropicResponse([AnthropicBlock("text", text="thinking out loud")])
        response = ModelResponse(stop_reason="end_turn", content=[], raw=raw)
        client = RecordingClient()
        AnthropicProvider(client).complete(
            config(),
            system="s",
            messages=[user_message("hi"), assistant_message(response)],
            tools=[],
        )
        sent = client.messages.calls[0]["messages"]
        assert sent[0] == {"role": "user", "content": "hi"}
        assert sent[1]["content"] is raw.content

    def test_tool_results_are_filed_under_the_user_turn(self):
        client = RecordingClient()
        AnthropicProvider(client).complete(
            config(),
            system="s",
            messages=[
                tool_results_message(
                    [{"id": "toolu_1", "content": "{}", "is_error": True}]
                )
            ],
            tools=[],
        )
        block = client.messages.calls[0]["messages"][0]
        assert block["role"] == "user"
        assert block["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "{}",
                "is_error": True,
            }
        ]


class TestAnthropicResponses:
    def test_blocks_and_usage_are_normalised(self):
        raw = AnthropicResponse(
            [
                AnthropicBlock("text", text="looking"),
                AnthropicBlock("tool_use", name="get_brew", input={"brew_id": "b1"}),
            ]
        )
        result = AnthropicProvider(RecordingClient(raw)).complete(
            config(), system="s", messages=[], tools=[]
        )
        assert result.text == "looking"
        assert [c.name for c in result.tool_calls] == ["get_brew"]
        assert result.tool_calls[0].input == {"brew_id": "b1"}
        assert result.usage == {"input_tokens": 100, "output_tokens": 50}
        assert result.raw is raw

    def test_a_refusal_keeps_its_name(self):
        raw = AnthropicResponse([], stop_reason="refusal")
        result = AnthropicProvider(RecordingClient(raw)).complete(
            config(), system="s", messages=[], tools=[]
        )
        assert result.stop_reason == "refusal"
        assert extract_recommendation(result).error == "model declined the request"


class TestUnsupportedParametersDegradeRatherThanFail:
    """Smaller models reject parameters the frontier ones accept.

    Without this the eval fails every pair identically on a model-capability
    mismatch — an expensive way to learn that `effort` isn't universal.
    """

    def call(self, client, effort=None, force_tool=None):
        return AnthropicProvider(client).complete(
            config(effort=effort),
            system="s",
            messages=[],
            tools=[],
            force_tool=force_tool,
        )

    def test_effort_is_dropped_and_the_call_retried(self):
        client = RecordingClient(
            error=bad_request("output_config.effort: unsupported on this model")
        )
        self.call(client, effort="high")
        first, second = client.messages.calls
        assert first["output_config"] == {"effort": "high"}
        assert "output_config" not in second

    def test_dropping_effort_is_announced(self, capsys):
        client = RecordingClient(error=bad_request("output_config.effort: unsupported"))
        self.call(client, effort="high")
        assert "not comparable" in capsys.readouterr().err

    def test_the_warning_fires_once_across_many_calls(self, capsys):
        for _ in range(5):
            self.call(
                RecordingClient(error=bad_request("output_config.effort: bad")),
                effort="high",
            )
        assert capsys.readouterr().err.count("warning:") == 1

    def test_a_forced_tool_is_still_dropped_on_its_own_error(self):
        client = RecordingClient(error=bad_request("tool_choice: not allowed here"))
        self.call(client, force_tool=SUBMIT_TOOL)
        assert "tool_choice" not in client.messages.calls[1]

    def test_an_unrelated_bad_request_still_raises(self):
        """Only the two known-survivable rejections are absorbed."""
        client = RecordingClient(error=bad_request("messages: must not be empty"))
        with pytest.raises(anthropic.BadRequestError):
            self.call(client, effort="high")

    def test_nothing_is_dropped_when_effort_was_never_sent(self):
        client = RecordingClient(error=bad_request("output_config.effort: unsupported"))
        with pytest.raises(anthropic.BadRequestError):
            self.call(client)


# --- Cohere ----------------------------------------------------------------


def cohere_response(
    tool_calls=(), text="", plan="", finish="TOOL_CALL", billed=(30, 12)
):
    """A v2 chat response, shaped the way the SDK returns one."""
    return {
        "finish_reason": finish,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}] if text else None,
            "tool_plan": plan,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
                for call_id, name, args in tool_calls
            ],
        },
        "usage": {
            "billed_units": {"input_tokens": billed[0], "output_tokens": billed[1]},
            "tokens": {"input_tokens": 999, "output_tokens": 999},
        },
    }


class RecordingChat:
    def __init__(self, response=None, error=None):
        self._response = response if response is not None else cohere_response()
        self._error = error
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._error and len(self.calls) == 1:
            raise self._error
        return self._response


class TestCohereRequests:
    def test_the_system_prompt_becomes_the_first_message(self):
        client = RecordingChat()
        CohereProvider(client).complete(
            config("cohere"), system="you diagnose brews", messages=[user_message("hi")],
            tools=[],
        )
        sent = client.calls[0]["messages"]
        assert sent[0] == {"role": "system", "content": "you diagnose brews"}
        assert sent[1] == {"role": "user", "content": "hi"}

    def test_strict_tools_is_always_on(self):
        """It is what replaces the schema-retry the harness leans on."""
        client = RecordingChat()
        CohereProvider(client).complete(
            config("cohere"), system="s", messages=[], tools=[]
        )
        assert client.calls[0]["strict_tools"] is True

    def test_forcing_a_tool_can_only_ask_for_one_of_them(self):
        client = RecordingChat()
        CohereProvider(client).complete(
            config("cohere"),
            system="s",
            messages=[],
            tools=[SUBMIT_RECOMMENDATION],
            force_tool=SUBMIT_TOOL,
        )
        assert client.calls[0]["tool_choice"] == "REQUIRED"

    def test_forcing_with_several_tools_offered_is_announced(self, capsys):
        """REQUIRED means 'some tool'. With one offered that is the same thing;
        with several it silently is not."""
        CohereProvider(RecordingChat()).complete(
            config("cohere"),
            system="s",
            messages=[],
            tools=[SUBMIT_RECOMMENDATION, CLASSIFY_TASTE],
            force_tool=SUBMIT_TOOL,
        )
        assert "may answer with a different one" in capsys.readouterr().err

    def test_effort_is_reported_as_not_applied(self, capsys):
        CohereProvider(RecordingChat()).complete(
            config("cohere", effort="high"), system="s", messages=[], tools=[]
        )
        assert "effort is an Anthropic parameter" in capsys.readouterr().err

    def test_tools_are_wrapped_as_functions(self):
        client = RecordingChat()
        CohereProvider(client).complete(
            config("cohere"), system="s", messages=[], tools=[SUBMIT_RECOMMENDATION]
        )
        tool = client.calls[0]["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == SUBMIT_TOOL
        assert tool["function"]["parameters"]["type"] == "object"

    def test_an_assistant_turn_carries_its_plan_and_calls(self):
        response = ModelResponse(
            stop_reason="tool_use",
            content=[
                TextBlock(text="checking their history"),
                ToolUseBlock(id="c1", name="get_brew", input={"brew_id": "b1"}),
            ],
        )
        client = RecordingChat()
        CohereProvider(client).complete(
            config("cohere"),
            system="s",
            messages=[assistant_message(response)],
            tools=[],
        )
        sent = client.calls[0]["messages"][1]
        assert sent["role"] == "assistant"
        # Cohere keeps pre-tool reasoning here rather than in `content`.
        assert sent["tool_plan"] == "checking their history"
        assert sent["tool_calls"] == [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "get_brew",
                    "arguments": json.dumps({"brew_id": "b1"}),
                },
            }
        ]

    def test_a_silent_assistant_turn_still_gets_a_plan(self):
        """Cohere rejects an empty tool_plan alongside tool calls."""
        response = ModelResponse(
            stop_reason="tool_use",
            content=[ToolUseBlock(id="c1", name="get_brew", input={})],
        )
        client = RecordingChat()
        CohereProvider(client).complete(
            config("cohere"), system="s", messages=[assistant_message(response)],
            tools=[],
        )
        assert client.calls[0]["messages"][1]["tool_plan"]

    def test_a_prose_turn_goes_in_content_not_a_plan(self):
        response = ModelResponse(
            stop_reason="end_turn", content=[TextBlock(text="grind finer")]
        )
        client = RecordingChat()
        CohereProvider(client).complete(
            config("cohere"), system="s", messages=[assistant_message(response)],
            tools=[],
        )
        assert client.calls[0]["messages"][1] == {
            "role": "assistant",
            "content": "grind finer",
        }

    def test_every_tool_result_gets_its_own_message(self):
        client = RecordingChat()
        CohereProvider(client).complete(
            config("cohere"),
            system="s",
            messages=[
                tool_results_message(
                    [
                        {"id": "c1", "content": '{"brew": 1}', "is_error": False},
                        {"id": "c2", "content": '{"error": "boom"}', "is_error": True},
                    ]
                )
            ],
            tools=[],
        )
        sent = client.calls[0]["messages"][1:]
        assert sent == [
            {"role": "tool", "tool_call_id": "c1", "content": '{"brew": 1}'},
            {"role": "tool", "tool_call_id": "c2", "content": '{"error": "boom"}'},
        ]

    def test_a_rejected_parameter_is_dropped_and_retried(self, capsys):
        client = RecordingChat(error=RuntimeError("400: strict_tools not supported"))
        CohereProvider(client).complete(
            config("cohere"), system="s", messages=[], tools=[]
        )
        assert "strict_tools" in client.calls[0]
        assert "strict_tools" not in client.calls[1]
        assert "no longer guaranteed" in capsys.readouterr().err

    def test_an_unrelated_failure_still_raises(self):
        client = RecordingChat(error=RuntimeError("502 bad gateway"))
        with pytest.raises(RuntimeError):
            CohereProvider(client).complete(
                config("cohere"), system="s", messages=[], tools=[]
            )


class TestCohereResponses:
    def test_tool_arguments_are_parsed_out_of_their_json_string(self):
        """The whole answer arrives as a string here, not as an object."""
        raw = cohere_response(
            tool_calls=[("c1", SUBMIT_TOOL, json.dumps({
                "grind_setting": "485",
                "primary_lever": "grind_setting",
                "reasoning": "their good brews sit at 485",
            }))]
        )
        result = CohereProvider(RecordingChat(raw)).complete(
            config("cohere"), system="s", messages=[], tools=[]
        )
        rec = extract_recommendation(result)
        assert rec.grind_setting == "485"
        assert rec.primary_lever == "grind_setting"

    def test_unparseable_arguments_fail_loudly(self):
        """`{}` would score as an abstention and look like a real opinion."""
        raw = cohere_response(tool_calls=[("c1", SUBMIT_TOOL, "{not json")])
        with pytest.raises(ValueError, match="unparseable"):
            CohereProvider(RecordingChat(raw)).complete(
                config("cohere"), system="s", messages=[], tools=[]
            )

    def test_the_tool_plan_is_recorded_as_the_assistant_text(self):
        raw = cohere_response(
            tool_calls=[("c1", "get_brew", "{}")], plan="I will look up the brew"
        )
        result = CohereProvider(RecordingChat(raw)).complete(
            config("cohere"), system="s", messages=[], tools=[]
        )
        assert result.text == "I will look up the brew"

    def test_billed_units_are_what_the_trace_records(self):
        """They differ from the raw counts, and the bill is the useful one."""
        result = CohereProvider(RecordingChat(cohere_response(billed=(30, 12)))).complete(
            config("cohere"), system="s", messages=[], tools=[]
        )
        assert result.usage == {"input_tokens": 30, "output_tokens": 12}

    @pytest.mark.parametrize(
        "finish,expected",
        [
            ("COMPLETE", "end_turn"),
            ("TOOL_CALL", "tool_use"),
            ("MAX_TOKENS", "max_tokens"),
            ("STOP_SEQUENCE", "stop_sequence"),
            ("ERROR", "error"),
        ],
    )
    def test_finish_reasons_are_mapped(self, finish, expected):
        result = CohereProvider(RecordingChat(cohere_response(finish=finish))).complete(
            config("cohere"), system="s", messages=[], tools=[]
        )
        assert result.stop_reason == expected

    def test_a_missing_answer_reads_the_same_as_on_the_other_wire(self):
        """Cohere has no refusal reason, so a decline arrives as a plain turn."""
        raw = cohere_response(text="I would rather not", finish="COMPLETE")
        result = CohereProvider(RecordingChat(raw)).complete(
            config("cohere"), system="s", messages=[], tools=[]
        )
        assert extract_recommendation(result).error == (
            f"no {SUBMIT_TOOL} call (stop_reason=end_turn)"
        )


# --- schema translation ----------------------------------------------------


class TestSchemaTranslation:
    """`nullable()` is built out of `anyOf`, which Cohere's strict mode rejects."""

    def test_nullable_fields_lose_their_anyOf(self):
        converted = cohere_schema(SUBMIT_RECOMMENDATION["input_schema"])
        assert converted["properties"]["grind_setting"] == {
            "type": "string",
            "description": SUBMIT_RECOMMENDATION["input_schema"]["properties"][
                "grind_setting"
            ]["description"],
        }
        assert "anyOf" not in json.dumps(converted)

    def test_nullable_fields_stop_being_required(self):
        """Required under strict mode means the model must invent a value.

        Left in, "change the grind" would come back as a new dose, yield,
        temperature and time as well — and no scoring column could tell that
        from a recommendation that meant it.
        """
        converted = cohere_schema(SUBMIT_RECOMMENDATION["input_schema"])
        assert converted["required"] == ["primary_lever", "reasoning"]

    def test_something_stays_required(self):
        """Cohere refuses a strict tool whose parameters are all optional."""
        for tool in (SUBMIT_RECOMMENDATION, CLASSIFY_TASTE):
            assert cohere_schema(tool["input_schema"])["required"]

    def test_unsupported_keywords_are_stripped(self):
        converted = cohere_schema(SUBMIT_RECOMMENDATION["input_schema"])
        assert "additionalProperties" not in converted

    def test_enums_and_descriptions_survive(self):
        converted = cohere_schema(SUBMIT_RECOMMENDATION["input_schema"])
        lever = converted["properties"]["primary_lever"]
        assert lever["enum"][0] == "grind_setting"
        assert lever["description"]

    def test_a_plain_schema_is_left_alone(self):
        plain = {
            "type": "object",
            "properties": {"brew_id": {"type": "string", "description": "uuid"}},
            "required": ["brew_id"],
        }
        assert cohere_schema(plain) == plain

    def test_an_unrecognised_union_is_not_guessed_at(self):
        """Not the nullable() shape: left to fail at the API, not reinterpreted."""
        odd = {
            "type": "object",
            "properties": {
                "x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}
            },
            "required": ["x"],
        }
        assert cohere_schema(odd)["properties"]["x"] == odd["properties"]["x"]
        assert cohere_schema(odd)["required"] == ["x"]


def test_an_omitted_field_and_a_null_one_mean_the_same_thing():
    """The two wires spell 'leave this alone' differently.

    Anthropic's schema asks for an explicit null, Cohere's for an absent key.
    Both have to arrive at the same recommendation or the arms are not
    comparable across providers.
    """
    from brew_agent.models import Recommendation

    spelled_null = Recommendation.from_tool_input(
        {"grind_setting": "485", "water_temp": None, "primary_lever": "grind_setting"}
    )
    left_out = Recommendation.from_tool_input(
        {"grind_setting": "485", "primary_lever": "grind_setting"}
    )
    assert spelled_null.to_dict() == left_out.to_dict()
