"""Round-trip an excerpt of one recorded Assay provider response."""
import json
import math
from pathlib import Path

import httpx

from jig.jev import (
    ChoiceAnswer, ChoiceQuestion, JevClient, NoulAnswer, NoulQuestion,
    ScoreAnswer, ScoreQuestion,
)
from jig.jev.wire import build_request_body

FIXTURE = Path(__file__).parent / "fixtures/jev/recorded-run"


def load(name):
    return json.loads((FIXTURE / name).read_text())


# Hand-transcribed from the catalogue whose digest is pinned in provenance.json.
# The catalogue is an offline source only: tests and runtime do not parse it.
QUESTIONS = [
    NoulQuestion(
        "topic_observability",
        "Is the post's subject observability for operating agents?",
        {"true": {"what": "Tracing", "logging": None, "monitoring": None,
                  "debugging": None, "telemetry": None,
                  "or diagnosis of running agents.": None},
         "false": {"what": "No agent observability topic is present."}},
    ),
    ChoiceQuestion(
        "account_type",
        "Classify the author account from name and handle only. This annotation must not affect relevance.",
        {"individual_practitioner": {"what": "A person who builds or operates software themselves."},
         "brand_or_company": {"what": "A product", "vendor": None, "organization": None,
                              "or company account.": None},
         "aggregator_or_feed": {"what": "A news feed", "link aggregator": None,
                                "digest": None, "or auto-poster.": None},
         "executive_or_founder_voice": {"what": "A named executive or founder posting in that capacity."},
         "bot": {"what": "An automated account with no human author."}},
    ),
    ScoreQuestion(
        "band",
        {"question": "Classify the post's own text into the highest-precedence relevance situation.",
         "apply_in_order": ["hard exclusion", "substantive point", "on-topic pointer",
                            "general building", "out of scope"],
         "note": "Parent context can clarify the post but cannot supply a missing point."},
        [{"summary": "out_of_scope", "signals": ["unrelated content", "hard exclusion applies"]},
         {"summary": "building", "signals": ["general agent-building content", "no operational point"]},
         {"summary": "pointer", "signals": ["on-topic resource or announcement", "no point of its own"]},
         {"summary": "substantive", "signals": ["specific operational claim", "result", "failure",
                                                "reasoned practice", "or concrete question in the post text"]}],
    ),
]


async def test_recorded_roundtrip():
    request = load("request.json")
    response = load("response.json")
    provenance = load("provenance.json")
    assert "RECONSTRUCTED" in provenance["request_note"]
    assert set(request) == {"state", "model", "questions"}
    assert set(response) == {"model", "request_id", "answers", "usage"}
    assert request["state"] == {}
    assert json.dumps(build_request_body({}, QUESTIONS, "jev-latest"), sort_keys=True) == json.dumps(request, sort_keys=True)
    assert math.isclose(math.fsum(response["answers"]["band"]["probabilities"].values()), 0.99)

    def handler(sent):
        assert json.loads(sent.content) == request
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with JevClient(api_key="test-only", http=http) as client:
            result = await client.evaluate(request["state"], QUESTIONS)

    assert result.model == provenance["resolved_model"] == "jev-1.13.0"
    assert request["model"] == provenance["requested_model"] == "jev-latest"
    assert result.provider_request_id == response["request_id"]
    assert result.call_id != result.provider_request_id
    assert result.usage.input_tokens == response["usage"]["input_tokens"]
    assert result.usage.output_tokens == response["usage"]["output_tokens"]
    assert result.answers.keys() == response["answers"].keys()
    for key, recorded in response["answers"].items():
        answer = result.answers[key]
        if isinstance(answer, NoulAnswer):
            assert answer.noul == recorded["noul"]
        elif isinstance(answer, ChoiceAnswer):
            assert answer.choice == recorded["choice"]
            assert answer.probabilities == recorded["probabilities"]
            assert answer.confidence == recorded["confidence"]
        elif isinstance(answer, ScoreAnswer):
            assert answer.score == recorded["score"]
            assert answer.probabilities == recorded["probabilities"]
            assert answer.legend == recorded["legend"]
            assert answer.confidence == recorded["confidence"]
        else:
            raise AssertionError(type(answer))
