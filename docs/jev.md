# Jev client

`jig.jev` submits typed questions to TypeSafe Jev and returns native answers. Supply `TYPESAFE_API_KEY` or pass `api_key`; the client uses `jev-latest` by default. The returned `model` is the provider's resolved version. Keep your application's decision rule outside jig.

```python jev-example
from jig.jev import JevClient, NoulQuestion, ChoiceQuestion, ScoreQuestion

async def evaluate_example():
    questions = [
        NoulQuestion("relevant", "Is this about operating agents?"),
        ChoiceQuestion("kind", "Which kind?", {"practice": "Practice", "incident": "Incident"}),
        ScoreQuestion("quality", "Rate its quality", ["low", "high"]),
    ]
    async with JevClient(api_key="example-key") as client:
        result = await client.evaluate({"subject": "agent operations"}, questions)
    probability = result.answers["relevant"].noul
    choice = result.answers["kind"].choice
    score = result.answers["quality"].score
    return result, probability, choice, score
```

The answer mapping is keyed by question ID. Choice and Score answers also retain the full probability distribution; Score retains the provider's structured `legend`. Do not reduce a Jev Score answer to `jig.Score`, which represents a grading result.

## Record a trace

Start a `SpanKind.PROVIDER_CALL` span named `"jev.call"` before `evaluate`. On success, record `result.call_id`, `result.provider_request_id`, resolved `result.model`, `result.latency_ms`, and status `"ok"` in span metadata, then end the span with `usage=to_jig_usage(result.usage)`. On `JevError`, record status `"error"`, its available IDs and elapsed time, and end the span with the error. Keep private state and answer values out of trace metadata. `to_jig_usage` has no cost estimate.

See [the Jev contract](jev-contract.md) for validation and retry rules.
