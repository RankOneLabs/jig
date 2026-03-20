# jig

Minimal agent framework. Five interfaces, two execution models, everything swappable.

A jig holds the pieces while you work on them. It doesn't have opinions about your pieces — just the shapes of the plugs between them.

## Install

```bash
uv add jig                          # core only
uv add 'jig[anthropic]'             # + Anthropic adapter
uv add 'jig[ollama]'                # + Ollama adapter (homelab)
uv add 'jig[all]'                   # everything
```

## Quick start

```python
from jig import AgentConfig, run_agent
from jig.llm import AnthropicClient
from jig.memory import LocalMemory
from jig.feedback import SQLiteFeedbackLoop
from jig.tracing import StdoutTracer
from jig.tools import ToolRegistry

config = AgentConfig(
    name="my-agent",
    description="A simple agent",
    system_prompt="You are a helpful assistant.",
    llm=AnthropicClient(model="claude-sonnet-4-20250514"),
    memory=LocalMemory(),
    feedback=SQLiteFeedbackLoop(),
    tracer=StdoutTracer(),
    tools=ToolRegistry(),
)

result = await run_agent(config, "What's the weather like?")
print(result.output)
```

## Interfaces

| Interface | Purpose | Adapters |
|---|---|---|
| `LLMClient` | LLM completions | Anthropic, OpenAI, Ollama |
| `AgentMemory` | Storage + retrieval | Local (SQLite + embeddings), Honcho, Zep |
| `FeedbackLoop` | Score tracking + eval export | SQLite |
| `Grader` | Auto-score outputs | LLM Judge, Heuristic, Ground Truth, Composite |
| `TracingLogger` | Structured spans | SQLite, Stdout |

Plus `ToolRegistry` (concrete) and `Tool` (abstract) for tool use.

## Two execution models

### `run_agent()` — LLM-in-the-loop

The LLM decides what to do next. You provide tools and a system prompt; jig runs the completion + tool loop.

1. Start trace
2. Resolve system prompt (string or async callable)
3. Query memory for relevant context
4. Query feedback for quality signals from past runs
5. Assemble `CompletionParams` (system separate from messages)
6. LLM call + tool execution loop
7. Store output in memory
8. Auto-grade if grader configured
9. Close trace, return `AgentResult`

### `run_pipeline()` — orchestrator-controlled

You define the step sequence; jig wraps it with tracing, grading, and feedback. The LLM (if any) is a transform *within* a step, not the decision-maker.

```python
from jig import PipelineConfig, Step, run_pipeline
from jig.tracing import StdoutTracer

async def fetch(ctx):
    return await get_document(ctx["input"])

async def summarize(ctx):
    return await llm_summarize(ctx["fetch"])

async def score(ctx):
    return evaluate_quality(ctx["summarize"])

result = await run_pipeline(
    PipelineConfig(
        name="summarizer",
        steps=[
            Step(name="fetch", fn=fetch),
            Step(name="summarize", fn=summarize),
            Step(name="score", fn=score),
        ],
        tracer=StdoutTracer(),
    ),
    input="https://example.com/article",
)
print(result.output)            # score result
print(result.step_outputs)      # {"fetch": ..., "summarize": ..., "score": ...}
```

Each step receives a context dict (`ctx`) and returns anything. The framework:
- Stores the return value in `ctx[step.name]` for downstream steps
- Traces every step as a `PIPELINE_STEP` span
- Short-circuits on error if `is_err` / `extract_err` are configured
- Skips steps conditionally via `skip_when`
- Grades per-step (via `Step.grader`) or pipeline-wide (via `PipelineConfig.grader`)
- Stores graded results via `FeedbackLoop` if configured

#### `map_pipeline()`

Runs `run_pipeline` per item with a shared parent trace. Optionally grades the batch.

```python
from jig import map_pipeline

result = await map_pipeline(config, items=[doc1, doc2, doc3])
# result.results — list of PipelineResult, one per item
```

#### Nested pipelines

A step can call `run_pipeline` internally. Pass `ctx["_tracer"]` and `ctx["_span_id"]` to nest spans under the parent trace.

```python
async def inner_step(ctx):
    return await run_pipeline(
        inner_config,
        input=ctx["previous_step"],
        _parent_span_id=ctx["_span_id"],
    )
```

## CompletionParams

LLM adapters receive a single `CompletionParams` object — typed core fields plus a provider escape hatch:

```python
CompletionParams(
    messages=[...],              # universal
    system="You are...",         # universal, handled per-provider
    tools=[...],                 # universal
    temperature=0.7,             # universal
    max_tokens=4096,             # universal
    provider_params={"top_k": 40},  # forwarded to SDK, provider-specific
)
```

## Project layout

```
src/jig/
├── core/           # types, runner, pipeline, errors, retry, prompt builder
├── llm/            # anthropic, openai, ollama adapters
├── memory/         # local (sqlite+embeddings), honcho, zep
├── feedback/       # feedback loop, llm judge, heuristic, ground truth, composite
├── tracing/        # sqlite tracer, stdout tracer
└── tools/          # registry + common tools
```
