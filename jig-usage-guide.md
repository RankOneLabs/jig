# jig — usage guide

jig is a minimal agent orchestration framework. It defines the shapes between components, runs your execution model, and gets out of the way. Your agents, tools, and services can be anything — jig just holds the pieces while you work on them.

jig has two execution models:

- **`run_agent`** — LLM-in-the-loop. The model decides what to do next via a completion + tool loop.
- **`run_pipeline`** — Orchestrator-controlled. You define the step sequence; jig wraps it with tracing, grading, and feedback.

## Install

```bash
# Clone
git clone https://github.com/rankonelabs/jig.git
cd jig

# Editable install with the providers you need
uv pip install -e ".[anthropic]"         # just Anthropic
uv pip install -e ".[anthropic,ollama]"  # Anthropic + local models
uv pip install -e ".[all]"              # everything
```

From another repo (e.g. your agents repo):

```toml
# pyproject.toml
[project]
dependencies = [
    "jig[anthropic,ollama] @ git+https://github.com/rankonelabs/jig.git",
]
```

## Quick start — agent

```python
from jig import AgentConfig, run_agent
from jig.llm import AnthropicClient
from jig.memory import LocalMemory
from jig.feedback import SQLiteFeedbackLoop
from jig.tracing import SQLiteTracer
from jig.tools import ToolRegistry

config = AgentConfig(
    name="my-agent",
    description="Does a thing",
    system_prompt="You are a helpful agent that does the thing.",
    llm=AnthropicClient(model="claude-sonnet-4-20250514"),
    memory=LocalMemory(db_path="./data/memory.db"),
    feedback=SQLiteFeedbackLoop(db_path="./data/feedback.db"),
    tracer=SQLiteTracer(db_path="./data/traces.db"),
    tools=ToolRegistry(),
)

result = await run_agent(config, "Do the thing")
print(result.output)
```

## Quick start — pipeline

```python
from jig import PipelineConfig, Step, run_pipeline
from jig.tracing import StdoutTracer

async def fetch(ctx):
    return await get_document(ctx["input"])

async def summarize(ctx):
    return await llm_summarize(ctx["fetch"])

async def evaluate(ctx):
    return score_quality(ctx["summarize"])

result = await run_pipeline(
    PipelineConfig(
        name="summarizer",
        steps=[
            Step(name="fetch", fn=fetch),
            Step(name="summarize", fn=summarize),
            Step(name="evaluate", fn=evaluate),
        ],
        tracer=StdoutTracer(),
    ),
    input="https://example.com/article",
)
print(result.output)            # evaluate result
print(result.step_outputs)      # {"fetch": ..., "summarize": ..., "evaluate": ...}
```

## What `run_agent` does

Every call to `run_agent(config, input)` executes the same loop:

1. **Start trace** — opens a root span for the entire run
2. **Resolve system prompt** — evaluates static string or callable
3. **Query memory** — semantic search for relevant past context *(skipped if `include_memory_in_prompt=False`)*
4. **Query feedback** — retrieves quality signals from past similar runs *(skipped if `include_feedback_in_prompt=False`)*
5. **Assemble CompletionParams** — system prompt (with memory context + feedback signals) passed separately, messages = session history + user input
6. **LLM call + tool loop** — calls the model, executes any tool calls, loops until the model returns a final response or `max_tool_calls` is hit
7. **Store in memory** — saves the output for future runs
8. **Auto-grade** — if a grader is configured, scores the output and stores scores in the feedback loop
9. **Close trace** — finalizes all spans
10. **Return `AgentResult`** — output, trace ID, usage stats, scores, duration

You don't call these steps. You configure the components and `run_agent` handles the rest.

## What `run_pipeline` does

Every call to `run_pipeline(config, input)` executes a fixed sequence of steps you define:

1. **Start trace** — opens a root span (`PIPELINE_RUN`). If `_parent_span_id` is provided, creates a child span instead (for nesting).
2. **Init context** — `ctx = {"input": input, "_tracer": config.tracer, "_span_id": root.id, **context}`. Steps read their inputs from `ctx` and their outputs are stored back into it.
3. **For each step:**
   - If `skip_when(ctx)` returns `True` → record a skipped span, continue
   - Open a `PIPELINE_STEP` span
   - Call `step.fn(ctx)`
   - Store the return value in `ctx[step.name]` and `step_outputs`
   - If `is_err(result)` → end span with error from `extract_err`, short-circuit (remaining steps don't run)
   - If `step.grader` is set → grade in a `GRADING` sub-span, store in `step_scores`. If `feedback` is configured, call `feedback.score()`.
4. **Pipeline grader** — if `config.grader` is set and no short-circuit, grade the overall (input, final output)
5. **Close trace**, return `PipelineResult`

The key difference from `run_agent`: **you** define the step sequence. The LLM (if any) is a transform inside a step function, not the decision-maker. jig provides tracing, grading, and feedback — the same infrastructure as `run_agent`, without the completion loop.

### `PipelineConfig`

```python
from jig import PipelineConfig, Step

config = PipelineConfig(
    name="my-pipeline",                 # required — trace name
    steps=[...],                        # required — sequence of Step
    tracer=SQLiteTracer(),              # required — same TracingLogger interface

    grader=my_grader,                   # optional — grades overall output
    feedback=my_feedback_loop,          # optional — stores per-step scores
    is_err=lambda r: hasattr(r, "error"),   # optional — detect errors in step output
    extract_err=lambda r: r.error,          # optional — extract error message
    metadata={"version": "1.2"},        # optional — attached to root span
)
```

Only `name`, `steps`, and `tracer` are required. Everything else is opt-in.

### `Step`

```python
Step(
    name="summarize",                    # stored in ctx under this key
    fn=summarize,                        # async fn(ctx) -> Any
    grader=quality_grader,               # optional — grade this step's output
    skip_when=lambda ctx: ctx.get("cached"),  # optional — skip if predicate is True
)
```

Step functions receive the context dict and return anything. The framework doesn't constrain the types — type safety lives in your functions, not the framework.

### `PipelineResult`

```python
result = await run_pipeline(config, input="some input")

result.output           # Any — return value of the last step that ran
result.trace_id         # str — look up full trace in the tracer
result.step_outputs     # dict[str, Any] — {step_name: return_value}
result.scores           # list[Score] | None — pipeline-level grading
result.step_scores      # dict[str, list[Score]] — per-step grading
result.duration_ms      # float — wall clock time
result.short_circuited  # bool — True if is_err triggered early exit
result.error_step       # str | None — name of the step that errored
```

### Context dict

The context dict (`ctx`) is the data bus between steps. It starts with:

```python
{
    "input": input,          # the value passed to run_pipeline
    "_tracer": config.tracer,  # for nested pipelines
    "_span_id": root.id,      # for nested pipelines
}
```

Plus any extra keys from the `context` parameter. After each step runs, `ctx[step.name]` is set to its return value. Steps downstream read from these keys.

### Short-circuiting on errors

By default, `run_pipeline` runs all steps to completion. To enable early exit on errors, provide `is_err` and `extract_err`:

```python
from dataclasses import dataclass

@dataclass
class Err:
    error: str

config = PipelineConfig(
    name="fallible",
    steps=[step_a, step_b, step_c],
    tracer=tracer,
    is_err=lambda r: isinstance(r, Err),
    extract_err=lambda r: r.error,
)
```

This teaches the pipeline how to detect errors in your types without importing them. If `step_b` returns an `Err`, `step_c` never runs, and `result.short_circuited` is `True`.

### Conditional steps with `skip_when`

```python
Step(
    name="cache_lookup",
    fn=lookup_cache,
)
Step(
    name="expensive_compute",
    fn=compute,
    skip_when=lambda ctx: ctx.get("cache_lookup") is not None,
)
```

When `skip_when` returns `True`, the step is recorded as a skipped span (output: `"skipped"`) and doesn't appear in `step_outputs`.

### Per-step grading

Attach a `Grader` to individual steps to score intermediate outputs:

```python
Step(
    name="summarize",
    fn=summarize,
    grader=HeuristicGrader(checks=[
        Check(name="min_length", pattern=lambda i, o: min(1.0, len(o) / 100)),
    ]),
)
```

Per-step scores appear in `result.step_scores["summarize"]`. If `config.feedback` is also set, scores are stored via `feedback.score()` with a result ID of `{trace_id}:{step_name}`.

### `map_pipeline`

Runs the same pipeline over a sequence of items. Each item gets its own `run_pipeline` call as a child span under a shared parent trace.

```python
from jig import map_pipeline

result = await map_pipeline(config, items=[doc1, doc2, doc3])

result.results       # list[PipelineResult] — one per item
result.trace_id      # str — parent trace ID
result.duration_ms   # float — total wall clock time
result.scores        # list[Score] | None — batch-level grading
```

Optional `batch_grader` grades the entire batch after all items complete:

```python
result = await map_pipeline(
    config,
    items=[doc1, doc2, doc3],
    batch_grader=my_batch_grader,
)
```

### Nested pipelines

A step can call `run_pipeline` internally to compose pipelines. Pass `_parent_span_id` to nest the child trace under the parent:

```python
async def enrich_step(ctx):
    inner_result = await run_pipeline(
        enrich_config,
        input=ctx["fetch"],
        _parent_span_id=ctx["_span_id"],
    )
    return inner_result.output
```

The inner pipeline's spans appear as children of the outer pipeline's root span. This nests indefinitely — pipelines are fractal.

## Components

### Required

Every `AgentConfig` must provide these five components. There are no defaults — you choose every piece.

#### `llm: LLMClient`

The model that powers the agent. One adapter per provider.

| Adapter | Provider | Install extra | Notes |
|---------|----------|--------------|-------|
| `AnthropicClient` | Anthropic Messages API | `anthropic` | Sonnet, Opus, Haiku |
| `OpenAIClient` | OpenAI Chat Completions | `openai` | GPT-4, GPT-5, o-series |
| `OllamaClient` | Ollama (local) | `ollama` | Any model on your hardware. Cost is always `None`. |

```python
from jig.llm import AnthropicClient

llm = AnthropicClient(
    model="claude-sonnet-4-20250514",
    api_key="sk-...",          # or set ANTHROPIC_API_KEY env var
)
```

Temperature, max_tokens, and provider-specific parameters are set per-call via `CompletionParams`, not on the client:

```python
from jig import CompletionParams, Message, Role

params = CompletionParams(
    messages=[Message(role=Role.USER, content="Hello")],
    system="You are helpful.",
    temperature=0.0,
    max_tokens=4096,
    provider_params={"top_k": 10},  # passed directly to the SDK
)
response = await llm.complete(params)
```

When using `run_agent`, these are assembled automatically — you only configure the client.

#### `memory: AgentMemory`

Where the agent stores and retrieves context across runs. Semantic search over past outputs, plus session history for conversational agents.

| Adapter | Backend | Install extra | Notes |
|---------|---------|--------------|-------|
| `LocalMemory` | SQLite + Ollama embeddings | `ollama` | Zero external dependencies. Embeddings via `nomic-embed-text`. |
| `HonchoMemory` | Honcho API | `honcho` | Sessions, metamessages, hosted collections. |
| `ZepMemory` | Zep API | `zep` | Memory search with built-in relevance scoring. |

```python
from jig.memory import LocalMemory

memory = LocalMemory(
    db_path="./data/memory.db",
    embed_model="nomic-embed-text",   # runs on your local Ollama
    ollama_host="http://localhost:11434",
)
```

#### `feedback: FeedbackLoop`

Stores quality scores against past outputs. Serves two purposes:

- **Fast loop** — `get_signals()` injects past quality signals into the prompt so the agent knows what worked before
- **Slow loop** — `export_eval_set()` produces promptfoo-compatible test cases for batch evaluation

| Adapter | Backend | Notes |
|---------|---------|-------|
| `SQLiteFeedbackLoop` | SQLite + Ollama embeddings | Uses cosine similarity to find relevant past results. |

```python
from jig.feedback import SQLiteFeedbackLoop

feedback = SQLiteFeedbackLoop(db_path="./data/feedback.db")
```

#### `tracer: TracingLogger`

Structured logging with parent-child spans. Every LLM call, tool execution, memory query, and grading step is recorded.

| Adapter | Backend | Notes |
|---------|---------|-------|
| `SQLiteTracer` | SQLite | Queryable. Use for dashboards and debugging. |
| `StdoutTracer` | Console | Pretty-prints spans with indentation. Dev only. |

```python
from jig.tracing import SQLiteTracer

tracer = SQLiteTracer(db_path="./data/traces.db")
```

#### `tools: ToolRegistry`

Holds the tools the agent can call. Tools can call anything — local functions, HTTP APIs, Rust binaries, external services. jig doesn't care what's behind the `execute()` method.

```python
from jig.tools import ToolRegistry

tools = ToolRegistry([
    farcaster_search_tool,
    web_search_tool,
    profile_lookup_tool,
])
```

An empty registry is valid if your agent doesn't use tools:

```python
tools = ToolRegistry()
```

### Optional

These components have sensible defaults (off) and can be added when you need them.

#### `grader: Grader`

**Default: `None` (no auto-grading)**

Automatically scores output after each run. In `run_agent`, scores are stored in the feedback loop and available to future runs via `get_signals()`. In `run_pipeline`, graders can be attached per-step (via `Step.grader`) or pipeline-wide (via `PipelineConfig.grader`).

Graders accept `Any` for input and output, so they work with both string-based agent outputs and typed pipeline step outputs.

| Grader | How it works | Cost |
|--------|-------------|------|
| `LLMJudge` | Separate LLM call rates the output against a rubric | API cost per run |
| `HeuristicGrader` | Regex patterns and callable checks | Free, instant |
| `GroundTruthGrader` | Compares output to a known-correct answer | Free, requires expected answer |
| `CompositeGrader` | Runs multiple graders, merges all scores | Sum of component costs |

```python
from jig.feedback import HeuristicGrader, Check

grader = HeuristicGrader(checks=[
    Check(name="has_findings", pattern=r"signal:\s*[1-5]"),
    Check(name="has_source", pattern=r"https?://"),
    Check(name="min_length", pattern=lambda i, o: min(1.0, len(o) / 200)),
])
```

Graders compose:

```python
from jig.feedback import CompositeGrader, HeuristicGrader, LLMJudge, Check
from jig.llm import AnthropicClient

grader = CompositeGrader([
    HeuristicGrader(checks=[...]),       # fast structural checks
    LLMJudge(                            # deeper quality assessment
        llm=AnthropicClient(model="claude-haiku-4-5-20251001"),
        dimensions=["relevance", "actionability"],
        rubric="Rate whether the findings would be useful to a security consultant.",
    ),
])
```

#### `system_prompt` as callable

**Default behavior: static string**

If your system prompt needs dynamic content (current time, live config, external data), pass a callable instead of a string. Can be sync or async.

```python
config = AgentConfig(
    system_prompt=lambda: f"Current time: {datetime.now().isoformat()}. You are...",
    # ...
)
```

```python
async def build_prompt():
    portfolio = await fetch_current_portfolio()
    return f"Current positions: {portfolio}. You are the decision layer..."

config = AgentConfig(
    system_prompt=build_prompt,
    # ...
)
```

#### `session_id: str`

**Default: `None` (stateless)**

Enables conversational continuity. When set, `run_agent` loads session history from memory before the LLM call and stores the exchange after.

```python
config = AgentConfig(
    session_id="scout-daily-2026-03-19",
    # ...
)
```

#### `include_memory_in_prompt: bool`

**Default: `True`**

Set to `False` to skip the memory query step. Useful for agents where past context would be noise (e.g. a stateless calculator tool).

#### `include_feedback_in_prompt: bool`

**Default: `True`**

Set to `False` to skip injecting quality signals. Useful early on when the feedback database is empty, or for agents where past scores aren't informative.

#### `max_tool_calls: int`

**Default: `10`**

Hard limit on tool executions per run. Prevents infinite loops. When hit, the agent receives "Max tool calls reached. Provide final answer." and must respond without further tool use.

## Writing tools

A tool is anything that implements `Tool`. The `definition` property provides the JSON Schema sent to the LLM. The `execute` method does the work.

```python
from jig import Tool, ToolDefinition

class FarcasterSearch(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="farcaster_search",
            description="Search Farcaster posts by keyword",
            parameters={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        )

    async def execute(self, args: dict) -> str:
        # Call whatever you want here — HTTP API, local binary, database
        results = await self.client.search(args["query"], limit=args.get("limit", 10))
        return json.dumps([r.to_dict() for r in results])
```

Tools return strings. If your tool produces structured data, serialize it — the LLM will parse it.

Tools catch their own errors. If `execute()` raises, jig wraps the exception into a `ToolResult` with the error field set and passes it back to the LLM so it can recover.

## Working with results

### `AgentResult`

```python
result = await run_agent(config, "Find agent auth discussions on Farcaster")

result.output          # str — the agent's final response
result.trace_id        # str — look up full trace in the tracer
result.duration_ms     # float — wall clock time
result.scores          # list[Score] | None — if grader was configured
result.usage           # dict:
                       #   total_input_tokens: int
                       #   total_output_tokens: int
                       #   total_cost: float (USD)
                       #   llm_calls: int
                       #   tool_calls: int
```

### `PipelineResult`

```python
result = await run_pipeline(config, input=my_input)

result.output           # Any — last step's return value
result.trace_id         # str — look up full trace in the tracer
result.step_outputs     # dict[str, Any] — every step's return value by name
result.scores           # list[Score] | None — pipeline-level grading
result.step_scores      # dict[str, list[Score]] — per-step grading
result.duration_ms      # float — wall clock time
result.short_circuited  # bool — True if is_err triggered early exit
result.error_step       # str | None — which step errored
```

### `MapResult`

```python
result = await map_pipeline(config, items=[a, b, c])

result.results       # list[PipelineResult] — one per item
result.trace_id      # str — parent trace ID
result.duration_ms   # float — total wall clock time
result.scores        # list[Score] | None — batch-level grading
```

What you do with the result is your business. jig returns it and gets out of the way.

## Eval with promptfoo

jig agents are testable via promptfoo Python providers. Export eval sets from the feedback loop, or write test cases by hand.

```yaml
# evals/scout.yaml
prompts:
  - "Find Farcaster discussions about {{topic}} from the last 24 hours"

providers:
  - id: python:agents/scout.py:run_scout_eval
    label: scout-sonnet

tests:
  - vars:
      topic: "agent authentication"
    assert:
      - type: contains
        value: "signal:"
      - type: llm-rubric
        value: "The response contains specific Farcaster posts with relevance scores"
```

The eval provider is a thin wrapper:

```python
# agents/scout.py
from jig import run_agent
from agents.scout import scout_config

async def run_scout_eval(prompt: str, options: dict) -> str:
    result = await run_agent(scout_config, prompt)
    return result.output
```

Export past runs as test cases:

```python
eval_cases = await feedback.export_eval_set(since=last_week, min_score=0.8)
# Write to YAML for promptfoo consumption
```

## Writing adapters

If you need a new LLM provider, memory backend, or tracer, implement the interface. The contract is:

1. **Receive** the native API response in whatever shape the external service returns
2. **Translate** into jig types (`LLMResponse`, `MemoryEntry`, `Span`, etc.)
3. **Return** the jig type — the rest of the framework never sees the native shape
4. **Errors** — wrap provider-specific exceptions in jig error types (`JigLLMError`, `JigMemoryError`). Set `retryable=True` for transient failures.

The adapter's job is translation, not logic. Keep them thin.
