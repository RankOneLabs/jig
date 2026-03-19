# jig

Minimal agent framework. Five interfaces, one run function, everything swappable.

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

## How it works

`run_agent()` is the entire execution model:

1. Start trace
2. Resolve system prompt (string or async callable)
3. Query memory for relevant context
4. Query feedback for quality signals from past runs
5. Assemble `CompletionParams` (system separate from messages)
6. LLM call + tool execution loop
7. Store output in memory
8. Auto-grade if grader configured
9. Close trace, return `AgentResult`

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
├── core/           # types, runner, errors, retry, prompt builder
├── llm/            # anthropic, openai, ollama adapters
├── memory/         # local (sqlite+embeddings), honcho, zep
├── feedback/       # feedback loop, llm judge, heuristic, ground truth, composite
├── tracing/        # sqlite tracer, stdout tracer
└── tools/          # registry + common tools
```
