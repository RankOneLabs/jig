from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from jig.core.prompt import build_system_message
from jig.core.types import (
    AgentMemory,
    CompletionParams,
    FeedbackLoop,
    Grader,
    LLMClient,
    MemoryEntry,
    Message,
    Role,
    Score,
    ScoredResult,
    SpanKind,
    TracingLogger,
)
from jig.tools.registry import ToolRegistry


@dataclass
class AgentConfig:
    name: str
    description: str
    system_prompt: str | Callable[[], str | Awaitable[str]]

    llm: LLMClient
    memory: AgentMemory
    feedback: FeedbackLoop
    tracer: TracingLogger
    tools: ToolRegistry

    grader: Grader | None = None
    max_tool_calls: int = 10
    include_memory_in_prompt: bool = True
    include_feedback_in_prompt: bool = True
    session_id: str | None = None


@dataclass
class AgentResult:
    output: str
    trace_id: str
    usage: dict[str, Any]
    scores: list[Score] | None
    duration_ms: float


async def run_agent(config: AgentConfig, input: str) -> AgentResult:
    start = time.time()

    # 1. Start trace
    trace = config.tracer.start_trace(config.name, {"input": input}, kind=SpanKind.AGENT_RUN)

    # 2. Resolve system prompt
    if callable(config.system_prompt):
        result = config.system_prompt()
        system_prompt = await result if hasattr(result, "__await__") else result
    else:
        system_prompt = config.system_prompt

    # 3. Query memory
    memory_context: list[MemoryEntry] = []
    if config.include_memory_in_prompt:
        mem_span = config.tracer.start_span(
            trace.id, SpanKind.MEMORY_QUERY, "query_memory", {"query": input}
        )
        memory_context = await config.memory.query(input, limit=5, session_id=config.session_id)
        config.tracer.end_span(mem_span.id, [e.content for e in memory_context])

    # 4. Query feedback signals
    feedback_signals: list[ScoredResult] = []
    if config.include_feedback_in_prompt:
        fb_span = config.tracer.start_span(
            trace.id, SpanKind.MEMORY_QUERY, "query_feedback", {"query": input}
        )
        feedback_signals = await config.feedback.get_signals(input, limit=3, min_score=0.7)
        config.tracer.end_span(fb_span.id, [s.content[:100] for s in feedback_signals])

    # 5. Assemble messages (system prompt is separate, not in messages list)
    system_message = build_system_message(system_prompt, memory_context, feedback_signals)
    messages: list[Message] = []
    if config.session_id:
        history = await config.memory.get_session(config.session_id)
        messages.extend(history)
    messages.append(Message(role=Role.USER, content=input))

    # 6. LLM call + tool loop
    total_usage: dict[str, Any] = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "llm_calls": 0,
        "tool_calls": 0,
    }
    tool_call_count = 0
    final_output = ""

    while True:
        llm_span = config.tracer.start_span(trace.id, SpanKind.LLM_CALL, "completion")
        params = CompletionParams(
            messages=messages,
            system=system_message,
            tools=config.tools.list() or None,
        )
        response = await config.llm.complete(params)
        config.tracer.end_span(
            llm_span.id,
            {"content": response.content[:200], "tool_calls": len(response.tool_calls or [])},
        )

        total_usage["total_input_tokens"] += response.usage.input_tokens
        total_usage["total_output_tokens"] += response.usage.output_tokens
        total_usage["total_cost"] += response.usage.cost or 0.0
        total_usage["llm_calls"] += 1

        if not response.tool_calls:
            final_output = response.content
            break

        # Execute tool calls
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
        )

        for call in response.tool_calls:
            if tool_call_count >= config.max_tool_calls:
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content="Max tool calls reached. Provide final answer.",
                        tool_call_id=call.id,
                    )
                )
                continue

            tool_span = config.tracer.start_span(
                trace.id, SpanKind.TOOL_CALL, call.name, call.arguments
            )
            result = await config.tools.execute(call)
            config.tracer.end_span(tool_span.id, result.output[:500], error=result.error)

            messages.append(
                Message(role=Role.TOOL, content=result.output, tool_call_id=call.id)
            )
            tool_call_count += 1
            total_usage["tool_calls"] += 1

    # 7. Store in memory
    result_id = await config.memory.add(
        final_output,
        {"agent": config.name, "input": input, "trace_id": trace.trace_id},
    )

    if config.session_id:
        await config.memory.add_to_session(
            config.session_id, Message(role=Role.USER, content=input)
        )
        await config.memory.add_to_session(
            config.session_id, Message(role=Role.ASSISTANT, content=final_output)
        )

    # 8. Auto-grade
    scores: list[Score] | None = None
    if config.grader:
        grade_span = config.tracer.start_span(
            trace.id, SpanKind.GRADING, "auto_grade", {"input": input}
        )
        scores = await config.grader.grade(input, final_output)
        await config.feedback.score(result_id, scores)
        config.tracer.end_span(
            grade_span.id, [{"dim": s.dimension, "val": s.value} for s in scores]
        )

    # 9. Close trace
    duration = (time.time() - start) * 1000
    config.tracer.end_span(trace.id, {"output": final_output[:200], "scores": scores})
    await config.tracer.flush()

    return AgentResult(
        output=final_output,
        trace_id=trace.trace_id,
        usage=total_usage,
        scores=scores,
        duration_ms=duration,
    )
