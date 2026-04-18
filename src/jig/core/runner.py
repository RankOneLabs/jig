from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from jig.core.errors import JigLLMError
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
    ToolCall,
    ToolDefinition,
    TracingLogger,
)
from jig.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 3

# Reserved tool name the runner injects when output_schema is set. The agent
# loop terminates when the model calls this tool; its arguments are validated
# against the schema to produce AgentResult.parsed.
SUBMIT_OUTPUT_TOOL = "submit_output"


@dataclass(frozen=True, kw_only=True)
class AgentConfig[T]:
    """Immutable agent configuration.

    Keyword-only construction. Frozen so configs can be shared across
    concurrent runs without aliasing bugs. Derive variants with
    :meth:`with_` rather than mutating — the generic ``T`` is preserved.
    """

    name: str
    description: str
    system_prompt: str | Callable[[], str | Awaitable[str]]

    llm: LLMClient
    memory: AgentMemory
    feedback: FeedbackLoop
    tracer: TracingLogger
    tools: ToolRegistry

    grader: Grader[T] | None = None
    max_tool_calls: int = 10
    # Absolute cap on LLM calls per run. ``max_tool_calls`` caps tool
    # *execution*, but a model that keeps emitting tool_use blocks after
    # being told "max reached" would otherwise loop forever. This bounds
    # the outer loop so a misbehaving model can't burn unbounded spend.
    max_llm_calls: int = 50
    max_llm_retries: int = _MAX_LLM_RETRIES
    include_memory_in_prompt: bool = True
    include_feedback_in_prompt: bool = True
    session_id: str | None = None

    # --- Structured output ---
    # Pydantic model the agent should produce. When set, the runner injects a
    # ``submit_output`` tool with the model's JSON schema; the loop ends when
    # the model calls it. Invalid args trigger a retry up to
    # ``max_parse_retries`` times before the loop gives up and leaves
    # ``AgentResult.parsed`` as None.
    output_schema: type[T] | None = None
    max_parse_retries: int = 2

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AgentConfig.name must be non-empty.")
        if self.max_tool_calls <= 0:
            raise ValueError(
                f"max_tool_calls must be positive, got {self.max_tool_calls}."
            )
        if self.max_llm_calls <= 0:
            raise ValueError(
                f"max_llm_calls must be positive, got {self.max_llm_calls}."
            )
        if self.max_llm_retries <= 0:
            raise ValueError(
                f"max_llm_retries must be positive, got {self.max_llm_retries}."
            )
        if self.max_parse_retries < 0:
            raise ValueError(
                f"max_parse_retries must be non-negative, got {self.max_parse_retries}."
            )

    def with_(self, **overrides: Any) -> AgentConfig[T]:
        """Return a new config with the given fields replaced.

        Preserves the generic parameter ``T`` so typed configs stay typed
        across variants::

            base = AgentConfig[StrategyOutput](..., grader=explore_grader)
            refined = base.with_(llm=other_client, max_tool_calls=15, grader=strict_grader)

        Any unknown field names raise ``TypeError`` (from ``dataclasses.replace``);
        ``__post_init__`` validation runs on the new instance.
        """
        return dataclasses.replace(self, **overrides)


@dataclass
class AgentResult[T]:
    output: str
    trace_id: str
    usage: dict[str, Any]
    scores: list[Score] | None
    duration_ms: float
    parsed: T | None = None


def _validate_output_schema(schema: type) -> None:
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(
            f"output_schema must be a pydantic BaseModel subclass, got {schema!r}"
        )


def _build_submit_output_tool(schema: type[BaseModel]) -> ToolDefinition:
    return ToolDefinition(
        name=SUBMIT_OUTPUT_TOOL,
        description=(
            "Submit your final answer. Call this exactly once when you have "
            "your result — do not produce a free-form text response as your "
            "final answer."
        ),
        parameters=schema.model_json_schema(),
    )


def _append_schema_instruction(system_message: str) -> str:
    return (
        f"{system_message}\n\n"
        f"When you have your final answer, call the `{SUBMIT_OUTPUT_TOOL}` "
        f"tool with your result matching the provided schema. Do not produce "
        f"a free-form text response as your final answer — always finish by "
        f"calling `{SUBMIT_OUTPUT_TOOL}`."
    )


async def run_agent[T](config: AgentConfig[T], input: str) -> AgentResult[T]:
    start = time.time()

    if config.output_schema is not None:
        _validate_output_schema(config.output_schema)

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
    if config.output_schema is not None:
        system_message = _append_schema_instruction(system_message)
    messages: list[Message] = []
    if config.session_id:
        history = await config.memory.get_session(config.session_id)
        messages.extend(history)
    messages.append(Message(role=Role.USER, content=input))

    # Build the tool list — user tools + submit_output when a schema is set.
    user_tools = config.tools.list()
    extra_tools: list[ToolDefinition] = []
    if config.output_schema is not None:
        if any(t.name == SUBMIT_OUTPUT_TOOL for t in user_tools):
            raise ValueError(
                f"Tool name {SUBMIT_OUTPUT_TOOL!r} is reserved by the runner "
                f"when output_schema is set. Rename the user tool."
            )
        extra_tools.append(_build_submit_output_tool(config.output_schema))
    tools_for_llm = (user_tools + extra_tools) or None

    # 6. LLM call + tool loop
    total_usage: dict[str, Any] = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "llm_calls": 0,
        "tool_calls": 0,
    }
    tool_call_count = 0
    consecutive_llm_errors = 0
    parse_retries = 0
    final_output = ""
    parsed: T | None = None

    while True:
        if total_usage["llm_calls"] >= config.max_llm_calls:
            final_output = (
                f"[agent terminated: exceeded max_llm_calls "
                f"({config.max_llm_calls})]"
            )
            break

        llm_span = config.tracer.start_span(trace.id, SpanKind.LLM_CALL, "completion")
        params = CompletionParams(
            messages=messages,
            system=system_message,
            tools=tools_for_llm,
        )

        try:
            response = await config.llm.complete(params)
        except JigLLMError as e:
            consecutive_llm_errors += 1
            config.tracer.end_span(llm_span.id, None, error=str(e))
            logger.warning(
                "LLM call failed (%d/%d): %s",
                consecutive_llm_errors, config.max_llm_retries, e,
            )
            if consecutive_llm_errors >= config.max_llm_retries:
                final_output = f"[agent terminated: {consecutive_llm_errors} consecutive LLM errors, last: {e}]"
                break
            # Inject an error message so the model can see it on next turn
            messages.append(Message(
                role=Role.USER,
                content=f"[system: LLM call failed: {e}. Please continue.]",
            ))
            continue

        consecutive_llm_errors = 0
        config.tracer.end_span(
            llm_span.id,
            {"content": response.content[:200], "tool_calls": len(response.tool_calls or [])},
            usage=response.usage,
        )

        total_usage["total_input_tokens"] += response.usage.input_tokens
        total_usage["total_output_tokens"] += response.usage.output_tokens
        total_usage["total_cost"] += response.usage.cost or 0.0
        total_usage["llm_calls"] += 1

        # --- Structured-output termination path ---
        # When a schema is set, submit_output must be the ONLY tool call in
        # its turn. Otherwise, silently terminating on submit_output would
        # drop the other tool executions the model just requested — bugs
        # that look like "my RAG lookup result vanished."
        submit_calls: list[ToolCall] = []
        other_tool_calls: list[ToolCall] = []
        if config.output_schema is not None and response.tool_calls:
            for call in response.tool_calls:
                if call.name == SUBMIT_OUTPUT_TOOL:
                    submit_calls.append(call)
                else:
                    other_tool_calls.append(call)

        # Ambiguous turn: submit_output combined with other tool calls, or
        # emitted more than once. Ask the model to retry with submit_output
        # alone. Counts against max_parse_retries — same failure budget as
        # invalid args, since both are "model didn't follow the schema
        # contract."
        if submit_calls and (len(submit_calls) > 1 or other_tool_calls):
            parse_retries += 1
            logger.info(
                "submit_output emitted alongside other tool calls "
                "(%d/%d)", parse_retries, config.max_parse_retries,
            )
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                if call.name == SUBMIT_OUTPUT_TOOL:
                    content = (
                        f"Ambiguous turn: when calling `{SUBMIT_OUTPUT_TOOL}`,"
                        f" it must be the only tool call. Retry with "
                        f"`{SUBMIT_OUTPUT_TOOL}` alone."
                    )
                else:
                    content = (
                        f"[skipped: `{SUBMIT_OUTPUT_TOOL}` must be the only "
                        f"call in its turn. Run this tool in a separate turn "
                        f"before submitting.]"
                    )
                messages.append(
                    Message(role=Role.TOOL, content=content, tool_call_id=call.id)
                )
            if parse_retries > config.max_parse_retries:
                final_output = (
                    f"[agent terminated: model combined "
                    f"{SUBMIT_OUTPUT_TOOL} with other tool calls "
                    f"{parse_retries} times]"
                )
                break
            continue

        submit_call: ToolCall | None = submit_calls[0] if submit_calls else None

        if submit_call is not None:
            extract_span = config.tracer.start_span(
                trace.id,
                SpanKind.TOOL_CALL,
                SUBMIT_OUTPUT_TOOL,
                submit_call.arguments,
            )
            try:
                parsed = config.output_schema.model_validate(submit_call.arguments)  # type: ignore[union-attr]
            except ValidationError as ve:
                parse_retries += 1
                config.tracer.end_span(extract_span.id, None, error=str(ve))
                logger.info(
                    "submit_output validation failed (%d/%d): %s",
                    parse_retries, config.max_parse_retries, ve,
                )
                # Record the assistant's attempt and feed the validation
                # errors back so the model can correct. Do not execute any
                # other tools in this turn — the model is in retry mode.
                messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                # Every tool call in this turn must receive a response
                # (providers reject assistant-with-tool_calls followed by
                # user text without matching tool messages in between).
                for call in response.tool_calls:
                    if call.name == SUBMIT_OUTPUT_TOOL:
                        content = (
                            f"Validation failed: {ve.error_count()} error(s).\n"
                            f"{ve}\n"
                            f"Call {SUBMIT_OUTPUT_TOOL} again with corrected arguments."
                        )
                    else:
                        content = (
                            f"[skipped: submit_output retry in progress — "
                            f"please call {SUBMIT_OUTPUT_TOOL} with valid "
                            f"arguments]"
                        )
                    messages.append(
                        Message(role=Role.TOOL, content=content, tool_call_id=call.id)
                    )
                if parse_retries > config.max_parse_retries:
                    final_output = (
                        f"[agent terminated: submit_output validation failed "
                        f"{parse_retries} times, last error: {ve}]"
                    )
                    break
                continue

            # Validation succeeded — finalize.
            config.tracer.end_span(extract_span.id, submit_call.arguments)
            final_output = parsed.model_dump_json()
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            break

        # --- Plain-text termination path (no schema, or schema set but model
        # didn't call submit_output this turn). ---
        if not response.tool_calls:
            if config.output_schema is not None:
                # Model ignored the schema instruction. Nudge and retry, up
                # to max_parse_retries attempts.
                parse_retries += 1
                if parse_retries > config.max_parse_retries:
                    # Fail closed: the caller explicitly asked for a typed
                    # output, so returning the model's free-form content
                    # would make a non-compliant run look successful. Leave
                    # parsed as None and use a deterministic failure marker
                    # — same shape as the validation-error path.
                    final_output = (
                        "[agent terminated: model did not call "
                        f"{SUBMIT_OUTPUT_TOOL} within retry budget]"
                    )
                    break
                messages.append(
                    Message(role=Role.ASSISTANT, content=response.content)
                )
                messages.append(
                    Message(
                        role=Role.USER,
                        content=(
                            f"You must call the `{SUBMIT_OUTPUT_TOOL}` tool "
                            f"with your final answer. Do not respond with "
                            f"plain text."
                        ),
                    )
                )
                continue
            final_output = response.content
            break

        # --- Execute user-tool calls (no submit_output in this turn). ---
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

            # Surface tool errors to the model so it can react
            if result.error:
                content = f"[tool error: {result.error}]"
                if result.output:
                    content = f"{result.output}\n{content}"
            else:
                content = result.output

            messages.append(
                Message(role=Role.TOOL, content=content, tool_call_id=call.id)
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

    # 8. Auto-grade — pass parsed output when available, raw otherwise.
    # Graders that want the raw string even when parsed is present can read
    # it from context["raw_output"].
    scores: list[Score] | None = None
    if config.grader:
        grade_span = config.tracer.start_span(
            trace.id, SpanKind.GRADING, "auto_grade", {"input": input}
        )
        grade_output: Any = parsed if parsed is not None else final_output
        scores = await config.grader.grade(
            input, grade_output, context={"raw_output": final_output}
        )
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
        parsed=parsed,
    )
