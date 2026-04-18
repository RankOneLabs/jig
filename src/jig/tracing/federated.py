"""Federated tracer: local SQLite for writes, rollup for cross-machine reads.

Phase 9 pairs jig-side spans (written here via :class:`SQLiteTracer`)
with worker-side spans (written on smithers workers, queried via the
rollup service on willie). ``FederatedTracer`` hides that federation
from callers of :func:`jig.replay` and :func:`jig.trace_diff`: they
get one merged span list whether they ran locally or dispatched out.

Write path is **always 100% local**. Worker spans never come through
this tracer — smithers workers own their own SQLite and the rollup
aggregates them.

Read path is ``local ∪ rollup``: caller spans from the local DB,
worker spans from the rollup. Duplicates (same id) collapse with a
warning. When the rollup is unreachable, this falls back to local-only
and logs — a degraded read is almost always more useful than an error.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import httpx

from jig.core.types import Span, SpanKind, TracingLogger, Usage
from jig.tracing.sqlite import SQLiteTracer

logger = logging.getLogger(__name__)


_DEFAULT_ROLLUP_URL = "http://willie:8902"


def _span_from_row(row: dict[str, Any]) -> Span | None:
    """Convert a worker-side row dict to a :class:`Span`.

    Returns ``None`` for rows that are missing load-bearing fields — we
    log and skip rather than crash the whole read, since the rollup
    union should serve whatever we can parse.
    """
    try:
        sid = row["id"]
        trace_id = row["trace_id"]
        kind_value = row["kind"]
        name = row["name"]
        started_at = datetime.fromisoformat(row["started_at"])
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("federated: skipping malformed worker span row: %s", e)
        return None

    try:
        kind = SpanKind(kind_value)
    except ValueError:
        # Worker-origin kinds (``task_run``) aren't in jig's SpanKind.
        # Round-trip via the enum's value so comparison stays stable,
        # then fall back to the closest agent-side kind. Callers that
        # care about task spans should special-case by ``name`` instead.
        logger.debug(
            "federated: worker kind %r not in jig SpanKind; "
            "substituting TOOL_CALL for span %s",
            kind_value, sid,
        )
        kind = SpanKind.TOOL_CALL

    ended_at = (
        datetime.fromisoformat(row["ended_at"])
        if row.get("ended_at") else None
    )
    usage: Usage | None = None
    if row.get("usage_input_tokens") is not None or row.get("usage_output_tokens") is not None:
        usage = Usage(
            input_tokens=row.get("usage_input_tokens") or 0,
            output_tokens=row.get("usage_output_tokens") or 0,
            cost=row.get("usage_cost"),
        )

    return Span(
        id=sid,
        trace_id=trace_id,
        kind=kind,
        name=name,
        started_at=started_at,
        parent_id=row.get("parent_id"),
        input=_maybe_json(row.get("input")),
        output=_maybe_json(row.get("output")),
        ended_at=ended_at,
        duration_ms=row.get("duration_ms"),
        metadata=_maybe_json(row.get("metadata")),
        error=row.get("error"),
        usage=usage,
    )


def _maybe_json(value: Any) -> Any:
    """Decode a JSON string if it looks like one; otherwise pass through."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class RollupClient:
    """Thin httpx wrapper around the rollup's ``/traces/{trace_id}``.

    Returns ``list[Span]`` parsed from the aggregate response's ``spans``
    array. Connection errors raise :class:`RollupUnreachableError` so
    ``FederatedTracer.get_trace`` can decide whether to partial-read
    or propagate.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_ROLLUP_URL,
        *,
        http: httpx.AsyncClient | None = None,
        timeout: float = 5.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient()

    async def get_trace(self, trace_id: str) -> list[Span]:
        try:
            response = await self._http.get(
                f"{self._base_url}/traces/{trace_id}",
                timeout=self._timeout,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            raise RollupUnreachableError(
                f"rollup at {self._base_url!r} unreachable: {e}",
            ) from e
        except httpx.HTTPError as e:
            raise RollupUnreachableError(
                f"rollup at {self._base_url!r} request failed: {e}",
            ) from e

        if response.status_code != 200:
            raise RollupUnreachableError(
                f"rollup at {self._base_url!r} returned HTTP "
                f"{response.status_code}",
            )

        # Any shape failure from the rollup funnels through
        # RollupUnreachableError so the federated reader's fallback-to-
        # local path catches it the same way a connection error would.
        # Otherwise a malformed rollup response would bubble a raw
        # ValueError / AttributeError out of get_trace and skip the
        # "serve local-only + warn" branch.
        try:
            body = response.json()
        except ValueError as e:
            raise RollupUnreachableError(
                f"rollup at {self._base_url!r} returned malformed JSON: {e}",
            ) from e
        if not isinstance(body, dict):
            raise RollupUnreachableError(
                f"rollup at {self._base_url!r} returned non-object "
                f"payload: {type(body).__name__}",
            )
        rows = body.get("spans") or []
        spans: list[Span] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            span = _span_from_row(row)
            if span is not None:
                spans.append(span)
        return spans

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()


class RollupUnreachableError(Exception):
    """Rollup service is down or timed out.

    :meth:`FederatedTracer.get_trace` catches this and returns local-
    only spans rather than failing the whole read. Callers that *need*
    the federated view can use :class:`RollupClient` directly.
    """


class FederatedTracer(TracingLogger):
    """SQLiteTracer-backed writer with rollup-powered reads.

    Delegates every writer method to a local :class:`SQLiteTracer` —
    worker spans are never written here. On read, unions the local
    trace with whatever the rollup returns, deduping by span id and
    sorting by ``started_at``. When the rollup is unreachable, reads
    fall back to local-only with a warning log.
    """

    def __init__(
        self,
        local: SQLiteTracer,
        rollup: RollupClient | None = None,
    ) -> None:
        self._local = local
        self._rollup = rollup

    # --- write path: pure delegation ---

    def start_trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        kind: SpanKind = SpanKind.AGENT_RUN,
    ) -> Span:
        return self._local.start_trace(name, metadata, kind)

    def start_span(
        self, parent_id: str, kind: SpanKind, name: str, input: Any = None,  # noqa: A002
    ) -> Span:
        return self._local.start_span(parent_id, kind, name, input)

    def end_span(
        self,
        span_id: str,
        output: Any = None,
        error: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        return self._local.end_span(span_id, output, error, usage)

    async def flush(self) -> None:
        await self._local.flush()

    async def close(self) -> None:
        """Close both the local tracer and the rollup client.

        Rollup close errors don't propagate — the local side is the
        authority and should always get a clean shutdown.
        """
        try:
            await self._local.close()
        finally:
            if self._rollup is not None:
                try:
                    await self._rollup.close()
                except Exception:
                    logger.exception("federated: rollup close failed")

    # --- read path: union of local + rollup ---

    async def get_trace(self, trace_id: str) -> list[Span]:
        local_spans = await self._local.get_trace(trace_id)
        if self._rollup is None:
            return local_spans

        try:
            remote_spans = await self._rollup.get_trace(trace_id)
        except RollupUnreachableError as e:
            logger.warning(
                "federated: rollup unavailable, serving local-only trace: %s",
                e,
            )
            return local_spans

        merged: dict[str, Span] = {s.id: s for s in local_spans}
        for span in remote_spans:
            if span.id in merged:
                # Local wins — the jig-side caller has first-class
                # context (full input/output objects, not worker
                # metadata-only). But log the overlap; it shouldn't
                # happen in practice (local ids vs worker ids don't
                # collide statistically).
                logger.warning(
                    "federated: span id %s present in both local and rollup; "
                    "keeping local copy",
                    span.id,
                )
                continue
            merged[span.id] = span
        return sorted(merged.values(), key=lambda s: s.started_at)

    async def list_traces(
        self,
        since: datetime | None = None,
        limit: int = 50,
        name: str | None = None,
    ) -> list[Span]:
        # Root AGENT_RUN spans always live on the caller side. Worker
        # task spans aren't roots — they reparent under caller spans.
        # So ``list_traces`` is purely local and doesn't need the
        # rollup, which saves a network hop on the common "list my
        # recent runs" path.
        return await self._local.list_traces(since=since, limit=limit, name=name)


__all__ = ["FederatedTracer", "RollupClient", "RollupUnreachableError"]
