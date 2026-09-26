"""Bridge Jev usage into jig trace spans."""

from jig.core.types import Usage
from jig.jev.models import JevUsage


def to_jig_usage(usage: JevUsage) -> Usage:
    """Convert provider token counts for a jig span, without assigning a price.

    To record a Jev call, start a span with ``SpanKind.PROVIDER_CALL`` and a
    name such as ``"jev.call"`` before invoking the client. On success, put
    ``result.call_id``, ``result.provider_request_id``, the resolved
    ``result.model``, ``result.latency_ms``, and ``"ok"`` status in its
    metadata; end the span with ``usage=to_jig_usage(result.usage)``. On
    failure, record ``"error"`` status and the available call ID, provider
    request ID, and elapsed time from the exception; end the span with its
    error text. Keep the provider usage separate from jig's priced LLM usage.
    """
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost=None,
    )
