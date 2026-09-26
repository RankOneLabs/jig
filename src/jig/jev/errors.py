"""Structured failures for an in-flight Jev request."""

from typing import Literal, TypeAlias

from jig.core.errors import JigError

JevErrorKind: TypeAlias = Literal[
    "auth", "invalid_request", "rate_limited", "overloaded", "timeout",
    "transport", "invalid_response",
]

# Maximum characters retained from untrusted error detail, including in str(error).
JEV_ERROR_DETAIL_MAX_LENGTH = 512


class JevError(JigError):
    def __init__(
        self,
        kind: JevErrorKind,
        status_code: int | None,
        call_id: str,
        provider_request_id: str | None,
        detail: str,
        attempts: int,
        elapsed_ms: float,
    ) -> None:
        self.kind = kind
        self.status_code = status_code
        self.call_id = call_id
        self.provider_request_id = provider_request_id
        self.detail = detail[:JEV_ERROR_DETAIL_MAX_LENGTH]
        self.attempts = attempts
        self.elapsed_ms = elapsed_ms
        super().__init__(
            f"Jev {kind} (status={status_code}, call_id={call_id}): {self.detail}"
        )
