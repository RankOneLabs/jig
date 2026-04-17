class JigError(Exception):
    pass


class JigLLMError(JigError):
    def __init__(
        self,
        message: str,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class JigMemoryError(JigError):
    pass


class JigToolError(JigError):
    pass


class JigBudgetError(JigError):
    """Raised when a :class:`BudgetTracker` observes spend exceeding its limit."""

    def __init__(self, message: str, spent_usd: float, limit_usd: float):
        super().__init__(message)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
