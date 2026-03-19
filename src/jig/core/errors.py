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
