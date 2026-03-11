# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(
        self,
        detail: str | None = None,
        *args,
        suppress_context: bool = False,
        **kwargs,
    ):
        base_message = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501
        self.detail = detail
        if detail:
            message = f"{base_message}\nEngineCore fatal detail:\n{detail}"
        else:
            message = base_message

        super().__init__(message, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
