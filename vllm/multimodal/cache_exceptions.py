# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured errors for multimodal references missing their raw payload."""


class MultiModalCacheMissError(ValueError):
    """A missing reference discovered before preprocessing cache updates.

    ``identifier`` is the effective multimodal hash, including processor options,
    rather than necessarily the caller-provided UUID. The request has not been
    registered with the output processor or submitted to EngineCore.
    """

    phase: str = "input_preprocessing"
    admitted: bool = False

    def __init__(self, modality: str, index: int, identifier: str) -> None:
        self.modality = modality
        self.index = index
        self.identifier = identifier
        super().__init__(
            f"Cache miss for {modality} at index {index} but data is not provided."
        )
