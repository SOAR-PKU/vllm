# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from bisect import bisect_right
from collections.abc import Callable, Mapping
from copy import copy
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
    PrefillContextMetadata,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        prefill_context: PrefillContextMetadata | None = None,
        copy_prompt_token_ids_to_all: bool = True,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        if self.prompt_token_ids is None:
            self._all_token_ids = [0] * self.num_prompt_tokens
            self._all_token_ids_shares_prompt = False
        elif copy_prompt_token_ids_to_all:
            self._all_token_ids = self.prompt_token_ids.copy()
            self._all_token_ids_shares_prompt = False
        else:
            # Streaming logical Requests are immutable coordinator inputs.
            # Sharing avoids one full cumulative-prefix copy. If such a
            # Request is scheduled by the direct baseline, the first generated
            # token detaches this list before mutation.
            self._all_token_ids = self.prompt_token_ids
            self._all_token_ids_shares_prompt = True
        self.num_output_placeholders = 0  # Used in async scheduling.
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.num_encoder_inputs = len(self.mm_features)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        self.prefill_context = prefill_context
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1
        self.local_cached_tokens = 0
        self.lmcache_hit_tokens = 0
        self.lmcache_total_prompt_tokens = 0
        self.lmcache_need_to_load_tokens = 0
        self.lmcache_hit_rate = 0.0

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of requests being preempted by the scheduler
        self.num_preemptions = 0

        # Interrupt offload / reschedule state. These fields are intentionally
        # scheduler-owned and preserve continuation metadata across LMCache
        # offload, verdict wait, and resume admission.
        self.interrupt_seq: int | None = None
        self.suspend_reason: str | None = None
        self.interrupt_work_kind: str | None = None
        self.interrupt_ready_ts: float = 0.0
        self.preemptive_admit_eligible: bool = False
        self.recovery_protected_until_step: int = 0
        self.recovery_protected_until_ts: float = 0.0
        self.saved_num_computed_tokens: int = 0
        self.saved_stream_seq: int = 0
        self.saved_output_token_count: int = 0
        self.full_continuation_token_ids: list[int] = []
        self.lmcache_store_token_ids: list[int] | None = None
        self.lmcache_lookup_token_ids: list[int] | None = None
        self.offloaded_restore_pending: bool = False
        self.remote_kv_origin_interrupt: bool = False
        self.remote_kv_origin_status: "RequestStatus | None" = None

        self._block_hasher = block_hasher
        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            prefill_context=request.prefill_context,
            copy_prompt_token_ids_to_all=not (
                request.prefill_context is not None
                and request.prefill_context.kind == "streaming"
            ),
        )

    @classmethod
    def from_request_prefix(
        cls,
        source: "Request",
        *,
        request_id: str,
        target_num_prompt_tokens: int,
        block_size: int,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        prefill_context: PrefillContextMetadata | None = None,
        mm_features_are_validated_prefix: bool = False,
    ) -> "Request":
        """Create an independently mutable request for a prompt prefix.

        The returned request reuses the source request's hashes for complete
        prompt blocks. It still owns separate prompt/all-token lists and a
        separate block-hash list because generated tokens may extend both
        ``_all_token_ids`` and ``block_hashes``.

        This factory is intended for internal requests derived from an already
        validated prefix of ``source``. It rejects targets that split a
        multimodal placeholder and validates any caller-prepared feature list
        using only the fields that affect prefix identity.
        """

        prompt_token_ids = source.prompt_token_ids
        if prompt_token_ids is None or source.prompt_embeds is not None:
            raise ValueError(
                "Request prefix reuse requires prompt token IDs without "
                "prompt embeddings"
            )
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0 < target_num_prompt_tokens <= source.num_prompt_tokens:
            raise ValueError(
                f"target_num_prompt_tokens must be in [1, {source.num_prompt_tokens}]"
            )

        prefix_feature_count = bisect_right(
            source.mm_features,
            target_num_prompt_tokens,
            key=lambda feature: (
                feature.mm_position.offset + feature.mm_position.length
            ),
        )
        if prefix_feature_count < len(source.mm_features):
            next_feature = source.mm_features[prefix_feature_count]
            position = next_feature.mm_position
            if (
                position.offset
                < target_num_prompt_tokens
                < position.offset + position.length
            ):
                raise ValueError(
                    "target_num_prompt_tokens cannot split a multimodal "
                    f"feature: identifier={next_feature.identifier!r}, "
                    f"offset={position.offset}, length={position.length}"
                )

        if mm_features is None:
            source_prefix_features = source.mm_features[
                :prefix_feature_count
            ]
            prefix_features = [copy(feature) for feature in source_prefix_features]
        elif mm_features_are_validated_prefix:
            prefix_features = mm_features
        else:
            source_prefix_features = source.mm_features[
                :prefix_feature_count
            ]
            prefix_features = mm_features
            expected_identity = cls._mm_feature_identity(source_prefix_features)
            actual_identity = cls._mm_feature_identity(prefix_features)
            if actual_identity != expected_identity:
                raise ValueError(
                    "mm_features must preserve the source prefix's modality, "
                    "identifier, offset, length, and order"
                )

        prefix_request = cls(
            request_id=request_id,
            client_index=source.client_index,
            prompt_token_ids=prompt_token_ids[:target_num_prompt_tokens],
            prompt_embeds=None,
            mm_features=prefix_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            eos_token_id=source.eos_token_id,
            arrival_time=source.arrival_time,
            lora_request=source.lora_request,
            cache_salt=source.cache_salt,
            priority=source.priority,
            trace_headers=source.trace_headers,
            block_hasher=None,
            prefill_context=prefill_context,
            copy_prompt_token_ids_to_all=False,
        )

        block_hasher = source._block_hasher
        if block_hasher is None:
            return prefix_request

        expected_source_blocks = source.num_tokens // block_size
        if len(source.block_hashes) != expected_source_blocks:
            raise ValueError(
                "block_size does not match the source request's block-hash "
                f"state: expected={expected_source_blocks}, "
                f"available={len(source.block_hashes)}"
            )

        num_full_blocks = target_num_prompt_tokens // block_size
        prefix_request._block_hasher = block_hasher
        prefix_request.block_hashes = source.block_hashes[:num_full_blocks]
        prefix_request.get_hash_new_full_blocks = partial(block_hasher, prefix_request)
        return prefix_request

    @staticmethod
    def _mm_feature_identity(
        features: list[MultiModalFeatureSpec],
    ) -> tuple[tuple[str, str, int, int], ...]:
        return tuple(
            (
                feature.modality,
                feature.identifier,
                feature.mm_position.offset,
                feature.mm_position.length,
            )
            for feature in features
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if self._all_token_ids_shares_prompt:
            self._all_token_ids = self._all_token_ids.copy()
            self.all_token_ids = ConstantList(self._all_token_ids)
            self._all_token_ids_shares_prompt = False
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        num_tokens = self.mm_features[input_id].mm_position.length
        return num_tokens

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    INTERRUPT_PREEMPT_REQUESTED = enum.auto()
    OFFLOADING_TO_LMCACHE = enum.auto()
    PENDING_INTERRUPT_VERDICT = enum.auto()
    Q_INTERRUPT_WAITING = enum.auto()
    Q_NORMAL_WAITING = enum.auto()
    RECOVERY_PROTECTED = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ABORTED_BY_TRUE_INTERRUPT = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_ABORTED_BY_TRUE_INTERRUPT: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
