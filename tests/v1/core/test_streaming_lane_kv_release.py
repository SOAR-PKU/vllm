# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only lifecycle regressions with the real scheduler and KV allocator.

The synthetic runner output acknowledges execution; no model, tokenizer, GPU
allocation, or connector is needed. Allocation before that acknowledgement must
remain live even though the scheduler advances its token counter optimistically.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config import CacheConfig, SchedulerConfig
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _initialize_block_hashes() -> None:
    init_none_hash(sha256)


def _scheduler(
    *,
    usable_blocks: int = 16,
    token_budget: int = 128,
    max_num_seqs: int = 4,
    chunk_size: int = 0,
    policy: str = "fcfs",
    prefix_caching: bool = True,
) -> Scheduler:
    """Use real scheduling/cache configs without loading a model config."""
    num_blocks = usable_blocks + 1  # The null block is not allocatable.
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        max_model_len=256,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=chunk_size,
        policy=policy,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        enable_prefix_caching=prefix_caching,
    )
    cache_config.num_gpu_blocks = num_blocks
    config = SimpleNamespace(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        model_config=SimpleNamespace(is_encoder_decoder=False, max_model_len=256),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        lora_config=None,
        kv_events_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        speculative_config=None,
    )
    registry = Mock()
    registry.supports_multimodal_inputs.return_value = False
    structured_output_manager = Mock()
    structured_output_manager.should_advance.return_value = False
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(BLOCK_SIZE, 1, 1, torch.float32, False),
            )
        ],
    )
    return Scheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=structured_output_manager,
        block_size=BLOCK_SIZE,
        mm_registry=registry,
        log_stats=False,
    )


def _lane(
    request_id: str,
    tokens: int,
    *,
    token_id: int = 1,
    arrival: float = 1.0,
    mm_features: list[MultiModalFeatureSpec] | None = None,
) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=[token_id] * tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        priority=1,
        arrival_time=arrival,
        mm_features=mm_features,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    request.start_streaming_anchor()
    return request


def _complete(scheduler: Scheduler, output: SchedulerOutput):
    req_ids = list(output.num_scheduled_tokens)
    return scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=[np.array([7], dtype=np.int32) for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _blocks(scheduler: Scheduler, lane: Request):
    return list(scheduler.kv_cache_manager.get_blocks(lane.request_id).blocks[0])


def test_idle_release_waits_for_execution_and_keeps_logical_lane() -> None:
    scheduler = _scheduler(usable_blocks=4)
    lane = _lane("lane", 64)
    scheduler.add_request(lane)

    step = scheduler.schedule()
    blocks = _blocks(scheduler, lane)
    assert len(blocks) == 4
    assert all(block.ref_cnt == 1 for block in blocks)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    assert lane in scheduler.running

    result = _complete(scheduler, step)

    assert not result  # A streaming quantum is not a client completion.
    assert scheduler.requests[lane.request_id] is lane
    assert not lane.is_finished()
    assert lane.streaming_anchor_parked
    assert lane not in scheduler.running
    assert not _blocks(scheduler, lane)
    assert all(block.ref_cnt == 0 for block in blocks)
    assert all(block.block_hash is not None for block in blocks)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 4
    assert not scheduler.has_requests()
    assert scheduler.has_unfinished_requests()
    assert lane.request_id not in scheduler.finished_req_ids
    for _ in range(3):
        empty = scheduler.schedule()
        assert not empty.num_scheduled_tokens
        assert lane.request_id not in empty.finished_req_ids


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_equal_priority_waiting_lane_progresses_with_spare_sequence_slots(
    policy: str,
) -> None:
    scheduler = _scheduler(usable_blocks=4, policy=policy)
    older = _lane("older", 64)
    newer = _lane("newer", 16, token_id=2, arrival=2.0)
    scheduler.add_request(older)
    _complete(scheduler, scheduler.schedule())
    scheduler.add_request(newer)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {newer.request_id: 16}
    assert older.streaming_anchor_parked
    assert not _blocks(scheduler, older)
    _complete(scheduler, step)
    assert not scheduler.has_requests()


def test_slot_pressure_cannot_reclaim_an_inflight_idle_snapshot() -> None:
    scheduler = _scheduler(usable_blocks=4, max_num_seqs=1)
    older = _lane("older", 64)
    newer = _lane("newer", 16, token_id=2, arrival=2.0)
    scheduler.add_request(older)
    first = scheduler.schedule()
    blocks = _blocks(scheduler, older)
    scheduler.add_request(newer)

    blocked = scheduler.schedule()

    assert not blocked.num_scheduled_tokens
    assert _blocks(scheduler, older) == blocks
    assert all(block.ref_cnt == 1 for block in blocks)
    _complete(scheduler, first)
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {newer.request_id: 16}
    _complete(scheduler, second)


def test_idle_append_resumes_cached_prefix_without_finishing_lane() -> None:
    scheduler = _scheduler(usable_blocks=6, max_num_seqs=1)
    lane = _lane("lane", 64)
    scheduler.add_request(lane)
    first = scheduler.schedule()
    initial_blocks = _blocks(scheduler, lane)
    _complete(scheduler, first)

    assert lane.append_streaming_prompt_chunk(prompt_token_ids=[2] * 16)
    assert scheduler.has_requests()
    second = scheduler.schedule()

    assert second.num_scheduled_tokens == {lane.request_id: 16}
    assert _blocks(scheduler, lane)[:4] == initial_blocks
    assert lane.request_id not in second.finished_req_ids
    assert second.scheduled_cached_reqs.resumed_req_ids == {lane.request_id}
    assert _complete(scheduler, second) == {}
    assert lane.streaming_anchor_parked
    assert not _blocks(scheduler, lane)


@pytest.mark.parametrize("is_final", [False, True])
def test_append_during_execution_is_not_lost_or_completed_early(
    is_final: bool,
) -> None:
    scheduler = _scheduler()
    lane = _lane("lane", 64)
    scheduler.add_request(lane)
    first = scheduler.schedule()
    initial_blocks = _blocks(scheduler, lane)

    lane.append_streaming_prompt_chunk(prompt_token_ids=[2] * 16, is_final=is_final)
    assert all(block.ref_cnt == 1 for block in initial_blocks)
    assert _complete(scheduler, first) == {}
    assert not lane.streaming_anchor_parked
    assert not lane.is_finished()
    assert lane.prompt_token_ids == [1] * 64 + [2] * 16
    assert scheduler.has_requests()

    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {lane.request_id: 16}
    result = _complete(scheduler, second)
    if is_final:
        assert lane.is_finished()
        assert result[0].outputs[0].request_id == lane.request_id
        assert result[0].outputs[0].finish_reason is not None
    else:
        assert not result
        assert lane.streaming_anchor_parked
        assert not _blocks(scheduler, lane)


def test_consecutive_chunks_keep_allocations_until_caught_up() -> None:
    scheduler = _scheduler(token_budget=32)
    lane = _lane("lane", 96)
    scheduler.add_request(lane)
    first = scheduler.schedule()
    initial_blocks = _blocks(scheduler, lane)
    _complete(scheduler, first)

    assert not lane.streaming_anchor_parked
    assert _blocks(scheduler, lane) == initial_blocks
    assert all(block.ref_cnt == 1 for block in initial_blocks)
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {lane.request_id: 32}
    assert _blocks(scheduler, lane)[:2] == initial_blocks
    assert not second.scheduled_cached_reqs.resumed_req_ids
    _complete(scheduler, second)
    _complete(scheduler, scheduler.schedule())

    assert lane.streaming_anchor_parked
    assert not _blocks(scheduler, lane)


def test_unselected_backlog_lane_yields_its_allocations() -> None:
    scheduler = _scheduler(token_budget=64, chunk_size=32)
    first_lane = _lane("first", 96)
    second_lane = _lane("second", 96, token_id=2, arrival=2.0)
    scheduler.add_request(first_lane)
    scheduler.add_request(second_lane)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"first": 32, "second": 32}
    _complete(scheduler, first)
    second_lane_blocks = _blocks(scheduler, second_lane)
    assert second_lane_blocks

    scheduler.max_num_scheduled_tokens = 32
    second = scheduler.schedule()

    assert second.num_scheduled_tokens == {"first": 32}
    assert not _blocks(scheduler, second_lane)
    assert all(block.ref_cnt == 0 for block in second_lane_blocks)
    assert not second_lane.streaming_anchor_parked
    assert not second_lane.is_finished()
    assert second_lane not in scheduler.running
    assert _complete(scheduler, second) == {}

    scheduler.max_num_scheduled_tokens = 64
    third = scheduler.schedule()
    assert third.num_scheduled_tokens == {"first": 32, "second": 32}
    assert second_lane.request_id in third.scheduled_cached_reqs.resumed_req_ids
    assert _blocks(scheduler, second_lane)[:2] == second_lane_blocks
    assert _complete(scheduler, third) == {}
    fourth = scheduler.schedule()
    assert fourth.num_scheduled_tokens == {"second": 32}
    assert _complete(scheduler, fourth) == {}
    assert not scheduler.has_requests()


def test_idle_release_only_drops_the_lanes_own_shared_block_refs() -> None:
    scheduler = _scheduler(usable_blocks=8)
    lane = _lane("lane", 64)
    scheduler.add_request(lane)
    step = scheduler.schedule()
    lane_blocks = _blocks(scheduler, lane)

    # Model an independent prefix consumer already holding the same blocks.
    consumer = _lane("consumer", 65)
    manager = scheduler.kv_cache_manager
    cached_blocks, cached_tokens = manager.get_computed_blocks(consumer)
    assert cached_tokens == 64
    allocation = manager.allocate_slots(
        consumer,
        1,
        num_new_computed_tokens=cached_tokens,
        new_computed_blocks=cached_blocks,
    )
    assert allocation is not None
    assert all(block.ref_cnt == 2 for block in lane_blocks)

    _complete(scheduler, step)

    assert not _blocks(scheduler, lane)
    assert all(block.ref_cnt == 1 for block in lane_blocks)
    assert _blocks(scheduler, consumer)[:4] == lane_blocks
    manager.free(consumer)
    assert all(block.ref_cnt == 0 for block in lane_blocks)


def test_released_prefix_can_be_overwritten_then_recomputed_on_append() -> None:
    scheduler = _scheduler(usable_blocks=4)
    lane = _lane("lane", 32)
    other = _lane("other", 64, token_id=2, arrival=2.0)
    scheduler.add_request(lane)
    _complete(scheduler, scheduler.schedule())
    scheduler.add_request(other)
    _complete(scheduler, scheduler.schedule())

    lane.append_streaming_prompt_chunk(prompt_token_ids=[3] * 16)
    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {lane.request_id: 48}
    assert _complete(scheduler, step) == {}
    assert lane.streaming_anchor_parked
    assert other.streaming_anchor_parked
    assert not scheduler.has_requests()


def test_idle_append_without_prefix_cache_recomputes_safely() -> None:
    scheduler = _scheduler(prefix_caching=False)
    lane = _lane("lane", 32)
    scheduler.add_request(lane)
    _complete(scheduler, scheduler.schedule())
    assert not _blocks(scheduler, lane)

    lane.append_streaming_prompt_chunk(prompt_token_ids=[2] * 16)
    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {lane.request_id: 48}
    assert _complete(scheduler, step) == {}
    assert lane.streaming_anchor_parked


@pytest.mark.parametrize("with_encoder", [False, True])
@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_resource_pressure_preserves_selected_work_and_inflight_lane(
    with_encoder: bool,
    policy: str,
) -> None:
    scheduler = _scheduler(
        usable_blocks=3, token_budget=32, chunk_size=16, policy=policy
    )
    features = None
    if with_encoder:
        features = [
            MultiModalFeatureSpec(
                data=None,
                modality="audio",
                identifier="first-audio",
                mm_position=PlaceholderRange(offset=16, length=16),
            )
        ]
        scheduler.max_num_encoder_input_tokens = 16
        scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    first = _lane("first", 64, mm_features=features)
    second = _lane("second", 64, token_id=2, arrival=2.0)
    scheduler.add_request(first)
    scheduler.add_request(second)
    initial = scheduler.schedule()
    assert initial.num_scheduled_tokens == {"first": 16, "second": 16}
    _complete(scheduler, initial)

    # Represent an older worker snapshot that still protects the later lane.
    # A's next block fits. B's growth cannot revoke A's selected work, and
    # cannot reclaim B's own blocks while its previous execution is in flight.
    second.streaming_anchor_inflight_steps = 1
    second_blocks = _blocks(scheduler, second)
    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"first": 16}
    assert step.total_num_scheduled_tokens == 16
    assert step.scheduled_cached_reqs.req_ids == ["first"]
    assert step.scheduled_encoder_inputs == ({"first": [0]} if with_encoder else {})
    assert first in scheduler.running
    assert first not in scheduler.waiting
    first_blocks = _blocks(scheduler, first)
    assert len(first_blocks) == 2
    assert all(block.ref_cnt == 1 for block in first_blocks)
    assert first.num_computed_tokens == 32
    assert second.num_computed_tokens == 16
    assert _blocks(scheduler, second) == second_blocks
    assert all(block.ref_cnt == 1 for block in second_blocks)
    assert second.streaming_anchor_inflight_steps == 1
    first_block_ids = {block.block_id for block in first_blocks}
    for groups in step.scheduled_cached_reqs.new_block_ids:
        assert groups is not None
        assert {block_id for group in groups for block_id in group} <= first_block_ids

    assert _complete(scheduler, step) == {}
    assert first.streaming_anchor_confirmed_tokens == 32
    assert _blocks(scheduler, first) == first_blocks
    assert second.streaming_anchor_inflight_steps == 1
    if with_encoder:
        # A actually executed its encoder work before this reuse check.
        # Its now-reusable output must not have been canceled as a reservation.
        encoder_inputs, _, _, _ = scheduler._try_schedule_encoder_inputs(
            first, 16, 16, 16
        )
        assert encoder_inputs == []
        assert first.request_id in scheduler.encoder_cache_manager.cached["first-audio"]


def test_unsupported_connector_rejects_lane_before_admission_mutates_state() -> None:
    scheduler = _scheduler()
    scheduler.connector = SimpleNamespace()
    scheduler._streaming_lane_kv_release_enabled = False
    lane = _lane("lane", 32)

    with pytest.raises(ValueError, match="does not guarantee step-safe release"):
        scheduler.add_request(lane)

    assert lane.request_id not in scheduler.requests
    assert not scheduler.waiting
    assert not scheduler.running
    assert not _blocks(scheduler, lane)
