# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regressions for demand-driven parked-lane KV reclamation.

These tests use the real scheduler, prefix cache, block references, and synthetic
worker acknowledgements. Scheduling a snapshot is not an execution completion.
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
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

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
    priority: int = 1,
    arrival: float = 1.0,
    mm_features: list[MultiModalFeatureSpec] | None = None,
    streaming: bool = True,
) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=[token_id] * tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        priority=priority,
        arrival_time=arrival,
        mm_features=mm_features,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    if streaming:
        request.start_streaming_anchor()
    return request


def _complete(
    scheduler: Scheduler,
    output: SchedulerOutput,
    *,
    kv_connector_output: KVConnectorOutput | None = None,
):
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
            kv_connector_output=kv_connector_output,
        ),
    )


def _blocks(scheduler: Scheduler, lane: Request):
    return list(scheduler.kv_cache_manager.get_blocks(lane.request_id).blocks[0])


def _run_lane(scheduler: Scheduler, lane: Request) -> None:
    scheduler.add_request(lane)
    step = scheduler.schedule()
    assert lane.request_id in step.num_scheduled_tokens
    assert _complete(scheduler, step) == {}
    assert lane.streaming_anchor_parked


def test_no_pressure_keeps_allocation_across_idle_steps_and_appends() -> None:
    scheduler = _scheduler(usable_blocks=8)
    lane = _lane("lane", 64)
    _run_lane(scheduler, lane)
    original_blocks = _blocks(scheduler, lane)

    assert len(original_blocks) == 4
    assert all(block.ref_cnt == 1 for block in original_blocks)
    assert lane in scheduler.running
    assert lane.streaming_anchor_confirmed_tokens == 64
    assert not scheduler.has_requests()
    assert scheduler.has_unfinished_requests()
    for _ in range(3):
        assert scheduler.schedule().num_scheduled_tokens == {}
        assert _blocks(scheduler, lane) == original_blocks

    for target in (80, 96):
        lane.append_streaming_prompt_chunk(prompt_token_ids=[2] * 16)
        step = scheduler.schedule()
        assert step.num_scheduled_tokens == {"lane": 16}
        assert not step.scheduled_cached_reqs.resumed_req_ids
        assert _blocks(scheduler, lane)[:4] == original_blocks
        assert _complete(scheduler, step) == {}
        assert lane.streaming_anchor_confirmed_tokens == target
        assert lane.streaming_anchor_parked
    assert lane.num_preemptions == 0
    assert lane.num_streaming_kv_releases == 0


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
@pytest.mark.parametrize("idle_priority", [0, 1, 2])
def test_waiting_admission_reclaims_idle_lane_with_spare_sequence_slots(
    policy: str, idle_priority: int
) -> None:
    scheduler = _scheduler(usable_blocks=4, policy=policy)
    idle = _lane("idle", 64, priority=idle_priority)
    waiter = _lane("waiter", 16, token_id=2, arrival=2.0)
    _run_lane(scheduler, idle)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    scheduler.add_request(waiter)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"waiter": 16}
    assert idle.status == RequestStatus.PREEMPTED
    assert idle.streaming_anchor_parked
    assert idle.streaming_anchor_confirmed_tokens == 64
    assert idle.num_computed_tokens == 0
    assert idle.num_streaming_kv_releases == 1
    assert not _blocks(scheduler, idle)
    assert idle in scheduler.waiting
    assert _complete(scheduler, step) == {}
    for _ in range(3):
        assert scheduler.schedule().num_scheduled_tokens == {}
        assert not _blocks(scheduler, idle)
    assert not scheduler.has_requests()
    assert not idle.is_finished()


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_sequence_slot_pressure_reclaims_completed_higher_priority_idle_lane(
    policy: str,
) -> None:
    scheduler = _scheduler(usable_blocks=8, max_num_seqs=1, policy=policy)
    idle = _lane("idle", 32, priority=0)
    waiter = _lane("waiter", 16, priority=2, token_id=2, arrival=2.0)
    _run_lane(scheduler, idle)
    scheduler.add_request(waiter)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"waiter": 16}
    assert idle.streaming_anchor_parked
    assert not _blocks(scheduler, idle)
    assert _complete(scheduler, step) == {}
    assert scheduler.schedule().num_scheduled_tokens == {}


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_running_growth_reclaims_idle_lane_before_active_target(
    policy: str,
) -> None:
    scheduler = _scheduler(usable_blocks=4, policy=policy)
    idle = _lane("idle", 48, priority=0)
    active = _lane("active", 16, priority=2, token_id=2, arrival=2.0)
    _run_lane(scheduler, idle)
    _run_lane(scheduler, active)
    active_blocks = _blocks(scheduler, active)
    active.append_streaming_prompt_chunk(prompt_token_ids=[2] * 16)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"active": 16}
    assert _blocks(scheduler, active)[:1] == active_blocks
    assert active.num_preemptions == 0
    assert idle.streaming_anchor_parked
    assert not _blocks(scheduler, idle)
    assert _complete(scheduler, step) == {}
    assert active.streaming_anchor_confirmed_tokens == 32


@pytest.mark.parametrize("max_num_seqs", [1, 4])
@pytest.mark.parametrize("waiter_priority", [0, 1])
def test_optimistically_parked_inflight_snapshot_cannot_be_reclaimed(
    max_num_seqs: int, waiter_priority: int
) -> None:
    scheduler = _scheduler(
        usable_blocks=4, max_num_seqs=max_num_seqs, policy="priority"
    )
    idle = _lane("idle", 64)
    waiter = _lane(
        "waiter", 16, priority=waiter_priority, token_id=2, arrival=2.0
    )
    scheduler.add_request(idle)
    first = scheduler.schedule()
    blocks = _blocks(scheduler, idle)
    assert idle.streaming_anchor_parked
    assert idle.streaming_anchor_inflight_steps == 1
    scheduler.add_request(waiter)

    blocked = scheduler.schedule()

    assert blocked.num_scheduled_tokens == {}
    assert _blocks(scheduler, idle) == blocks
    assert all(block.ref_cnt == 1 for block in blocks)
    assert idle.streaming_anchor_inflight_steps == 1
    assert _complete(scheduler, first) == {}
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"waiter": 16}
    assert _complete(scheduler, second) == {}


@pytest.mark.parametrize("protection", ["step", "time", "restore"])
def test_recovery_protected_parked_lane_is_not_a_pressure_victim(
    protection: str,
) -> None:
    scheduler = _scheduler(usable_blocks=4, policy="priority")
    idle = _lane("idle", 64)
    waiter = _lane("waiter", 16, priority=0, token_id=2, arrival=2.0)
    _run_lane(scheduler, idle)
    blocks = _blocks(scheduler, idle)
    if protection == "step":
        idle.recovery_protected_until_step = scheduler._interrupt_scheduler_step + 10
    elif protection == "time":
        idle.recovery_protected_until_ts = float("inf")
    else:
        idle.offloaded_restore_pending = True
    scheduler.add_request(waiter)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {}
    assert _blocks(scheduler, idle) == blocks
    assert all(block.ref_cnt == 1 for block in blocks)
    assert idle in scheduler.running


def test_shared_prefix_requires_multiple_idle_victims_before_admission() -> None:
    scheduler = _scheduler(usable_blocks=5)
    first = _lane("first", 64)
    second = _lane("second", 65, arrival=2.0)
    waiter = _lane("waiter", 32, token_id=3, arrival=3.0)
    _run_lane(scheduler, first)
    shared_blocks = _blocks(scheduler, first)
    _run_lane(scheduler, second)
    assert _blocks(scheduler, second)[:4] == shared_blocks
    assert all(block.ref_cnt == 2 for block in shared_blocks)
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 0
    scheduler.add_request(waiter)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"waiter": 32}
    assert not _blocks(scheduler, first)
    assert not _blocks(scheduler, second)
    assert first.streaming_anchor_parked
    assert second.streaming_anchor_parked
    assert first.num_streaming_kv_releases == 1
    assert second.num_streaming_kv_releases == 1
    assert len(_blocks(scheduler, waiter)) == 2
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 3
    assert _complete(scheduler, step) == {}


@pytest.mark.parametrize("max_num_seqs", [1, 4])
def test_encoder_budget_exhaustion_does_not_release_idle_allocations(
    max_num_seqs: int,
) -> None:
    scheduler = _scheduler(usable_blocks=4, max_num_seqs=max_num_seqs)
    idle = _lane("idle", 64)
    _run_lane(scheduler, idle)
    blocks = _blocks(scheduler, idle)
    scheduler.max_num_encoder_input_tokens = 0
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    waiter = _lane(
        "waiter",
        16,
        token_id=2,
        arrival=2.0,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="audio",
                identifier="waiter-audio",
                mm_position=PlaceholderRange(offset=0, length=16),
            )
        ],
    )
    scheduler.add_request(waiter)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {}
    assert _blocks(scheduler, idle) == blocks
    assert all(block.ref_cnt == 1 for block in blocks)


def test_exhausted_token_budget_does_not_release_other_idle_lane() -> None:
    scheduler = _scheduler(usable_blocks=6, token_budget=32)
    idle = _lane("idle", 32, priority=0)
    active = _lane("active", 32, token_id=2, arrival=2.0)
    waiter = _lane("waiter", 16, token_id=3, arrival=3.0)
    _run_lane(scheduler, idle)
    _run_lane(scheduler, active)
    blocks = _blocks(scheduler, idle)
    active.append_streaming_prompt_chunk(prompt_token_ids=[2] * 32)
    scheduler.add_request(waiter)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"active": 32}
    assert _blocks(scheduler, idle) == blocks
    assert all(block.ref_cnt == 1 for block in blocks)
    assert not _blocks(scheduler, waiter)
    assert _complete(scheduler, step) == {}


def test_unselected_backlog_retains_kv_without_allocation_pressure() -> None:
    scheduler = _scheduler(token_budget=64, chunk_size=32)
    first = _lane("first", 96)
    second = _lane("second", 96, token_id=2, arrival=2.0)
    scheduler.add_request(first)
    scheduler.add_request(second)
    initial = scheduler.schedule()
    assert initial.num_scheduled_tokens == {"first": 32, "second": 32}
    assert _complete(scheduler, initial) == {}
    second_blocks = _blocks(scheduler, second)
    scheduler.max_num_scheduled_tokens = 32

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"first": 32}
    assert _blocks(scheduler, second) == second_blocks
    assert second in scheduler.running
    assert second.num_preemptions == 0
    assert _complete(scheduler, step) == {}


def test_slot_reclaimed_lane_reuses_unmodified_local_prefix_on_append() -> None:
    scheduler = _scheduler(usable_blocks=8, max_num_seqs=1)
    first = _lane("first", 64)
    second = _lane("second", 16, token_id=2, arrival=2.0)
    _run_lane(scheduler, first)
    initial_blocks = _blocks(scheduler, first)
    _run_lane(scheduler, second)
    assert not _blocks(scheduler, first)
    assert all(block.ref_cnt == 0 for block in initial_blocks)
    assert first.streaming_anchor_confirmed_tokens == 64
    first.append_streaming_prompt_chunk(prompt_token_ids=[3] * 16)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"first": 16}
    assert step.scheduled_cached_reqs.resumed_req_ids == {"first"}
    assert _blocks(scheduler, first)[:4] == initial_blocks
    assert first.streaming_anchor_confirmed_tokens == 64
    assert _complete(scheduler, step) == {}
    assert first.streaming_anchor_confirmed_tokens == 80
    assert first.streaming_anchor_parked
    assert first in scheduler.running


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_overwritten_prefix_recomputes_without_rewinding_logical_completion(
    prefix_caching: bool,
) -> None:
    scheduler = _scheduler(usable_blocks=4, prefix_caching=prefix_caching)
    first = _lane("first", 32)
    second = _lane("second", 64, token_id=2, arrival=2.0)
    _run_lane(scheduler, first)
    _run_lane(scheduler, second)
    assert not _blocks(scheduler, first)
    assert first.streaming_anchor_confirmed_tokens == 32
    first.append_streaming_prompt_chunk(prompt_token_ids=[3] * 16)

    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"first": 48}
    assert first.streaming_anchor_confirmed_tokens == 32
    assert second.streaming_anchor_parked
    assert not _blocks(scheduler, second)
    assert _complete(scheduler, step) == {}
    assert first.streaming_anchor_confirmed_tokens == 48
    assert not first.is_finished()


def test_recomputation_chunks_do_not_rewind_confirmed_frontier() -> None:
    scheduler = _scheduler(usable_blocks=4)
    first = _lane("first", 32)
    second = _lane("second", 64, token_id=2, arrival=2.0)
    _run_lane(scheduler, first)
    _run_lane(scheduler, second)
    first.append_streaming_prompt_chunk(prompt_token_ids=[3] * 16)
    scheduler.max_num_scheduled_tokens = 16

    for physical_frontier in (16, 32, 48):
        step = scheduler.schedule()
        assert step.num_scheduled_tokens == {"first": 16}
        assert first.num_computed_tokens == physical_frontier
        assert _complete(scheduler, step) == {}
        assert first.streaming_anchor_confirmed_tokens == max(32, physical_frontier)

    assert first.streaming_anchor_parked
    assert first in scheduler.running
    assert not scheduler.has_requests()


def test_user_query_can_reclaim_idle_lane_and_complete_normally() -> None:
    scheduler = _scheduler(usable_blocks=4, policy="priority")
    idle = _lane("idle", 64)
    query = _lane(
        "query", 32, priority=0, token_id=2, arrival=2.0, streaming=False
    )
    _run_lane(scheduler, idle)
    scheduler.add_request(query)

    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"query": 32}
    result = _complete(scheduler, step)

    assert query.is_finished()
    assert result[0].outputs[0].request_id == "query"
    assert result[0].outputs[0].finish_reason is not None
    assert not _blocks(scheduler, query)
    assert idle.streaming_anchor_parked
    assert not _blocks(scheduler, idle)
    assert idle.streaming_anchor_confirmed_tokens == 64
    assert scheduler.schedule().num_scheduled_tokens == {}


@pytest.mark.parametrize("is_final", [False, True])
def test_append_during_execution_keeps_work_and_final_completion(
    is_final: bool,
) -> None:
    scheduler = _scheduler(usable_blocks=8)
    lane = _lane("lane", 32)
    scheduler.add_request(lane)
    first = scheduler.schedule()
    original_blocks = _blocks(scheduler, lane)
    lane.append_streaming_prompt_chunk(prompt_token_ids=[2] * 16, is_final=is_final)

    assert _complete(scheduler, first) == {}
    assert not lane.streaming_anchor_parked
    assert lane.streaming_anchor_confirmed_tokens == 32
    assert _blocks(scheduler, lane) == original_blocks
    assert lane.num_streaming_kv_releases == 0
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"lane": 16}
    result = _complete(scheduler, second)

    if is_final:
        assert lane.is_finished()
        assert result[0].outputs[0].request_id == "lane"
        assert not _blocks(scheduler, lane)
    else:
        assert result == {}
        assert lane.streaming_anchor_parked
        assert _blocks(scheduler, lane)[:2] == original_blocks
        assert lane.streaming_anchor_confirmed_tokens == 48


def test_final_append_after_pressure_eviction_finishes_once() -> None:
    scheduler = _scheduler(usable_blocks=4)
    lane = _lane("lane", 32)
    other = _lane("other", 64, token_id=2, arrival=2.0)
    _run_lane(scheduler, lane)
    _run_lane(scheduler, other)
    assert not _blocks(scheduler, lane)
    lane.append_streaming_prompt_chunk(prompt_token_ids=[3] * 16, is_final=True)

    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"lane": 48}
    result = _complete(scheduler, step)

    assert lane.is_finished()
    assert result[0].outputs[0].request_id == "lane"
    assert result[0].outputs[0].finish_reason is not None
    assert len(result[0].outputs) == 1
    assert not _blocks(scheduler, lane)
    assert lane.request_id not in scheduler.requests
    assert other.streaming_anchor_parked
    for _ in range(3):
        empty = scheduler.schedule()
        assert not empty.num_scheduled_tokens
        assert _complete(scheduler, empty) == {}


def test_failed_kv_load_does_not_confirm_the_scheduled_endpoint() -> None:
    scheduler = _scheduler(usable_blocks=4)
    lane = _lane("lane", 32)
    _run_lane(scheduler, lane)
    lane.append_streaming_prompt_chunk(prompt_token_ids=[2] * 16)
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"lane": 16}

    # Inject the worker's sync-load failure report after a cached prefix was
    # selected. This tests commit accounting without needing a remote connector.
    lane.num_cached_tokens = 32
    invalid_block = _blocks(scheduler, lane)[0].block_id
    result = _complete(
        scheduler,
        step,
        kv_connector_output=KVConnectorOutput(invalid_block_ids={invalid_block}),
    )

    assert result == {}
    assert lane.streaming_anchor_inflight_steps == 0
    assert lane.streaming_anchor_confirmed_tokens == 32
    assert lane.num_computed_tokens == 0
    assert not lane.streaming_anchor_parked
    assert not lane.is_finished()
    retry = scheduler.schedule()
    assert retry.num_scheduled_tokens == {"lane": 48}
    assert _complete(scheduler, retry) == {}
    assert lane.streaming_anchor_confirmed_tokens == 48


def test_unsupported_connector_rejects_lane_without_mutating_admission() -> None:
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


def test_slot_blocked_waiter_does_not_pin_running_lanes_needed_encoder_cache() -> None:
    scheduler = _scheduler(token_budget=32, chunk_size=16, max_num_seqs=1)
    scheduler.max_num_encoder_input_tokens = 16
    scheduler.encoder_cache_manager = EncoderCacheManager(cache_size=16)
    active = _lane(
        "active",
        96,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="audio",
                identifier="active-audio",
                mm_position=PlaceholderRange(offset=32, length=16),
            )
        ],
    )
    waiter = _lane(
        "waiter",
        16,
        token_id=2,
        arrival=2.0,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="audio",
                identifier="waiter-audio",
                mm_position=PlaceholderRange(offset=0, length=16),
            )
        ],
    )
    scheduler.encoder_cache_manager.allocate(waiter, 0)
    scheduler.encoder_cache_manager.free(waiter)
    scheduler.add_request(active)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"active": 16}
    assert _complete(scheduler, first) == {}
    scheduler.add_request(waiter)

    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"active": 16}
    assert _complete(scheduler, second) == {}
    third = scheduler.schedule()

    assert third.num_scheduled_tokens == {"active": 16}
    assert third.scheduled_encoder_inputs == {"active": [0]}
    assert _complete(scheduler, third) == {}


@pytest.mark.parametrize("with_encoder", [False, True])
@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_selected_work_and_inflight_active_lane_are_not_revoked(
    with_encoder: bool, policy: str
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
    assert _complete(scheduler, initial) == {}
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
    assert second.streaming_anchor_inflight_steps == 1
    assert _complete(scheduler, step) == {}
    assert first.streaming_anchor_confirmed_tokens == 32
    assert _blocks(scheduler, first) == first_blocks
    if with_encoder:
        encoder_inputs, _, _, _ = scheduler._try_schedule_encoder_inputs(
            first, 16, 16, 16
        )
        assert encoder_inputs == []
        assert first.request_id in scheduler.encoder_cache_manager.cached["first-audio"]
