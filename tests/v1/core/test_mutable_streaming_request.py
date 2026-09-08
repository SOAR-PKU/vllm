# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    MutableRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


def _mm_feature(identifier: str, offset: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=object(),
        modality="audio",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=1),
    )


def test_request_accepts_only_append_only_prompt_and_mm_snapshots() -> None:
    feature_0 = _mm_feature("mm-0", 1)
    request = Request(
        request_id="anchor",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        mm_features=[feature_0],
    )
    request.start_streaming_anchor()

    feature_1 = _mm_feature("mm-1", 3)
    changed = request.update_streaming_anchor(
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=[feature_0, feature_1],
    )

    assert changed
    assert request.prompt_revision == 1
    assert request.prompt_token_ids == [1, 2, 3, 4, 5]
    assert [feature.identifier for feature in request.mm_features] == [
        "mm-0",
        "mm-1",
    ]
    assert not request.streaming_anchor_parked

    with pytest.raises(ValueError, match="complete token prefix"):
        request.update_streaming_anchor(
            prompt_token_ids=[1, 9, 3, 4, 5],
            mm_features=[feature_0, feature_1],
        )

    with pytest.raises(ValueError, match="multimodal prefix"):
        request.update_streaming_anchor(
            prompt_token_ids=[1, 2, 3, 4, 5, 6],
            mm_features=[feature_1, feature_0],
        )


def test_empty_final_cannot_leave_an_idle_lane_without_schedulable_work() -> None:
    request = Request(
        request_id="lane",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
    )
    request.start_streaming_anchor()
    request.num_computed_tokens = 2
    request.streaming_anchor_parked = True
    with pytest.raises(ValueError, match="must contain tokens"):
        request.append_streaming_prompt_chunk(prompt_token_ids=[], is_final=True)
    with pytest.raises(ValueError, match="must contain tokens"):
        request.update_streaming_anchor(prompt_token_ids=[1, 2], is_final=True)
    assert request.streaming_anchor_open
    assert request.streaming_anchor_parked


def test_dispatch_snapshots_do_not_alias_mutable_prompt_descriptors() -> None:
    request = Request(
        request_id="lane",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        mm_features=[_mm_feature("mm-0", 1)],
    )
    request.start_streaming_anchor()
    new_snapshot = NewRequestData.from_request(request, ([],))
    cached_snapshot = MutableRequestData.from_request(request)

    request.append_streaming_prompt_chunk(prompt_token_ids=[3, 4])
    for snapshot in (new_snapshot, cached_snapshot):
        assert snapshot.prompt_token_ids == [1, 2]
        snapshot.mm_features[0].data = None
    assert request.prompt_token_ids == [1, 2, 3, 4]
    assert request.mm_features[0].data is not None


def test_scheduler_parks_anchor_discards_sample_and_sends_full_update() -> None:
    request = Request(
        request_id="anchor",
        prompt_token_ids=[0, 0, 0, 0],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
    )
    request.start_streaming_anchor()

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = []
    scheduler.interrupt_waiting = []
    scheduler.remote_kv_ready_interrupt = []
    scheduler.offloading = {}
    scheduler.remote_kv_loading = {}
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.connector = None
    scheduler.log_stats = False
    scheduler.max_model_len = 64
    scheduler.kv_cache_manager = SimpleNamespace(take_events=lambda: None)

    first_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={request.request_id: 4},
        total_num_scheduled_tokens=4,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        streaming_anchor_snapshots={request.request_id: (0, 4)},
    )
    scheduler._update_after_schedule(first_output)
    assert request.streaming_anchor_parked
    assert not scheduler.has_requests()
    assert scheduler.has_unfinished_requests()
    assert scheduler.get_num_unfinished_requests() == 1

    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        sampled_token_ids=[np.array([42])],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    engine_outputs = scheduler.update_from_output(first_output, model_output)
    assert not engine_outputs
    assert request.num_output_tokens == 0
    assert request.streaming_anchor_parked

    request.update_streaming_anchor(
        prompt_token_ids=[0, 0, 0, 0, 1, 1],
        mm_features=[],
    )
    assert scheduler.has_requests()

    request.worker_prompt_revision = 0
    scheduler.use_pp = False
    scheduler.prev_step_scheduled_req_ids = {request.request_id}

    class Blocks:
        def get_block_ids(self, allow_none=False):
            return ([],)

    cached_data = scheduler._make_cached_request_data(
        running_reqs=[request],
        resumed_reqs=[],
        num_scheduled_tokens={request.request_id: 2},
        spec_decode_tokens={},
        req_to_new_blocks={request.request_id: Blocks()},
    )
    update = cached_data.prompt_updates[request.request_id]
    assert update.revision == 1
    assert update.prompt_token_ids == [0, 0, 0, 0, 1, 1]


def test_invalid_kv_rollback_unparks_anchor() -> None:
    request = Request(
        request_id="anchor",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
    )
    request.start_streaming_anchor()
    request.status = RequestStatus.RUNNING
    request.num_cached_tokens = 4
    request.num_computed_tokens = 4
    request.streaming_anchor_parked = True

    scheduler = object.__new__(Scheduler)
    scheduler.block_size = 4
    scheduler.kv_cache_manager = SimpleNamespace(
        get_block_ids=lambda request_id: ([7],)
    )
    affected, _ = scheduler._update_requests_with_invalid_blocks([request], {7})

    assert affected == {request.request_id}
    assert request.num_computed_tokens == 0
    assert not request.streaming_anchor_parked


def test_worker_applies_cached_snapshot_with_native_mrope_reinitialization() -> None:
    feature_0 = _mm_feature("mm-0", 1)
    feature_1 = _mm_feature("mm-1", 3)
    req_state = CachedRequestState(
        req_id="anchor",
        prompt_token_ids=[1, 2, 3],
        mm_features=[feature_0],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[99],
        prompt_revision=0,
    )
    update = MutableRequestData(
        revision=1,
        prompt_token_ids=[1, 2, 3, 4, 5],
        mm_features=[feature_0, feature_1],
    )

    runner = object.__new__(GPUModelRunner)
    runner.uses_mrope = True
    initialized = []
    runner._init_mrope_positions = initialized.append

    assert runner._apply_cached_prompt_update(req_state, update)
    assert req_state.prompt_revision == 1
    assert req_state.prompt_token_ids == [1, 2, 3, 4, 5]
    assert req_state.output_token_ids == []
    assert initialized == [req_state]
    assert not runner._apply_cached_prompt_update(req_state, update)


def test_batch_queue_keeps_revision_snapshots_until_delayed_outputs_return() -> None:
    def model_output(token_id: int) -> ModelRunnerOutput:
        return ModelRunnerOutput(
            req_ids=["anchor"],
            req_id_to_index={"anchor": 0},
            sampled_token_ids=[np.array([token_id])],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    request = Request(
        request_id="anchor",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
    )
    request.start_streaming_anchor()
    request.status = RequestStatus.RUNNING

    scheduler = object.__new__(Scheduler)
    scheduler.requests = {request.request_id: request}
    scheduler.running = [request]
    scheduler.waiting = []
    scheduler.interrupt_waiting = []
    scheduler.remote_kv_ready_interrupt = []
    scheduler.offloading = {}
    scheduler.remote_kv_loading = {}
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler.connector = None
    scheduler.log_stats = False
    scheduler.max_model_len = 64
    scheduler.kv_cache_manager = SimpleNamespace(take_events=lambda: None)
    scheduler.use_pp = False
    scheduler.prev_step_scheduled_req_ids = set()

    class Blocks:
        def get_block_ids(self, allow_none: bool = False):
            return ([],)

    class QueueScheduler:
        """Build the three immutable ledgers that an async queue retains."""

        def __init__(self) -> None:
            self.outputs: list[SchedulerOutput] = []

        def has_requests(self) -> bool:
            return len(self.outputs) < 3

        def schedule(self) -> SchedulerOutput:
            num_scheduled_tokens = {
                request.request_id: (
                    request.num_prompt_tokens - request.num_computed_tokens
                )
            }
            assert num_scheduled_tokens[request.request_id] == 2

            if not self.outputs:
                new_data = NewRequestData.from_request(request, ([],))
                request.worker_prompt_revision = new_data.prompt_revision
                scheduled_new_reqs = [new_data]
                cached_data = CachedRequestData.make_empty()
            else:
                scheduled_new_reqs = []
                cached_data = scheduler._make_cached_request_data(
                    running_reqs=[request],
                    resumed_reqs=[],
                    num_scheduled_tokens=num_scheduled_tokens,
                    spec_decode_tokens={},
                    req_to_new_blocks={request.request_id: Blocks()},
                )

            output = SchedulerOutput(
                scheduled_new_reqs=scheduled_new_reqs,
                scheduled_cached_reqs=cached_data,
                num_scheduled_tokens=num_scheduled_tokens,
                total_num_scheduled_tokens=2,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[],
                finished_req_ids=set(),
                free_encoder_mm_hashes=[],
                streaming_anchor_snapshots={
                    request.request_id: (
                        request.prompt_revision,
                        request.num_prompt_tokens,
                    )
                },
            )
            scheduler.prev_step_scheduled_req_ids = {request.request_id}
            scheduler._update_after_schedule(output)
            self.outputs.append(output)
            return output

        @staticmethod
        def get_grammar_bitmask(_output: SchedulerOutput):
            return None

        @staticmethod
        def update_from_output(
            output: SchedulerOutput,
            model_output: ModelRunnerOutput,
        ):
            return scheduler.update_from_output(output, model_output)

    runner = object.__new__(GPUModelRunner)
    runner.uses_mrope = False
    worker_state: CachedRequestState | None = None

    class ApplyingExecutor:
        """Apply worker snapshots now but delay each sampler result."""

        def __init__(self) -> None:
            self.sample_futures: list[Future[ModelRunnerOutput]] = []
            self.applied_snapshots: list[tuple[int, list[int], list[int]]] = []

        def execute_model(
            self,
            output: SchedulerOutput,
            non_block: bool,
        ) -> Future[None]:
            nonlocal worker_state
            assert non_block

            if output.scheduled_new_reqs:
                data = output.scheduled_new_reqs[0]
                assert data.prompt_token_ids is not None
                worker_state = CachedRequestState(
                    req_id=data.req_id,
                    prompt_token_ids=data.prompt_token_ids,
                    mm_features=data.mm_features,
                    sampling_params=data.sampling_params,
                    pooling_params=data.pooling_params,
                    generator=None,
                    block_ids=data.block_ids,
                    num_computed_tokens=data.num_computed_tokens,
                    output_token_ids=[],
                    prompt_revision=data.prompt_revision,
                )

            assert worker_state is not None
            update = output.scheduled_cached_reqs.prompt_updates.get(request.request_id)
            if update is not None:
                assert runner._apply_cached_prompt_update(worker_state, update)

            self.applied_snapshots.append(
                (
                    worker_state.prompt_revision,
                    worker_state.prompt_token_ids.copy(),
                    worker_state.output_token_ids.copy(),
                )
            )
            completed: Future[None] = Future()
            completed.set_result(None)
            return completed

        def sample_tokens(
            self,
            _grammar_output,
            non_block: bool,
        ) -> Future[ModelRunnerOutput]:
            assert non_block
            assert worker_state is not None
            # ModelRunner caches a frontier sample before EngineCore receives it.
            worker_state.output_token_ids.append(700 + len(self.sample_futures))
            future: Future[ModelRunnerOutput] = Future()
            self.sample_futures.append(future)
            return future

    queue_scheduler = QueueScheduler()
    executor = ApplyingExecutor()
    engine = object.__new__(EngineCore)
    engine.batch_queue_size = 3
    engine.batch_queue = deque(maxlen=engine.batch_queue_size)
    engine.scheduler = queue_scheduler
    engine.model_executor = executor
    engine.streaming_prefill_coordinator = SimpleNamespace(
        has_pending_policy_work=lambda: False
    )
    engine.ec_producer = False
    engine.is_pooling_model = False
    engine._maybe_log_mm_feature_prefetch_metrics = lambda: None
    engine._poll_mm_feature_prefetch_acks = lambda: None
    engine._update_streaming_prefill_after_step = lambda *_args: None
    engine._finish_drained_mm_feature_abort_dispatches = lambda *_args: None
    engine._log_err_callback = lambda _output: (lambda _future: None)
    engine.log_error_detail = lambda _output: nullcontext()

    # rev0 is queued and its sampler future has not returned.
    assert engine._step_with_batch_queue() == (None, True)
    assert len(engine.batch_queue) == 1
    assert not executor.sample_futures[0].done()

    # rev1 and rev2 are both appended while older batches remain in flight.
    request.update_streaming_anchor(
        prompt_token_ids=[10, 11, 12, 13],
        mm_features=[],
    )
    assert engine._step_with_batch_queue() == (None, True)
    assert not executor.sample_futures[0].done()
    assert not executor.sample_futures[1].done()

    request.update_streaming_anchor(
        prompt_token_ids=[10, 11, 12, 13, 14, 15],
        mm_features=[],
    )
    executor.sample_futures[0].set_result(model_output(800))
    assert engine._step_with_batch_queue()[1]

    # EngineCore drains FIFO. Every queued iteration still sees its own captured
    # revision, so none of the three dummy samples enters scheduler state.
    executor.sample_futures[1].set_result(model_output(801))
    assert engine._step_with_batch_queue() == ({}, False)
    executor.sample_futures[2].set_result(model_output(802))
    assert engine._step_with_batch_queue() == ({}, False)
    assert not engine.batch_queue

    assert [
        output.streaming_anchor_snapshots[request.request_id]
        for output in queue_scheduler.outputs
    ] == [(0, 2), (1, 4), (2, 6)]
    assert [
        (
            update.revision,
            update.prompt_token_ids,
        )
        for output in queue_scheduler.outputs[1:]
        if (update := output.scheduled_cached_reqs.prompt_updates[request.request_id])
    ] == [
        (1, [10, 11, 12, 13]),
        (2, [10, 11, 12, 13, 14, 15]),
    ]
    assert executor.applied_snapshots == [
        (0, [10, 11], []),
        (1, [10, 11, 12, 13], []),
        (2, [10, 11, 12, 13, 14, 15], []),
    ]
    assert request.prompt_revision == 2
    assert request.prompt_token_ids == [10, 11, 12, 13, 14, 15]
    assert list(request.all_token_ids) == request.prompt_token_ids
    assert request.num_computed_tokens == request.num_prompt_tokens == 6
    assert request.num_output_tokens == 0
    assert request.streaming_anchor_parked


@pytest.mark.parametrize("cache_kind", ["custom", "native"])
def test_worker_mm_cache_materializes_cached_prompt_updates(
    cache_kind: str,
) -> None:
    feature = MultiModalFeatureSpec(
        data=None,
        modality="audio",
        identifier="cached-mm",
        mm_position=PlaceholderRange(offset=0, length=1),
    )
    cached_data = CachedRequestData.make_empty()
    cached_data.prompt_updates["anchor"] = MutableRequestData(
        revision=1,
        prompt_token_ids=[1],
        mm_features=[feature],
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_data,
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    class PrefetchCache:
        def materialize_features(self, features) -> None:
            for mm_feature in features:
                mm_feature.data = object()

    class ReceiverCache:
        def get_and_update_features(self, features):
            for mm_feature in features:
                mm_feature.data = object()
            return features

    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.mm_feature_prefetch_cache = (
        PrefetchCache() if cache_kind == "custom" else None
    )
    wrapper.mm_receiver_cache = ReceiverCache() if cache_kind == "native" else None
    wrapper._apply_mm_cache(scheduler_output)

    assert feature.data is not None
