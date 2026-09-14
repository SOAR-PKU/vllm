# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    MutableRequestData,
    SchedulerOutput,
)
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def runner(monkeypatch):
    runner = object.__new__(GPUModelRunner)
    runner.input_batch = InputBatch(
        max_num_reqs=4,
        max_model_len=128,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=128,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )
    runner.requests = {}
    runner.encoder_cache = {"retained-mm": object()}
    runner.use_async_scheduling = False
    runner.uses_mrope = True
    runner.valid_sampled_token_count_event = None
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    monkeypatch.setattr(
        gpu_model_runner,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    return runner


def _add_lane(runner, req_id="lane"):
    state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(32)),
        prompt_revision=2,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0),
        pooling_params=None,
        generator=None,
        block_ids=([1, 2],),
        num_computed_tokens=16,
        output_token_ids=[],
        mrope_positions=torch.arange(32).repeat(3, 1),
        mrope_position_delta=0,
    )
    runner.requests[req_id] = state
    runner.input_batch.add_request(state)
    runner.input_batch.refresh_metadata()
    return state


def _output(*, resumed=True, prompt_update=None, num_output_tokens=0):
    req_data = CachedRequestData(
        req_ids=["lane"],
        resumed_req_ids={"lane"} if resumed else set(),
        new_token_ids=[[]],
        all_token_ids={"lane": list(range(32)) + [42] * num_output_tokens},
        new_block_ids=[([7, 8],)],
        num_computed_tokens=[16],
        num_output_tokens=[num_output_tokens],
        prompt_updates={} if prompt_update is None else {"lane": prompt_update},
    )
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=req_data,
        num_scheduled_tokens={"lane": 16},
        total_num_scheduled_tokens=16,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _assert_blocks(runner, expected):
    req_index = runner.input_batch.req_id_to_index["lane"]
    table = runner.input_batch.block_table[0]
    assert table.num_blocks_per_row[req_index] == len(expected)
    assert table.block_table.np[req_index, : len(expected)].tolist() == expected
    assert runner.requests["lane"].block_ids == (expected,)


@pytest.mark.parametrize("skipped_step", [False, True])
def test_resume_replaces_blocks_without_rebuilding_prompt(runner, skipped_step):
    state = _add_lane(runner)
    prompt = state.prompt_token_ids
    features = state.mm_features
    positions = state.mrope_positions
    encoder_output = runner.encoder_cache["retained-mm"]
    if skipped_step:
        runner.input_batch.remove_request("lane")

    runner._update_states(_output())

    assert runner.requests["lane"] is state
    assert state.prompt_token_ids is prompt
    assert state.mm_features is features
    assert state.mrope_positions is positions
    assert state.prompt_revision == 2
    assert runner.encoder_cache["retained-mm"] is encoder_output
    assert runner.input_batch.num_reqs == 1
    _assert_blocks(runner, [7, 8])


def test_consecutive_resume_and_append_refresh_one_batch_row(runner, monkeypatch):
    state = _add_lane(runner)
    updates = []
    monkeypatch.setattr(runner, "_init_mrope_positions", updates.append)
    prompt_update = MutableRequestData(
        revision=3,
        prompt_token_ids=list(range(48)),
        mm_features=[],
    )

    runner._update_states(_output(prompt_update=prompt_update))

    assert updates == [state]
    assert state.prompt_revision == 3
    assert state.num_prompt_tokens == 48
    assert runner.input_batch.num_reqs == 1
    req_index = runner.input_batch.req_id_to_index["lane"]
    assert runner.input_batch.num_prompt_tokens[req_index] == 48
    assert runner.input_batch.token_ids_cpu[req_index, :48].tolist() == list(range(48))
    _assert_blocks(runner, [7, 8])


def test_running_without_resume_keeps_append_block_semantics(runner):
    state = _add_lane(runner)

    runner._update_states(_output(resumed=False))

    assert runner.requests["lane"] is state
    _assert_blocks(runner, [1, 2, 7, 8])


def test_consecutive_resume_rebuilds_async_output_state(runner):
    state = _add_lane(runner)
    runner.use_async_scheduling = True
    state.prev_num_draft_len = 3
    state.output_token_ids[:] = [99, 100]

    runner._update_states(_output(num_output_tokens=1))

    assert state.prev_num_draft_len == 0
    assert state.output_token_ids == [42]
    req_index = runner.input_batch.req_id_to_index["lane"]
    assert runner.input_batch.token_ids_cpu[req_index, 32] == 42
    _assert_blocks(runner, [7, 8])


def test_consecutive_resume_does_not_drop_another_scheduled_lane(runner):
    _add_lane(runner)
    other = _add_lane(runner, "other")
    output = _output()
    output.num_scheduled_tokens["other"] = 16
    output.total_num_scheduled_tokens += 16
    req_data = output.scheduled_cached_reqs
    req_data.req_ids.append("other")
    req_data.new_token_ids.append([])
    req_data.new_block_ids.append(None)
    req_data.num_computed_tokens.append(16)
    req_data.num_output_tokens.append(0)

    runner._update_states(output)

    assert runner.input_batch.num_reqs == 2
    assert runner.requests["other"] is other
    other_index = runner.input_batch.req_id_to_index["other"]
    assert runner.input_batch.block_table[0].block_table.np[
        other_index, :2
    ].tolist() == [1, 2]
    _assert_blocks(runner, [7, 8])
