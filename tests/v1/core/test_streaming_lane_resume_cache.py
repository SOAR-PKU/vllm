# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only resume-cache regressions against the real prefix allocator.

Allocation records hashes before model execution. Helpers explicitly acknowledge
the synthetic token frontier before allowing a release-time residency snapshot.
No model weights, device tensors, tokenizer, or connector are needed.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


@pytest.fixture(autouse=True)
def _initialize(monkeypatch: pytest.MonkeyPatch) -> None:
    init_none_hash(sha256)
    monkeypatch.setenv("VLLM_STREAMING_LANE_RESUME_CACHE", "1")


def _manager(
    *,
    usable_blocks: int = 16,
    enable_caching: bool = True,
    use_eagle: bool = False,
    dcp_world_size: int = 1,
    num_groups: int = 1,
    sliding: bool = False,
) -> KVCacheManager:
    window_spec = SlidingWindowSpec(BLOCK_SIZE, 1, 1, torch.float32, 32)
    spec = (
        window_spec
        if sliding
        else FullAttentionSpec(BLOCK_SIZE, 1, 1, torch.float32, False)
    )
    config = KVCacheConfig(
        num_blocks=usable_blocks + 1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], window_spec if i else spec)
            for i in range(num_groups)
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=256,
        enable_caching=enable_caching,
        use_eagle=use_eagle,
        dcp_world_size=dcp_world_size,
    )


def _lane(
    request_id: str = "lane",
    tokens: int = 64,
    *,
    token_id: int = 1,
    block_size: int = BLOCK_SIZE,
    streaming: bool = True,
) -> Request:
    request = Request(
        request_id=request_id,
        prompt_token_ids=[token_id] * tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )
    if streaming:
        request.start_streaming_anchor()
    return request


def _compute(manager: KVCacheManager, request: Request) -> list:
    assert manager.allocate_slots(request, request.num_tokens) is not None
    request.num_computed_tokens = request.num_tokens
    request.streaming_anchor_confirmed_tokens = request.num_tokens
    return list(manager.get_blocks(request.request_id).blocks[0])


def _release(manager: KVCacheManager, request: Request) -> None:
    manager.free(request, cache_streaming_prefix=True)
    request.num_computed_tokens = 0


def _native_hit(manager: KVCacheManager, request: Request) -> tuple:
    return manager.coordinator.find_longest_cache_hit(
        request.block_hashes, request.num_tokens - 1
    )


def test_append_keeps_prefix_without_directory_or_generation_scan() -> None:
    manager = _manager()
    request = _lane()
    initial = _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])

    class NoGenerationReads(list):
        def __getitem__(self, key):
            raise AssertionError("Unchanged eviction epoch needs no block scan")

    manager.block_pool.block_cache_generations = NoGenerationReads(
        manager.block_pool.block_cache_generations
    )
    with patch.object(
        manager.block_pool,
        "get_cached_block",
        wraps=manager.block_pool.get_cached_block,
    ) as lookup:
        blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == 64
    assert list(blocks.blocks[0]) == initial
    lookup.assert_not_called()
    assert all(block.ref_cnt == 0 for block in initial)


def test_lookup_does_not_pin_and_allocate_reacquires_references() -> None:
    manager = _manager(usable_blocks=6)
    request = _lane()
    initial = _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2] * 16)
    before_free = manager.block_pool.get_num_free_blocks()

    blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == 64
    assert manager.block_pool.get_num_free_blocks() == before_free == 6
    assert all(block.ref_cnt == 0 for block in initial)
    assert (
        manager.allocate_slots(
            request, 16, num_new_computed_tokens=tokens, new_computed_blocks=blocks
        )
        is not None
    )
    assert all(block.ref_cnt == 1 for block in initial)
    assert manager.block_pool.get_num_free_blocks() == 1


@pytest.mark.parametrize("evicted", [1, 2, 4])
def test_overwrite_preserves_only_contiguous_valid_prefix(evicted: int) -> None:
    manager = _manager(usable_blocks=4)
    request = _lane()
    initial = _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    pool = manager.block_pool
    epoch = pool.cache_eviction_epoch
    generations = pool.block_cache_generations.copy()

    overwritten = pool.get_new_blocks(evicted)

    assert overwritten == list(reversed(initial))[:evicted]
    assert pool.cache_eviction_epoch > epoch
    for block in overwritten:
        assert (
            pool.block_cache_generations[block.block_id] > generations[block.block_id]
        )
    expected, expected_tokens = _native_hit(manager, request)
    with patch.object(pool, "get_cached_block", wraps=pool.get_cached_block) as lookup:
        blocks, tokens = manager.get_computed_blocks(request)
    assert tokens == expected_tokens == (4 - evicted) * BLOCK_SIZE
    assert list(blocks.blocks[0]) == expected[0] == initial[: 4 - evicted]
    assert lookup.call_count == 1
    assert lookup.call_args.args[0] == request.block_hashes[4 - evicted]


def test_unrelated_eviction_does_not_force_prefix_directory_lookup() -> None:
    manager = _manager(usable_blocks=5)
    unrelated = _lane("other", 16, token_id=3)
    unrelated_blocks = _compute(manager, unrelated)
    manager.free(unrelated)
    request = _lane()
    initial = _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    assert manager.block_pool.get_new_blocks(1) == unrelated_blocks

    with patch.object(
        manager.block_pool,
        "get_cached_block",
        wraps=manager.block_pool.get_cached_block,
    ) as lookup:
        blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == 64
    assert list(blocks.blocks[0]) == initial
    lookup.assert_not_called()


def test_evicted_snapshot_falls_back_to_duplicate_physical_block() -> None:
    manager = _manager(usable_blocks=8)
    request = _lane()
    initial = _compute(manager, request)
    duplicate = _lane("duplicate")
    duplicate_blocks = _compute(manager, duplicate)
    assert set(b.block_id for b in initial).isdisjoint(
        b.block_id for b in duplicate_blocks
    )
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    assert manager.block_pool.get_new_blocks(1) == [initial[-1]]

    with patch.object(
        manager.block_pool,
        "get_cached_block",
        wraps=manager.block_pool.get_cached_block,
    ) as lookup:
        blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == 64
    assert list(blocks.blocks[0]) == initial[:3] + duplicate_blocks[3:]
    assert lookup.call_count == 1
    assert blocks.blocks[0][-1] is not initial[-1]


def test_append_searches_new_suffix_cached_by_another_lane() -> None:
    manager = _manager(usable_blocks=12)
    request = _lane()
    initial = _compute(manager, request)
    _release(manager, request)
    donor = _lane("donor", 96)
    donor_blocks = _compute(manager, donor)
    request.append_streaming_prompt_chunk(prompt_token_ids=[1] * 33)

    with patch.object(
        manager.block_pool,
        "get_cached_block",
        wraps=manager.block_pool.get_cached_block,
    ) as lookup:
        blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == 96
    assert list(blocks.blocks[0]) == initial + donor_blocks[4:]
    assert [call.args[0] for call in lookup.call_args_list] == request.block_hashes[4:6]


@pytest.mark.parametrize(
    ("initial_tokens", "appended_tokens", "expected_tokens"),
    [(64, 0, 48), (64, 1, 64), (64, 16, 64), (63, 1, 48), (63, 17, 48)],
)
def test_native_full_block_and_last_token_clipping(
    initial_tokens: int, appended_tokens: int, expected_tokens: int
) -> None:
    manager = _manager()
    request = _lane(tokens=initial_tokens)
    _compute(manager, request)
    _release(manager, request)
    if appended_tokens:
        request.append_streaming_prompt_chunk(prompt_token_ids=[2] * appended_tokens)

    expected, native_tokens = _native_hit(manager, request)
    blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == native_tokens == expected_tokens
    assert list(blocks.blocks[0]) == expected[0]


def test_successful_reset_invalidates_snapshot_and_all_block_generations() -> None:
    manager = _manager()
    request = _lane()
    initial = _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    epoch = manager.block_pool.cache_eviction_epoch
    generations = manager.block_pool.block_cache_generations.copy()

    assert manager.reset_prefix_cache()

    assert manager.block_pool.cache_eviction_epoch > epoch
    assert all(
        manager.block_pool.block_cache_generations[block.block_id]
        > generations[block.block_id]
        for block in initial
    )
    assert not manager._streaming_prefix_snapshots
    blocks, tokens = manager.get_computed_blocks(request)
    assert tokens == 0
    assert not blocks.blocks[0]


def test_failed_reset_preserves_valid_snapshot() -> None:
    manager = _manager()
    request = _lane()
    initial = _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    _compute(manager, _lane("busy", 16, token_id=3))
    epoch = manager.block_pool.cache_eviction_epoch
    generations = manager.block_pool.block_cache_generations.copy()

    assert not manager.reset_prefix_cache()

    assert manager.block_pool.cache_eviction_epoch == epoch
    assert manager.block_pool.block_cache_generations == generations
    with patch.object(
        manager.block_pool,
        "get_cached_block",
        wraps=manager.block_pool.get_cached_block,
    ) as lookup:
        blocks, tokens = manager.get_computed_blocks(request)
    assert tokens == 64
    assert list(blocks.blocks[0]) == initial
    lookup.assert_not_called()


def test_non_append_prefix_change_uses_native_lookup() -> None:
    manager = _manager()
    request = _lane()
    _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    # Simulate a future non-append API that correctly recomputes chained hashes.
    replacement = _lane("replacement", request.num_tokens, token_id=9)
    request.prompt_token_ids[:] = replacement.prompt_token_ids
    request._all_token_ids[:] = replacement.prompt_token_ids
    request.block_hashes[:] = replacement.block_hashes

    with patch.object(
        manager.coordinator,
        "find_longest_cache_hit",
        wraps=manager.coordinator.find_longest_cache_hit,
    ) as lookup:
        blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == 0
    assert not blocks.blocks[0]
    lookup.assert_called_once()


def test_same_request_id_does_not_reuse_another_request_snapshot() -> None:
    manager = _manager()
    request = _lane()
    _compute(manager, request)
    _release(manager, request)
    replacement = _lane(request.request_id, 65)

    with patch.object(
        manager.coordinator,
        "find_longest_cache_hit",
        wraps=manager.coordinator.find_longest_cache_hit,
    ) as lookup:
        _, tokens = manager.get_computed_blocks(replacement)

    assert tokens == 64  # Native content reuse is still correct.
    lookup.assert_called_once()


def test_ordinary_free_forgets_existing_residency_snapshot() -> None:
    manager = _manager()
    request = _lane()
    _compute(manager, request)
    _release(manager, request)
    assert request.request_id in manager._streaming_prefix_snapshots

    manager.free(request)

    assert request.request_id not in manager._streaming_prefix_snapshots


@pytest.mark.parametrize(
    "options",
    [
        {"enable_caching": False},
        {"use_eagle": True},
        {"dcp_world_size": 2},
        {"num_groups": 2},
        {"sliding": True},
    ],
)
def test_unsupported_managers_keep_native_behavior(options: dict) -> None:
    manager = _manager(**options)
    request = _lane(block_size=BLOCK_SIZE * options.get("dcp_world_size", 1))
    _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    assert not manager._streaming_prefix_snapshots

    expected, expected_tokens = _native_hit(manager, request)
    blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == expected_tokens
    assert tuple(list(group) for group in blocks.blocks) == expected


def test_env_disabled_uses_native_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_STREAMING_LANE_RESUME_CACHE", "0")
    manager = _manager()
    request = _lane()
    _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    assert not manager._streaming_prefix_snapshots

    with patch.object(
        manager.coordinator,
        "find_longest_cache_hit",
        wraps=manager.coordinator.find_longest_cache_hit,
    ) as lookup:
        _, tokens = manager.get_computed_blocks(request)
    assert tokens == 64
    lookup.assert_called_once()


@pytest.mark.parametrize("streaming", [False, True])
def test_non_lane_or_inflight_release_does_not_save_snapshot(streaming: bool) -> None:
    manager = _manager()
    request = _lane(streaming=streaming)
    _compute(manager, request)
    if streaming:
        request.streaming_anchor_inflight_steps = 1

    _release(manager, request)

    assert not manager._streaming_prefix_snapshots


def test_skip_reading_prefix_cache_bypasses_snapshot() -> None:
    manager = _manager()
    request = _lane()
    _compute(manager, request)
    _release(manager, request)
    request.append_streaming_prompt_chunk(prompt_token_ids=[2])
    request.skip_reading_prefix_cache = True

    blocks, tokens = manager.get_computed_blocks(request)

    assert tokens == 0
    assert not blocks.blocks[0]


@pytest.mark.parametrize("confirmed", [0, 32])
def test_snapshot_excludes_unconfirmed_optimistic_hashes(confirmed: int) -> None:
    manager = _manager()
    request = _lane()
    _compute(manager, request)
    request.streaming_anchor_confirmed_tokens = confirmed

    _release(manager, request)

    if confirmed:
        snapshot = manager._streaming_prefix_snapshots[request.request_id]
        assert len(snapshot.blocks) == confirmed // BLOCK_SIZE
    else:
        assert not manager._streaming_prefix_snapshots
