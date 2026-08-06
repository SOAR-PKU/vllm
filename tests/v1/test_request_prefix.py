# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

import pytest

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.request import Request


def _request(
    token_count: int,
    *,
    block_hasher=None,
    mm_features: list[MultiModalFeatureSpec] | None = None,
) -> Request:
    return Request(
        request_id="source",
        client_index=3,
        prompt_token_ids=list(range(token_count)),
        mm_features=mm_features,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        arrival_time=123.0,
        cache_salt="salt",
        priority=-1,
        block_hasher=block_hasher,
    )


def _feature(
    identifier: str,
    *,
    modality: str,
    offset: int,
    length: int,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def test_request_prefix_reuses_hashes_and_extends_independently() -> None:
    init_none_hash(sha256)
    hash_inputs = []

    def counting_hash(value):
        hash_inputs.append(value)
        return sha256(value)

    block_size = 4
    block_hasher = get_request_block_hasher(block_size, counting_hash)
    source = _request(10, block_hasher=block_hasher)
    source_hashes = source.block_hashes.copy()
    calls_after_source = len(hash_inputs)

    prefix = Request.from_request_prefix(
        source,
        request_id="physical",
        target_num_prompt_tokens=6,
        block_size=block_size,
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert len(hash_inputs) == calls_after_source
    assert prefix.block_hashes == source_hashes[:1]
    assert prefix.block_hashes is not source.block_hashes
    assert prefix.prompt_token_ids == list(range(6))
    assert list(prefix.all_token_ids) == list(range(6))
    assert prefix._all_token_ids is prefix.prompt_token_ids

    prefix.append_output_token_ids([100, 101])

    assert len(hash_inputs) == calls_after_source + 1
    assert prefix.block_hashes[0] == source_hashes[0]
    assert prefix.block_hashes[1] == sha256((source_hashes[0], (4, 5, 100, 101), None))
    assert source.block_hashes == source_hashes
    assert source.prompt_token_ids == list(range(10))
    assert prefix.prompt_token_ids == list(range(6))
    assert prefix._all_token_ids is not prefix.prompt_token_ids
    assert list(prefix.all_token_ids) == [
        *range(6),
        100,
        101,
    ]


def test_request_prefix_preserves_and_validates_mm_identity() -> None:
    init_none_hash(sha256)
    block_size = 4
    block_hasher = get_request_block_hasher(block_size, sha256)
    first = _feature("audio-a", modality="audio", offset=0, length=4)
    second = _feature("audio-b", modality="audio", offset=8, length=4)
    source = _request(
        12,
        block_hasher=block_hasher,
        mm_features=[first, second],
    )

    automatic = Request.from_request_prefix(
        source,
        request_id="automatic",
        target_num_prompt_tokens=8,
        block_size=block_size,
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert len(automatic.mm_features) == 1
    assert automatic.mm_features[0] is not first
    assert automatic.mm_features[0].identifier == first.identifier
    assert automatic.block_hashes == source.block_hashes[:2]
    fresh = _request(
        8,
        block_hasher=block_hasher,
        mm_features=[copy(first)],
    )
    assert automatic.block_hashes == fresh.block_hashes

    prepared = copy(first)
    accepted = Request.from_request_prefix(
        source,
        request_id="prepared",
        target_num_prompt_tokens=8,
        block_size=block_size,
        sampling_params=SamplingParams(max_tokens=1),
        mm_features=[prepared],
    )
    assert accepted.mm_features == [prepared]

    changed = copy(first)
    changed.identifier = "different-audio"
    with pytest.raises(ValueError, match="must preserve"):
        Request.from_request_prefix(
            source,
            request_id="changed",
            target_num_prompt_tokens=8,
            block_size=block_size,
            sampling_params=SamplingParams(max_tokens=1),
            mm_features=[changed],
        )

    with pytest.raises(ValueError, match="cannot split"):
        Request.from_request_prefix(
            source,
            request_id="split",
            target_num_prompt_tokens=10,
            block_size=block_size,
            sampling_params=SamplingParams(max_tokens=1),
        )


def test_request_prefix_without_source_hasher_keeps_hashing_disabled() -> None:
    source = _request(10)

    prefix = Request.from_request_prefix(
        source,
        request_id="physical",
        target_num_prompt_tokens=6,
        block_size=4,
        sampling_params=SamplingParams(max_tokens=1),
    )

    assert prefix.block_hashes == []
    assert prefix.get_hash_new_full_blocks is None
    prefix.append_output_token_ids([100, 101])
    assert prefix.block_hashes == []


def test_request_prefix_rejects_incomplete_source_hashes() -> None:
    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(4, sha256)
    source = _request(8, block_hasher=block_hasher)
    source.block_hashes.pop()

    with pytest.raises(ValueError, match="expected=2, available=1"):
        Request.from_request_prefix(
            source,
            request_id="physical",
            target_num_prompt_tokens=8,
            block_size=4,
            sampling_params=SamplingParams(max_tokens=1),
        )


def test_request_prefix_rejects_mismatched_block_size() -> None:
    init_none_hash(sha256)
    block_hasher = get_request_block_hasher(4, sha256)
    source = _request(8, block_hasher=block_hasher)

    with pytest.raises(ValueError, match="expected=4, available=2"):
        Request.from_request_prefix(
            source,
            request_id="physical",
            target_num_prompt_tokens=8,
            block_size=2,
            sampling_params=SamplingParams(max_tokens=1),
        )
