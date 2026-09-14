# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def _media_features(media, tokenization_kwargs):
    processor = SimpleNamespace(info=SimpleNamespace(model_id="test-model"))
    features = []
    for uuid, offset in media:
        identifiers = BaseMultiModalProcessor._hash_mm_items(
            processor,
            {"audio": [object()]},
            {"use_audio_in_video": False},
            tokenization_kwargs or {},
            mm_uuids={"audio": [uuid]},
        )
        features.append(
            MultiModalFeatureSpec(
                data=object(),
                modality="audio",
                identifier=identifiers["audio"][0],
                mm_position=PlaceholderRange(offset=offset, length=13),
            )
        )
    return features


def _request(request_id, tokens, params, features=None):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(tokens),
        mm_features=features,
        sampling_params=params,
        pooling_params=None,
        eos_token_id=0,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _frontend():
    processed: list[tuple[str, dict[str, Any] | None]] = []
    sent: list[EngineCoreRequest] = []

    class Processor:
        @staticmethod
        def process_inputs(request_id, prompt, params, *args):
            tokenization_kwargs = args[2]
            processed.append((request_id, tokenization_kwargs))
            return _request(
                request_id,
                prompt["prompt_token_ids"],
                params,
                _media_features(prompt.get("media", ()), tokenization_kwargs),
            )

    class EngineClient:
        resources = SimpleNamespace(engine_dead=False)

        @staticmethod
        def shutdown():
            pass

        @staticmethod
        async def add_request_async(request):
            sent.append(request)

    async def add_processed_request(request, prompt, parent, index, queue):
        sent.append(request)
        queue.put(
            RequestOutput(
                request_id=request.request_id,
                prompt=None,
                prompt_token_ids=request.prompt_token_ids,
                prompt_logprobs=None,
                outputs=[],
                finished=True,
            )
        )

    llm = object.__new__(AsyncLLM)
    llm.model_config = SimpleNamespace(max_model_len=256)
    llm.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    llm.engine_core = EngineClient()
    llm.processor = Processor()
    llm.output_handler = None
    llm.log_requests = False
    llm._run_output_handler = lambda: None
    llm._add_request = add_processed_request
    return llm, processed, sent


async def _open_append_query(params, kwargs):
    llm, processed, sent = _frontend()
    await llm.add_streaming_prompt_chunk(
        {"prompt_token_ids": list(range(64)), "media": [("audio-0", 34)]},
        params,
        "lane",
        is_first_chunk=True,
        tokenization_kwargs=kwargs,
    )
    await llm.add_streaming_prompt_chunk(
        {"prompt_token_ids": list(range(64, 128)), "media": [("audio-1", 2)]},
        params,
        "lane",
        is_first_chunk=False,
        tokenization_kwargs=kwargs,
    )
    outputs = [
        output
        async for output in llm.generate(
            {
                "prompt_token_ids": list(range(147)),
                "media": [("audio-0", 34), ("audio-1", 66)],
            },
            params,
            "query",
            tokenization_kwargs=kwargs,
        )
    ]
    assert len(outputs) == 1 and outputs[0].finished
    return processed, sent


@pytest.mark.parametrize(
    "truncate,kwargs,expected",
    [
        (None, None, {"truncation": False}),
        (64, None, {"truncation": True, "max_length": 64}),
        (-1, None, {"truncation": True, "max_length": 256}),
        (None, {}, {}),
        (
            64,
            {"truncation": False, "add_special_tokens": False},
            {"truncation": False, "add_special_tokens": False},
        ),
        (
            128,
            {"truncation": True, "max_length": 32},
            {"truncation": True, "max_length": 32},
        ),
    ],
)
def test_open_append_generate_forward_identical_tokenization_kwargs(
    truncate, kwargs, expected
):
    original = None if kwargs is None else kwargs.copy()
    processed, _ = asyncio.run(
        _open_append_query(
            SamplingParams(max_tokens=1, truncate_prompt_tokens=truncate), kwargs
        )
    )
    assert [request_id for request_id, _ in processed] == ["lane", "lane", "query"]
    assert all(actual == expected for _, actual in processed)
    assert all(actual is not kwargs for _, actual in processed)
    assert len({id(actual) for _, actual in processed}) == 3
    assert kwargs == original


@pytest.mark.parametrize("entrypoint", ["open", "query"])
def test_invalid_default_truncation_rejected_before_processing(entrypoint):
    async def exercise():
        llm, processed, sent = _frontend()
        params = SamplingParams(max_tokens=1, truncate_prompt_tokens=257)
        with pytest.raises(ValueError, match="greater than max_model_len"):
            if entrypoint == "open":
                await llm.add_streaming_prompt_chunk(
                    {"prompt_token_ids": [1]}, params, "lane", is_first_chunk=True
                )
            else:
                async for _ in llm.generate({"prompt_token_ids": [1]}, params, "query"):
                    pytest.fail("Invalid input must not produce an output")
        assert processed == []
        assert sent == []

    asyncio.run(exercise())


def test_preprocessed_native_chunk_and_query_do_not_rehash_media():
    async def exercise():
        llm, processed, sent = _frontend()
        params = SamplingParams(max_tokens=1)
        features = _media_features([("audio-0", 34)], {})
        native = _request("lane", range(64), params, features)
        query = _request("query", range(80), params, features)
        await llm.add_streaming_prompt_chunk(
            native,
            SamplingParams(max_tokens=1, truncate_prompt_tokens=257),
            "lane",
            is_first_chunk=True,
        )
        async for _ in llm.generate(query, params, "query"):
            pass
        assert processed == []
        assert sent[0] is native and sent[1] is query
        assert native.mm_features is features
        assert query.mm_features is features
        assert native.prompt_token_ids == list(range(64))

    asyncio.run(exercise())


def test_default_native_append_and_query_share_complete_kv_hash_prefix():
    _, sent = asyncio.run(_open_append_query(SamplingParams(max_tokens=1), None))
    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)
    lane = Request.from_engine_core_request(sent[0], hasher)
    lane.start_streaming_anchor()
    lane.append_streaming_prompt_chunk(
        prompt_token_ids=sent[1].prompt_token_ids,
        mm_features=sent[1].mm_features,
    )
    query = Request.from_engine_core_request(sent[2], hasher)
    assert lane.prompt_token_ids == query.prompt_token_ids[:128]
    assert Request._mm_feature_identity(lane.mm_features) == (
        Request._mm_feature_identity(query.mm_features)
    )
    assert len(lane.block_hashes) == 8
    assert lane.block_hashes == query.block_hashes[:8]

    # The old native default omitted truncation=False from each media identity.
    legacy = Request.from_engine_core_request(
        _request(
            "legacy",
            range(128),
            SamplingParams(max_tokens=1),
            _media_features([("audio-0", 34), ("audio-1", 66)], {}),
        ),
        hasher,
    )
    assert legacy.prompt_token_ids == lane.prompt_token_ids
    assert legacy.block_hashes[:2] == query.block_hashes[:2]
    assert legacy.block_hashes[2] != query.block_hashes[2]


def test_meaningful_explicit_tokenization_options_remain_in_media_identity():
    default = _media_features([("audio-0", 34)], {"truncation": False})
    explicit = _media_features(
        [("audio-0", 34)], {"truncation": False, "add_special_tokens": False}
    )
    assert default[0].identifier != explicit[0].identifier
