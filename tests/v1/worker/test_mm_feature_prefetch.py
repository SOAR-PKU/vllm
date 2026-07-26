# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
    PrefillContextMetadata,
)
from vllm.v1.engine.prefill_coordinator import StreamingPrefillCoordinator
from vllm.v1.request import Request
from vllm.v1.worker.mm_feature_prefetch import (
    MMFeaturePrefetchProtocolError,
    WorkerMMFeaturePrefetchCache,
)
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


def _item(size: int = 8) -> MultiModalKwargsItem:
    return MultiModalKwargsItem.from_elems(
        [
            MultiModalFieldElem(
                modality="audio",
                key="input_features",
                data=torch.arange(size, dtype=torch.int8),
                field=MultiModalSharedField(1),
            )
        ]
    )


def _feature(
    identifier: str = "feature-a",
    *,
    data: MultiModalKwargsItem | None = None,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=_item() if data is None else data,
        modality="audio",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=0, length=2),
    )


def _request(
    request_id: str,
    token_count: int,
    *,
    version: int,
    kind: str = "streaming",
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(token_count)),
        mm_features=[_feature()],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        prefill_context=PrefillContextMetadata(
            kind=kind,
            router_session_id=7,
            context_lifetime_id=11,
            context_version=version,
            context_role="active",
        ),
    )


def test_worker_cache_is_bounded_idempotent_and_non_evicting() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    item = _item()

    assert cache.put("a", item)
    assert cache.used_bytes == 8
    assert cache.put("a", _item())
    assert cache.used_bytes == 8
    assert not cache.put("b", _item())
    assert cache.get("a") is item
    with pytest.raises(MMFeaturePrefetchProtocolError):
        cache.get("b")


def test_worker_cache_materializes_explicit_reference() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    item = _item()
    assert cache.put("a", item)
    feature = _feature("a")
    feature.data = None

    cache.materialize_features([feature])

    assert feature.data is item


def test_worker_wrapper_materializes_before_formal_execution() -> None:
    wrapper = WorkerWrapperBase.__new__(WorkerWrapperBase)
    wrapper.mm_feature_prefetch_cache = WorkerMMFeaturePrefetchCache(
        max_bytes=8
    )
    wrapper.mm_receiver_cache = None
    item = _item()
    assert wrapper.mm_feature_prefetch_cache.put("a", item)
    feature = _feature("a")
    feature.data = None
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(mm_features=[feature])]
    )

    wrapper._apply_mm_cache(scheduler_output)

    assert feature.data is item


def test_coordinator_uses_reference_only_after_all_rank_acks() -> None:
    admitted: list[Request] = []
    submitted: list[str] = []

    def prefetch(features: list[MultiModalFeatureSpec]) -> set[str]:
        identifiers = {feature.identifier for feature in features}
        submitted.extend(identifiers)
        return identifiers

    coordinator = StreamingPrefillCoordinator(
        chunk_token_cap=8,
        block_size=2,
        request_block_hasher=None,
        admit_request=admitted.append,
        feature_prefetch_world_size=2,
        prefetch_mm_features=prefetch,
    )
    coordinator.add_streaming(_request("logical-1", 4, version=1))
    coordinator.add_streaming(_request("logical-2", 8, version=2))

    assert submitted == ["feature-a"]
    assert admitted[0].mm_features[0].data is not None

    coordinator.update_mm_prefetch_acks(
        [(0, "feature-a", True), (1, "feature-a", True)]
    )
    first = admitted[0]
    outputs = {
        0: EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id=first.request_id,
                    new_token_ids=[1],
                    finish_reason=FinishReason.LENGTH,
                )
            ]
        )
    }
    coordinator.update_after_step(
        SimpleNamespace(
            num_scheduled_tokens={first.request_id: 4},
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id=first.request_id,
                    num_computed_tokens=0,
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                num_computed_tokens=[],
            ),
        ),
        outputs,
    )

    assert len(admitted) == 2
    assert admitted[1].mm_features[0].data is None


def test_query_references_ready_features_but_never_waits_for_acks() -> None:
    def prefetch(features: list[MultiModalFeatureSpec]) -> set[str]:
        return {feature.identifier for feature in features}

    coordinator = StreamingPrefillCoordinator(
        chunk_token_cap=8,
        block_size=2,
        request_block_hasher=None,
        admit_request=lambda _request: None,
        feature_prefetch_world_size=2,
        prefetch_mm_features=prefetch,
    )
    first_query = _request(
        "query-inline",
        8,
        version=1,
        kind="user_query",
    )
    coordinator.observe_query(first_query)
    assert first_query.mm_features[0].data is not None

    coordinator.update_mm_prefetch_acks(
        [(0, "feature-a", True), (1, "feature-a", True)]
    )
    second_query = _request(
        "query-reference",
        8,
        version=2,
        kind="user_query",
    )
    coordinator.observe_query(second_query)
    assert second_query.mm_features[0].data is None


def test_disabled_prefetch_never_creates_feature_references() -> None:
    coordinator = StreamingPrefillCoordinator(
        chunk_token_cap=8,
        block_size=2,
        request_block_hasher=None,
        admit_request=lambda _request: None,
    )
    query = _request(
        "query-inline",
        8,
        version=1,
        kind="user_query",
    )

    coordinator.observe_query(query)

    assert query.mm_features[0].data is not None
