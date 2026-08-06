# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.mm_feature_prefetch as mm_feature_prefetch
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


def test_worker_cache_duplicate_generation_is_idempotent_and_pinned() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    item = _item()

    assert cache.put("a", 1, item)
    assert cache.used_bytes == 8
    assert cache.put("a", 1, _item())
    assert cache.used_bytes == 8
    assert not cache.put("b", 1, _item())
    assert cache.get("a") is item
    with pytest.raises(MMFeaturePrefetchProtocolError):
        cache.get("b")


def test_worker_cache_materializes_explicit_reference() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    item = _item()
    assert cache.put("a", 1, item)
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
    assert wrapper.mm_feature_prefetch_cache.put("a", 1, item)
    feature = _feature("a")
    feature.data = None
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(mm_features=[feature])]
    )

    wrapper._apply_mm_cache(scheduler_output)

    assert feature.data is item


def test_worker_cache_evicts_released_entry_in_lru_order() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=16)
    item_a = _item()
    item_b = _item()
    item_c = _item()

    assert cache.put("a", 1, item_a)
    assert cache.put("b", 1, item_b)
    assert cache.release("a", 1)
    assert cache.release("b", 1)

    # Materializing ``a`` makes it newer than ``b`` without re-pinning it.
    feature_a = _feature("a")
    feature_a.data = None
    cache.materialize_features([feature_a])
    assert feature_a.data is item_a

    assert cache.put("c", 1, item_c)
    assert "a" in cache
    assert "b" not in cache
    assert cache.get("c") is item_c


def test_worker_cache_never_evicts_pinned_entries() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    assert cache.put("a", 1, _item())

    assert not cache.put("b", 1, _item())
    assert "a" in cache
    assert "b" not in cache


def test_worker_cache_stale_release_does_not_unpin_new_generation() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    assert cache.put("a", 2, _item())

    assert not cache.release("a", 1)
    assert not cache.put("b", 1, _item())
    assert cache.release("a", 2)
    assert cache.put("b", 1, _item())
    assert "a" not in cache
    assert "b" in cache


def test_worker_cache_new_generation_replaces_and_repins() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    old_item = _item()
    new_item = _item()
    assert cache.put("a", 1, old_item)
    assert cache.release("a", 1)

    assert cache.put("a", 2, new_item)
    assert cache.get("a") is new_item
    assert not cache.put("a", 1, old_item)
    assert not cache.put("b", 1, _item())


def test_worker_cache_keeps_per_item_lru_eviction_at_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

    def record_debug(message: str, *args: object) -> None:
        messages.append(message % args)

    monkeypatch.setattr(
        mm_feature_prefetch.logger,
        "debug",
        record_debug,
    )
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    assert cache.put("a", 3, _item())
    assert cache.release("a", 3)

    assert cache.put("b", 4, _item())

    assert len(messages) == 1
    assert "identifier=a" in messages[0]
    assert "generation=3" in messages[0]
    assert "reason=lru_unpinned_capacity" in messages[0]
    metrics = cache.snapshot_metrics()
    assert metrics["evicted_entries"] == 1
    assert metrics["evicted_bytes"] == 8
    assert metrics["evicted_before_use_entries"] == 1
    assert metrics["evicted_before_use_bytes"] == 8


def test_worker_cache_logs_and_resets_interval_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []

    def record_info(message: str, *args: object) -> None:
        messages.append(message % args)

    monkeypatch.setattr(mm_feature_prefetch.logger, "info", record_info)
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    assert cache.put("a", 1, _item())

    cache.log_metrics(rank=3)

    assert len(messages) == 1
    assert '"rank":3' in messages[0]
    assert '"put_accepted_entries":1' in messages[0]
    assert cache.snapshot_metrics().get("put_accepted_entries", 0) == 0


def test_worker_cache_does_not_count_materialized_entry_as_evicted_before_use(
) -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    item = _item()
    assert cache.put("a", 1, item)
    feature = _feature("a")
    feature.data = None
    cache.materialize_features([feature])
    assert feature.data is item
    assert cache.release("a", 1)

    assert cache.put("b", 1, _item())

    metrics = cache.snapshot_metrics()
    assert metrics["materialize_hit_entries"] == 1
    assert metrics["materialize_hit_bytes"] == 8
    assert metrics["evicted_entries"] == 1
    assert metrics.get("evicted_before_use_entries", 0) == 0


def test_worker_cache_materializes_mixed_features_atomically() -> None:
    cache = WorkerMMFeaturePrefetchCache(max_bytes=8)
    cached_item = _item()
    inline_item = _item()
    assert cache.put("cached", 1, cached_item)
    cached = _feature("cached")
    cached.data = None
    missing = _feature("missing")
    missing.data = None
    inline = _feature("inline", data=inline_item)

    cache.materialize_features([cached, missing, inline])

    assert cached.data is cached_item
    assert missing.data is None
    assert inline.data is inline_item


def test_worker_wrapper_releases_exact_prefetch_generation() -> None:
    wrapper = WorkerWrapperBase.__new__(WorkerWrapperBase)
    wrapper.mm_feature_prefetch_cache = WorkerMMFeaturePrefetchCache(
        max_bytes=8
    )
    assert wrapper.cache_prefetched_mm_feature("a", 2, _item())

    assert not wrapper.release_prefetched_mm_feature("a", 1)
    assert wrapper.release_prefetched_mm_feature("a", 2)


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
        [(0, "feature-a", 1, True), (1, "feature-a", 1, True)]
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
        [(0, "feature-a", 1, True), (1, "feature-a", 1, True)]
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
