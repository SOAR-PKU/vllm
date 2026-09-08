# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.mm_feature_prefetch import MMFeaturePrefetchProtocolError
from vllm.v1.worker.worker_base import WorkerWrapperBase

pytestmark = pytest.mark.cpu_test


def _feature(identifier, offset, data=None):
    return MultiModalFeatureSpec(
        identifier=identifier,
        modality="audio",
        data=data,
        mm_position=PlaceholderRange(offset=offset, length=1),
    )


@pytest.mark.parametrize("cache_kind", ["custom", "native", "inline"])
def test_only_selected_payloads_materialize_and_cached_metadata_stays_light(cache_kind):
    data = object()
    descriptor = _feature("current", 1, data={"audio_feature_lengths": 1})
    future = _feature("future", 3)
    payload = _feature("current", 1, data=data if cache_kind == "inline" else None)
    output = SimpleNamespace(
        scheduled_encoder_inputs={"lane": [0]},
        scheduled_mm_features={"lane": {0: payload}},
        scheduled_new_reqs=[SimpleNamespace(mm_features=[descriptor, future])],
    )
    visited = []

    def materialize(features):
        visited.extend(feature.identifier for feature in features)
        for feature in features:
            feature.data = data
        return features

    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.mm_feature_prefetch_cache = (
        SimpleNamespace(materialize_features=materialize)
        if cache_kind == "custom"
        else None
    )
    wrapper.mm_receiver_cache = (
        SimpleNamespace(get_and_update_features=materialize)
        if cache_kind == "native"
        else None
    )
    wrapper._apply_mm_cache(output)
    assert visited == ([] if cache_kind == "inline" else ["current"])

    runner = object.__new__(GPUModelRunner)
    runner.requests = {"lane": SimpleNamespace(mm_features=[descriptor, future])}
    kwargs, positions = runner._batch_mm_kwargs_from_scheduler(output)
    assert kwargs == [data]
    assert positions == [("current", payload.mm_position)]
    assert descriptor.data == {"audio_feature_lengths": 1}
    assert future.data is None


def test_compact_dispatch_rejects_unresolved_current_payload():
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.mm_feature_prefetch_cache = None
    wrapper.mm_receiver_cache = None
    output = SimpleNamespace(
        scheduled_mm_features={"lane": {0: _feature("current", 1)}}
    )
    with pytest.raises(MMFeaturePrefetchProtocolError, match="unresolved"):
        wrapper._apply_mm_cache(output)


@pytest.mark.parametrize("mismatch", ["missing", "identifier", "offset", "unresolved"])
def test_encoder_payload_is_validated_against_scheduler_and_prompt(mismatch):
    descriptor = _feature("current", 1)
    payload = _feature(
        "different" if mismatch == "identifier" else "current",
        2 if mismatch == "offset" else 1,
        data=None if mismatch == "unresolved" else object(),
    )
    output = SimpleNamespace(
        scheduled_encoder_inputs={"lane": [0]},
        scheduled_mm_features={} if mismatch == "missing" else {"lane": {0: payload}},
    )
    runner = object.__new__(GPUModelRunner)
    runner.requests = {"lane": SimpleNamespace(mm_features=[descriptor])}
    with pytest.raises(RuntimeError, match="On-demand MM"):
        runner._batch_mm_kwargs_from_scheduler(output)


def test_empty_compact_dispatch_does_not_materialize_future_descriptors():
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.mm_feature_prefetch_cache = None
    wrapper.mm_receiver_cache = None
    future = _feature("future", 100)
    output = SimpleNamespace(
        scheduled_mm_features={},
        scheduled_encoder_inputs={},
        scheduled_new_reqs=[SimpleNamespace(mm_features=[future])],
    )
    wrapper._apply_mm_cache(output)
    runner = object.__new__(GPUModelRunner)
    assert runner._batch_mm_kwargs_from_scheduler(output) == ([], [])
    assert future.data is None
