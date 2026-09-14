# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import PrefillContextMetadata
from vllm.v1.engine import core as core_module
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.lane_prefetch import NativeLaneFeaturePrefetch
from vllm.v1.engine.prefill_coordinator import MMFeaturePrefetchTracker
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test


def _lane(*, tokens=32, version=1):
    return Request(
        request_id="lane",
        prompt_token_ids=[1] * tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        mm_features=[
            MultiModalFeatureSpec(
                data=object(),
                modality="audio",
                identifier=f"v{version}-f{offset}",
                mm_position=PlaceholderRange(offset=offset, length=2),
            )
            for offset in range(0, tokens, 4)
        ],
        prefill_context=PrefillContextMetadata(
            kind="streaming",
            router_session_id=7,
            context_lifetime_id=11,
            context_version=version,
        ),
        is_streaming_prompt=True,
    )


def _prefetch(*, horizon=2):
    submitted = []

    def submit(items):
        submitted.extend(items)
        return {(item.identifier, item.generation) for item in items}

    tracker = MMFeaturePrefetchTracker(
        world_size=2,
        chunk_token_cap=4,
        block_size=1,
        horizon_chunks=horizon,
        submit=submit,
        release=lambda items: None,
    )
    return NativeLaneFeaturePrefetch(tracker), submitted


@pytest.mark.parametrize("physical, expected", [(0, 20), (8, 20), (20, 28)])
def test_horizon_uses_confirmed_or_inflight_frontier(physical, expected):
    manager, _ = _prefetch()
    lane = _lane()
    lane.start_streaming_anchor()
    lane.streaming_anchor_confirmed_tokens = 12
    lane.num_computed_tokens = physical

    assert manager._horizon(lane) == expected


def test_release_preserves_future_feature_prefetch_window():
    manager, submitted = _prefetch()
    lane = _lane()
    manager.offer_early(lane)
    lane.start_streaming_anchor()
    manager.apply_update(lane, lane)
    assert [item.identifier for item in submitted] == ["v1-f0", "v1-f4"]

    lane.streaming_anchor_confirmed_tokens = 12
    lane.num_computed_tokens = 0
    lane.num_streaming_kv_releases = 1
    manager._refresh(lane)

    assert manager.tracker.horizon_target((7, 11)) == 20
    assert [item.identifier for item in submitted] == [
        "v1-f0",
        "v1-f4",
        "v1-f8",
        "v1-f12",
        "v1-f16",
    ]
    assert lane.num_computed_tokens == 0
    assert lane.streaming_anchor_confirmed_tokens == 12


def test_idle_released_lane_reserves_horizon_for_early_appends():
    manager, submitted = _prefetch()
    lane = _lane(tokens=4)
    manager.offer_early(lane)
    lane.start_streaming_anchor()
    manager.apply_update(lane, lane)
    lane.streaming_anchor_confirmed_tokens = 4
    lane.num_computed_tokens = 0
    manager._refresh(lane)

    assert manager.tracker.horizon_target((7, 11)) == 12
    for version in (2, 3, 4):
        manager.offer_early(_lane(tokens=4, version=version))

    assert [item.identifier for item in submitted] == ["v1-f0", "v2-f0", "v3-f0"]


def test_disabled_prefetch_remains_disabled_after_release():
    manager, submitted = _prefetch(horizon=0)
    lane = _lane()
    lane.start_streaming_anchor()
    lane.streaming_anchor_confirmed_tokens = 12
    lane.num_computed_tokens = 0

    manager._refresh(lane)

    assert manager._horizon(lane) == 0
    assert submitted == []


@pytest.mark.parametrize("confirmed, expected", [(0, 0), (12, 12), (24, 16)])
def test_progress_log_uses_confirmed_frontier_after_allocation_release(
    monkeypatch, confirmed, expected
):
    lane = _lane()
    lane.start_streaming_anchor()
    lane.streaming_anchor_confirmed_tokens = confirmed
    lane.num_computed_tokens = 0
    lane.num_streaming_kv_releases = 3
    engine = object.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(requests={"lane": lane})
    engine._optimized_coalesce_enabled = True
    engine._coordinator_cpu_metrics_enabled = True
    output = SimpleNamespace(streaming_anchor_snapshots={"lane": (1, 16)})
    records = []
    monkeypatch.setattr(
        core_module,
        "logger",
        SimpleNamespace(
            isEnabledFor=lambda level: True,
            info=lambda message, payload: records.append(json.loads(payload)),
        ),
    )
    monkeypatch.setattr(core_module.time, "time_ns", lambda: 123456)

    engine._log_native_streaming_prompt_progress(output)

    assert records == [
        {
            "request_id": "lane",
            "router_session_id": 7,
            "context_lifetime_id": 11,
            "prompt_revision": 1,
            "snapshot_prompt_tokens": 16,
            "confirmed_tokens": expected,
            "kv_allocation_releases": 3,
            "completed_wall_ns": 123456,
        }
    ]
    assert lane.num_computed_tokens == 0
    assert lane.streaming_anchor_confirmed_tokens == confirmed
