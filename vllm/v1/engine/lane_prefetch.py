# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MM feature transport for append-only lanes and independent query snapshots.

The input thread owns only locked, provisional transport bookkeeping. Core owns
validated lanes and dispatches. Neither path waits for a feature ACK. References
are substituted in dispatch copies only, after Scheduler/connector metadata has
been constructed; exact-generation leases last until the step result commits.
"""

from __future__ import annotations

import threading
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.v1.engine.prefill_coordinator import (
    ContextKey,
    MMFeaturePrefetchTracker,
    _feature_descriptor,
    _feature_end,
    _first_feature_ending_after,
    _prefetch_horizon_target,
)

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.request import Request


class NativeLaneFeaturePrefetch:
    """Reuse the baseline bounded FIFO/all-TP-ACK transport without carriers."""

    def __init__(
        self,
        tracker: MMFeaturePrefetchTracker,
        *,
        reference_dispatch: bool = True,
        snapshot_metadata_keys: frozenset[str] | None = None,
    ) -> None:
        self.tracker = tracker
        self.reference_dispatch = reference_dispatch
        self.snapshot_metadata_keys = snapshot_metadata_keys
        self._input_lock = threading.RLock()
        self._input_frontiers: dict[str, tuple[ContextKey, int, int]] = {}
        self._pending_updates: dict[str, tuple[ContextKey, str]] = {}
        self._retired: set[ContextKey] = set()
        self._lanes: dict[str, Request] = {}
        self._queries: dict[str, Request] = {}
        self._feature_counts: dict[str, int] = {}
        self._inline: dict[str, object] = {}
        self._step_leases: dict[int, list[str]] = {}
        self._step_sequence = 0

    @staticmethod
    def _key(request: Request) -> ContextKey | None:
        metadata = request.prefill_context
        if metadata is None:
            return None
        return metadata.router_session_id, metadata.context_lifetime_id

    @staticmethod
    def _operation(request: Request) -> str:
        metadata = request.prefill_context
        assert metadata is not None
        return (
            f"lane-update:{len(request.request_id)}:{request.request_id}:"
            f"{metadata.context_version}:{int(request.is_streaming_prompt_finished)}"
        )

    @staticmethod
    def _query_owner(request_id: str) -> str:
        return f"lane-query:{request_id}"

    def offer_early(self, request: Request) -> None:
        """Offer delta-local features using a transport-only cumulative offset."""

        key = self._key(request)
        if key is None:
            return
        metadata = request.prefill_context
        assert metadata is not None
        if metadata.kind == "user_query":
            # A query owns its snapshot independently of context retirement.
            self.tracker.hold_provisional_features(
                self._query_owner(request.request_id), request.mm_features
            )
            return
        if not request.is_streaming_prompt or metadata.kind != "streaming":
            return
        with self._input_lock:
            if key in self._retired:
                return
            previous = self._input_frontiers.get(request.request_id)
            base = 0
            if previous is not None:
                old_key, version, base = previous
                if old_key != key or metadata.context_version <= version:
                    return
            total = base + len(request.prompt_token_ids or ())
            self._input_frontiers[request.request_id] = (
                key,
                metadata.context_version,
                total,
            )
            horizon = self.tracker.horizon_target(key)
            if horizon is None:
                # Only the first OPEN needs a provisional horizon. Later
                # APPENDs use the scalar published by Core's physical progress.
                if base:
                    return
                horizon = self._horizon(request)
            features = []
            for feature in request.mm_features:
                if base + _feature_end(feature) <= horizon:
                    normalized = copy(feature)
                    normalized.mm_position = replace(
                        feature.mm_position, offset=base + feature.mm_position.offset
                    )
                    features.append(normalized)
            operation = self._operation(request)
            self._pending_updates[operation] = (key, request.request_id)
            self.tracker.hold_provisional_features(operation, features)

    def _restore(self, features: list[MultiModalFeatureSpec]) -> None:
        for feature in features:
            if feature.data is None:
                feature.data = self._inline.get(feature.identifier)
            if feature.data is not None:
                self._inline[feature.identifier] = feature.data

    def apply_update(self, lane: Request, incoming: Request) -> None:
        """Confirm a validated APPEND without changing any in-flight snapshot."""

        self._lanes[lane.request_id] = lane
        added_count = len(incoming.mm_features)
        if added_count:
            self._restore(lane.mm_features[-added_count:])
        self._refresh(lane)
        if incoming.prefill_context is not None:
            with self._input_lock:
                operation = self._operation(incoming)
                self.tracker.cancel_request(operation)
                self._pending_updates.pop(operation, None)

    def observe_query(self, request: Request) -> None:
        self._restore(request.mm_features)
        self._queries[request.request_id] = request
        self.tracker.hold_provisional_features(
            self._query_owner(request.request_id), request.mm_features
        )

    def _horizon(self, lane: Request) -> int:
        # Releasing GPU blocks must not rewind the look-ahead window to zero.
        # Actual encoder dispatch still has per-feature inline fallback if a
        # missing KV prefix needs recomputation after cache eviction.
        frontier = max(lane.num_computed_tokens, lane.streaming_anchor_confirmed_tokens)
        target = _prefetch_horizon_target(
            request=lane,
            computed_frontier=frontier,
            inflight_target=None,
            horizon_chunks=self.tracker.horizon_chunks,
            chunk_token_cap=self.tracker.chunk_token_cap,
            block_size=self.tracker.block_size,
        )
        if self.tracker.horizon_chunks and target == lane.num_prompt_tokens:
            # Reserve the unused part of the bounded horizon for deltas that
            # have reached the input thread but have not yet reached Core.
            # Only complete MM features inside that window are offered early.
            target = max(
                target,
                frontier + self.tracker.horizon_chunks * self.tracker.chunk_token_cap,
            )
        return target

    def _refresh(self, lane: Request) -> None:
        key = self._key(lane)
        if key is None:
            return
        target = self._horizon(lane)
        self.tracker.publish_horizon(key, target)
        count = _first_feature_ending_after(lane.mm_features, target)
        previous = self._feature_counts.get(lane.request_id, 0)
        if count > previous:
            metadata = lane.prefill_context
            assert metadata is not None
            self.tracker.offer_for_context(
                key,
                lane.mm_features[previous:count],
                context_version=metadata.context_version,
                total_feature_count=count,
                terminal_feature=_feature_descriptor(lane.mm_features[count - 1]),
            )
            self._feature_counts[lane.request_id] = count
        bounded = lane.mm_features[:count]
        if self.tracker.has_retryable_features(bounded):
            self.tracker.offer(bounded)

    def prepare_dispatch(
        self, output: SchedulerOutput, requests: dict[str, Request]
    ) -> None:
        """Pin READY data for this step; retain per-feature inline fallback."""

        for request_id in output.num_scheduled_tokens:
            request = requests.get(request_id)
            if request is not None and request.is_streaming_anchor:
                # Scheduler has already advanced its in-flight token frontier.
                self._refresh(request)
            elif request_id in self._queries:
                self.tracker.offer(self._queries[request_id].mm_features)
        self._step_sequence += 1
        leases: list[str] = []
        self._step_leases[id(output)] = leases

        def features_for_step(
            request_id: str, features: list[MultiModalFeatureSpec]
        ) -> list[MultiModalFeatureSpec]:
            features = [copy(feature) for feature in features]
            if not self.reference_dispatch or not features:
                return features
            lease = f"lane-step:{self._step_sequence}:{request_id}"
            ready = self.tracker.acquire_ready_dispatch(
                lease, [feature.identifier for feature in features]
            )
            leases.append(lease)
            for feature in features:
                if feature.identifier in ready:
                    feature.data = None
            return features

        output.scheduled_new_reqs = [copy(item) for item in output.scheduled_new_reqs]
        for item in output.scheduled_new_reqs:
            item.mm_features = (
                self._metadata_snapshot(item.mm_features)
                if self.snapshot_metadata_keys is not None
                else features_for_step(item.req_id, item.mm_features)
            )
        # The connector may retain the original prompt_updates objects. Do not
        # replace their payloads with references after it has built metadata.
        cached = copy(output.scheduled_cached_reqs)
        cached.prompt_updates = {
            request_id: copy(update)
            for request_id, update in cached.prompt_updates.items()
        }
        for request_id, update in cached.prompt_updates.items():
            update.mm_features = (
                self._metadata_snapshot(update.mm_features)
                if self.snapshot_metadata_keys is not None
                else features_for_step(request_id, update.mm_features)
            )
        output.scheduled_cached_reqs = cached
        if self.snapshot_metadata_keys is not None:
            # Unlike the complete position metadata, encoder payloads are
            # bounded by this step's actual encoder scheduling decisions.
            output.scheduled_mm_features = {}
            for request_id, indices in output.scheduled_encoder_inputs.items():
                features = features_for_step(
                    request_id,
                    [requests[request_id].mm_features[index] for index in indices],
                )
                output.scheduled_mm_features[request_id] = dict(zip(indices, features))

    def _metadata_snapshot(
        self, features: list[MultiModalFeatureSpec]
    ) -> list[MultiModalFeatureSpec]:
        assert self.snapshot_metadata_keys is not None
        result = []
        for feature in features:
            descriptor = copy(feature)
            if feature.data is not None:
                elements = [
                    value
                    for key, value in feature.data.items()
                    if key in self.snapshot_metadata_keys
                ]
                descriptor.data = (
                    MultiModalKwargsItem.from_elems(elements) if elements else None
                )
            result.append(descriptor)
        return result

    def reset_after_worker_cache_clear(self) -> None:
        """Rebuild bounded offers after the shared tracker has been reset."""

        self._feature_counts.clear()
        for lane in self._lanes.values():
            self._refresh(lane)
        for query in self._queries.values():
            self.tracker.offer(query.mm_features)

    def finish_step(
        self, output: SchedulerOutput, results: dict[int, EngineCoreOutputs]
    ) -> None:
        for lease in self._step_leases.pop(id(output), ()):
            self.tracker.release_dispatch(lease)
        self.finish_requests(
            [
                item.request_id
                for result in results.values()
                for item in result.outputs
                if item.finished
            ]
        )

    def finish_requests(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            self._queries.pop(request_id, None)
            self.tracker.cancel_request(self._query_owner(request_id))
            lane = self._lanes.pop(request_id, None)
            self._feature_counts.pop(request_id, None)
            if lane is not None:
                key = self._key(lane)
                if key is not None:
                    self.tracker.release_context(key)
            with self._input_lock:
                self._input_frontiers.pop(request_id, None)
                for operation, (_key, owner) in list(self._pending_updates.items()):
                    if owner == request_id:
                        self.tracker.cancel_request(operation)
                        self._pending_updates.pop(operation)
        if request_ids:
            self._prune_inline()

    def retire_context(self, key: ContextKey) -> None:
        with self._input_lock:
            self._retired.add(key)
            for request_id, (context, _, _) in list(self._input_frontiers.items()):
                if context == key:
                    self._input_frontiers.pop(request_id)
            for operation, (context, _) in list(self._pending_updates.items()):
                if context == key:
                    self.tracker.cancel_request(operation)
                    self._pending_updates.pop(operation)
        self.finish_requests(
            [
                request_id
                for request_id, lane in self._lanes.items()
                if self._key(lane) == key
            ]
        )
        self.tracker.release_context(key)

    def _prune_inline(self) -> None:
        live = {
            feature.identifier
            for request in (*self._lanes.values(), *self._queries.values())
            for feature in request.mm_features
        }
        self._inline = {key: data for key, data in self._inline.items() if key in live}
