# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EngineCore-side coordination for cumulative streaming-prefill requests.

The application submits every logical media-context version.  This coordinator
keeps those original request IDs as completion obligations while admitting at
most one bounded, full-prefix physical request per context to the Scheduler.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
    LogicalRequestCompletion,
    PrefillContextMetadata,
)
from vllm.v1.engine.coordinator_metrics import (
    COORDINATOR_MATERIALIZE_METRIC,
    COORDINATOR_PHYSICAL_BUILD_METRIC,
    COORDINATOR_PREFIX_IDENTITY_METRIC,
    COORDINATOR_QUERY_COVERAGE_METRIC,
    COORDINATOR_QUERY_METRIC,
    COORDINATOR_STREAMING_ADD_METRIC,
    COORDINATOR_VERSION_VALIDATE_METRIC,
    EngineCoreCoordinatorMetrics,
    thread_cpu_time_ns,
)
from vllm.v1.request import Request
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.mm_feature_prefetch import (
    MMFeaturePrefetchAck,
    MMFeaturePrefetchItem,
    MMFeaturePrefetchRelease,
)

logger = init_logger(__name__)

_STREAMING_KIND = "streaming"
_QUERY_KIND = "user_query"
_PHYSICAL_ID_PREFIX = "__llm_rtc_streaming_prefill__"

ContextKey = tuple[int, int]
FeatureIdentity = tuple[str, str, int, int]


def _feature_end(feature: MultiModalFeatureSpec) -> int:
    position = feature.mm_position
    return position.offset + position.length


def _feature_descriptor(feature: MultiModalFeatureSpec) -> FeatureIdentity:
    return (
        feature.modality,
        feature.identifier,
        feature.mm_position.offset,
        feature.mm_position.length,
    )


def _lora_identity(request: Request) -> tuple[int, str, str] | None:
    lora = request.lora_request
    if lora is None:
        return None
    return (lora.lora_int_id, lora.lora_name, lora.lora_path)


@dataclass(frozen=True, slots=True)
class _PrefixIdentity:
    target: int
    full_block_count: int
    terminal_block_hash: BlockHash | None
    tail_tokens: tuple[int, ...]
    tail_features: tuple[FeatureIdentity, ...]
    completed_feature_count: int
    terminal_feature: FeatureIdentity | None
    fallback_token_digest: bytes | None
    fallback_feature_digest: bytes | None
    cache_salt: str | None
    lora: tuple[int, str, str] | None


def _fallback_token_digest(tokens: list[int], target: int) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for token_index in range(target):
        token_id = tokens[token_index]
        digest.update(token_id.to_bytes(8, "little", signed=False))
    return digest.digest()


def _fallback_feature_digest(
    features: list[MultiModalFeatureSpec],
    target: int,
) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for feature in features:
        if _feature_end(feature) > target:
            continue
        for value in (feature.modality, feature.identifier):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(
            feature.mm_position.offset.to_bytes(8, "little", signed=False)
        )
        digest.update(
            feature.mm_position.length.to_bytes(8, "little", signed=False)
        )
    return digest.digest()


def _first_feature_ending_after(
    features: list[MultiModalFeatureSpec],
    token_index: int,
) -> int:
    """Return a lower bound for features intersecting a bounded token tail.

    vLLM requires multimodal features to be ordered by prompt position. Media
    placeholders are non-overlapping, so their end positions are monotonic too.
    """

    low = 0
    high = len(features)
    while low < high:
        middle = (low + high) // 2
        if _feature_end(features[middle]) <= token_index:
            low = middle + 1
        else:
            high = middle
    return low


def _safe_physical_target(
    *,
    request: Request,
    frontier: int,
    target: int,
    block_size: int,
) -> int:
    """Apply the exact MM-boundary and block-alignment physical chunk rules."""

    latest = request.num_prompt_tokens
    split_index = _first_feature_ending_after(
        request.mm_features,
        target,
    )
    if split_index < len(request.mm_features):
        feature = request.mm_features[split_index]
        start = feature.mm_position.offset
        end = _feature_end(feature)
        if start < target < end:
            target = start if start > frontier else min(end, latest)

    if target < latest:
        aligned = target - (target % block_size)
        split_index = _first_feature_ending_after(
            request.mm_features,
            aligned,
        )
        splits_feature = False
        if split_index < len(request.mm_features):
            feature = request.mm_features[split_index]
            splits_feature = (
                feature.mm_position.offset < aligned < _feature_end(feature)
            )
        if aligned > frontier and not splits_feature:
            target = aligned
    return min(target, latest)


def _prefetch_horizon_target(
    *,
    request: Request,
    computed_frontier: int,
    inflight_target: int | None,
    horizon_chunks: int,
    chunk_token_cap: int,
    block_size: int,
) -> int:
    """Return the end of N not-yet-admitted physical chunks.

    An already admitted physical request is the starting point rather than one
    of the ahead chunks. Reusing ``_safe_physical_target`` keeps prefetch and
    formal physical-request boundaries identical for MM placeholders larger
    than, or intersecting, the nominal token cap.
    """

    if horizon_chunks <= 0:
        return 0
    cursor = max(
        0,
        computed_frontier,
        0 if inflight_target is None else inflight_target,
    )
    for _ in range(horizon_chunks):
        if cursor >= request.num_prompt_tokens:
            break
        tentative = min(
            request.num_prompt_tokens,
            cursor + chunk_token_cap,
        )
        next_target = _safe_physical_target(
            request=request,
            frontier=cursor,
            target=tentative,
            block_size=block_size,
        )
        if next_target <= cursor:
            break
        cursor = next_target
    return cursor


def _tail_feature_identity(
    features: list[MultiModalFeatureSpec],
    tail_start: int,
    target: int,
) -> tuple[FeatureIdentity, ...]:
    identities: list[FeatureIdentity] = []
    index = _first_feature_ending_after(features, tail_start)
    while index < len(features):
        feature = features[index]
        if feature.mm_position.offset >= target:
            break
        if _feature_end(feature) <= target:
            identities.append(_feature_descriptor(feature))
        index += 1
    return tuple(identities)


def _prefix_identity(
    request: Request,
    block_size: int,
    target: int | None = None,
) -> _PrefixIdentity:
    tokens = request.prompt_token_ids
    if tokens is None:
        raise ValueError("Streaming-prefill identity requires prompt token IDs")
    if target is None:
        target = len(tokens)
    if not 0 <= target <= len(tokens):
        raise ValueError(
            f"Invalid streaming-prefill identity target {target} for "
            f"{len(tokens)} prompt tokens"
        )

    full_block_count = target // block_size
    has_block_checkpoint = (
        full_block_count == 0 or len(request.block_hashes) >= full_block_count
    )
    if has_block_checkpoint:
        terminal_block_hash = (
            request.block_hashes[full_block_count - 1]
            if full_block_count
            else None
        )
        tail_start = full_block_count * block_size
        fallback_token_digest = None
        fallback_feature_digest = None
    else:
        # Compatibility for tests and unsupported callers without prefix
        # caching. Production coordinator configurations require vLLM prefix
        # caching, so normal ingress never takes this full-prefix slow path.
        terminal_block_hash = None
        tail_start = target
        fallback_token_digest = _fallback_token_digest(tokens, target)
        fallback_feature_digest = _fallback_feature_digest(
            request.mm_features,
            target,
        )
    completed_feature_count = _first_feature_ending_after(
        request.mm_features,
        target,
    )
    terminal_feature = (
        _feature_descriptor(request.mm_features[completed_feature_count - 1])
        if completed_feature_count
        else None
    )

    return _PrefixIdentity(
        target=target,
        full_block_count=full_block_count,
        terminal_block_hash=terminal_block_hash,
        tail_tokens=tuple(tokens[tail_start:target]),
        tail_features=_tail_feature_identity(
            request.mm_features,
            tail_start,
            target,
        ),
        completed_feature_count=completed_feature_count,
        terminal_feature=terminal_feature,
        fallback_token_digest=fallback_token_digest,
        fallback_feature_digest=fallback_feature_digest,
        cache_salt=request.cache_salt,
        lora=_lora_identity(request),
    )


@dataclass(slots=True)
class _LogicalPrefill:
    request_id: str
    client_index: int
    context_version: int
    identity: _PrefixIdentity
    trace_headers: Mapping[str, str] | None
    sequence: int

    @property
    def target(self) -> int:
        return self.identity.target


@dataclass(slots=True)
class _PhysicalPrefix:
    request_id: str
    request: Request
    target: int
    source: Request


@dataclass(slots=True)
class _QueryFence:
    request: Request
    key: ContextKey
    context_version: int
    coverage_limit: int = 0
    confirmed_frontier: int = 0
    snapshot_checked: bool = False

    @property
    def request_id(self) -> str:
        return self.request.request_id


@dataclass(slots=True)
class _ContextState:
    key: ContextKey
    role: str
    latest_version: int
    latest_target: Request
    version_identities: dict[int, _PrefixIdentity] = field(default_factory=dict)
    feature_count: int = 0
    terminal_feature: FeatureIdentity | None = None
    logical: OrderedDict[str, _LogicalPrefill] = field(
        default_factory=OrderedDict
    )
    logical_by_target: list[tuple[int, int, str]] = field(default_factory=list)
    next_logical_sequence: int = 0
    computed_frontier: int = 0
    inflight: _PhysicalPrefix | None = None
    covering_queries: dict[str, _QueryFence] = field(default_factory=dict)
    next_physical_sequence: int = 0
    prefetch_horizon_target: int = 0
    prefetch_feature_count: int = 0


class _FeatureTransferStatus(Enum):
    INFLIGHT = auto()
    READY = auto()


class _FeatureDeliveryMode(Enum):
    ALL_INLINE = auto()
    MIXED = auto()
    ALL_REFERENCE = auto()


@dataclass(slots=True)
class _FeatureTransferState:
    generation: int
    status: _FeatureTransferStatus
    ready_ranks: set[int] = field(default_factory=set)
    reference_count: int = 0


@dataclass(frozen=True, slots=True)
class _ContextFeatureProgress:
    version: int
    feature_count: int
    terminal_feature: FeatureIdentity | None


class MMFeaturePrefetchTracker:
    """Thread-safe transport and residency state for prefetched MM features.

    The EngineCore input thread may offer data while the EngineCore main loop
    is executing a model step. Context/coalescing state remains owned by the
    main loop; this tracker owns only feature generations, rank ACKs, and the
    leases that make a worker-side cache entry safe to reference.
    """

    def __init__(
        self,
        *,
        world_size: int,
        chunk_token_cap: int,
        block_size: int,
        horizon_chunks: int,
        submit: (
            Callable[
                [list[MMFeaturePrefetchItem]],
                set[str] | set[tuple[str, int]],
            ]
            | None
        ),
        release: (
            Callable[[list[MMFeaturePrefetchRelease]], None] | None
        ),
    ) -> None:
        self.world_size = world_size
        self.chunk_token_cap = chunk_token_cap
        self.block_size = block_size
        self.horizon_chunks = horizon_chunks
        self.submit = submit
        self.release = release
        self._lock = threading.RLock()
        self._states: dict[str, _FeatureTransferState] = {}
        self._last_generation: dict[str, int] = {}
        # A bounded submit queue can reject a side-channel item transiently.
        # Keep only identifiers that remain context-owned, so a later refresh
        # of the same physical horizon can retry without widening that horizon.
        self._retryable_identifiers: set[str] = set()

        # Provisional request ownership is registered by the input thread and
        # converted to context ownership only after the main loop validates the
        # request. Dispatch ownership protects references already admitted to
        # the Scheduler even if their context is retired concurrently.
        self._provisional: dict[str, set[str]] = {}
        self._confirmed_requests: set[str] = set()
        self._context_requests: dict[ContextKey, set[str]] = {}
        self._context_features: dict[ContextKey, set[str]] = {}
        self._dispatch_features: dict[
            str, set[tuple[str, int]]
        ] = {}
        self._owner_counts: dict[str, int] = {}
        # Updated only after main-thread prefix validation. The input thread
        # uses this checkpoint to offer the append-only suffix instead of
        # rescanning every cumulative feature on every media version.
        self._confirmed_context_progress: dict[
            ContextKey, _ContextFeatureProgress
        ] = {}
        # Input preprocessing can run while the EngineCore main thread is in a
        # model step. Track the newest structurally compatible version already
        # offered by that input thread so a burst of cumulative versions sends
        # only each new suffix once instead of repeatedly scanning from the
        # last main-thread confirmation.
        self._offered_context_progress: dict[
            ContextKey, _ContextFeatureProgress
        ] = {}
        # Main-thread physical planning publishes the current token horizon.
        # The input thread reads only this locked scalar checkpoint; Scheduler
        # and coalescing state remain main-thread owned.
        self._context_horizon_targets: dict[ContextKey, int] = {}
        self._metrics: dict[str, int] = {}

    def _record_locked(self, name: str, value: int = 1) -> None:
        self._metrics[name] = self._metrics.get(name, 0) + int(value)

    def snapshot_metrics(self) -> dict[str, int]:
        with self._lock:
            snapshot = dict(self._metrics)
            snapshot["transfer_states"] = len(self._states)
            snapshot["provisional_requests"] = len(self._provisional)
            snapshot["dispatch_leases"] = sum(
                len(leases) for leases in self._dispatch_features.values()
            )
            snapshot["retryable_features"] = len(self._retryable_identifiers)
            return snapshot

    def drain_metrics(self) -> dict[str, int]:
        """Return interval counters while retaining live transfer state.

        The EngineCore emits this snapshot periodically so an externally
        terminated benchmark still has useful data.  Only counters are reset:
        ownership, generations, and readiness are intentionally untouched.
        """

        with self._lock:
            snapshot = dict(self._metrics)
            self._metrics.clear()
            snapshot["transfer_states"] = len(self._states)
            snapshot["provisional_requests"] = len(self._provisional)
            snapshot["dispatch_leases"] = sum(
                len(leases) for leases in self._dispatch_features.values()
            )
            snapshot["retryable_features"] = len(self._retryable_identifiers)
            return snapshot

    def has_retryable_features(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> bool:
        """Check whether this already-bounded prefix has a resend candidate."""

        with self._lock:
            return any(
                feature.identifier in self._retryable_identifiers
                for feature in features
                if feature.identifier
            )

    def publish_horizon(self, key: ContextKey, target: int) -> None:
        """Publish a monotonic token boundary for input-thread early offers."""

        with self._lock:
            self._context_horizon_targets[key] = max(
                target,
                self._context_horizon_targets.get(key, 0),
            )

    def offer_for_context(
        self,
        key: ContextKey,
        features: list[MultiModalFeatureSpec],
        *,
        context_version: int,
        total_feature_count: int,
        terminal_feature: FeatureIdentity | None,
    ) -> None:
        """Own and offer a validated prefix newly entering the horizon."""

        identifiers = {
            feature.identifier for feature in features if feature.identifier
        }
        with self._lock:
            context_owned = self._context_features.setdefault(key, set())
            for identifier in identifiers - context_owned:
                context_owned.add(identifier)
                self._add_owner(identifier)
            progress = _ContextFeatureProgress(
                version=context_version,
                feature_count=total_feature_count,
                terminal_feature=terminal_feature,
            )
            confirmed = self._confirmed_context_progress.get(key)
            if (
                confirmed is None
                or context_version > confirmed.version
                or (
                    context_version == confirmed.version
                    and total_feature_count >= confirmed.feature_count
                )
            ):
                self._confirmed_context_progress[key] = progress
            offered = self._offered_context_progress.get(key)
            if (
                offered is None
                or context_version > offered.version
                or (
                    context_version == offered.version
                    and total_feature_count >= offered.feature_count
                )
            ):
                self._offered_context_progress[key] = progress
        # Scan the whole bounded prefix. ``offer`` deduplicates READY/INFLIGHT
        # identifiers and gives a previously rejected item a later retry.
        self.offer(features)

    def offer_early(self, request: Request) -> None:
        """Register provisional ownership and submit immutable data snapshots."""

        metadata = request.prefill_context
        if (
            metadata is None
            or metadata.kind not in {_STREAMING_KIND, _QUERY_KIND}
        ):
            return
        key = (
            metadata.router_session_id,
            metadata.context_lifetime_id,
        )
        eligible_features = request.mm_features
        if metadata.kind == _STREAMING_KIND:
            with self._lock:
                horizon_target = self._context_horizon_targets.get(key)
            if horizon_target is None:
                horizon_target = _prefetch_horizon_target(
                    request=request,
                    computed_frontier=0,
                    inflight_target=None,
                    horizon_chunks=self.horizon_chunks,
                    chunk_token_cap=self.chunk_token_cap,
                    block_size=self.block_size,
                )
            eligible_count = _first_feature_ending_after(
                request.mm_features,
                horizon_target,
            )
            eligible_features = request.mm_features[:eligible_count]
        with self._lock:
            confirmed_progress = self._confirmed_context_progress.get(key)
            progress = self._offered_context_progress.get(
                key,
                confirmed_progress,
            )
            offered_features, compatible = self._feature_delta(
                eligible_features,
                metadata=metadata,
                progress=progress,
            )
            if not compatible and progress is not confirmed_progress:
                offered_features, compatible = self._feature_delta(
                    eligible_features,
                    metadata=metadata,
                    progress=confirmed_progress,
                )

            if metadata.kind == _STREAMING_KIND and compatible:
                current = self._offered_context_progress.get(key)
                if (
                    current is None
                    or metadata.context_version >= current.version
                ):
                    self._offered_context_progress[key] = (
                        _ContextFeatureProgress(
                            version=metadata.context_version,
                            feature_count=len(eligible_features),
                            terminal_feature=(
                                _feature_descriptor(eligible_features[-1])
                                if eligible_features
                                else None
                            ),
                        )
                    )

            if request.request_id in self._confirmed_requests:
                return
            identifiers = {
                feature.identifier
                for feature in offered_features
                if feature.identifier
            }
            owned = self._provisional.setdefault(request.request_id, set())
            for identifier in identifiers - owned:
                owned.add(identifier)
                self._add_owner(identifier)
            self._record_locked(
                "early_offer_eligible_features",
                len(eligible_features),
            )
            self._record_locked(
                "early_offer_deferred_features",
                len(request.mm_features) - len(eligible_features),
            )
        self.offer(offered_features)

    def confirm_request(
        self,
        request_id: str,
        key: ContextKey,
        features: list[MultiModalFeatureSpec],
        *,
        context_version: int,
        total_feature_count: int,
        terminal_feature: FeatureIdentity | None,
        update_context_progress: bool,
    ) -> None:
        """Convert an input-thread provisional owner into a context owner."""

        identifiers = {
            feature.identifier for feature in features if feature.identifier
        }
        releases: list[MMFeaturePrefetchRelease] = []
        with self._lock:
            provisional = self._provisional.pop(request_id, set())
            context_owned = self._context_features.setdefault(key, set())
            for identifier in identifiers:
                if identifier not in context_owned:
                    context_owned.add(identifier)
                    self._add_owner(identifier)
            for identifier in provisional:
                self._remove_owner(identifier)
                release = self._release_if_unused(identifier)
                if release is not None:
                    releases.append(release)
            self._confirmed_requests.add(request_id)
            self._context_requests.setdefault(key, set()).add(request_id)
            if update_context_progress:
                progress = _ContextFeatureProgress(
                    version=context_version,
                    feature_count=total_feature_count,
                    terminal_feature=terminal_feature,
                )
                self._confirmed_context_progress[key] = progress
                offered = self._offered_context_progress.get(key)
                if offered is None or offered.version <= context_version:
                    self._offered_context_progress[key] = progress
        self._send_releases(releases)
        self.offer(features)

    def cancel_request(self, request_id: str) -> None:
        """Drop an input-thread provisional request that failed validation."""

        releases: list[MMFeaturePrefetchRelease] = []
        with self._lock:
            for identifier in self._provisional.pop(request_id, set()):
                self._remove_owner(identifier)
                release = self._release_if_unused(identifier)
                if release is not None:
                    releases.append(release)
        self._send_releases(releases)

    def release_context(self, key: ContextKey) -> None:
        """Release a validated context's ownership of its feature set."""

        releases: list[MMFeaturePrefetchRelease] = []
        with self._lock:
            for request_id in self._context_requests.pop(key, set()):
                self._confirmed_requests.discard(request_id)
            for identifier in self._context_features.pop(key, set()):
                self._remove_owner(identifier)
                release = self._release_if_unused(identifier)
                if release is not None:
                    releases.append(release)
            self._confirmed_context_progress.pop(key, None)
            self._offered_context_progress.pop(key, None)
            self._context_horizon_targets.pop(key, None)
        self._send_releases(releases)

    def acquire_dispatch(
        self,
        request_id: str,
        identifiers: list[str],
    ) -> None:
        """Pin the exact generations referenced by one formal request."""

        self.acquire_ready_dispatch(request_id, identifiers)

    def acquire_ready_dispatch(
        self,
        request_id: str,
        identifiers: list[str],
    ) -> set[str]:
        """Atomically partition ready identifiers and pin their generations."""

        ready: set[str] = set()
        with self._lock:
            leases = self._dispatch_features.setdefault(request_id, set())
            for identifier in identifiers:
                state = self._states.get(identifier)
                if (
                    state is None
                    or state.status is not _FeatureTransferStatus.READY
                ):
                    self._record_locked(
                        "dispatch_fallback_inflight_features"
                        if state is not None
                        else "dispatch_fallback_absent_features"
                    )
                    continue
                ready.add(identifier)
                state.reference_count += 1
                lease = (identifier, state.generation)
                if lease in leases:
                    continue
                leases.add(lease)
                self._add_owner(identifier)
            if not leases:
                self._dispatch_features.pop(request_id, None)
            self._record_locked("dispatch_requested_features", len(identifiers))
            self._record_locked("dispatch_reference_hit_features", len(ready))
            self._record_locked(
                "dispatch_inline_fallback_features",
                len(identifiers) - len(ready),
            )
        return ready

    def release_dispatch(self, request_id: str) -> None:
        """Release references after execution finished or an abort is final."""

        releases: list[MMFeaturePrefetchRelease] = []
        with self._lock:
            for identifier, _generation in self._dispatch_features.pop(
                request_id, set()
            ):
                self._remove_owner(identifier)
                release = self._release_if_unused(identifier)
                if release is not None:
                    releases.append(release)
        self._send_releases(releases)

    def dispatch_request_ids(self) -> set[str]:
        """Return request IDs whose worker-cache generations remain leased."""

        with self._lock:
            return set(self._dispatch_features)

    def offer(self, features: list[MultiModalFeatureSpec]) -> None:
        """Reserve generations before non-blocking submission to avoid lost ACKs."""

        submit = self.submit
        if submit is None or self.world_size <= 0:
            return

        candidates: list[MMFeaturePrefetchItem] = []
        seen: set[str] = set()
        with self._lock:
            for feature in features:
                identifier = feature.identifier
                if (
                    not identifier
                    or identifier in seen
                    or feature.data is None
                    or identifier in self._states
                ):
                    continue
                generation = self._last_generation.get(identifier, 0) + 1
                self._last_generation[identifier] = generation
                self._states[identifier] = _FeatureTransferState(
                    generation=generation,
                    status=_FeatureTransferStatus.INFLIGHT,
                )
                candidates.append(
                    MMFeaturePrefetchItem(
                        identifier=identifier,
                        generation=generation,
                        data=feature.data,
                    )
                )
                seen.add(identifier)
            self._record_locked("offer_candidate_features", len(candidates))
        if not candidates:
            return

        submit_failed = False
        try:
            with record_function_or_nullcontext(
                "streaming_prefill: mm_feature_prefetch_submit"
            ):
                accepted_raw = submit(candidates)
        except Exception:
            submit_failed = True
            logger.warning_once(
                "MM feature prefetch submission failed; affected physical "
                "requests will keep inline feature data."
            )
            accepted_raw = set()

        accepted: set[tuple[str, int]] = set()
        for item in candidates:
            if (
                item.identifier in accepted_raw
                or (item.identifier, item.generation) in accepted_raw
            ):
                accepted.add((item.identifier, item.generation))

        releases: list[MMFeaturePrefetchRelease] = []
        with self._lock:
            self._record_locked("offer_accepted_features", len(accepted))
            self._record_locked(
                "offer_rejected_features",
                len(candidates) - len(accepted),
            )
            # ``MultiprocExecutor.prefetch_mm_features`` returns this subset
            # specifically when its bounded submit queue has no free slots.
            # Keep a separately named counter so horizon tuning can distinguish
            # queue pressure from readiness/fallback at formal dispatch.
            self._record_locked(
                "submit_rejected_features",
                len(candidates) - len(accepted),
            )
            for item in candidates:
                key = (item.identifier, item.generation)
                state = self._states.get(item.identifier)
                if key in accepted:
                    self._retryable_identifiers.discard(item.identifier)
                    if (
                        state is None
                        or state.generation != item.generation
                    ):
                        # A concurrent reset/release won after the data item was
                        # accepted. Unpin that now-unreachable generation after
                        # the FIFO-delivered PUT.
                        releases.append(
                            MMFeaturePrefetchRelease(
                                item.identifier,
                                item.generation,
                            )
                        )
                        if self._owner_counts.get(item.identifier, 0) > 0:
                            self._retryable_identifiers.add(item.identifier)
                    continue
                if state is not None and state.generation == item.generation:
                    self._states.pop(item.identifier, None)
                    if self._owner_counts.get(item.identifier, 0) > 0:
                        self._retryable_identifiers.add(item.identifier)
                    if (
                        submit_failed
                        or self._owner_counts.get(item.identifier, 0) == 0
                    ):
                        releases.append(
                            MMFeaturePrefetchRelease(
                                item.identifier,
                                item.generation,
                            )
                        )
        self._send_releases(releases)

    def update_acks(self, acks: list[MMFeaturePrefetchAck]) -> None:
        """Record ACKs only for the current generation."""

        if self.world_size <= 0:
            return
        releases: list[MMFeaturePrefetchRelease] = []
        with self._lock:
            for rank, identifier, generation, success in acks:
                state = self._states.get(identifier)
                if (
                    state is None
                    or state.generation != generation
                    or state.status is not _FeatureTransferStatus.INFLIGHT
                ):
                    continue
                if not success or not 0 <= rank < self.world_size:
                    self._record_locked("rank_ack_failed")
                    self._states.pop(identifier, None)
                    if self._owner_counts.get(identifier, 0) > 0:
                        self._retryable_identifiers.add(identifier)
                    releases.append(
                        MMFeaturePrefetchRelease(identifier, generation)
                    )
                    continue
                self._record_locked("rank_ack_succeeded")
                state.ready_ranks.add(rank)
                if len(state.ready_ranks) == self.world_size:
                    state.status = _FeatureTransferStatus.READY
                    self._record_locked("all_tp_ready_features")
        self._send_releases(releases)

    def is_ready(self, identifier: str) -> bool:
        with self._lock:
            state = self._states.get(identifier)
            return (
                state is not None
                and state.status is _FeatureTransferStatus.READY
            )

    def reset(self) -> None:
        """Forget residency after workers explicitly clear their caches."""

        with self._lock:
            self._states.clear()
            self._retryable_identifiers.update(self._owner_counts)
            self._confirmed_context_progress.clear()
            self._offered_context_progress.clear()
            self._context_horizon_targets.clear()
            for leases in self._dispatch_features.values():
                for identifier, _generation in leases:
                    self._remove_owner(identifier)
            self._dispatch_features.clear()

    @staticmethod
    def _feature_delta(
        features: list[MultiModalFeatureSpec],
        *,
        metadata: PrefillContextMetadata,
        progress: _ContextFeatureProgress | None,
    ) -> tuple[list[MultiModalFeatureSpec], bool]:
        if progress is None:
            return features, True
        if (
            metadata.kind == _STREAMING_KIND
            and metadata.context_version < progress.version
        ):
            return [], True
        if (
            metadata.context_version != progress.version
            and metadata.kind == _QUERY_KIND
        ):
            # An older query snapshot is infrequent and must not use a suffix
            # boundary derived from a newer streaming version.
            return features, False
        count = progress.feature_count
        if len(features) < count:
            return features, False
        if count and (
            _feature_descriptor(features[count - 1])
            != progress.terminal_feature
        ):
            return features, False
        return features[count:], True

    def _add_owner(self, identifier: str) -> None:
        self._owner_counts[identifier] = (
            self._owner_counts.get(identifier, 0) + 1
        )

    def _remove_owner(self, identifier: str) -> None:
        remaining = self._owner_counts.get(identifier, 0) - 1
        if remaining > 0:
            self._owner_counts[identifier] = remaining
        else:
            self._owner_counts.pop(identifier, None)

    def _release_if_unused(
        self,
        identifier: str,
    ) -> MMFeaturePrefetchRelease | None:
        if self._owner_counts.get(identifier, 0) > 0:
            return None
        state = self._states.pop(identifier, None)
        self._retryable_identifiers.discard(identifier)
        if state is None:
            return None
        if (
            state.status is _FeatureTransferStatus.READY
            and state.reference_count == 0
        ):
            self._record_locked("ready_never_referenced_features")
        return MMFeaturePrefetchRelease(identifier, state.generation)

    def _send_releases(
        self,
        releases: list[MMFeaturePrefetchRelease],
    ) -> None:
        if not releases or self.release is None:
            return
        try:
            self.release(releases)
        except Exception:
            # A missed release leaks bounded cache space, but does not make an
            # acknowledged feature disappear underneath a formal request.
            logger.warning_once(
                "MM feature prefetch release submission failed; entries "
                "remain pinned for correctness."
            )


class StreamingPrefillCoordinator:
    """Serialize logical coalescing decisions inside the EngineCore loop."""

    def __init__(
        self,
        *,
        chunk_token_cap: int,
        block_size: int,
        request_block_hasher: (
            Callable[[Request], list[BlockHash]] | None
        ),
        admit_request: Callable[[Request], None],
        feature_prefetch_world_size: int = 0,
        prefetch_mm_features: (
            Callable[
                [list[MMFeaturePrefetchItem]],
                set[str] | set[tuple[str, int]],
            ]
            | None
        ) = None,
        release_mm_features: (
            Callable[[list[MMFeaturePrefetchRelease]], None] | None
        ) = None,
        feature_prefetch_horizon_chunks: int = 2,
        max_query_fences: int = 1024,
    ) -> None:
        if chunk_token_cap <= 0:
            raise ValueError("chunk_token_cap must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_query_fences <= 0:
            raise ValueError("max_query_fences must be positive")
        if (
            isinstance(feature_prefetch_horizon_chunks, bool)
            or not isinstance(feature_prefetch_horizon_chunks, int)
            or feature_prefetch_horizon_chunks < 0
        ):
            raise ValueError(
                "feature_prefetch_horizon_chunks must be a non-negative integer"
            )
        self.chunk_token_cap = chunk_token_cap
        self.block_size = block_size
        self.feature_prefetch_horizon_chunks = (
            feature_prefetch_horizon_chunks
        )
        self.request_block_hasher = request_block_hasher
        self.admit_request = admit_request
        self.feature_prefetch_world_size = feature_prefetch_world_size
        self.mm_feature_tracker = MMFeaturePrefetchTracker(
            world_size=feature_prefetch_world_size,
            chunk_token_cap=chunk_token_cap,
            block_size=block_size,
            horizon_chunks=feature_prefetch_horizon_chunks,
            submit=prefetch_mm_features,
            release=release_mm_features,
        )
        self.max_query_fences = max_query_fences
        self.contexts: dict[ContextKey, _ContextState] = {}
        self.retired_contexts: set[ContextKey] = set()
        self.logical_to_context: dict[str, ContextKey] = {}
        self.physical_to_context: dict[str, ContextKey] = {}
        self.query_to_context: dict[str, ContextKey] = {}
        self.pending_queries: dict[str, _QueryFence] = {}
        self.pending_queries_by_context: dict[
            ContextKey, dict[str, _QueryFence]
        ] = {}
        self._query_fence_order: OrderedDict[str, None] = OrderedDict()
        self._pending_completions: dict[
            int, list[LogicalRequestCompletion]
        ] = {}
        # Inline feature objects are retained by content identifier while at
        # least one live context refers to them.  This lets later cumulative
        # requests restore inline data after the frontend/native cache replaces
        # repeated items with ``data=None``.
        self._feature_inline_data: dict[str, Any] = {}
        self._feature_context_refcounts: dict[str, int] = {}
        self._context_feature_identifiers: dict[ContextKey, set[str]] = {}
        # EngineCore replaces this with its process-wide collector. Keeping an
        # explicit default preserves standalone unit-test construction.
        self.cpu_metrics: EngineCoreCoordinatorMetrics | None = None
        self._prefetch_metrics: dict[str, int] = {}

    def _record_prefetch_metric(self, name: str, value: int = 1) -> None:
        self._prefetch_metrics[name] = (
            self._prefetch_metrics.get(name, 0) + int(value)
        )

    def mm_prefetch_metrics_snapshot(self) -> dict[str, int]:
        snapshot = dict(self._prefetch_metrics)
        snapshot.update(self.mm_feature_tracker.snapshot_metrics())
        snapshot["configured_horizon_chunks"] = (
            self.feature_prefetch_horizon_chunks
        )
        return snapshot

    def drain_mm_prefetch_metrics(self) -> dict[str, int]:
        """Return interval prefetch metrics without perturbing execution."""

        snapshot = dict(self._prefetch_metrics)
        self._prefetch_metrics.clear()
        snapshot.update(self.mm_feature_tracker.drain_metrics())
        snapshot["configured_horizon_chunks"] = (
            self.feature_prefetch_horizon_chunks
        )
        return snapshot

    def log_mm_prefetch_metrics(self, *, interval: bool = False) -> None:
        """Emit bounded aggregate diagnostics, never one record per feature."""

        metrics = (
            self.drain_mm_prefetch_metrics()
            if interval
            else self.mm_prefetch_metrics_snapshot()
        )
        if not any(
            value
            for key, value in metrics.items()
            if key != "configured_horizon_chunks"
        ):
            return
        logger.info(
            "LLM_RTC_ENGINECORE_MM_FEATURE_PREFETCH_METRICS %s",
            json.dumps(metrics, separators=(",", ":"), sort_keys=True),
        )

    @staticmethod
    def _key(metadata: PrefillContextMetadata) -> ContextKey:
        return (
            metadata.router_session_id,
            metadata.context_lifetime_id,
        )

    def handles_streaming(self, request: Request) -> bool:
        metadata = request.prefill_context
        return metadata is not None and metadata.kind == _STREAMING_KIND

    def observes_query(self, request: Request) -> bool:
        metadata = request.prefill_context
        return metadata is not None and metadata.kind == _QUERY_KIND

    def _enabled_cpu_metrics(
        self,
    ) -> EngineCoreCoordinatorMetrics | None:
        metrics = self.cpu_metrics
        return metrics if metrics is not None and metrics.enabled else None

    def _build_prefix_identity(
        self,
        request: Request,
        target: int | None = None,
    ) -> _PrefixIdentity:
        metrics = self._enabled_cpu_metrics()
        if metrics is None:
            return _prefix_identity(request, self.block_size, target)

        started_cpu_ns = thread_cpu_time_ns()
        identity = _prefix_identity(request, self.block_size, target)
        metrics.record_cpu_ns(
            COORDINATOR_PREFIX_IDENTITY_METRIC,
            thread_cpu_time_ns() - started_cpu_ns,
        )
        metrics.record_counter("prefix_identity_count")
        metrics.record_counter(
            "prefix_tail_tokens_checked",
            len(identity.tail_tokens),
        )
        metrics.record_counter(
            "prefix_tail_features_checked",
            len(identity.tail_features),
        )
        if identity.fallback_token_digest is None:
            metrics.record_counter("prefix_block_checkpoint_count")
        else:
            metrics.record_counter("prefix_full_digest_fallback_count")
            metrics.record_counter(
                "prefix_full_digest_fallback_tokens",
                identity.target,
            )
        return identity

    def _validate_version(
        self,
        context: _ContextState,
        request: Request,
        metadata: PrefillContextMetadata,
        identity: _PrefixIdentity,
    ) -> bool:
        metrics = self._enabled_cpu_metrics()
        if metrics is None:
            return self._accept_version(
                context,
                request,
                metadata,
                identity,
            )

        started_cpu_ns = thread_cpu_time_ns()
        accepted = self._accept_version(
            context,
            request,
            metadata,
            identity,
        )
        metrics.record_cpu_ns(
            COORDINATOR_VERSION_VALIDATE_METRIC,
            thread_cpu_time_ns() - started_cpu_ns,
        )
        metrics.record_counter(
            "version_validation_accepted"
            if accepted
            else "version_validation_rejected"
        )
        return accepted

    def _feature_delta_for_context(
        self,
        context: _ContextState,
        features: list[MultiModalFeatureSpec],
    ) -> list[MultiModalFeatureSpec]:
        """Return only the append-only suffix beyond a validated checkpoint.

        Prefix validation has already proved that the newer Request has the
        same token/MM prefix as ``context.latest_target``.  The feature count
        and terminal descriptor therefore identify the slicing boundary in
        O(1).  If the boundary is unavailable or inconsistent, return the full
        list as a correctness-preserving fallback; the transport tracker still
        deduplicates identifiers that are already resident or in flight.
        """

        count = context.feature_count
        if count == 0:
            return features

        metrics = self._enabled_cpu_metrics()
        if (
            len(features) < count
            or _feature_descriptor(features[count - 1])
            != context.terminal_feature
        ):
            if metrics is not None:
                metrics.record_counter("mm_feature_delta_fallback_count")
                metrics.record_counter(
                    "mm_feature_delta_fallback_features",
                    len(features),
                )
            return features

        delta = features[count:]
        if metrics is not None:
            metrics.record_counter("mm_feature_delta_checkpoint_count")
            metrics.record_counter(
                "mm_feature_delta_skipped_features",
                count,
            )
        return delta

    def add_streaming(self, request: Request) -> None:
        metrics = self._enabled_cpu_metrics()
        if metrics is None:
            self._add_streaming_impl(request)
            return
        started_cpu_ns = thread_cpu_time_ns()
        try:
            self._add_streaming_impl(request)
        finally:
            metrics.record_cpu_ns(
                COORDINATOR_STREAMING_ADD_METRIC,
                thread_cpu_time_ns() - started_cpu_ns,
            )

    def _add_streaming_impl(self, request: Request) -> None:
        metadata = request.prefill_context
        assert metadata is not None and metadata.kind == _STREAMING_KIND
        if request.request_id in self.logical_to_context:
            self.mm_feature_tracker.cancel_request(request.request_id)
            logger.warning(
                "Ignoring duplicate logical streaming prefill request %s",
                request.request_id,
            )
            return
        if request.prompt_token_ids is None or request.prompt_embeds is not None:
            self.mm_feature_tracker.cancel_request(request.request_id)
            self._finish_logical(request, FinishReason.ABORT)
            return

        key = self._key(metadata)
        if key in self.retired_contexts:
            self.mm_feature_tracker.cancel_request(request.request_id)
            logger.warning(
                "Rejecting streaming prefill for retired context: "
                "session=%d lifetime=%d request=%s",
                key[0],
                key[1],
                request.request_id,
            )
            self._finish_logical(request, FinishReason.ABORT)
            return

        identity = self._build_prefix_identity(request)
        context = self.contexts.get(key)
        new_features = request.mm_features
        update_feature_progress = False
        if context is None:
            context = _ContextState(
                key=key,
                role=metadata.context_role,
                latest_version=metadata.context_version,
                latest_target=request,
                version_identities={metadata.context_version: identity},
                feature_count=len(request.mm_features),
                terminal_feature=(
                    _feature_descriptor(request.mm_features[-1])
                    if request.mm_features
                    else None
                ),
            )
            self.contexts[key] = context
            update_feature_progress = True
        elif not self._validate_version(
            context,
            request,
            metadata,
            identity,
        ):
            self.mm_feature_tracker.cancel_request(request.request_id)
            logger.error(
                "Streaming-prefill context received a contradictory update: "
                "session=%d lifetime=%d old_version=%d new_version=%d",
                metadata.router_session_id,
                metadata.context_lifetime_id,
                context.latest_version,
                metadata.context_version,
            )
            self._finish_logical(request, FinishReason.ABORT)
            return
        elif metadata.context_version > context.latest_version:
            new_features = self._feature_delta_for_context(
                context,
                request.mm_features,
            )
            context.latest_version = metadata.context_version
            context.latest_target = request
            context.role = metadata.context_role
            context.feature_count = len(request.mm_features)
            context.terminal_feature = (
                _feature_descriptor(request.mm_features[-1])
                if request.mm_features
                else None
            )
            update_feature_progress = True
        else:
            # Duplicate or late versions are exact prefixes already owned by
            # this Context and therefore add no new feature residency.
            new_features = []

        (
            prefetched_features,
            prefetched_feature_count,
            prefetched_terminal_feature,
        ) = self._prefetch_subset(
            context,
            request,
            new_features,
        )
        self._offer_mm_features(
            key,
            request.request_id,
            prefetched_features,
            context_version=metadata.context_version,
            total_feature_count=prefetched_feature_count,
            terminal_feature=prefetched_terminal_feature,
            update_context_progress=update_feature_progress,
            remember_features=new_features,
        )
        metrics = self._enabled_cpu_metrics()
        if metrics is not None:
            metrics.record_counter(
                "mm_features_seen",
                len(request.mm_features),
            )
            metrics.record_counter("mm_features_new", len(new_features))
        with record_function_or_nullcontext(
            "streaming_prefill: logical_register"
        ):
            context.next_logical_sequence += 1
            logical = _LogicalPrefill(
                request_id=request.request_id,
                client_index=request.client_index,
                context_version=metadata.context_version,
                identity=identity,
                trace_headers=request.trace_headers,
                sequence=context.next_logical_sequence,
            )
            context.logical[request.request_id] = logical
            heapq.heappush(
                context.logical_by_target,
                (logical.target, logical.sequence, logical.request_id),
            )
            self.logical_to_context[request.request_id] = key
        self._refresh_query_fences(context)
        self._complete_covered_logicals(context)
        self._maybe_admit_next(context)
        self._refresh_prefetch_horizon(context)

    def observe_query(self, request: Request) -> None:
        metrics = self._enabled_cpu_metrics()
        if metrics is None:
            self._observe_query_impl(request)
            return
        started_cpu_ns = thread_cpu_time_ns()
        try:
            self._observe_query_impl(request)
        finally:
            metrics.record_cpu_ns(
                COORDINATOR_QUERY_METRIC,
                thread_cpu_time_ns() - started_cpu_ns,
            )

    def _observe_query_impl(self, request: Request) -> None:
        metadata = request.prefill_context
        assert metadata is not None and metadata.kind == _QUERY_KIND

        key = self._key(metadata)
        if key in self.retired_contexts:
            self.mm_feature_tracker.cancel_request(request.request_id)
            return
        self._drop_query_fence(request.request_id)
        context = self.contexts.get(key)
        query_features = (
            request.mm_features
            if context is None
            else self._feature_delta_for_context(
                context,
                request.mm_features,
            )
        )
        self._offer_mm_features(
            key,
            request.request_id,
            query_features,
            context_version=metadata.context_version,
            total_feature_count=len(request.mm_features),
            terminal_feature=(
                _feature_descriptor(request.mm_features[-1])
                if request.mm_features
                else None
            ),
            update_context_progress=False,
        )
        metrics = self._enabled_cpu_metrics()
        if metrics is not None:
            metrics.record_counter(
                "mm_features_seen",
                len(request.mm_features),
            )
            metrics.record_counter("mm_features_new", len(query_features))
        self._prepare_features_for_dispatch(
            request.mm_features,
            dispatch_request_id=request.request_id,
        )
        if request.prompt_token_ids is None:
            return

        query = _QueryFence(
            request=request,
            key=key,
            context_version=metadata.context_version,
        )
        self._query_fence_order[request.request_id] = None
        self._add_pending_query(query)
        if context is not None:
            self._refresh_query_fences(context)
        self._enforce_query_fence_bound()

    def _query_coverage_limit(
        self,
        context: _ContextState,
        query: _QueryFence,
    ) -> int | None:
        metrics = self._enabled_cpu_metrics()
        started_cpu_ns = (
            thread_cpu_time_ns() if metrics is not None else 0
        )
        identity = context.version_identities.get(query.context_version)
        if identity is None:
            coverage_limit = None
        elif identity.target > query.request.num_prompt_tokens:
            coverage_limit = 0
        else:
            query_identity = self._build_prefix_identity(
                query.request,
                identity.target,
            )
            coverage_limit = (
                identity.target if query_identity == identity else 0
            )

        if metrics is not None:
            metrics.record_cpu_ns(
                COORDINATOR_QUERY_COVERAGE_METRIC,
                thread_cpu_time_ns() - started_cpu_ns,
            )
            if identity is None:
                metrics.record_counter("query_snapshot_checkpoint_miss")
            else:
                metrics.record_counter("query_snapshot_versions_checked")
                metrics.record_counter(
                    "query_snapshot_match"
                    if coverage_limit
                    else "query_snapshot_mismatch"
                )
        return coverage_limit

    def _refresh_query_fences(self, context: _ContextState) -> None:
        """Attach queries when their explicit snapshot checkpoint is available."""

        pending = self.pending_queries_by_context.get(context.key)
        if not pending:
            return
        for request_id, query in list(pending.items()):
            coverage_limit = query.coverage_limit
            if coverage_limit == 0 and not query.snapshot_checked:
                checked_limit = self._query_coverage_limit(context, query)
                if checked_limit is None:
                    continue
                query.snapshot_checked = True
                coverage_limit = checked_limit
            if coverage_limit == 0:
                continue
            query.coverage_limit = coverage_limit
            context.computed_frontier = max(
                context.computed_frontier,
                min(query.coverage_limit, query.confirmed_frontier),
            )
            if context.computed_frontier >= query.coverage_limit:
                continue
            self._remove_pending_query(request_id)
            context.covering_queries[request_id] = query
            self.query_to_context[request_id] = context.key

    def _add_pending_query(self, query: _QueryFence) -> None:
        self.pending_queries[query.request_id] = query
        self.pending_queries_by_context.setdefault(query.key, {})[
            query.request_id
        ] = query

    def _remove_pending_query(self, request_id: str) -> _QueryFence | None:
        query = self.pending_queries.pop(request_id, None)
        if query is None:
            return None
        by_context = self.pending_queries_by_context.get(query.key)
        if by_context is not None:
            by_context.pop(request_id, None)
            if not by_context:
                self.pending_queries_by_context.pop(query.key, None)
        return query

    def _drop_query_fence(self, request_id: str) -> ContextKey | None:
        query = self._remove_pending_query(request_id)
        key = None if query is None else query.key

        attached_key = self.query_to_context.pop(request_id, None)
        if attached_key is not None:
            key = attached_key
            context = self.contexts.get(attached_key)
            if context is not None:
                context.covering_queries.pop(request_id, None)

        self._query_fence_order.pop(request_id, None)
        if (
            key is not None
            and key not in self.contexts
            and key not in self.pending_queries_by_context
        ):
            self._release_context_features(key)
        return key

    def _enforce_query_fence_bound(self) -> None:
        while len(self._query_fence_order) > self.max_query_fences:
            request_id = next(iter(self._query_fence_order))
            key = self._drop_query_fence(request_id)
            logger.warning(
                "Evicting oldest streaming-prefill query fence after "
                "reaching max_query_fences=%d: request=%s",
                self.max_query_fences,
                request_id,
            )
            if key is not None:
                context = self.contexts.get(key)
                if context is not None:
                    self._maybe_admit_next(context)

    def _accept_version(
        self,
        context: _ContextState,
        request: Request,
        metadata: PrefillContextMetadata,
        identity: _PrefixIdentity,
    ) -> bool:
        """Validate monotonic version/length and cumulative-prefix identity."""

        version = metadata.context_version
        known_identity = context.version_identities.get(version)
        if known_identity is not None:
            return known_identity == identity

        latest_version = context.latest_version
        latest_target = context.latest_target
        if version < latest_version:
            valid = request.num_prompt_tokens <= latest_target.num_prompt_tokens
            if valid:
                latest_prefix = self._build_prefix_identity(
                    latest_target,
                    request.num_prompt_tokens,
                )
                valid = latest_prefix == identity
        elif version > latest_version:
            valid = request.num_prompt_tokens >= latest_target.num_prompt_tokens
            if valid:
                latest_identity = context.version_identities[latest_version]
                request_prefix = self._build_prefix_identity(
                    request,
                    latest_target.num_prompt_tokens,
                )
                valid = request_prefix == latest_identity
        else:
            # A previously unseen identity for the current version contradicts
            # the version already represented by latest_target.
            valid = False

        if valid:
            context.version_identities[version] = identity
        return valid

    def update_mm_prefetch_acks(
        self,
        acks: list[MMFeaturePrefetchAck],
    ) -> None:
        """Record rank-local readiness without ever gating admission."""

        self.mm_feature_tracker.update_acks(acks)

    def reset_mm_prefetch_state(self) -> None:
        """Forget ACK state when worker caches are explicitly reset."""

        self.mm_feature_tracker.reset()

    def offer_mm_features_early(self, request: Request) -> None:
        """Offer feature snapshots from the EngineCore input thread."""

        self.mm_feature_tracker.offer_early(request)

    def _prefetch_target_for_context(
        self,
        context: _ContextState,
        request: Request,
    ) -> int:
        inflight_target = (
            None if context.inflight is None else context.inflight.target
        )
        return _prefetch_horizon_target(
            request=request,
            computed_frontier=context.computed_frontier,
            inflight_target=inflight_target,
            horizon_chunks=self.feature_prefetch_horizon_chunks,
            chunk_token_cap=self.chunk_token_cap,
            block_size=self.block_size,
        )

    def _prefetch_subset(
        self,
        context: _ContextState,
        request: Request,
        features: list[MultiModalFeatureSpec],
    ) -> tuple[list[MultiModalFeatureSpec], int, FeatureIdentity | None]:
        if self.feature_prefetch_horizon_chunks == 0:
            return [], 0, None
        horizon_target = self._prefetch_target_for_context(context, request)
        total_feature_count = _first_feature_ending_after(
            request.mm_features,
            horizon_target,
        )
        eligible = [
            feature
            for feature in features
            if _feature_end(feature) <= horizon_target
        ]
        terminal_feature = (
            _feature_descriptor(
                request.mm_features[total_feature_count - 1]
            )
            if total_feature_count
            else None
        )
        return eligible, total_feature_count, terminal_feature

    def _refresh_prefetch_horizon(self, context: _ContextState) -> None:
        """Offer newly eligible features after physical planning advances."""

        request = context.latest_target
        if self.feature_prefetch_horizon_chunks == 0:
            self.mm_feature_tracker.publish_horizon(context.key, 0)
            return

        horizon_target = self._prefetch_target_for_context(context, request)
        feature_count = _first_feature_ending_after(
            request.mm_features,
            horizon_target,
        )
        if (
            horizon_target == context.prefetch_horizon_target
            and feature_count == context.prefetch_feature_count
        ):
            # A full submit queue can reject a feature even though no new
            # media tokens arrived and the horizon therefore did not advance.
            # Retry only those explicit rejects, over the same bounded prefix;
            # this neither waits for readiness nor changes physical planning.
            bounded_features = request.mm_features[:feature_count]
            if not self.mm_feature_tracker.has_retryable_features(
                bounded_features,
            ):
                return
            retry_features = [copy(feature) for feature in bounded_features]
            self._restore_inline_features(retry_features)
            self.mm_feature_tracker.offer(retry_features)
            self._record_prefetch_metric("horizon_same_target_retry_count")
            self._record_prefetch_metric(
                "horizon_same_target_retry_features_scanned",
                len(retry_features),
            )
            return

        previous_feature_count = context.prefetch_feature_count
        context.prefetch_horizon_target = horizon_target
        context.prefetch_feature_count = feature_count
        self.mm_feature_tracker.publish_horizon(
            context.key,
            horizon_target,
        )
        features = [
            copy(feature)
            for feature in request.mm_features[:feature_count]
        ]
        # Missing side-channel data never gates planning. Restore what remains
        # available in the EngineCore ledger; ``offer`` skips unresolved items.
        self._restore_inline_features(features)
        self.mm_feature_tracker.offer_for_context(
            context.key,
            features,
            context_version=context.latest_version,
            total_feature_count=feature_count,
            terminal_feature=(
                _feature_descriptor(features[-1]) if features else None
            ),
        )
        self._record_prefetch_metric("horizon_refresh_count")
        self._record_prefetch_metric(
            "horizon_newly_eligible_features",
            max(0, feature_count - previous_feature_count),
        )
        self._record_prefetch_metric(
            "horizon_deferred_feature_observations",
            max(0, len(request.mm_features) - feature_count),
        )
        self._prefetch_metrics["horizon_target_tokens_max"] = max(
            self._prefetch_metrics.get("horizon_target_tokens_max", 0),
            horizon_target,
        )

    def _offer_mm_features(
        self,
        key: ContextKey,
        request_id: str,
        features: list[MultiModalFeatureSpec],
        *,
        context_version: int,
        total_feature_count: int,
        terminal_feature: FeatureIdentity | None,
        update_context_progress: bool,
        remember_features: list[MultiModalFeatureSpec] | None = None,
    ) -> None:
        self._remember_inline_features(
            key,
            features if remember_features is None else remember_features,
        )
        self.mm_feature_tracker.confirm_request(
            request_id,
            key,
            features,
            context_version=context_version,
            total_feature_count=total_feature_count,
            terminal_feature=terminal_feature,
            update_context_progress=update_context_progress,
        )

    def _remember_inline_features(
        self,
        key: ContextKey,
        features: list[MultiModalFeatureSpec],
    ) -> None:
        owned = self._context_feature_identifiers.setdefault(key, set())
        for feature in features:
            identifier = feature.identifier
            if not identifier:
                continue
            if identifier not in owned:
                owned.add(identifier)
                self._feature_context_refcounts[identifier] = (
                    self._feature_context_refcounts.get(identifier, 0) + 1
                )
            if feature.data is not None:
                self._feature_inline_data[identifier] = feature.data

    def _restore_inline_features(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> bool:
        for feature in features:
            if feature.data is None:
                inline_data = self._feature_inline_data.get(
                    feature.identifier
                )
                if inline_data is not None:
                    feature.data = inline_data
        return all(feature.data is not None for feature in features)

    def _release_context_features(self, key: ContextKey) -> None:
        for identifier in self._context_feature_identifiers.pop(key, ()):
            remaining = self._feature_context_refcounts[identifier] - 1
            if remaining > 0:
                self._feature_context_refcounts[identifier] = remaining
                continue
            self._feature_context_refcounts.pop(identifier, None)
            self._feature_inline_data.pop(identifier, None)
        self.mm_feature_tracker.release_context(key)

    def _prepare_features_for_dispatch(
        self,
        features: list[MultiModalFeatureSpec],
        *,
        dispatch_request_id: str,
    ) -> _FeatureDeliveryMode:
        referenced = self.mm_feature_tracker.acquire_ready_dispatch(
            dispatch_request_id,
            [feature.identifier for feature in features],
        )
        inline_count = 0
        for feature in features:
            if feature.identifier in referenced:
                feature.data = None
                continue
            if feature.data is None:
                inline_data = self._feature_inline_data.get(
                    feature.identifier
                )
                if inline_data is not None:
                    feature.data = inline_data
            inline_count += 1

        if referenced and inline_count:
            return _FeatureDeliveryMode.MIXED
        if referenced:
            return _FeatureDeliveryMode.ALL_REFERENCE
        return _FeatureDeliveryMode.ALL_INLINE

    def abort(self, request_ids: list[str]) -> list[str]:
        """Remove logical obligations and return physical IDs to abort."""

        physical_to_abort: list[str] = []
        touched_contexts: set[ContextKey] = set()
        for request_id in request_ids:
            key = self.logical_to_context.pop(request_id, None)
            if key is not None:
                context = self.contexts.get(key)
                if context is not None:
                    logical = context.logical.pop(request_id, None)
                    if logical is not None:
                        self._queue_completion(
                            logical.client_index,
                            LogicalRequestCompletion(
                                request_id=logical.request_id,
                                finish_reason=FinishReason.ABORT,
                            ),
                        )
                    touched_contexts.add(key)
                continue

            if (
                request_id in self.pending_queries
                or request_id in self.query_to_context
            ):
                key = self._drop_query_fence(request_id)
                if key is not None and key in self.contexts:
                    touched_contexts.add(key)

        for key in touched_contexts:
            context = self.contexts.get(key)
            if context is None:
                continue
            if not context.logical and context.inflight is not None:
                physical_to_abort.append(context.inflight.request_id)
                self.physical_to_context.pop(context.inflight.request_id, None)
                context.inflight = None
            self._maybe_admit_next(context)
        return physical_to_abort

    def retire_context(
        self,
        router_session_id: int,
        context_lifetime_id: int,
    ) -> list[str]:
        """Permanently retire a context and abort its internal physical work.

        Retirement is idempotent.  The tombstone is retained so late messages
        for the same lifetime are explicitly aborted instead of recreating the
        context.
        """

        key = (router_session_id, context_lifetime_id)
        if key in self.retired_contexts:
            return []
        self.retired_contexts.add(key)

        for request_id in list(
            self.pending_queries_by_context.get(key, ())
        ):
            self._drop_query_fence(request_id)

        context = self.contexts.get(key)
        if context is None:
            self._release_context_features(key)
            return []

        for request_id in list(context.covering_queries):
            self._drop_query_fence(request_id)
        self.contexts.pop(key, None)
        self._release_context_features(key)

        physical_to_abort: list[str] = []
        if context.inflight is not None:
            physical_to_abort.append(context.inflight.request_id)
            self.physical_to_context.pop(context.inflight.request_id, None)
            context.inflight = None

        for logical in context.logical.values():
            self._queue_completion(
                logical.client_index,
                LogicalRequestCompletion(
                    request_id=logical.request_id,
                    finish_reason=FinishReason.ABORT,
                ),
            )
            self.logical_to_context.pop(logical.request_id, None)
        context.logical.clear()
        context.logical_by_target.clear()

        return physical_to_abort

    def update_after_step(
        self,
        scheduler_output: SchedulerOutput,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        """Advance coverage only from this completed scheduler step.

        ``Scheduler.schedule`` advances live ``Request.num_computed_tokens``
        optimistically.  Reading that mutable value here is incorrect when the
        engine uses a batch queue because it can already include later queued
        steps.  The SchedulerOutput contains the immutable pre-step value sent
        to workers, so confirmed progress is exactly that snapshot plus this
        successful step's scheduled-token count.
        """

        dispatch_ids = self.mm_feature_tracker.dispatch_request_ids()
        if not (
            self.physical_to_context
            or self.query_to_context
            or self.pending_queries
            or dispatch_ids
        ):
            # The coordinator is installed globally but most scheduler steps may
            # contain no streaming/query work. Preserve protection against a
            # late internal physical output while skipping all frontier maps.
            if any(
                output.request_id.startswith(_PHYSICAL_ID_PREFIX)
                for outputs in engine_core_outputs.values()
                for output in outputs.outputs
            ):
                self._strip_physical_outputs(engine_core_outputs)
            self._append_pending_outputs(engine_core_outputs)
            return

        def is_tracked(request_id: str) -> bool:
            return (
                request_id in self.physical_to_context
                or request_id in self.query_to_context
                or request_id in self.pending_queries
                or request_id in dispatch_ids
            )

        output_by_request: dict[str, EngineCoreOutput] = {}
        for outputs in engine_core_outputs.values():
            for output in outputs.outputs:
                if is_tracked(output.request_id):
                    output_by_request[output.request_id] = output

        scheduled_ids = scheduler_output.num_scheduled_tokens
        tracked_scheduled_ids = {
            request_id for request_id in scheduled_ids if is_tracked(request_id)
        }
        pre_step_frontiers = self._pre_step_frontiers(
            scheduler_output,
            tracked_scheduled_ids,
        )
        touched_contexts: set[ContextKey] = set()
        for request_id in tracked_scheduled_ids:
            num_scheduled_tokens = scheduled_ids[request_id]
            pre_step_frontier = pre_step_frontiers.get(request_id)
            if pre_step_frontier is None:
                logger.error(
                    "SchedulerOutput omitted pre-step progress for request %s; "
                    "coverage will not advance",
                    request_id,
                )
                continue
            output = output_by_request.get(request_id)
            step_succeeded = (
                output is None
                or output.finish_reason != FinishReason.ABORT
            )
            confirmed_after_step = (
                pre_step_frontier + num_scheduled_tokens
                if step_succeeded
                else pre_step_frontier
            )

            pending_query = self.pending_queries.get(request_id)
            if pending_query is not None:
                if step_succeeded:
                    pending_query.confirmed_frontier = max(
                        pending_query.confirmed_frontier,
                        confirmed_after_step,
                    )
                if output is not None and output.finished:
                    self._drop_query_fence(request_id)
                continue

            key = self.physical_to_context.get(request_id)
            if key is not None:
                context = self.contexts.get(key)
                if context is None or context.inflight is None:
                    continue
                physical = context.inflight

                if step_succeeded:
                    context.computed_frontier = max(
                        context.computed_frontier,
                        min(physical.target, confirmed_after_step),
                    )
                if output is not None and output.finished:
                    self.physical_to_context.pop(request_id, None)
                    context.inflight = None
                touched_contexts.add(key)
                continue

            key = self.query_to_context.get(request_id)
            if key is None:
                continue
            context = self.contexts.get(key)
            query = (
                None
                if context is None
                else context.covering_queries.get(request_id)
            )
            if context is None or query is None:
                continue
            if step_succeeded:
                query.confirmed_frontier = max(
                    query.confirmed_frontier,
                    confirmed_after_step,
                )
                progress = min(
                    query.coverage_limit,
                    query.confirmed_frontier,
                )
                context.computed_frontier = max(
                    context.computed_frontier,
                    progress,
                )
            if output is not None and output.finished:
                self._drop_query_fence(request_id)
            elif context.computed_frontier >= query.coverage_limit:
                context.covering_queries.pop(request_id, None)
                self.query_to_context.pop(request_id, None)
                self._add_pending_query(query)
            touched_contexts.add(key)

        for request_id, output in output_by_request.items():
            if not output.finished:
                continue
            self.mm_feature_tracker.release_dispatch(request_id)
            if (
                request_id in self.pending_queries
                or request_id in self.query_to_context
            ):
                key = self._drop_query_fence(request_id)
                if key is not None and key in self.contexts:
                    touched_contexts.add(key)

        for key in touched_contexts:
            context = self.contexts.get(key)
            if context is None:
                continue
            self._complete_covered_logicals(context)
            self._maybe_admit_next(context)
            self._refresh_prefetch_horizon(context)

        self._strip_physical_outputs(engine_core_outputs)
        self._append_pending_outputs(engine_core_outputs)

    def finish_aborted_dispatches(self, request_ids: list[str]) -> None:
        """Release leases after synchronous Scheduler aborts are committed."""

        for request_id in request_ids:
            self.mm_feature_tracker.release_dispatch(request_id)

    @staticmethod
    def _pre_step_frontiers(
        scheduler_output: SchedulerOutput,
        tracked_ids: set[str] | None = None,
    ) -> dict[str, int]:
        frontiers = {
            request.req_id: request.num_computed_tokens
            for request in scheduler_output.scheduled_new_reqs
            if tracked_ids is None or request.req_id in tracked_ids
        }
        cached = scheduler_output.scheduled_cached_reqs
        frontiers.update(
            (request_id, num_computed_tokens)
            for request_id, num_computed_tokens in zip(
                cached.req_ids,
                cached.num_computed_tokens,
            )
            if tracked_ids is None or request_id in tracked_ids
        )
        return frontiers

    def take_pending_outputs(self) -> dict[int, EngineCoreOutputs]:
        outputs: dict[int, EngineCoreOutputs] = {}
        self._append_pending_outputs(outputs)
        return outputs

    def _maybe_admit_next(self, context: _ContextState) -> None:
        if context.inflight is not None or not context.logical:
            return
        if any(
            query.confirmed_frontier < query.coverage_limit
            for query in context.covering_queries.values()
        ):
            return

        target_request = context.latest_target
        remaining = target_request.num_prompt_tokens - context.computed_frontier
        if remaining <= 0:
            self._complete_covered_logicals(context)
            return

        target = min(
            target_request.num_prompt_tokens,
            context.computed_frontier + self.chunk_token_cap,
        )
        target = self._safe_target(
            request=target_request,
            frontier=context.computed_frontier,
            target=target,
        )
        if target <= context.computed_frontier:
            self._fail_context(context)
            return

        context.next_physical_sequence += 1
        request_id = (
            f"{_PHYSICAL_ID_PREFIX}:{context.key[0]}:{context.key[1]}:"
            f"{context.next_physical_sequence}:{target}"
        )
        sampling_params = target_request.sampling_params
        if sampling_params is None:
            self._fail_context(context)
            return
        sampling_params = sampling_params.clone()
        sampling_params.max_tokens = 1

        metrics = self._enabled_cpu_metrics()
        physical_started_cpu_ns = (
            thread_cpu_time_ns() if metrics is not None else 0
        )
        materialize_started_cpu_ns = (
            physical_started_cpu_ns if metrics is not None else 0
        )
        feature_count = _first_feature_ending_after(
            target_request.mm_features,
            target,
        )
        features = [
            copy(feature)
            for feature in target_request.mm_features[:feature_count]
        ]
        delivery_mode = self._prepare_features_for_dispatch(
            features,
            dispatch_request_id=request_id,
        )
        physical_request = Request.from_request_prefix(
            source=target_request,
            request_id=request_id,
            target_num_prompt_tokens=target,
            block_size=self.block_size,
            mm_features=features,
            sampling_params=sampling_params,
            pooling_params=None,
            prefill_context=None,
            mm_features_are_validated_prefix=True,
        )
        if metrics is not None:
            metrics.record_cpu_ns(
                COORDINATOR_MATERIALIZE_METRIC,
                thread_cpu_time_ns() - materialize_started_cpu_ns,
            )
            metrics.record_counter("physical_request_count")
            metrics.record_counter(
                "physical_prompt_tokens_materialized",
                target,
            )
            metrics.record_counter(
                "physical_mm_features_materialized",
                len(features),
            )
            metrics.record_counter(
                "physical_block_hashes_reused",
                len(physical_request.block_hashes),
            )
            metrics.record_counter(
                {
                    _FeatureDeliveryMode.ALL_REFERENCE: (
                        "physical_delivery_all_reference"
                    ),
                    _FeatureDeliveryMode.MIXED: "physical_delivery_mixed",
                    _FeatureDeliveryMode.ALL_INLINE: (
                        "physical_delivery_all_inline"
                    ),
                }[delivery_mode]
            )
        physical = _PhysicalPrefix(
            request_id=request_id,
            request=physical_request,
            target=target,
            source=target_request,
        )
        context.inflight = physical
        self.physical_to_context[request_id] = context.key
        scope_name = {
            _FeatureDeliveryMode.ALL_REFERENCE: (
                "streaming_prefill: physical_admit_reference"
            ),
            _FeatureDeliveryMode.MIXED: (
                "streaming_prefill: physical_admit_mixed"
            ),
            _FeatureDeliveryMode.ALL_INLINE: (
                "streaming_prefill: physical_admit_inline"
            ),
        }[delivery_mode]
        with record_function_or_nullcontext(scope_name):
            self.admit_request(physical_request)
        if metrics is not None:
            metrics.record_cpu_ns(
                COORDINATOR_PHYSICAL_BUILD_METRIC,
                thread_cpu_time_ns() - physical_started_cpu_ns,
            )

    def _safe_target(
        self,
        *,
        request: Request,
        frontier: int,
        target: int,
    ) -> int:
        return _safe_physical_target(
            request=request,
            frontier=frontier,
            target=target,
            block_size=self.block_size,
        )

    def _complete_covered_logicals(
        self,
        context: _ContextState,
    ) -> None:
        completed: list[_LogicalPrefill] = []
        while (
            context.logical_by_target
            and context.logical_by_target[0][0] <= context.computed_frontier
        ):
            _target, sequence, request_id = heapq.heappop(
                context.logical_by_target
            )
            logical = context.logical.get(request_id)
            if logical is None or logical.sequence != sequence:
                continue
            completed.append(logical)

        # The target heap discovers only newly covered obligations. Preserve
        # ingress order in the externally visible ACK batch.
        completed.sort(key=lambda logical: logical.sequence)
        for logical in completed:
            self._queue_completion(
                logical.client_index,
                LogicalRequestCompletion(
                    request_id=logical.request_id,
                    finish_reason=FinishReason.LENGTH,
                ),
            )

        if completed:
            with record_function_or_nullcontext(
                "streaming_prefill: coverage_complete"
            ):
                for logical in completed:
                    context.logical.pop(logical.request_id, None)
                    self.logical_to_context.pop(logical.request_id, None)

    def _finish_logical(
        self,
        request: Request,
        finish_reason: FinishReason,
    ) -> None:
        self._queue_completion(
            request.client_index,
            LogicalRequestCompletion(
                request_id=request.request_id,
                finish_reason=finish_reason,
            ),
        )

    def _fail_context(self, context: _ContextState) -> None:
        for logical in list(context.logical.values()):
            self._queue_completion(
                logical.client_index,
                LogicalRequestCompletion(
                    request_id=logical.request_id,
                    finish_reason=FinishReason.ABORT,
                ),
            )
            self.logical_to_context.pop(logical.request_id, None)
        context.logical.clear()
        context.logical_by_target.clear()
        query_ids = list(context.covering_queries)
        query_ids.extend(
            self.pending_queries_by_context.get(context.key, ())
        )
        for request_id in query_ids:
            self._drop_query_fence(request_id)
        self.contexts.pop(context.key, None)
        self.retired_contexts.add(context.key)
        self._release_context_features(context.key)

    def _queue_completion(
        self,
        client_index: int,
        completion: LogicalRequestCompletion,
    ) -> None:
        self._pending_completions.setdefault(client_index, []).append(
            completion
        )

    def _append_pending_outputs(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        for client_index, pending in self._pending_completions.items():
            outputs = engine_core_outputs.setdefault(
                client_index,
                EngineCoreOutputs(),
            )
            outputs.logical_request_completions.extend(pending)
            if outputs.finished_requests is None:
                outputs.finished_requests = set()
            outputs.finished_requests.update(
                completion.request_id for completion in pending
            )
        self._pending_completions.clear()

    def _strip_physical_outputs(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        for outputs in engine_core_outputs.values():
            outputs.outputs = [
                output
                for output in outputs.outputs
                if not output.request_id.startswith(_PHYSICAL_ID_PREFIX)
            ]
            if outputs.finished_requests:
                outputs.finished_requests = {
                    request_id
                    for request_id in outputs.finished_requests
                    if not request_id.startswith(_PHYSICAL_ID_PREFIX)
                }
