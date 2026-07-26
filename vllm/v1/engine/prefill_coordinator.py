# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EngineCore-side coordination for cumulative streaming-prefill requests.

The application submits every logical media-context version.  This coordinator
keeps those original request IDs as completion obligations while admitting at
most one bounded, full-prefix physical request per context to the Scheduler.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass, field
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
from vllm.v1.request import Request
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.mm_feature_prefetch import MMFeaturePrefetchAck

logger = init_logger(__name__)

_STREAMING_KIND = "streaming"
_QUERY_KIND = "user_query"
_PHYSICAL_ID_PREFIX = "__llm_rtc_streaming_prefill__"

ContextKey = tuple[int, int]


def _feature_end(feature: MultiModalFeatureSpec) -> int:
    position = feature.mm_position
    return position.offset + position.length


def _feature_identity(
    features: list[MultiModalFeatureSpec],
    target: int,
) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (
            feature.modality,
            feature.identifier,
            feature.mm_position.offset,
            feature.mm_position.length,
        )
        for feature in features
        if _feature_end(feature) <= target
    )


def _lora_identity(request: Request) -> tuple[int, str, str] | None:
    lora = request.lora_request
    if lora is None:
        return None
    return (lora.lora_int_id, lora.lora_name, lora.lora_path)


@dataclass(frozen=True, slots=True)
class _PrefixIdentity:
    target: int
    token_digest: bytes
    features: tuple[tuple[str, str, int, int], ...]
    cache_salt: str | None
    lora: tuple[int, str, str] | None


def _token_digest(tokens: list[int]) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for token_id in tokens:
        digest.update(token_id.to_bytes(8, "little", signed=False))
    return digest.digest()


def _prefix_identity(request: Request, target: int | None = None) -> _PrefixIdentity:
    tokens = request.prompt_token_ids
    if tokens is None:
        raise ValueError("Streaming-prefill identity requires prompt token IDs")
    if target is None:
        target = len(tokens)
    return _PrefixIdentity(
        target=target,
        token_digest=_token_digest(tokens[:target]),
        features=_feature_identity(request.mm_features, target),
        cache_salt=request.cache_salt,
        lora=_lora_identity(request),
    )


def _is_exact_prefix(shorter: Request, longer: Request) -> bool:
    short_tokens = shorter.prompt_token_ids
    long_tokens = longer.prompt_token_ids
    if short_tokens is None or long_tokens is None:
        return False
    short_len = len(short_tokens)
    if short_len > len(long_tokens):
        return False
    if shorter.cache_salt != longer.cache_salt:
        return False
    if _lora_identity(shorter) != _lora_identity(longer):
        return False
    if short_tokens != long_tokens[:short_len]:
        return False
    return _feature_identity(
        shorter.mm_features,
        short_len,
    ) == _feature_identity(longer.mm_features, short_len)


@dataclass(slots=True)
class _LogicalPrefill:
    request_id: str
    client_index: int
    context_version: int
    identity: _PrefixIdentity
    trace_headers: Mapping[str, str] | None
    queued_timestamp: float
    scheduled_timestamp: float | None = None

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
    logical: OrderedDict[str, _LogicalPrefill] = field(
        default_factory=OrderedDict
    )
    computed_frontier: int = 0
    inflight: _PhysicalPrefix | None = None
    covering_queries: dict[str, _QueryFence] = field(default_factory=dict)
    next_physical_sequence: int = 0


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
            Callable[[list[MultiModalFeatureSpec]], set[str]] | None
        ) = None,
        max_query_fences: int = 1024,
    ) -> None:
        if chunk_token_cap <= 0:
            raise ValueError("chunk_token_cap must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if max_query_fences <= 0:
            raise ValueError("max_query_fences must be positive")
        self.chunk_token_cap = chunk_token_cap
        self.block_size = block_size
        self.request_block_hasher = request_block_hasher
        self.admit_request = admit_request
        self.feature_prefetch_world_size = feature_prefetch_world_size
        self.prefetch_mm_features = prefetch_mm_features
        self.max_query_fences = max_query_fences
        self.contexts: dict[ContextKey, _ContextState] = {}
        self.retired_contexts: set[ContextKey] = set()
        self.logical_to_context: dict[str, ContextKey] = {}
        self.physical_to_context: dict[str, ContextKey] = {}
        self.query_to_context: dict[str, ContextKey] = {}
        self.pending_queries: dict[str, _QueryFence] = {}
        self._query_fence_order: OrderedDict[str, None] = OrderedDict()
        self._pending_completions: dict[
            int, list[LogicalRequestCompletion]
        ] = {}
        self._feature_ready_ranks: dict[str, set[int]] = {}
        self._feature_prefetch_inflight: set[str] = set()
        self._feature_prefetch_failed: set[str] = set()
        # Inline feature objects are retained by content identifier while at
        # least one live context refers to them.  This lets later cumulative
        # requests restore inline data after the frontend/native cache replaces
        # repeated items with ``data=None``.
        self._feature_inline_data: dict[str, Any] = {}
        self._feature_context_refcounts: dict[str, int] = {}
        self._context_feature_identifiers: dict[ContextKey, set[str]] = {}

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

    def add_streaming(self, request: Request) -> None:
        metadata = request.prefill_context
        assert metadata is not None and metadata.kind == _STREAMING_KIND
        if request.request_id in self.logical_to_context:
            logger.warning(
                "Ignoring duplicate logical streaming prefill request %s",
                request.request_id,
            )
            return
        if request.prompt_token_ids is None or request.prompt_embeds is not None:
            self._finish_logical(request, FinishReason.ABORT)
            return

        key = self._key(metadata)
        if key in self.retired_contexts:
            logger.warning(
                "Rejecting streaming prefill for retired context: "
                "session=%d lifetime=%d request=%s",
                key[0],
                key[1],
                request.request_id,
            )
            self._finish_logical(request, FinishReason.ABORT)
            return

        now = time.monotonic()
        identity = _prefix_identity(request)
        context = self.contexts.get(key)
        if context is None:
            context = _ContextState(
                key=key,
                role=metadata.context_role,
                latest_version=metadata.context_version,
                latest_target=request,
                version_identities={metadata.context_version: identity},
            )
            self.contexts[key] = context
        elif not self._accept_version(context, request, metadata, identity):
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
            context.latest_version = metadata.context_version
            context.latest_target = request
            context.role = metadata.context_role

        self._offer_mm_features(key, request.mm_features)
        with record_function_or_nullcontext(
            "streaming_prefill: logical_register"
        ):
            logical = _LogicalPrefill(
                request_id=request.request_id,
                client_index=request.client_index,
                context_version=metadata.context_version,
                identity=identity,
                trace_headers=request.trace_headers,
                queued_timestamp=now,
            )
            context.logical[request.request_id] = logical
            self.logical_to_context[request.request_id] = key
        self._refresh_query_fences(context)
        self._complete_covered_logicals(context)
        self._maybe_admit_next(context)

    def observe_query(self, request: Request) -> None:
        metadata = request.prefill_context
        assert metadata is not None and metadata.kind == _QUERY_KIND

        key = self._key(metadata)
        if key in self.retired_contexts:
            return
        self._drop_query_fence(request.request_id)
        self._offer_mm_features(key, request.mm_features)
        self._prepare_features_for_dispatch(request.mm_features)
        if request.prompt_token_ids is None:
            return

        query = _QueryFence(
            request=request,
            key=key,
            context_version=metadata.context_version,
        )
        self._query_fence_order[request.request_id] = None
        self.pending_queries[request.request_id] = query
        context = self.contexts.get(key)
        if context is not None:
            self._refresh_query_fences(context)
        self._enforce_query_fence_bound()

    def _query_coverage_limit(
        self,
        context: _ContextState,
        query: _QueryFence,
    ) -> int:
        coverage_limit = 0
        for version, identity in context.version_identities.items():
            if (
                version > query.context_version
                or identity.target > query.request.num_prompt_tokens
            ):
                continue
            if _prefix_identity(query.request, identity.target) == identity:
                coverage_limit = max(coverage_limit, identity.target)
        return coverage_limit

    def _refresh_query_fences(self, context: _ContextState) -> None:
        """Attach late contexts and extend exact-prefix query coverage."""

        for query in context.covering_queries.values():
            query.coverage_limit = max(
                query.coverage_limit,
                self._query_coverage_limit(context, query),
            )
            context.computed_frontier = max(
                context.computed_frontier,
                min(query.coverage_limit, query.confirmed_frontier),
            )

        for request_id, query in list(self.pending_queries.items()):
            if query.key != context.key:
                continue
            coverage_limit = max(
                query.coverage_limit,
                self._query_coverage_limit(context, query),
            )
            if coverage_limit == 0:
                continue
            query.coverage_limit = coverage_limit
            context.computed_frontier = max(
                context.computed_frontier,
                min(query.coverage_limit, query.confirmed_frontier),
            )
            if context.computed_frontier >= query.coverage_limit:
                continue
            self.pending_queries.pop(request_id, None)
            context.covering_queries[request_id] = query
            self.query_to_context[request_id] = context.key

    def _drop_query_fence(self, request_id: str) -> ContextKey | None:
        query = self.pending_queries.pop(request_id, None)
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
            and not any(
                pending.key == key
                for pending in self.pending_queries.values()
            )
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
            valid = (
                request.num_prompt_tokens <= latest_target.num_prompt_tokens
                and _is_exact_prefix(request, latest_target)
            )
        elif version > latest_version:
            valid = (
                request.num_prompt_tokens >= latest_target.num_prompt_tokens
                and _is_exact_prefix(latest_target, request)
            )
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

        world_size = self.feature_prefetch_world_size
        if world_size <= 0:
            return
        for rank, identifier, success in acks:
            if identifier not in self._feature_prefetch_inflight:
                continue
            if not success or not 0 <= rank < world_size:
                self._feature_prefetch_failed.add(identifier)
                self._feature_prefetch_inflight.discard(identifier)
                self._feature_ready_ranks.pop(identifier, None)
                continue
            ranks = self._feature_ready_ranks.setdefault(identifier, set())
            ranks.add(rank)
            if len(ranks) == world_size:
                self._feature_prefetch_inflight.discard(identifier)

    def reset_mm_prefetch_state(self) -> None:
        """Forget ACK state when worker caches are explicitly reset."""

        self._feature_ready_ranks.clear()
        self._feature_prefetch_inflight.clear()
        self._feature_prefetch_failed.clear()

    def _offer_mm_features(
        self,
        key: ContextKey,
        features: list[MultiModalFeatureSpec],
    ) -> None:
        self._remember_inline_features(key, features)

        send = self.prefetch_mm_features
        if send is None or self.feature_prefetch_world_size <= 0:
            return

        candidates: list[MultiModalFeatureSpec] = []
        seen: set[str] = set()
        for feature in features:
            identifier = feature.identifier
            if (
                feature.data is None
                or not identifier
                or identifier in seen
                or identifier in self._feature_prefetch_inflight
                or identifier in self._feature_prefetch_failed
                or self._is_feature_ready(identifier)
            ):
                continue
            candidates.append(feature)
            seen.add(identifier)
        if not candidates:
            return

        try:
            with record_function_or_nullcontext(
                "streaming_prefill: mm_feature_prefetch_send"
            ):
                accepted = send(candidates)
        except Exception:
            logger.warning_once(
                "MM feature prefetch submission failed; affected physical "
                "requests will keep inline feature data."
            )
            return
        self._feature_prefetch_inflight.update(accepted)

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
            self._feature_ready_ranks.pop(identifier, None)
            self._feature_prefetch_inflight.discard(identifier)
            self._feature_prefetch_failed.discard(identifier)

    def _is_feature_ready(self, identifier: str) -> bool:
        return self.feature_prefetch_world_size > 0 and len(
            self._feature_ready_ranks.get(identifier, ())
        ) == (
            self.feature_prefetch_world_size
        )

    def _prepare_features_for_dispatch(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> bool:
        if features and all(
            self._is_feature_ready(feature.identifier)
            for feature in features
        ):
            for feature in features:
                feature.data = None
            return True

        # A request must use one consistent data path on every TP rank.  If any
        # identifier is not ready everywhere, restore all available inline
        # objects instead of mixing rank-local custom-cache references.
        self._restore_inline_features(features)
        return False

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

        for request_id, query in list(self.pending_queries.items()):
            if query.key == key:
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

        output_by_request: dict[str, EngineCoreOutput] = {}
        for outputs in engine_core_outputs.values():
            for output in outputs.outputs:
                output_by_request[output.request_id] = output

        pre_step_frontiers = self._pre_step_frontiers(scheduler_output)
        scheduled_ids = scheduler_output.num_scheduled_tokens
        touched_contexts: set[ContextKey] = set()
        for request_id, num_scheduled_tokens in scheduled_ids.items():
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
                now = time.monotonic()
                for logical in context.logical.values():
                    if (
                        logical.scheduled_timestamp is None
                        and logical.target <= physical.target
                    ):
                        logical.scheduled_timestamp = now

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
                self.pending_queries[request_id] = query
            touched_contexts.add(key)

        for request_id, output in output_by_request.items():
            if not output.finished:
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
            self._complete_covered_logicals(context)
            self._maybe_admit_next(context)

        self._strip_physical_outputs(engine_core_outputs)
        self._append_pending_outputs(engine_core_outputs)

    @staticmethod
    def _pre_step_frontiers(
        scheduler_output: SchedulerOutput,
    ) -> dict[str, int]:
        frontiers = {
            request.req_id: request.num_computed_tokens
            for request in scheduler_output.scheduled_new_reqs
        }
        cached = scheduler_output.scheduled_cached_reqs
        frontiers.update(zip(cached.req_ids, cached.num_computed_tokens))
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

        features = [
            copy(feature)
            for feature in target_request.mm_features
            if _feature_end(feature) <= target
        ]
        uses_prefetched_references = self._prepare_features_for_dispatch(
            features
        )
        physical_request = Request(
            request_id=request_id,
            client_index=target_request.client_index,
            prompt_token_ids=target_request.prompt_token_ids[:target],
            prompt_embeds=None,
            mm_features=features,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=target_request.eos_token_id,
            arrival_time=target_request.arrival_time,
            lora_request=target_request.lora_request,
            cache_salt=target_request.cache_salt,
            priority=target_request.priority,
            trace_headers=target_request.trace_headers,
            block_hasher=self.request_block_hasher,
            prefill_context=None,
        )
        physical = _PhysicalPrefix(
            request_id=request_id,
            request=physical_request,
            target=target,
            source=target_request,
        )
        context.inflight = physical
        self.physical_to_context[request_id] = context.key
        scope_name = (
            "streaming_prefill: physical_admit_reference"
            if uses_prefetched_references
            else "streaming_prefill: physical_admit_inline"
        )
        with record_function_or_nullcontext(scope_name):
            self.admit_request(physical_request)

    def _safe_target(
        self,
        *,
        request: Request,
        frontier: int,
        target: int,
    ) -> int:
        latest = request.num_prompt_tokens
        for feature in request.mm_features:
            start = feature.mm_position.offset
            end = _feature_end(feature)
            if start < target < end:
                target = start if start > frontier else min(end, latest)
                break

        if target < latest:
            aligned = target - (target % self.block_size)
            if aligned > frontier and not any(
                feature.mm_position.offset < aligned < _feature_end(feature)
                for feature in request.mm_features
            ):
                target = aligned
        return min(target, latest)

    def _complete_covered_logicals(
        self,
        context: _ContextState,
    ) -> None:
        completed: list[str] = []
        for request_id, logical in context.logical.items():
            if logical.target > context.computed_frontier:
                continue
            self._queue_completion(
                logical.client_index,
                LogicalRequestCompletion(
                    request_id=request_id,
                    finish_reason=FinishReason.LENGTH,
                ),
            )
            completed.append(request_id)

        if completed:
            with record_function_or_nullcontext(
                "streaming_prefill: coverage_complete"
            ):
                for request_id in completed:
                    context.logical.pop(request_id, None)
                    self.logical_to_context.pop(request_id, None)

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
        query_ids = list(context.covering_queries)
        query_ids.extend(
            request_id
            for request_id, query in self.pending_queries.items()
            if query.key == context.key
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
