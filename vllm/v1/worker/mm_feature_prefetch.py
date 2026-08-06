# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-local storage for opportunistically prefetched MM features.

This cache is deliberately separate from vLLM's mirrored multimodal processor
cache. A producer sends processed feature data before the corresponding
SchedulerOutput is on the execution critical path. Entries remain pinned until
EngineCore explicitly releases the matching transfer generation; only released
entries are eligible for LRU eviction.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from vllm.logger import init_logger
from vllm.multimodal.cache import MultiModalCache, MultiModalCacheValue

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem


logger = init_logger(__name__)


MMFeaturePrefetchAck: TypeAlias = tuple[int, str, int, bool]
MM_FEATURE_PREFETCH_CONFIG_KEY = (
    "llm_rtc_streaming_prefill_enginecore_coordinator_enabled"
)
MM_FEATURE_PREFETCH_HORIZON_CHUNKS_CONFIG_KEY = (
    "llm_rtc_mm_feature_prefetch_horizon_chunks"
)


@dataclass(slots=True)
class MMFeaturePrefetchItem:
    """One identifier-addressable feature sent on the prefetch data plane."""

    identifier: str
    generation: int
    data: Any


@dataclass(slots=True)
class MMFeaturePrefetchRelease:
    """Release one exact feature generation for worker-local LRU eviction."""

    identifier: str
    generation: int


class MMFeaturePrefetchProtocolError(RuntimeError):
    """A SchedulerOutput referenced data that the worker did not acknowledge."""


@dataclass(slots=True)
class _CacheEntry:
    generation: int
    data: Any
    size: int
    last_access: int
    pinned: bool
    ever_materialized: bool = False


class WorkerMMFeaturePrefetchCache:
    """Bounded, thread-safe cache of processed MM features.

    New and generation-updated entries are pinned. Capacity pressure may evict
    only entries whose exact generation has been explicitly released.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self._used_bytes = 0
        self._access_sequence = 0
        self._items: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._metrics: dict[str, int] = {}

    def _record_locked(self, name: str, value: int = 1) -> None:
        self._metrics[name] = self._metrics.get(name, 0) + int(value)

    def _record(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._record_locked(name, value)

    def snapshot_metrics(self, *, reset: bool = False) -> dict[str, int]:
        """Return bounded aggregate cache counters.

        These counters deliberately reuse the cache lock already required by
        PUT/materialize/release operations. They do not read a clock and do not
        emit one log record per feature.
        """

        with self._lock:
            snapshot = dict(self._metrics)
            snapshot["cache_used_bytes"] = self._used_bytes
            snapshot["cache_entries"] = len(self._items)
            snapshot["cache_pinned_entries"] = sum(
                entry.pinned for entry in self._items.values()
            )
            if reset:
                self._metrics.clear()
            return snapshot

    def log_metrics(self, *, rank: int | None = None) -> None:
        """Write interval counters and current cache gauges as one record."""

        with self._lock:
            if not self._metrics:
                return
            metrics = dict(self._metrics)
            self._metrics.clear()
            metrics["cache_used_bytes"] = self._used_bytes
            metrics["cache_entries"] = len(self._items)
            metrics["cache_pinned_entries"] = sum(
                entry.pinned for entry in self._items.values()
            )
        if rank is not None:
            metrics["rank"] = rank
        logger.info(
            "LLM_RTC_WORKER_MM_FEATURE_PREFETCH_METRICS %s",
            json.dumps(metrics, separators=(",", ":"), sort_keys=True),
        )

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def __contains__(self, identifier: str) -> bool:
        with self._lock:
            return identifier in self._items

    def _touch_locked(self, identifier: str, entry: _CacheEntry) -> None:
        self._access_sequence += 1
        entry.last_access = self._access_sequence
        self._items.move_to_end(identifier)

    def _evict_for_capacity_locked(
        self,
        required_bytes: int,
        *,
        incoming_identifier: str,
        incoming_generation: int,
    ) -> bool:
        free_bytes = self.max_bytes - self._used_bytes
        if required_bytes <= free_bytes:
            return True

        bytes_needed = required_bytes - free_bytes
        candidates: list[tuple[str, _CacheEntry]] = []
        reclaimable_bytes = 0
        for identifier, entry in self._items.items():
            if identifier == incoming_identifier or entry.pinned:
                continue
            candidates.append((identifier, entry))
            reclaimable_bytes += entry.size
            if reclaimable_bytes >= bytes_needed:
                break

        if reclaimable_bytes < bytes_needed:
            pinned_count = sum(entry.pinned for entry in self._items.values())
            unpinned_count = len(self._items) - pinned_count
            reason = (
                "capacity_all_pinned"
                if unpinned_count == 0
                else "capacity_insufficient_unpinned"
            )
            self._record_locked("put_rejected_capacity")
            self._record_locked(
                f"put_rejected_{reason}",
            )
            logger.warning(
                "Rejecting worker MM feature prefetch cache item: "
                "identifier=%s generation=%d size_bytes=%d used_bytes=%d "
                "max_bytes=%d pinned_entries=%d unpinned_entries=%d reason=%s",
                incoming_identifier,
                incoming_generation,
                required_bytes,
                self._used_bytes,
                self.max_bytes,
                pinned_count,
                unpinned_count,
                reason,
            )
            return False

        for identifier, entry in candidates:
            used_before = self._used_bytes
            del self._items[identifier]
            self._used_bytes -= entry.size
            self._record_locked("evicted_entries")
            self._record_locked("evicted_bytes", entry.size)
            if not entry.ever_materialized:
                self._record_locked("evicted_before_use_entries")
                self._record_locked("evicted_before_use_bytes", entry.size)
            logger.debug(
                "Evicting worker MM feature prefetch cache entry: "
                "identifier=%s generation=%d size_bytes=%d "
                "used_bytes_before=%d used_bytes_after=%d "
                "last_access=%d pinned=%s reason=lru_unpinned_capacity",
                identifier,
                entry.generation,
                entry.size,
                used_before,
                self._used_bytes,
                entry.last_access,
                entry.pinned,
            )
        return True

    def put(self, identifier: str, generation: int, data: Any) -> bool:
        """Store and pin one feature generation.

        Duplicate puts for the current generation are idempotent and update
        LRU recency without changing pin state. Older generations are rejected.
        A newer generation atomically replaces the same identifier and pins it.
        """

        if (
            not identifier
            or generation < 0
            or data is None
            or self.max_bytes == 0
        ):
            self._record("put_rejected_invalid")
            logger.warning(
                "Rejecting worker MM feature prefetch cache item: "
                "identifier=%s generation=%d max_bytes=%d reason=invalid",
                identifier,
                generation,
                self.max_bytes,
            )
            return False

        try:
            item_size = MultiModalCache.get_item_size(
                cast(MultiModalCacheValue, data)
            )
        except Exception:
            self._record("put_rejected_size_calculation")
            logger.warning(
                "Rejecting worker MM feature prefetch cache item: "
                "identifier=%s generation=%d reason=size_calculation_failed",
                identifier,
                generation,
                exc_info=True,
            )
            return False

        if item_size > self.max_bytes:
            self._record("put_rejected_oversize")
            self._record("put_rejected_oversize_bytes", item_size)
            logger.warning(
                "Rejecting worker MM feature prefetch cache item: "
                "identifier=%s generation=%d size_bytes=%d max_bytes=%d "
                "reason=oversize",
                identifier,
                generation,
                item_size,
                self.max_bytes,
            )
            return False

        with self._lock:
            existing = self._items.get(identifier)
            if existing is not None and generation == existing.generation:
                self._touch_locked(identifier, existing)
                self._record_locked("put_duplicate_entries")
                return True
            if existing is not None and generation < existing.generation:
                self._record_locked("put_rejected_stale_generation")
                logger.warning(
                    "Rejecting worker MM feature prefetch cache item: "
                    "identifier=%s generation=%d cached_generation=%d "
                    "reason=stale_generation",
                    identifier,
                    generation,
                    existing.generation,
                )
                return False

            replaced_size = 0 if existing is None else existing.size
            additional_bytes = max(0, item_size - replaced_size)
            if not self._evict_for_capacity_locked(
                additional_bytes,
                incoming_identifier=identifier,
                incoming_generation=generation,
            ):
                return False

            if existing is not None:
                self._used_bytes -= existing.size
            self._access_sequence += 1
            self._items[identifier] = _CacheEntry(
                generation=generation,
                data=data,
                size=item_size,
                last_access=self._access_sequence,
                pinned=True,
            )
            self._items.move_to_end(identifier)
            self._used_bytes += item_size
            self._record_locked("put_accepted_entries")
            self._record_locked("put_accepted_bytes", item_size)
            self._metrics["cache_used_bytes_max"] = max(
                self._metrics.get("cache_used_bytes_max", 0),
                self._used_bytes,
            )
            return True

    def get(self, identifier: str) -> MultiModalKwargsItem:
        with self._lock:
            entry = self._items.get(identifier)
            if entry is None:
                self._record_locked("get_miss_entries")
                raise MMFeaturePrefetchProtocolError(
                    "SchedulerOutput referenced prefetched multimodal feature "
                    f"{identifier!r}, but this worker did not cache it"
                )
            entry.ever_materialized = True
            self._touch_locked(identifier, entry)
            self._record_locked("get_hit_entries")
            self._record_locked("get_hit_bytes", entry.size)
            return cast("MultiModalKwargsItem", entry.data)

    def release(self, identifier: str, generation: int) -> bool:
        """Unpin only the exact cached generation.

        Stale releases are ignored so a delayed control message cannot make a
        newer generation eligible for eviction.
        """

        with self._lock:
            entry = self._items.get(identifier)
            if entry is None or entry.generation != generation:
                self._record_locked("release_stale_or_missing_entries")
                return False
            entry.pinned = False
            self._record_locked("release_entries")
            return True

    def materialize_features(
        self,
        features: list[MultiModalFeatureSpec],
    ) -> None:
        """Materialize only references owned by this custom cache.

        ``data=None`` is also the wire representation used by vLLM's native
        mirrored MM receiver cache.  A custom-cache miss therefore is not a
        protocol error: it must remain unresolved so the native cache can
        materialize it in the next stage.
        """

        with self._lock:
            for feature in features:
                if feature.data is not None:
                    continue
                entry = self._items.get(feature.identifier)
                if entry is not None:
                    feature.data = entry.data
                    entry.ever_materialized = True
                    self._touch_locked(feature.identifier, entry)
                    self._record_locked("materialize_hit_entries")
                    self._record_locked("materialize_hit_bytes", entry.size)
                else:
                    self._record_locked("materialize_miss_entries")

    def clear(self) -> None:
        """Explicitly invalidate all ready entries."""

        self.log_metrics()
        with self._lock:
            self._items.clear()
            self._used_bytes = 0
            self._metrics.clear()
