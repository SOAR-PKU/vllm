# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-local storage for opportunistically prefetched MM features.

This cache is deliberately separate from vLLM's mirrored multimodal processor
cache.  A producer sends processed feature data before the corresponding
SchedulerOutput is on the execution critical path.  Once a worker acknowledges
an item, the item remains addressable by identifier until an explicit cache
reset or worker shutdown.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from vllm.multimodal.cache import MultiModalCache, MultiModalCacheValue

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalKwargsItem


MMFeaturePrefetchAck: TypeAlias = tuple[int, str, bool]
MM_FEATURE_PREFETCH_CONFIG_KEY = (
    "llm_rtc_streaming_prefill_enginecore_coordinator_enabled"
)


@dataclass(slots=True)
class MMFeaturePrefetchItem:
    """One identifier-addressable feature sent on the prefetch data plane."""

    identifier: str
    data: Any


class MMFeaturePrefetchProtocolError(RuntimeError):
    """A SchedulerOutput referenced data that the worker did not acknowledge."""


class WorkerMMFeaturePrefetchCache:
    """Bounded, thread-safe and non-evicting cache of processed MM features."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = max_bytes
        self._used_bytes = 0
        self._items: dict[str, MultiModalKwargsItem] = {}
        self._lock = threading.RLock()

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def __contains__(self, identifier: str) -> bool:
        with self._lock:
            return identifier in self._items

    def put(self, identifier: str, data: Any) -> bool:
        """Store an item without evicting acknowledged entries.

        Duplicate puts are idempotent because ``identifier`` is the content
        identity used by the multimodal processor and encoder caches.  A full
        cache returns ``False`` immediately; it never stalls model execution
        and never evicts an item that was previously acknowledged as ready.
        """

        if not identifier or data is None or self.max_bytes == 0:
            return False

        try:
            item_size = MultiModalCache.get_item_size(
                cast(MultiModalCacheValue, data)
            )
        except Exception:
            return False

        with self._lock:
            if identifier in self._items:
                return True
            if item_size > self.max_bytes - self._used_bytes:
                return False
            self._items[identifier] = data
            self._used_bytes += item_size
            return True

    def get(self, identifier: str) -> MultiModalKwargsItem:
        with self._lock:
            item = self._items.get(identifier)
        if item is None:
            raise MMFeaturePrefetchProtocolError(
                "SchedulerOutput referenced prefetched multimodal feature "
                f"{identifier!r}, but this worker did not cache it"
            )
        return item

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

        for feature in features:
            if feature.data is not None:
                continue
            with self._lock:
                item = self._items.get(feature.identifier)
            if item is not None:
                feature.data = item

    def clear(self) -> None:
        """Explicitly invalidate all ready entries."""

        with self._lock:
            self._items.clear()
            self._used_bytes = 0
