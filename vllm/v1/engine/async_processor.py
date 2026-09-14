# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered, cancellation-safe frontend preprocessing and IPC commit."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial, wraps
from inspect import signature
from typing import Any, TypeVar

from vllm.kv_transfer_trace import record_kv_transfer_trace
from vllm.v1.engine import EngineCoreRequest

T = TypeVar("T")


class OrderedAsyncProcessor:
    """Keep sender-cache mutation and its IPC commit in one ordered transaction.

    Only CPU preprocessing runs in the executor. Commit callbacks run on the
    event loop, while the global transaction lock prevents cache references
    from overtaking the message that first carries their data. Admission is
    bounded; at most one executor job exists, irrespective of caller count.
    """

    def __init__(self, max_pending: int = 32) -> None:
        if max_pending < 1:
            raise ValueError("Async processor max_pending must be positive")
        self._admission = asyncio.Semaphore(max_pending)
        self._order = asyncio.Lock()
        self._cpu_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vllm-processor"
        )
        self._closed = False
        self._active = False
        self._cache_touched = False
        self._failure: BaseException | None = None

    @property
    def busy(self) -> bool:
        return self._active

    def mark_cache_mutation(self) -> None:
        self._cache_touched = True

    async def run(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        allow_failed: bool = False,
        recovery: bool = False,
    ) -> T:
        async with self._admission, self._order:
            if self._closed:
                raise RuntimeError("Async frontend processor is shut down")
            if self._failure is not None and not allow_failed:
                raise RuntimeError(
                    "Frontend MM cache commit failed; reset_mm_cache() is "
                    "required before admitting more requests"
                ) from self._failure

            self._active = True
            self._cache_touched = False

            async def transaction() -> T:
                try:
                    result = await operation()
                    if recovery:
                        self._failure = None
                    return result
                except BaseException as exc:
                    # Processing may have produced sender-cache entries that
                    # the receiver never saw. Fail closed until an ordered
                    # two-sided reset, rather than emit dangling references.
                    if self._cache_touched:
                        self._failure = exc
                    raise
                finally:
                    self._active = False

            task = asyncio.create_task(transaction())
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A running thread cannot be cancelled. Keep the lock
                    # through commit, then surface caller cancellation.
                    cancelled = True
                except BaseException:
                    break
            if cancelled:
                # Retrieve any transaction exception before raising, so it
                # cannot become an unhandled background task exception.
                if not task.cancelled():
                    task.exception()
                raise asyncio.CancelledError
            return task.result()

    async def process_inputs(self, processor: Any, *args: Any) -> Any:
        def process() -> Any:
            cache = getattr(processor, "mm_processor_cache", None)
            before = getattr(cache, "mutation_revision", None)
            try:
                return processor.process_inputs(*args)
            finally:
                # Built-in caches publish a revision for mutations, unlike
                # read-only membership tests. Early input validation errors
                # must not disable global admission. Unknown cache plugins
                # retain the conservative fail-closed behavior.
                self._cache_touched = cache is not None and (
                    before is None or cache.mutation_revision != before
                )

        result = await self.call_cpu(process)
        if self._closed:
            raise RuntimeError("Frontend shut down during preprocessing")
        return result

    async def call_cpu(self, function: Callable[..., T], *args: Any) -> T:
        context = copy_context()

        def guarded() -> T:
            with self._cpu_lock:
                return context.run(function, *args)

        return await asyncio.get_running_loop().run_in_executor(self._executor, guarded)

    def try_stat_mm_cache(self, processor: Any) -> Any:
        # Stats are optional and delta-based. Defer a sample instead of
        # blocking the output-drain loop behind preprocessing.
        if not self._cpu_lock.acquire(blocking=False):
            return None
        try:
            return processor.stat_mm_cache()
        finally:
            self._cpu_lock.release()

    def shutdown(self) -> None:
        self._closed = True
        # Do not synchronously join a potentially slow CPU operation on the
        # event loop. Its result is rejected before commit once closed.
        self._executor.shutdown(wait=False, cancel_futures=True)


def ordered_processor_call(
    method: Callable[..., Awaitable[T]] | None = None,
    *,
    allow_failed: bool = False,
    recovery: bool = False,
    stamp_arrival: bool = False,
) -> Any:
    """Fence admission/control methods without changing their public signature."""
    if method is None:
        return partial(
            ordered_processor_call,
            allow_failed=allow_failed,
            recovery=recovery,
            stamp_arrival=stamp_arrival,
        )

    positions = {name: index for index, name in enumerate(signature(method).parameters)}

    def argument(self_args: tuple, kwargs: dict, name: str) -> Any:
        index = positions[name]
        return self_args[index] if index < len(self_args) else kwargs.get(name)

    @wraps(method)
    async def ordered(self: Any, *args: Any, **kwargs: Any) -> T:
        entered = time.monotonic()
        request_id = None
        if stamp_arrival:
            self_args = (self, *args)
            request_id = argument(self_args, kwargs, "request_id")
            prompt = argument(self_args, kwargs, "prompt")
            if (
                not isinstance(prompt, EngineCoreRequest)
                and argument(self_args, kwargs, "arrival_time") is None
            ):
                index = positions["arrival_time"] - 1
                if index < len(args):
                    args = (*args[:index], time.time(), *args[index + 1 :])
                else:
                    kwargs["arrival_time"] = time.time()
            record_kv_transfer_trace(
                "async_llm_processor_ingress_enqueue",
                component="async_llm_frontend",
                request_id=request_id,
                method=method.__name__,
            )

        async def operation() -> T:
            if stamp_arrival:
                record_kv_transfer_trace(
                    "async_llm_processor_ingress_start",
                    component="async_llm_frontend",
                    request_id=request_id,
                    method=method.__name__,
                    queue_wait_ms=(time.monotonic() - entered) * 1000,
                )
            return await method(self, *args, **kwargs)

        ingress = getattr(self, "_processor_ingress", None)
        if ingress is None:
            return await operation()
        return await ingress.run(
            operation,
            allow_failed=allow_failed,
            recovery=recovery,
        )

    return ordered
