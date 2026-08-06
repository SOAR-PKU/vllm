# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead EngineCore/coordinator CPU instrumentation.

The instrumentation is carried through ``VllmConfig.additional_config`` so it
does not expand vLLM's public EngineArgs surface.  It is intentionally disabled
by default.  Callers should guard timing with ``metrics.enabled``; a disabled
hot path then pays only that boolean branch and does not read a clock or acquire
a lock.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENGINECORE_CPU_PROFILING_ENABLED_CONFIG_KEY = (
    "llm_rtc_enginecore_coordinator_cpu_profiling_enabled"
)
ENGINECORE_CPU_PROFILING_INTERVAL_S_CONFIG_KEY = (
    "llm_rtc_enginecore_coordinator_cpu_profiling_interval_s"
)
ENGINECORE_CPU_PROFILING_JSONL_DIR_CONFIG_KEY = (
    "llm_rtc_enginecore_coordinator_cpu_profiling_jsonl_dir"
)
ENGINECORE_INPUT_DRAIN_MAX_MESSAGES_CONFIG_KEY = (
    "llm_rtc_enginecore_input_drain_max_messages"
)
ENGINECORE_INPUT_DRAIN_MAX_TIME_MS_CONFIG_KEY = (
    "llm_rtc_enginecore_input_drain_max_time_ms"
)
ENGINECORE_DIRECT_BASELINE_ENABLED_CONFIG_KEY = (
    "llm_rtc_enginecore_direct_baseline_enabled"
)

ENGINECORE_CPU_PROFILING_ENABLED_ENV = (
    "LLM_RTC_ENGINECORE_COORDINATOR_CPU_PROFILING_ENABLED"
)
ENGINECORE_CPU_PROFILING_INTERVAL_S_ENV = (
    "LLM_RTC_ENGINECORE_COORDINATOR_CPU_PROFILING_INTERVAL_S"
)
ENGINECORE_CPU_PROFILING_JSONL_DIR_ENV = (
    "LLM_RTC_ENGINECORE_COORDINATOR_CPU_PROFILING_JSONL_DIR"
)

INPUT_PREPROCESS_METRIC = "input_preprocess"
ENGINECORE_ADD_METRIC = "enginecore_add"
COORDINATOR_STREAMING_ADD_METRIC = "coordinator_streaming_add"
COORDINATOR_QUERY_METRIC = "coordinator_query"
COORDINATOR_ACK_METRIC = "coordinator_ack"
COORDINATOR_PREFETCH_ACK_METRIC = "coordinator_prefetch_ack"
COORDINATOR_MATERIALIZE_METRIC = "coordinator_materialize"
COORDINATOR_PREFIX_IDENTITY_METRIC = "coordinator_prefix_identity"
COORDINATOR_VERSION_VALIDATE_METRIC = "coordinator_version_validate"
COORDINATOR_QUERY_COVERAGE_METRIC = "coordinator_query_coverage"
COORDINATOR_PHYSICAL_BUILD_METRIC = "coordinator_physical_build"

DEFAULT_CPU_PROFILING_INTERVAL_S = 5.0
CPU_PROFILE_LOG_PREFIX = "ENGINECORE_COORDINATOR_CPU_PROFILE_JSON "


def thread_cpu_time_ns() -> int:
    """Return CPU consumed by the current thread, excluding blocking waits."""

    return time.thread_time_ns()


def _additional_config(vllm_config: Any) -> dict[str, Any]:
    candidate = getattr(vllm_config, "additional_config", None)
    return candidate if isinstance(candidate, dict) else {}


def _optional_positive_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _optional_positive_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _optional_env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning(
        "Ignoring invalid boolean environment override %s=%r",
        name,
        value,
    )
    return None


@dataclass(frozen=True, slots=True)
class EngineCoreInstrumentationConfig:
    cpu_profiling_enabled: bool = False
    cpu_profiling_interval_s: float = DEFAULT_CPU_PROFILING_INTERVAL_S
    cpu_profiling_jsonl_dir: str = ""
    input_drain_max_messages: int | None = None
    input_drain_max_time_ns: int | None = None

    @property
    def input_drain_observability_enabled(self) -> bool:
        return (
            self.cpu_profiling_enabled
            or self.input_drain_max_messages is not None
            or self.input_drain_max_time_ns is not None
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> EngineCoreInstrumentationConfig:
        additional = _additional_config(vllm_config)
        environment_enabled = _optional_env_bool(
            ENGINECORE_CPU_PROFILING_ENABLED_ENV
        )
        interval_s = _optional_positive_float(
            os.environ.get(ENGINECORE_CPU_PROFILING_INTERVAL_S_ENV)
            or additional.get(ENGINECORE_CPU_PROFILING_INTERVAL_S_CONFIG_KEY)
        )
        drain_time_ms = _optional_positive_float(
            additional.get(ENGINECORE_INPUT_DRAIN_MAX_TIME_MS_CONFIG_KEY)
        )
        jsonl_dir = os.environ.get(
            ENGINECORE_CPU_PROFILING_JSONL_DIR_ENV
        )
        if jsonl_dir is None:
            jsonl_dir = additional.get(
                ENGINECORE_CPU_PROFILING_JSONL_DIR_CONFIG_KEY,
                "",
            )
        return cls(
            cpu_profiling_enabled=(
                bool(
                    additional.get(
                        ENGINECORE_CPU_PROFILING_ENABLED_CONFIG_KEY,
                        False,
                    )
                )
                if environment_enabled is None
                else environment_enabled
            ),
            cpu_profiling_interval_s=(
                DEFAULT_CPU_PROFILING_INTERVAL_S
                if interval_s is None
                else interval_s
            ),
            cpu_profiling_jsonl_dir=str(jsonl_dir or "").strip(),
            input_drain_max_messages=_optional_positive_int(
                additional.get(ENGINECORE_INPUT_DRAIN_MAX_MESSAGES_CONFIG_KEY)
            ),
            input_drain_max_time_ns=(
                None
                if drain_time_ms is None
                else max(1, int(drain_time_ms * 1_000_000))
            ),
        )


@dataclass(slots=True)
class _CpuStat:
    count: int = 0
    total_ns: int = 0
    max_ns: int = 0

    def add(self, duration_ns: int, count: int) -> None:
        self.count += count
        self.total_ns += duration_ns
        self.max_ns = max(self.max_ns, duration_ns)


class EngineCoreCoordinatorMetrics:
    """Aggregate CPU timings and emit one bounded JSONL record per interval."""

    def __init__(self, config: EngineCoreInstrumentationConfig) -> None:
        self.enabled = config.cpu_profiling_enabled
        self._interval_ns = max(
            1,
            int(config.cpu_profiling_interval_s * 1_000_000_000),
        )
        self._jsonl_dir = config.cpu_profiling_jsonl_dir
        # Do not allocate synchronization objects or read clocks in the
        # disabled configuration. Instrumented call sites then pay one boolean
        # branch and nothing else.
        self._lock = threading.Lock() if self.enabled else None
        self._stats: dict[str, _CpuStat] = {}
        self._counters: dict[str, int] = {}
        self._maxima: dict[str, int] = {}
        now = time.monotonic_ns() if self.enabled else 0
        self._interval_started_ns = now
        self._next_emit_ns = now + self._interval_ns
        self._writer_fd: int | None = None
        self._warned = False

    def record_cpu_ns(
        self,
        name: str,
        duration_ns: int,
        *,
        count: int = 1,
    ) -> None:
        """Record one named CPU duration.

        The caller is responsible for checking ``enabled`` before starting a
        timer.  Keeping this method fail-open makes optional profiling unable to
        fail serving.
        """

        if not self.enabled:
            return
        normalized_duration = max(0, int(duration_ns))
        normalized_count = max(0, int(count))
        if normalized_count == 0:
            return
        lock = self._lock
        assert lock is not None
        with lock:
            self._stats.setdefault(name, _CpuStat()).add(
                normalized_duration,
                normalized_count,
            )

    def record_counter(self, name: str, value: int = 1) -> None:
        if not self.enabled:
            return
        lock = self._lock
        assert lock is not None
        with lock:
            self._counters[name] = self._counters.get(name, 0) + int(value)

    def record_maximum(self, name: str, value: int) -> None:
        if not self.enabled:
            return
        normalized = max(0, int(value))
        lock = self._lock
        assert lock is not None
        with lock:
            self._maxima[name] = max(self._maxima.get(name, 0), normalized)

    def record_input_drain(
        self,
        *,
        message_count: int,
        cpu_ns: int,
        wall_ns: int,
        depth_before: int,
        depth_after: int,
        message_budget_hit: bool,
        time_budget_hit: bool,
    ) -> None:
        if not self.enabled:
            return
        lock = self._lock
        assert lock is not None
        with lock:
            self._stats.setdefault("input_drain", _CpuStat()).add(
                max(0, int(cpu_ns)),
                1,
            )
            self._counters["input_drain_message_count"] = (
                self._counters.get("input_drain_message_count", 0)
                + max(0, int(message_count))
            )
            self._counters["input_drain_wall_ns"] = (
                self._counters.get("input_drain_wall_ns", 0)
                + max(0, int(wall_ns))
            )
            if message_budget_hit:
                self._counters["input_drain_message_budget_hits"] = (
                    self._counters.get("input_drain_message_budget_hits", 0) + 1
                )
            if time_budget_hit:
                self._counters["input_drain_time_budget_hits"] = (
                    self._counters.get("input_drain_time_budget_hits", 0) + 1
                )
            self._maxima["input_queue_depth_before_max"] = max(
                self._maxima.get("input_queue_depth_before_max", 0),
                max(0, int(depth_before)),
            )
            self._maxima["input_queue_depth_after_max"] = max(
                self._maxima.get("input_queue_depth_after_max", 0),
                max(0, int(depth_after)),
            )

    def maybe_emit(self, *, force: bool = False) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        now = time.monotonic_ns()
        if not force and now < self._next_emit_ns:
            return None

        lock = self._lock
        assert lock is not None
        with lock:
            if not force and now < self._next_emit_ns:
                return None
            payload: dict[str, Any] = {
                "schema_version": 1,
                "event_type": "enginecore_coordinator_cpu_profile",
                "event_time_ns": time.time_ns(),
                "monotonic_time_ns": now,
                "pid": os.getpid(),
                "thread_name": threading.current_thread().name,
                "run_id": os.environ.get("FULL_DUPLEX_RUN_ID", "").strip()
                or None,
                "interval_started_monotonic_ns": self._interval_started_ns,
                "interval_duration_ns": max(
                    0,
                    now - self._interval_started_ns,
                ),
            }
            for name, stat in sorted(self._stats.items()):
                payload[f"{name}_count"] = stat.count
                payload[f"{name}_cpu_ns"] = stat.total_ns
                payload[f"{name}_cpu_ns_max"] = stat.max_ns
            payload.update(self._counters)
            payload.update(self._maxima)
            self._stats.clear()
            self._counters.clear()
            self._maxima.clear()
            self._interval_started_ns = now
            self._next_emit_ns = now + self._interval_ns

        self._write(payload)
        logger.info(
            "%s%s",
            CPU_PROFILE_LOG_PREFIX,
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )
        return payload

    def close(self) -> None:
        if not self.enabled:
            return
        self.maybe_emit(force=True)
        lock = self._lock
        assert lock is not None
        with lock:
            if self._writer_fd is not None:
                os.close(self._writer_fd)
                self._writer_fd = None

    def _write(self, payload: dict[str, Any]) -> None:
        if not self._jsonl_dir:
            return
        try:
            encoded = (
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            ).encode("utf-8")
            lock = self._lock
            assert lock is not None
            with lock:
                if self._writer_fd is None:
                    output_dir = Path(self._jsonl_dir).expanduser()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = (
                        output_dir
                        / f"enginecore-coordinator-cpu-{os.getpid()}.jsonl"
                    )
                    self._writer_fd = os.open(
                        output_path,
                        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                        0o644,
                    )
                os.write(self._writer_fd, encoded)
        except Exception as exc:  # pragma: no cover - profiling is fail-open.
            if not self._warned:
                self._warned = True
                logger.warning(
                    "EngineCore coordinator CPU profile write failed: %s",
                    exc,
                )
