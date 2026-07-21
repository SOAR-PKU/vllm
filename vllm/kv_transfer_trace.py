# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in structured timing trace for KV-transfer critical paths.

The tracer intentionally depends only on the Python standard library so it can
be called from the AsyncLLM frontend, EngineCore, scheduler, and worker
processes without introducing import cycles. It is disabled unless
``VLLM_KV_TRANSFER_TRACE_DIR`` is set.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TRACE_DIR_ENV = "VLLM_KV_TRANSFER_TRACE_DIR"
TRACE_RUN_ID_ENV = "FULL_DUPLEX_RUN_ID"
TRACE_POLL_INTERVAL_MS_ENV = "VLLM_KV_TRANSFER_TRACE_POLL_INTERVAL_MS"
DEFAULT_POLL_INTERVAL_MS = 1000.0

_writer_lock = threading.Lock()
_writer_key: tuple[int, str] | None = None
_writer_fd: int | None = None
_warned = False
_warned_lock = threading.Lock()


def kv_transfer_trace_enabled() -> bool:
    return bool(os.environ.get(TRACE_DIR_ENV, "").strip())


def kv_transfer_trace_poll_interval_ns() -> int:
    raw = os.environ.get(TRACE_POLL_INTERVAL_MS_ENV, "").strip()
    if not raw:
        return int(DEFAULT_POLL_INTERVAL_MS * 1_000_000)
    try:
        value_ms = float(raw)
    except ValueError:
        return int(DEFAULT_POLL_INTERVAL_MS * 1_000_000)
    if value_ms <= 0:
        return 0
    return int(value_ms * 1_000_000)


def _get_writer_fd(trace_dir_raw: str, pid: int) -> int:
    global _writer_fd, _writer_key
    key = (pid, trace_dir_raw)
    if _writer_fd is not None and _writer_key == key:
        return _writer_fd

    if _writer_fd is not None:
        os.close(_writer_fd)
        _writer_fd = None
        _writer_key = None

    trace_dir = Path(trace_dir_raw).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"kv-transfer-trace-{pid}.jsonl"
    _writer_fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    _writer_key = key
    return _writer_fd


def _json_default(value: object) -> str:
    return str(value)


def _warn_once(exc: BaseException) -> None:
    global _warned
    if _warned:
        return
    with _warned_lock:
        if _warned:
            return
        _warned = True
        logger.warning("KV-transfer timing trace write failed: %s", exc)


def record_kv_transfer_trace(
    event_type: str,
    *,
    component: str,
    request_id: str | None = None,
    engine_id: str | None = None,
    event_time_ns: int | None = None,
    monotonic_time_ns: int | None = None,
    **fields: Any,
) -> bool:
    """Append one trace event without allowing tracing to break serving."""

    trace_dir_raw = os.environ.get(TRACE_DIR_ENV, "").strip()
    if not trace_dir_raw:
        return False
    normalized_event_type = event_type.strip()
    normalized_component = component.strip()
    if not normalized_event_type or not normalized_component:
        return False

    try:
        pid = os.getpid()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "event_type": normalized_event_type,
            "event_time_ns": (
                time.time_ns() if event_time_ns is None else int(event_time_ns)
            ),
            "monotonic_time_ns": (
                time.monotonic_ns()
                if monotonic_time_ns is None
                else int(monotonic_time_ns)
            ),
            "component": normalized_component,
            "pid": pid,
            "thread_name": threading.current_thread().name,
            "run_id": os.environ.get(TRACE_RUN_ID_ENV, "").strip() or None,
            "request_id": request_id,
            "engine_id": engine_id,
        }
        payload.update(fields)
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                default=_json_default,
            )
            + "\n"
        ).encode("utf-8")
        with _writer_lock:
            fd = _get_writer_fd(trace_dir_raw, pid)
            os.write(fd, encoded)
        return True
    except Exception as exc:  # pragma: no cover - tracing must be fail-open.
        _warn_once(exc)
        return False
