# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace

import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreRequest,
    PrefillContextMetadata,
)
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


def _context(kind: str = "streaming", version: int = 0) -> PrefillContextMetadata:
    return PrefillContextMetadata(
        kind=kind,
        router_session_id=7,
        context_lifetime_id=11,
        context_version=version,
    )


def _feature(identifier: str, offset: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=object(),
        modality="audio",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=1),
    )


def _core_request(
    request_id: str,
    token_ids: list[int],
    *,
    context: PrefillContextMetadata | None,
    mm_features: list[MultiModalFeatureSpec] | None = None,
    is_final: bool = False,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=token_ids,
        mm_features=mm_features,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prefill_context=context,
        is_streaming_prompt=True,
        is_streaming_prompt_finished=is_final,
    )


class _Scheduler:
    def __init__(self) -> None:
        self.requests: dict[str, Request] = {}
        self.added: list[str] = []
        self.finished: list[tuple[list[str], RequestStatus]] = []

    def add_request(self, request: Request) -> None:
        assert request.request_id not in self.requests
        self.requests[request.request_id] = request
        self.added.append(request.request_id)

    def finish_requests(
        self,
        request_ids: list[str],
        status: RequestStatus,
    ) -> None:
        self.finished.append((list(request_ids), status))

    @staticmethod
    def get_kv_connector():
        return None


class _RejectingCoordinator:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def handles_streaming(self, _request: Request) -> bool:
        self.calls.append("handles_streaming")
        return True

    def add_streaming(self, _request: Request) -> None:
        self.calls.append("add_streaming")

    def observes_query(self, _request: Request) -> bool:
        self.calls.append("observes_query")
        return True

    def observe_query(self, _request: Request) -> None:
        self.calls.append("observe_query")

    def retire_context(self, *_args):
        self.calls.append("retire_context")
        return []

    def abort(self, _request_ids: list[str]) -> list[str]:
        self.calls.append("abort")
        return []


def _engine() -> tuple[EngineCore, _Scheduler, _RejectingCoordinator]:
    engine = object.__new__(EngineCore)
    scheduler = _Scheduler()
    coordinator = _RejectingCoordinator()
    engine.scheduler = scheduler
    engine.streaming_prefill_coordinator = coordinator
    engine._coordinator_cpu_metrics_enabled = False
    engine._optimized_coalesce_enabled = True
    engine._poll_mm_feature_prefetch_acks = lambda: None
    return engine, scheduler, coordinator


def test_engine_core_native_same_id_append_bypasses_coordinator_and_rebases_mm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.engine.core as core_module

    messages: list[str] = []
    monkeypatch.setattr(core_module.logger, "isEnabledFor", lambda _level: True)
    monkeypatch.setattr(
        core_module.logger,
        "info",
        lambda fmt, *args: messages.append(fmt % args),
    )

    engine, scheduler, coordinator = _engine()
    first = Request.from_engine_core_request(
        _core_request("anchor", [10, 11], context=_context(version=0)),
        None,
    )
    EngineCore._add_request_impl(engine, first)

    anchor = scheduler.requests["anchor"]
    assert anchor is first
    assert anchor.is_streaming_anchor
    assert anchor.streaming_anchor_open
    assert scheduler.added == ["anchor"]
    assert coordinator.calls == []

    delta = Request.from_engine_core_request(
        _core_request(
            "anchor",
            [12, 13],
            context=_context(version=1),
            mm_features=[_feature("audio-1", 0)],
        ),
        None,
    )
    EngineCore._add_request_impl(engine, delta)

    assert scheduler.requests["anchor"] is anchor
    assert scheduler.added == ["anchor"]
    assert anchor.prompt_token_ids == [10, 11, 12, 13]
    assert list(anchor.all_token_ids) == [10, 11, 12, 13]
    assert anchor.num_prompt_tokens == 4
    assert anchor.prompt_revision == 1
    assert anchor.mm_features[0].mm_position.offset == 2
    assert anchor.prefill_context == _context(version=1)
    assert coordinator.calls == []

    apply_events = [
        json.loads(message.removeprefix("native_streaming_prompt_apply "))
        for message in messages
        if message.startswith("native_streaming_prompt_apply ")
    ]
    assert [event["chunk_kind"] for event in apply_events] == ["OPEN", "APPEND"]
    expected_append_event = {
        "request_id": "anchor",
        "router_session_id": 7,
        "context_lifetime_id": 11,
        "context_version": 1,
        "delta_tokens": 2,
        "cumulative_tokens": 4,
        "prompt_revision": 1,
    }
    assert expected_append_event.items() <= apply_events[1].items()
    assert isinstance(apply_events[1]["applied_wall_ns"], int)


def test_context_retire_aborts_native_anchor_without_coordinator_carrier() -> None:
    engine, scheduler, coordinator = _engine()
    first = Request.from_engine_core_request(
        _core_request("anchor", [1, 2], context=_context()),
        None,
    )
    EngineCore._add_request_impl(engine, first)

    EngineCore.retire_streaming_prefill_context(engine, 7, 11)

    assert scheduler.finished == [
        (["anchor"], RequestStatus.FINISHED_ABORTED),
    ]
    assert engine._native_streaming_anchors_by_context == {}
    assert engine._native_streaming_context_by_anchor == {}
    assert (7, 11) in engine._retired_native_streaming_contexts
    assert coordinator.calls == []

    replacement = Request.from_engine_core_request(
        _core_request("replacement", [3], context=_context(version=2)),
        None,
    )
    with pytest.raises(ValueError, match="retired context"):
        EngineCore._add_request_impl(engine, replacement)


def test_native_lane_user_query_goes_directly_to_scheduler() -> None:
    engine, scheduler, coordinator = _engine()
    query = Request(
        request_id="query",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        eos_token_id=0,
        priority=0,
        prefill_context=_context(kind="user_query"),
    )

    EngineCore._add_request_impl(engine, query)

    assert scheduler.added == ["query"]
    assert scheduler.requests["query"] is query
    assert coordinator.calls == []


def test_async_llm_chunk_api_registers_output_processor_only_on_finish() -> None:
    asyncio.run(_test_async_llm_chunk_api_registers_output_processor_only_on_finish())


async def _test_async_llm_chunk_api_registers_output_processor_only_on_finish() -> None:
    sent: list[EngineCoreRequest] = []
    registered: list[str] = []
    utility_calls: list[tuple] = []

    class Processor:
        @staticmethod
        def process_inputs(request_id, prompt, params, *args):
            del args
            return _core_request(
                request_id,
                list(prompt["prompt_token_ids"]),
                context=prompt["context"],
                mm_features=prompt.get("mm_features"),
            )

    class EngineClient:
        resources = SimpleNamespace(engine_dead=False)

        @staticmethod
        def shutdown() -> None:
            return None

        @staticmethod
        async def add_request_async(request: EngineCoreRequest) -> None:
            sent.append(request)

        @staticmethod
        async def call_utility_async(*args) -> None:
            utility_calls.append(args)

    class OutputProcessor:
        @staticmethod
        def add_request(request, *_args) -> None:
            registered.append(request.request_id)

        @staticmethod
        def abort_requests(_request_ids):
            return []

    llm = object.__new__(AsyncLLM)
    llm.model_config = SimpleNamespace(max_model_len=128)
    llm.engine_core = EngineClient()
    llm.processor = Processor()
    llm.output_processor = OutputProcessor()
    llm.output_handler = None
    llm.log_requests = False
    llm._run_output_handler = lambda: None
    params = SamplingParams(max_tokens=1)

    first_tokens = await llm.add_streaming_prompt_chunk(
        {"prompt_token_ids": [10, 11], "context": _context(version=0)},
        params,
        "anchor",
        is_first_chunk=True,
        prefill_context=_context(version=0),
    )
    append_tokens = await llm.add_streaming_prompt_chunk(
        {"prompt_token_ids": [12], "context": _context(version=1)},
        params,
        "anchor",
        is_first_chunk=False,
        prefill_context=_context(version=1),
    )

    assert (first_tokens, append_tokens) == (2, 1)
    assert registered == []
    assert [request.prompt_token_ids for request in sent] == [[10, 11], [12]]
    assert all(request.is_streaming_prompt for request in sent)
    assert all(not request.is_streaming_prompt_finished for request in sent)
    assert all(not request.is_prompt_update for request in sent)

    with pytest.raises(ValueError, match="cannot be empty"):
        await llm.add_streaming_prompt_chunk(
            {"prompt_token_ids": [], "context": _context(version=2)},
            params,
            "anchor",
            is_first_chunk=False,
            is_final_chunk=True,
            prefill_context=_context(version=2),
        )
    assert len(sent) == 2
    assert registered == []

    final_tokens = await llm.add_streaming_prompt_chunk(
        {"prompt_token_ids": [13], "context": _context(version=2)},
        params,
        "anchor",
        is_first_chunk=False,
        is_final_chunk=True,
        prefill_context=_context(version=2),
    )
    assert final_tokens == 1
    assert registered == ["anchor"]
    assert sent[-1].is_streaming_prompt_finished

    with pytest.raises(ValueError, match="no open anchor"):
        await llm.add_streaming_prompt_chunk(
            {"prompt_token_ids": [14], "context": _context(version=3)},
            params,
            "anchor",
            is_first_chunk=False,
            prefill_context=_context(version=3),
        )

    # A separate open context is removed from frontend state by native retire.
    await llm.add_streaming_prompt_chunk(
        {"prompt_token_ids": [20], "context": _context(version=0)},
        params,
        "retired-anchor",
        is_first_chunk=True,
        prefill_context=_context(version=0),
    )
    await llm.retire_streaming_prefill_context(7, 11)
    assert utility_calls
    assert "retired-anchor" not in llm._streaming_prompt_open_request_ids
    assert llm._streaming_prompt_request_ids_by_context == {}
    assert (7, 11) in llm._retired_streaming_prompt_contexts
