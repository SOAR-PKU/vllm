# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from vllm.multimodal.cache import (
    MultiModalProcessorOnlyCache,
    MultiModalProcessorSenderCache,
)
from vllm.multimodal.inputs import (
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, PrefillContextMetadata
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.async_processor import OrderedAsyncProcessor
from vllm.v1.engine.core_client import AsyncMPClient

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def frontend():
    events = []
    started = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    sender_cache = set()
    receiver_cache = set()
    payloads = {}
    arrivals = {}
    processing_started = {}
    params = SamplingParams(max_tokens=1)

    class Processor:
        tokenizer = None
        mm_processor_cache = SimpleNamespace(mutation_revision=0)

        @staticmethod
        def process_inputs(request_id, prompt, request_params, *args):
            assert threading.get_ident() != loop_thread
            events.append(("process", request_id))
            arrivals[request_id] = args[0]
            processing_started[request_id] = time.time()
            if prompt.get("invalid"):
                raise ValueError("invalid input before cache update")
            if prompt.get("block"):
                started.set()
                assert release.wait(5), "test failed to release frontend worker"
            key = prompt.get("cache_key", request_id)
            Processor.mm_processor_cache.mutation_revision += 1
            payloads[request_id] = (key, key not in sender_cache)
            sender_cache.add(key)
            return EngineCoreRequest(
                request_id=request_id,
                prompt_token_ids=prompt.get("prompt_token_ids", [1, 2]),
                mm_features=None,
                sampling_params=request_params,
                pooling_params=None,
                eos_token_id=0,
                arrival_time=args[0],
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )

        @staticmethod
        def stat_mm_cache():
            return "stats"

        @staticmethod
        def clear_mm_cache():
            events.append(("clear", "sender"))
            Processor.mm_processor_cache.mutation_revision += 1
            sender_cache.clear()

    class Core:
        resources = SimpleNamespace(engine_dead=False)
        send_gate = None
        fail_send = False

        async def add_request_async(self, request):
            assert threading.get_ident() == loop_thread
            events.append(("send", request.request_id))
            if self.send_gate is not None:
                await self.send_gate.wait()
            if self.fail_send:
                raise RuntimeError("send failed")
            key, carries_data = payloads[request.request_id]
            if carries_data:
                receiver_cache.add(key)
            else:
                assert key in receiver_cache, "reference overtook its payload"
            events.append(("commit", request.request_id))

        @staticmethod
        async def abort_requests_async(request_ids):
            events.extend(("abort", item) for item in request_ids)

        @staticmethod
        async def call_utility_async(*args):
            events.append(("utility", args[0]))

        @staticmethod
        async def reset_mm_cache_async():
            events.append(("clear", "receiver"))
            receiver_cache.clear()

        @staticmethod
        def shutdown():
            pass

    class Output:
        @staticmethod
        def add_request(request, *_args):
            assert threading.get_ident() == loop_thread
            events.append(("register", request.request_id))

        @staticmethod
        def abort_requests(request_ids):
            return list(request_ids)

    llm = object.__new__(AsyncLLM)
    llm.model_config = SimpleNamespace(max_model_len=256)
    llm.processor = Processor()
    llm.engine_core = Core()
    llm.output_processor = Output()
    llm.output_handler = None
    llm.log_requests = False
    llm._run_output_handler = lambda: None
    llm._processor_ingress = OrderedAsyncProcessor(max_pending=2)
    yield SimpleNamespace(
        llm=llm,
        events=events,
        started=started,
        release=release,
        params=params,
        arrivals=arrivals,
        processing_started=processing_started,
    )
    release.set()
    llm._processor_ingress.shutdown()


async def wait_started(frontend):
    async with asyncio.timeout(2):
        while not frontend.started.is_set():
            await asyncio.sleep(0.001)


async def _off_loop_and_global_order(frontend):
    f = frontend
    f.llm.engine_core.send_gate = asyncio.Event()
    first = asyncio.create_task(
        f.llm.add_request("query", {"block": True, "cache_key": "shared"}, f.params)
    )
    await wait_started(f)
    second = asyncio.create_task(
        f.llm.add_streaming_prompt_chunk(
            {"cache_key": "shared"}, f.params, "lane", is_first_chunk=True
        )
    )
    # A blocked CPU function does not prevent the loop from servicing tasks.
    heartbeats = []
    for tick in range(5):
        await asyncio.sleep(0)
        heartbeats.append(tick)
    assert len(heartbeats) == 5
    assert f.events == [("process", "query")]
    f.release.set()
    async with asyncio.timeout(2):
        while ("send", "query") not in f.events:
            await asyncio.sleep(0.001)
    assert ("process", "lane") not in f.events
    f.llm.engine_core.send_gate.set()
    collector, tokens = await asyncio.gather(first, second)
    assert collector is not None and tokens == 2
    assert f.events.index(("commit", "query")) < f.events.index(("process", "lane"))


async def _active_cancellation_before_abort(frontend):
    f = frontend
    admission = asyncio.create_task(
        f.llm.add_request("query", {"block": True}, f.params)
    )
    await wait_started(f)
    admission.cancel()
    await asyncio.sleep(0)
    # Repeat cancellation must not release the order lock while CPU is active.
    admission.cancel()
    abort = asyncio.create_task(f.llm.abort("query"))
    await asyncio.sleep(0)
    assert not admission.done() and not abort.done()
    f.release.set()
    with pytest.raises(asyncio.CancelledError):
        await admission
    await abort
    assert f.events.index(("commit", "query")) < f.events.index(("abort", "query"))
    await f.llm.add_request("next", {"cache_key": "query"}, f.params)


async def _pending_cancellation(frontend):
    f = frontend
    active = asyncio.create_task(f.llm.add_request("active", {"block": True}, f.params))
    await wait_started(f)
    pending = asyncio.create_task(f.llm.add_request("pending", {}, f.params))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    f.release.set()
    await active
    assert ("process", "pending") not in f.events
    await f.llm.add_request("next", {}, f.params)


async def _open_append_finish_retire(frontend):
    f = frontend
    context = PrefillContextMetadata(
        kind="streaming",
        router_session_id=7,
        context_lifetime_id=11,
        context_version=0,
    )
    tasks = []
    for index in range(3):
        tasks.append(
            asyncio.create_task(
                f.llm.add_streaming_prompt_chunk(
                    {"block": index == 0},
                    f.params,
                    "lane",
                    is_first_chunk=index == 0,
                    is_final_chunk=index == 2,
                    prefill_context=context,
                )
            )
        )
        await asyncio.sleep(0)
    await wait_started(f)
    retire = asyncio.create_task(f.llm.retire_streaming_prefill_context(7, 11))
    f.release.set()
    assert await asyncio.gather(*tasks) == [2, 2, 2]
    await retire
    assert f.events[-1] == ("utility", "retire_streaming_prefill_context")
    assert f.events.count(("register", "lane")) == 1
    assert "lane" not in f.llm._streaming_prompt_open_request_ids


async def _reset_fence(frontend):
    f = frontend
    first = asyncio.create_task(
        f.llm.add_request("first", {"block": True, "cache_key": "shared"}, f.params)
    )
    await wait_started(f)
    reset = asyncio.create_task(f.llm.reset_mm_cache())
    await asyncio.sleep(0)
    second = asyncio.create_task(
        f.llm.add_request("second", {"cache_key": "shared"}, f.params)
    )
    f.release.set()
    await asyncio.gather(first, reset, second)
    assert f.events.index(("commit", "first")) < f.events.index(("clear", "sender"))
    assert f.events.index(("clear", "sender")) < f.events.index(("clear", "receiver"))
    assert f.events.index(("clear", "receiver")) < f.events.index(("process", "second"))


async def _failed_commit_recovery(frontend):
    f = frontend
    f.llm.engine_core.fail_send = True
    with pytest.raises(RuntimeError, match="send failed"):
        await f.llm.add_request("failed", {}, f.params)
    with pytest.raises(RuntimeError, match="reset_mm_cache"):
        await f.llm.add_request("blocked", {}, f.params)
    assert ("process", "blocked") not in f.events
    await f.llm.abort("failed")
    f.llm.engine_core.fail_send = False
    await f.llm.reset_mm_cache()
    await f.llm.add_request("recovered", {}, f.params)


async def _precache_validation_failure_does_not_poison_ingress(frontend):
    f = frontend
    with pytest.raises(ValueError, match="invalid input"):
        await f.llm.add_request("invalid", {"invalid": True}, f.params)
    await f.llm.add_request("valid", {}, f.params)
    assert ("commit", "valid") in f.events


async def _arrival_time_includes_frontend_queue(frontend):
    f = frontend
    active = asyncio.create_task(f.llm.add_request("active", {"block": True}, f.params))
    await wait_started(f)
    queued = asyncio.create_task(f.llm.add_request("queued", {}, f.params, None))
    streaming = asyncio.create_task(
        f.llm.add_streaming_prompt_chunk(
            {},
            f.params,
            "lane",
            is_first_chunk=True,
        )
    )
    await asyncio.sleep(0)
    while_queued = time.time()
    f.release.set()
    await asyncio.gather(active, queued, streaming)
    assert f.arrivals["queued"] <= while_queued <= f.processing_started["queued"]
    assert f.arrivals["lane"] <= while_queued <= f.processing_started["lane"]
    await f.llm.add_request("explicit", {}, f.params, arrival_time=123.0)
    assert f.arrivals["explicit"] == 123.0


async def _prebuilt_request_retains_arrival_time(frontend):
    f = frontend
    request = EngineCoreRequest(
        request_id="prebuilt",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=f.params,
        pooling_params=None,
        eos_token_id=0,
        arrival_time=42.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )

    async def send(incoming):
        assert incoming is request and incoming.arrival_time == 42.0

    f.llm.engine_core.add_request_async = send
    await f.llm.add_request("prebuilt", request, f.params)


async def _stats_and_tokenizer_guard(frontend):
    f = frontend
    active = asyncio.create_task(f.llm.add_request("active", {"block": True}, f.params))
    await wait_started(f)
    assert f.llm._processor_ingress.try_stat_mm_cache(f.llm.processor) is None
    with pytest.raises(RuntimeError, match="replace tokenizer"):
        f.llm.tokenizer = object()
    f.release.set()
    await active
    assert f.llm._processor_ingress.try_stat_mm_cache(f.llm.processor) == "stats"
    tokenizer = object()
    f.llm.tokenizer = tokenizer
    assert f.llm.tokenizer is tokenizer


async def _shutdown_running_job(frontend):
    f = frontend
    active = asyncio.create_task(f.llm.add_request("active", {"block": True}, f.params))
    await wait_started(f)
    f.llm._processor_ingress.shutdown()
    f.release.set()
    with pytest.raises(RuntimeError, match="shut down"):
        await active
    assert ("send", "active") not in f.events
    with pytest.raises(RuntimeError, match="shut down"):
        await f.llm.add_request("next", {}, f.params)


async def _retire_receipt_does_not_block_admission(frontend):
    f = frontend
    context = PrefillContextMetadata(
        kind="streaming",
        router_session_id=7,
        context_lifetime_id=11,
        context_version=0,
    )
    await f.llm.add_streaming_prompt_chunk(
        {},
        f.params,
        "lane",
        is_first_chunk=True,
        prefill_context=context,
    )
    receipt = asyncio.get_running_loop().create_future()

    async def submit_utility(method, *args):
        f.events.append(("utility_send", method))
        return receipt

    f.llm.engine_core.submit_utility_async = submit_utility
    retirement = asyncio.create_task(f.llm.retire_streaming_prefill_context(7, 11))
    async with asyncio.timeout(2):
        while (7, 11) not in f.llm._retired_streaming_prompt_contexts:
            await asyncio.sleep(0)
    assert not retirement.done()
    await f.llm.add_request("other_query", {}, f.params)
    assert ("commit", "other_query") in f.events
    with pytest.raises(ValueError, match="no open anchor"):
        await f.llm.add_streaming_prompt_chunk(
            {},
            f.params,
            "lane",
            is_first_chunk=False,
            prefill_context=context,
        )
    receipt.set_result(None)
    await retirement


async def _core_utility_send_and_receipt_are_separate(_frontend):
    client = object.__new__(AsyncMPClient)
    client.core_engine = b"engine"
    client.client_index = 0
    client.utility_results = {}
    client.encoder = SimpleNamespace(encode=lambda _message: [b"encoded"])
    client._ensure_output_queue_task = lambda: None
    sends = []

    async def send(message, engine, objects):
        sends.append((message, engine, objects))

    client._send_input_message = send
    receipt = await client.submit_utility_async("retire", 7, 11)
    assert len(sends) == 1 and not receipt.done()
    receipt.set_result("retired")
    assert await receipt == "retired"


@pytest.mark.parametrize(
    "scenario",
    [
        _off_loop_and_global_order,
        _active_cancellation_before_abort,
        _pending_cancellation,
        _open_append_finish_retire,
        _reset_fence,
        _failed_commit_recovery,
        _precache_validation_failure_does_not_poison_ingress,
        _arrival_time_includes_frontend_queue,
        _prebuilt_request_retains_arrival_time,
        _stats_and_tokenizer_guard,
        _shutdown_running_job,
        _retire_receipt_does_not_block_admission,
        _core_utility_send_and_receipt_are_separate,
    ],
    ids=lambda scenario: scenario.__name__,
)
def test_async_processor(frontend, scenario):
    asyncio.run(scenario(frontend))


@pytest.mark.parametrize(
    "cache_class", [MultiModalProcessorOnlyCache, MultiModalProcessorSenderCache]
)
def test_processor_cache_mutation_revision(cache_class):
    config = SimpleNamespace(
        get_multimodal_config=lambda: SimpleNamespace(mm_processor_cache_gb=0.001)
    )
    cache = cache_class(config)
    assert cache.mutation_revision == 0
    assert not cache.is_cached_item("image")
    assert cache.mutation_revision == 0
    item = MultiModalKwargsItem.from_elems(
        [
            MultiModalFieldElem(
                modality="image",
                key="pixels",
                data=torch.empty(1),
                field=MultiModalSharedField(1),
            )
        ]
    )
    cache.get_and_update_item((item, []), "image")
    assert cache.mutation_revision == 1
    assert cache.is_cached_item("image")
    assert cache.mutation_revision == 1
    cache.get_and_update_item(None, "image")
    assert cache.mutation_revision == 2
    cache.clear_cache()
    assert cache.mutation_revision == 3
