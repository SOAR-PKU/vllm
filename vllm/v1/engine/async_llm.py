# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket
import time
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any, cast

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import PromptType
from vllm.kv_transfer_trace import record_kv_transfer_trace
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.transformers_utils.tokenizer import AnyTokenizer, init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.utils.func_utils import deprecate_kwargs
from vllm.utils.math_utils import cdiv
from vllm.v1.engine import EngineCoreRequest, PrefillContextMetadata
from vllm.v1.engine.async_processor import (
    OrderedAsyncProcessor,
    ordered_processor_call,
)
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


def _kv_trace_request_fields(request: EngineCoreRequest) -> dict[str, Any]:
    # Keep tracing compatible with EngineCoreRequest variants that predate
    # remote-prefill fields (for example the router's NIXL warm-up request).
    params = getattr(request, "kv_transfer_params", None) or {}
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    return {
        "do_remote_prefill": bool(params.get("do_remote_prefill")),
        "do_remote_decode": bool(params.get("do_remote_decode")),
        "remote_engine_id": params.get("remote_engine_id"),
        "prompt_token_count": (
            0 if prompt_token_ids is None else len(prompt_token_ids)
        ),
    }


class AsyncLLM(EngineClient):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests

        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = init_tokenizer_from_configs(self.model_config)

        self.processor = Processor(self.vllm_config, tokenizer)
        self._processor_ingress = (
            OrderedAsyncProcessor(
                max_pending=int(
                    os.environ.get("VLLM_ASYNC_PROCESSOR_MAX_PENDING", "32")
                )
            )
            if os.environ.get("VLLM_ASYNC_PROCESSOR_ENABLED", "1") == "1"
            else None
        )
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = OutputProcessor(
            self.tokenizer, log_stats=self.log_stats, stream_interval=stream_interval
        )
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        self.output_handler: asyncio.Task | None = None
        # Front-end lifecycle validation for append-only native
        # streaming prompts. EngineCore remains the physical admission owner;
        # these sets only reject duplicate OPEN/FINISH and out-of-order APPEND
        # calls before they cross the process boundary.
        self._streaming_prompt_open_request_ids: set[str] = set()
        self._streaming_prompt_finished_request_ids: set[str] = set()
        self._streaming_prompt_context_by_request: dict[
            str, tuple[int, int] | None
        ] = {}
        self._streaming_prompt_request_ids_by_context: dict[
            tuple[int, int], set[str]
        ] = {}
        self._retired_streaming_prompt_contexts: set[tuple[int, int]] = set()
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        if envs.VLLM_TORCH_PROFILER_DIR:
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                envs.VLLM_TORCH_PROFILER_DIR,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    envs.VLLM_TORCH_PROFILER_DIR, worker_name=worker_name, use_gzip=True
                ),
            )
        else:
            self.profiler = None

    @classmethod
    @deprecate_kwargs(
        "disable_log_requests",
        additional_message=(
            "This argument will have no effect. Use `enable_log_requests` instead."
        ),
    )
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        disable_log_requests: bool = True,  # Deprecated, will be removed
    ) -> "AsyncLLM":
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""

        shutdown_prometheus()

        if ingress := getattr(self, "_processor_ingress", None):
            ingress.shutdown()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return await self.engine_core.get_supported_tasks_async()

    def _prepare_tokenization_kwargs(
        self,
        params: SamplingParams,
        tokenization_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Keep streaming and query multimodal cache identities consistent."""
        if tokenization_kwargs is not None:
            return dict(tokenization_kwargs)
        kwargs: dict[str, Any] = {}
        _validate_truncation_size(
            self.model_config.max_model_len,
            params.truncate_prompt_tokens,
            kwargs,
        )
        return kwargs

    @ordered_processor_call(stamp_arrival=True)
    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        prefill_context: "PrefillContextMetadata | None" = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        record_kv_transfer_trace(
            "async_llm_add_request_enter",
            component="async_llm_frontend",
            request_id=request_id,
            prompt_is_engine_core_request=isinstance(prompt, EngineCoreRequest),
        )
        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        # Create a new output collector for the request.
        queue = RequestOutputCollector(output_kind=params.output_kind)

        # Convert Input --> Request.
        if isinstance(prompt, EngineCoreRequest):
            request = prompt
        else:
            assert prompt_text is None
            logger.warning_once(
                "Processor has been moved under OpenAIServing and will "
                "be removed from AsyncLLM in v0.13."
            )
            record_kv_transfer_trace(
                "async_llm_process_inputs_enter",
                component="async_llm_frontend",
                request_id=request_id,
            )
            request = await self._process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
                prefill_context,
            )
            record_kv_transfer_trace(
                "async_llm_process_inputs_exit",
                component="async_llm_frontend",
                request_id=request_id,
                **_kv_trace_request_fields(request),
            )
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        # Get the updated SamplingParams from the request, which
        # were cloned/updated in processor.process_inputs above.
        parent_params = request.sampling_params
        assert parent_params is not None

        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request_id, parent_params)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    async def _process_inputs(self, *args: Any) -> EngineCoreRequest:
        ingress = getattr(self, "_processor_ingress", None)
        if ingress is None:
            return self.processor.process_inputs(*args)
        return await ingress.process_inputs(self.processor, *args)

    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # Add the request to OutputProcessor (this process).
        trace_fields = _kv_trace_request_fields(request)
        record_kv_transfer_trace(
            "async_llm_output_processor_add_enter",
            component="async_llm_frontend",
            request_id=request.request_id,
            **trace_fields,
        )
        self.output_processor.add_request(request, prompt, parent_req, index, queue)
        record_kv_transfer_trace(
            "async_llm_output_processor_add_exit",
            component="async_llm_frontend",
            request_id=request.request_id,
            **trace_fields,
        )

        # Add the EngineCoreRequest to EngineCore (separate process).
        record_kv_transfer_trace(
            "async_llm_engine_core_send_enter",
            component="async_llm_frontend",
            request_id=request.request_id,
            **trace_fields,
        )
        await self.engine_core.add_request_async(request)
        record_kv_transfer_trace(
            "async_llm_engine_core_send_exit",
            component="async_llm_frontend",
            request_id=request.request_id,
            **trace_fields,
        )

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    @staticmethod
    def _streaming_prompt_context_key(
        prefill_context: PrefillContextMetadata | None,
    ) -> tuple[int, int] | None:
        if prefill_context is None:
            return None
        if prefill_context.kind != "streaming":
            raise ValueError(
                "Streaming prompt chunks require prefill_context.kind='streaming'"
            )
        return (
            prefill_context.router_session_id,
            prefill_context.context_lifetime_id,
        )

    def _ensure_streaming_prompt_frontend_state(self) -> None:
        """Initialize lifecycle state for lightweight test/legacy instances."""

        if not hasattr(self, "_streaming_prompt_open_request_ids"):
            self._streaming_prompt_open_request_ids = set()
            self._streaming_prompt_finished_request_ids = set()
            self._streaming_prompt_context_by_request = {}
            self._streaming_prompt_request_ids_by_context = {}
            self._retired_streaming_prompt_contexts = set()

    def _forget_streaming_prompt_frontend_request(
        self,
        request_id: str,
        *,
        mark_finished: bool,
    ) -> None:
        self._streaming_prompt_open_request_ids.discard(request_id)
        if mark_finished:
            self._streaming_prompt_finished_request_ids.add(request_id)
        context = self._streaming_prompt_context_by_request.pop(request_id, None)
        if context is None:
            return
        request_ids = self._streaming_prompt_request_ids_by_context.get(context)
        if request_ids is None:
            return
        request_ids.discard(request_id)
        if not request_ids:
            self._streaming_prompt_request_ids_by_context.pop(context, None)

    @ordered_processor_call(stamp_arrival=True)
    async def add_streaming_prompt_chunk(
        self,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams,
        request_id: str,
        *,
        is_first_chunk: bool,
        is_final_chunk: bool = False,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 1,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        prefill_context: PrefillContextMetadata | None = None,
    ) -> int:
        """Submit one native append-only streaming-prompt chunk.

        The first call creates one Scheduler request using ``request_id``.
        Later calls send only the Processor-produced token/MM delta; EngineCore
        appends it to that same Request. Non-final chunks deliberately bypass
        OutputProcessor and have no per-version completion. The return value is
        the number of prompt tokens accepted by this call, not a cumulative
        prompt length and not a physical-prefill acknowledgement.

        Multimodal placeholder offsets are delta-local. EngineCore rebases them
        onto the cumulative prompt before publishing a full worker snapshot.
        """

        if self.errored:
            raise EngineDeadError()
        # Start the global drain loop for OPEN/APPEND too. Those
        # chunks have no per-request OutputProcessor state, but EngineCore's
        # empty/stat outputs must still be consumed so the IPC queue cannot
        # become the backlog boundary.
        self._run_output_handler()
        if not isinstance(is_first_chunk, bool):
            raise TypeError("is_first_chunk must be a boolean")
        if not isinstance(is_final_chunk, bool):
            raise TypeError("is_final_chunk must be a boolean")
        if params.n != 1:
            raise ValueError("Streaming prompt chunks require SamplingParams.n == 1")

        self._ensure_streaming_prompt_frontend_state()
        context_key = self._streaming_prompt_context_key(prefill_context)
        open_request_ids = self._streaming_prompt_open_request_ids
        finished_request_ids = self._streaming_prompt_finished_request_ids

        if is_first_chunk:
            if request_id in open_request_ids:
                raise ValueError(f"Streaming prompt {request_id!r} is already open")
            if request_id in finished_request_ids:
                raise ValueError(f"Streaming prompt {request_id!r} already finished")
            if context_key in self._retired_streaming_prompt_contexts:
                raise ValueError(
                    "Cannot open a streaming prompt for a retired context: "
                    f"request_id={request_id!r}, context={context_key}"
                )
        else:
            if request_id not in open_request_ids:
                raise ValueError(
                    f"Streaming prompt append has no open anchor: {request_id!r}"
                )
            expected_context_key = self._streaming_prompt_context_by_request.get(
                request_id
            )
            if context_key != expected_context_key:
                raise ValueError(
                    "Streaming prompt append changed context identity: "
                    f"request_id={request_id!r}, expected={expected_context_key}, "
                    f"received={context_key}"
                )

        if isinstance(prompt, EngineCoreRequest):
            request = prompt
            if request.request_id != request_id:
                raise ValueError(
                    "EngineCoreRequest request_id does not match the streaming "
                    f"request_id: {request.request_id!r} != {request_id!r}"
                )
        else:
            # MM hashes include tokenization kwargs, including explicit False.
            # Match generate() even when both paths leave the tokens unchanged.
            tokenization_kwargs = self._prepare_tokenization_kwargs(
                params, tokenization_kwargs
            )
            request = await self._process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
                prefill_context,
            )
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        prompt_token_ids = request.prompt_token_ids
        if prompt_token_ids is None or request.prompt_embeds is not None:
            raise ValueError(
                "Streaming prompt chunks require token IDs without prompt embeds"
            )
        num_delta_tokens = len(prompt_token_ids)
        if num_delta_tokens == 0:
            raise ValueError(
                "A streaming prompt chunk cannot be empty; retire idle lanes explicitly"
            )

        previous_mm_offset = -1
        for feature in request.mm_features or ():
            position = feature.mm_position
            if position.offset < 0 or position.length < 0:
                raise ValueError(
                    "Streaming delta multimodal positions must be non-negative"
                )
            if position.offset < previous_mm_offset:
                raise ValueError(
                    "Streaming delta multimodal features must be ordered by offset"
                )
            if position.offset + position.length > num_delta_tokens:
                raise ValueError(
                    "Streaming delta multimodal placeholder exceeds the delta prompt"
                )
            previous_mm_offset = position.offset

        request.is_streaming_prompt = True
        request.is_streaming_prompt_finished = is_final_chunk
        # False selects append mode. Full-prompt replacement/LCP
        # invalidation is intentionally outside this primitive.
        request.is_prompt_update = False

        output_registered = False
        if is_final_chunk:
            # OPEN/APPEND are invisible to the
            # OutputProcessor; FINISH installs exactly one terminal collector.
            queue = RequestOutputCollector(output_kind=params.output_kind)
            self.output_processor.add_request(
                request,
                prompt_text,
                None,
                0,
                queue,
            )
            output_registered = True

        try:
            await self.engine_core.add_request_async(request)
        except BaseException:
            if output_registered:
                self.output_processor.abort_requests((request_id,))
            raise

        if is_first_chunk:
            open_request_ids.add(request_id)
            self._streaming_prompt_context_by_request[request_id] = context_key
            if context_key is not None:
                self._streaming_prompt_request_ids_by_context.setdefault(
                    context_key, set()
                ).add(request_id)
        if is_final_chunk:
            open_request_ids.discard(request_id)
            finished_request_ids.add(request_id)

        if self.log_requests:
            lifecycle = "OPEN" if is_first_chunk else "APPEND"
            if is_final_chunk:
                lifecycle += "+FINISH"
            logger.info(
                "Accepted streaming prompt %s for %s (%d delta tokens).",
                lifecycle,
                request_id,
                num_delta_tokens,
            )
        return num_delta_tokens

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prefill_context: "PrefillContextMetadata | None" = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and sampling_params.prompt_logprobs
        ):
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            self._run_output_handler()

            tokenization_kwargs = self._prepare_tokenization_kwargs(
                sampling_params, tokenization_kwargs
            )

            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
                prefill_context=prefill_context,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                assert isinstance(out, RequestOutput)
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        logger_manager = self.logger_manager
        processor = self.processor
        processor_ingress = getattr(self, "_processor_ingress", None)

        async def output_handler():
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    if num_outputs <= envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                        slices = (outputs.outputs,)
                    else:
                        slices = np.array_split(
                            outputs.outputs,
                            cdiv(num_outputs, envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE),
                        )

                    for i, outputs_slice in enumerate(slices):
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice,
                            outputs.timestamp,
                            iteration_stats,
                            (outputs.logical_request_completions if i == 0 else None),
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        if i + 1 < len(slices):
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        await engine_core.abort_requests_async(
                            processed_outputs.reqs_to_abort
                        )

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_manager:
                        logger_manager.record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=(
                                processor.stat_mm_cache()
                                if processor_ingress is None
                                else processor_ingress.try_stat_mm_cache(processor)
                            ),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    @ordered_processor_call(allow_failed=True)
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        self._ensure_streaming_prompt_frontend_state()
        native_request_ids = set(request_ids).intersection(
            self._streaming_prompt_open_request_ids
        )
        all_request_ids = self.output_processor.abort_requests(request_ids)
        all_request_ids.extend(
            request_id
            for request_id in native_request_ids
            if request_id not in all_request_ids
        )
        await self.engine_core.abort_requests_async(all_request_ids)
        for native_request_id in native_request_ids:
            self._forget_streaming_prompt_frontend_request(
                native_request_id,
                mark_finished=True,
            )

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def retire_streaming_prefill_context(
        self,
        router_session_id: int,
        context_lifetime_id: int,
    ) -> None:
        """Permanently retire one streaming media-context lifetime."""

        completion = await self._submit_streaming_prefill_retirement(
            router_session_id, context_lifetime_id
        )
        if completion is not None:
            await completion

    @ordered_processor_call(allow_failed=True)
    async def _submit_streaming_prefill_retirement(
        self,
        router_session_id: int,
        context_lifetime_id: int,
    ) -> asyncio.Future[Any] | None:
        submit_utility = getattr(self.engine_core, "submit_utility_async", None)
        if submit_utility is None:
            # Compatibility with lightweight/legacy Core clients.
            await self.engine_core.call_utility_async(  # type: ignore[attr-defined]
                "retire_streaming_prefill_context",
                router_session_id,
                context_lifetime_id,
            )
            completion = None
        else:
            completion = await submit_utility(
                "retire_streaming_prefill_context",
                router_session_id,
                context_lifetime_id,
            )
            # A cancelled caller may never await the returned receipt, but
            # the tombstone and retirement send must still complete.
            completion.add_done_callback(
                lambda future: None if future.cancelled() else future.exception()
            )
        self._ensure_streaming_prompt_frontend_state()
        context = (router_session_id, context_lifetime_id)
        self._retired_streaming_prompt_contexts.add(context)
        request_ids = tuple(
            self._streaming_prompt_request_ids_by_context.get(context, ())
        )
        if request_ids:
            # OPEN/APPEND never registered with OutputProcessor; this is a
            # no-op for them and cleans a rare FINISH racing with retire.
            self.output_processor.abort_requests(request_ids)
        for request_id in request_ids:
            self._forget_streaming_prompt_frontend_request(
                request_id,
                mark_finished=True,
            )
        return completion

    @ordered_processor_call(allow_failed=True)
    async def preempt_to_lmcache(
        self,
        request_id: str,
        interrupt_seq: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Suspend a running request without aborting its output collector."""

        await self.engine_core.interrupt_preempt_to_lmcache_async(
            request_id,
            interrupt_seq,
            dict(metadata or {}),
        )
        return {"status": "accepted"}

    @ordered_processor_call(allow_failed=True)
    async def resume_from_lmcache(
        self,
        request_id: str,
        interrupt_seq: int,
    ) -> dict[str, str]:
        await self.engine_core.interrupt_resume_async(request_id, interrupt_seq)
        return {"status": "accepted"}

    @ordered_processor_call(allow_failed=True)
    async def confirm_interrupt(
        self,
        request_id: str,
        interrupt_seq: int,
        rollback_prompt_bytes_b64: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        await self.engine_core.interrupt_confirm_async(
            request_id,
            interrupt_seq,
            rollback_prompt_bytes_b64,
            dict(metadata or {}),
        )
        return {"status": "accepted"}

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            self._run_output_handler()

            if tokenization_kwargs is None:
                tokenization_kwargs = {}
            _validate_truncation_size(
                self.model_config.max_model_len,
                truncate_prompt_tokens,
                tokenization_kwargs,
            )

            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            await self.abort(request_id)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    @property
    def tokenizer(self) -> AnyTokenizer | None:
        return self.processor.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: AnyTokenizer | None) -> None:
        ingress = getattr(self, "_processor_ingress", None)
        if ingress is not None and ingress.busy:
            raise RuntimeError("Cannot replace tokenizer during frontend admission")
        self.processor.tokenizer = tokenizer

    async def get_tokenizer(self) -> AnyTokenizer:
        if self.tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because skip_tokenizer_init is True"
            )

        return self.tokenizer

    async def is_tracing_enabled(self) -> bool:
        return self.observability_config.otlp_traces_endpoint is not None  # type: ignore

    async def do_log_stats(self) -> None:
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        coros = [self.engine_core.profile_async(True)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    @ordered_processor_call(allow_failed=True, recovery=True)
    async def reset_mm_cache(self) -> None:
        ingress = getattr(self, "_processor_ingress", None)
        if ingress is None:
            self.processor.clear_mm_cache()
        else:
            ingress.mark_cache_mutation()
            await ingress.call_cpu(self.processor.clear_mm_cache)
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(self) -> None:
        await self.engine_core.reset_prefix_cache_async()

    async def sleep(self, level: int = 1) -> None:
        await self.reset_prefix_cache()
        await self.engine_core.sleep_async(level)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # Wait 1 second before checking again

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """
        Scale up or down the data parallel size by adding or removing
        engine cores.
        Args:
            new_data_parallel_size: The new number of data parallel workers
            drain_timeout:
                Maximum time to wait for requests to drain (seconds)
        """
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return
        logger.info(
            "Waiting for requests to drain before scaling up to %s engines...",
            new_data_parallel_size,
        )
        await self.wait_for_requests_to_drain(drain_timeout)
        logger.info(
            "Requests have been drained, proceeding with scale to %s engines",
            new_data_parallel_size,
        )
        await self.engine_core.scale_elastic_ep(new_data_parallel_size)
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size

        # recreate stat loggers
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()
