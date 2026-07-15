# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import math
import os
import threading
import time
from collections import deque
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.base_policy import (
    TRANSFER_POLICY_BASE,
    TRANSFER_POLICY_LEGACY_EAGER,
    VALID_TRANSFER_POLICIES,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
    TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip
from vllm.utils.torch_utils import current_stream

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32


def resolve_p2p_hostname(config: KVTransferConfig, hostname: str) -> str:
    """Use the endpoint advertised through KVTransferConfig when unspecified."""
    if hostname:
        return hostname
    configured_ip = getattr(config, "kv_ip", "")
    return configured_ip or get_ip()


class P2pTransferError(RuntimeError):
    """A request-scoped P2P transfer failed and must not be decoded."""


@dataclass(frozen=True)
class TransferFailure:
    reason: str


@dataclass(frozen=True)
class CpuStagedTensor:
    addr: int
    dtype: torch.dtype
    shape: torch.Size


@contextmanager
def set_p2p_nccl_context(num_channels: str):
    original_values: dict[str, Any] = {}
    env_vars = [
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CUMEM_ENABLE",
        "NCCL_BUFFSIZE",
        "NCCL_PROTO",  # LL,LL128,SIMPLE
        "NCCL_ALGO",  # RING,TREE
    ]

    for var in env_vars:
        original_values[var] = os.environ.get(var)

    logger.info("set_p2p_nccl_context, original_values: %s", original_values)

    try:
        os.environ["NCCL_MAX_NCHANNELS"] = num_channels
        os.environ["NCCL_MIN_NCHANNELS"] = num_channels
        os.environ["NCCL_CUMEM_ENABLE"] = "1"
        yield
    finally:
        for var in env_vars:
            if original_values[var] is not None:
                os.environ[var] = original_values[var]
            else:
                os.environ.pop(var, None)


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


class P2pNcclEngine:
    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "",
        port_offset: int = 0,
        library_path: str | None = None,
    ) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.nccl = NCCLLibrary(library_path)

        hostname = resolve_p2p_hostname(config, hostname)
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port

        # Each card corresponds to a ZMQ address.
        self.zmq_address = f"{self._hostname}:{self._port}"

        # If `proxy_ip` or `proxy_port` is `""`,
        # then the ping thread will not be enabled.
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
            self.http_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port
            # the `http_port` must be consistent with the port of OpenAI.
            http_port = self.config.get_from_extra_config("http_port", None)
            if http_port is None:
                example_cfg = {
                    "kv_connector": "P2pNcclConnector",
                    "kv_connector_extra_config": {"http_port": 8000},
                }
                example = (
                    f"--port=8000 --kv-transfer-config='{json.dumps(example_cfg)}'"
                )
                raise ValueError(
                    "kv_connector_extra_config.http_port is required. "
                    f"Example: {example}"
                )
            self.http_address = f"{self._hostname}:{http_port}"

        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")

        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)

        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()
        self._pool_lock = threading.Lock()

        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

        self.transfer_policy = self.config.get_from_extra_config(
            "transfer_policy", TRANSFER_POLICY_LEGACY_EAGER
        )
        if self.transfer_policy not in VALID_TRANSFER_POLICIES:
            raise ValueError(f"Unsupported P2P transfer policy: {self.transfer_policy}")

        if self.transfer_policy == TRANSFER_POLICY_BASE:
            mem_pool_size_gb = float(
                self.config.get_from_extra_config(
                    "cpu_staging_size_gib_per_tp_rank", DEFAULT_MEM_POOL_SIZE_GB
                )
            )
        else:
            mem_pool_size_gb = float(
                self.config.get_from_extra_config(
                    "mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB
                )
            )
        self._mem_pool_size_bytes = int(mem_pool_size_gb * 1024**3)

        # The legacy connector eagerly allocates this pinned pool on every
        # worker. Base mode only needs it on a consumer and allocates it lazily
        # on the first CPU-staged transfer.
        self.pool: TensorMemoryPool | None = None
        if self.transfer_policy == TRANSFER_POLICY_LEGACY_EAGER:
            self.pool = TensorMemoryPool(max_block_size=self._mem_pool_size_bytes)

        # The sending type includes tree mutually exclusive options:
        # PUT, GET, PUT_ASYNC.
        self.send_type = self.config.get_from_extra_config("send_type", "PUT_ASYNC")
        if self.send_type == "GET":
            # tensor_id: torch.Tensor
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            # PUT or PUT_ASYNC
            # tensor_id: torch.Tensor
            self.send_queue: deque[SendQueueItem] = deque()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(
                    target=self.send_async, daemon=True
                )
                self._send_thread.start()

        # tensor_id: torch.Tensor/(addr, dtype, shape)
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.recv_request_failures: dict[str, str] = {}
        self.send_request_failures: dict[str, str] = {}
        self._unreported_send_failures: set[str] = set()
        self._pending_send_counts: dict[str, int] = {}
        self._send_in_flight = 0
        self._finished_req_ids_pending: set[str] = set()
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.comms: dict[str, Any] = {}  # remote_address: (ncclComm_t, rank)

        self.buffer_size = 0
        self.buffer_size_threshold = float(
            self.config.get_from_extra_config(
                "gpu_staging_size_bytes", self.config.kv_buffer_size
            )
        )

        self.nccl_num_channels = self.config.get_from_extra_config(
            "nccl_num_channels", "8"
        )

        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True
        )
        self._listener_thread.start()

        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()

        logger.info(
            "💯P2pNcclEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, nccl_num_channels:%s",
            self.rank,
            self.local_rank,
            self.http_address,
            self.zmq_address,
            self.proxy_address,
            self.send_type,
            self.buffer_size_threshold,
            self.nccl_num_channels,
        )

    @property
    def is_base_policy(self) -> bool:
        return self.transfer_policy == TRANSFER_POLICY_BASE

    @staticmethod
    def _request_id_from_tensor_id(tensor_id: str) -> str:
        return tensor_id.split("#", 1)[0]

    def _get_or_create_cpu_pool(self) -> TensorMemoryPool:
        if self.pool is not None:
            return self.pool
        if not self.config.is_kv_consumer:
            raise P2pTransferError(
                "CPU staging is only available on a P2P KV consumer"
            )
        with self._pool_lock:
            if self.pool is None:
                self.pool = TensorMemoryPool(
                    max_block_size=self._mem_pool_size_bytes
                )
        return self.pool

    def _record_receive_failure(self, tensor_id: str, reason: str) -> None:
        request_id = self._request_id_from_tensor_id(tensor_id)
        with self.recv_store_cv:
            self.recv_store[tensor_id] = TransferFailure(reason)
            self.recv_request_failures[request_id] = reason
            self.have_received_tensor_id(tensor_id)
            self.recv_store_cv.notify_all()

    def _record_send_failure(self, tensor_id: str, reason: str) -> None:
        request_id = self._request_id_from_tensor_id(tensor_id)
        with self.send_queue_cv:
            self.send_request_failures[request_id] = reason
            self._unreported_send_failures.add(request_id)

    def _free_cpu_staging(self, staged: CpuStagedTensor | tuple[Any, ...]) -> None:
        addr = staged.addr if isinstance(staged, CpuStagedTensor) else staged[0]
        if self.pool is not None:
            self.pool.free(addr)

    def create_connect(self, remote_address: str | None = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            if remote_address in self.comms:
                logger.info(
                    "👋comm exists, remote_address:%s, comms:%s",
                    remote_address,
                    self.comms,
                )
                return sock, self.comms[remote_address]

            unique_id = self.nccl.ncclGetUniqueId()
            data = {"cmd": "NEW", "unique_id": bytes(unique_id.internal)}
            sock.send(msgpack.dumps(data))

            with torch.cuda.device(self.device):
                rank = 0
                with set_p2p_nccl_context(self.nccl_num_channels):
                    comm: ncclComm_t = self.nccl.ncclCommInitRank(2, unique_id, rank)
                self.comms[remote_address] = (comm, rank)
                logger.info(
                    "🤝ncclCommInitRank Success, %s👉%s, MyRank:%s",
                    self.zmq_address,
                    remote_address,
                    rank,
                )

        return self.socks[remote_address], self.comms[remote_address]

    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True

        item = SendQueueItem(
            tensor_id=tensor_id, remote_address=remote_address, tensor=tensor
        )

        if self.send_type == "PUT":
            success = self.send_sync(item)
            if not success and self.is_base_policy:
                self._record_send_failure(tensor_id, "peer rejected P2P transfer")
            return success

        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                request_id = self._request_id_from_tensor_id(tensor_id)
                self._pending_send_counts[request_id] = (
                    self._pending_send_counts.get(request_id, 0) + 1
                )
                self.send_queue_cv.notify()
            return True

        # GET
        with self.send_store_cv:
            tensor_size = tensor.element_size() * tensor.numel()
            if tensor_size > self.buffer_size_threshold:
                logger.warning(
                    "❗[GET]tensor_id:%s, tensor_size:%d, is greater than"
                    "buffer size threshold :%d, skip send to %s, rank:%d",
                    tensor_id,
                    tensor_size,
                    self.buffer_size_threshold,
                    remote_address,
                    self.rank,
                )
                return False
            while self.buffer_size + tensor_size > self.buffer_size_threshold:
                assert len(self.send_store) > 0
                oldest_tensor_id = next(iter(self.send_store))
                oldest_tensor = self.send_store.pop(oldest_tensor_id)
                oldest_tensor_size = (
                    oldest_tensor.element_size() * oldest_tensor.numel()
                )
                self.buffer_size -= oldest_tensor_size
                logger.debug(
                    "⛔[GET]Send to %s, tensor_id:%s, tensor_size:%d,"
                    " buffer_size:%d, oldest_tensor_size:%d, rank:%d",
                    remote_address,
                    tensor_id,
                    tensor_size,
                    self.buffer_size,
                    oldest_tensor_size,
                    self.rank,
                )

            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
            logger.debug(
                "🔵[GET]Send to %s, tensor_id:%s, tensor_size:%d, "
                "shape:%s, rank:%d, buffer_size:%d(%.2f%%)",
                remote_address,
                tensor_id,
                tensor_size,
                tensor.shape,
                self.rank,
                self.buffer_size,
                self.buffer_size / self.buffer_size_threshold * 100,
            )
        return True

    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self.recv_store[tensor_id]

            if isinstance(tensor, TransferFailure):
                raise P2pTransferError(
                    f"P2P receive failed for {tensor_id}: {tensor.reason}"
                )

            if tensor is not None:
                if isinstance(tensor, CpuStagedTensor):
                    pool = self._get_or_create_cpu_pool()
                    try:
                        tensor = pool.load_tensor(
                            tensor.addr, tensor.dtype, tensor.shape, self.device
                        )
                    except (RuntimeError, ValueError) as exc:
                        with suppress(ValueError):
                            pool.free(tensor.addr)
                        reason = f"failed to reload CPU-staged KV: {exc}"
                        self._record_receive_failure(tensor_id, reason)
                        raise P2pTransferError(
                            f"P2P receive failed for {tensor_id}: {reason}"
                        ) from exc
                elif isinstance(tensor, tuple):
                    # Backward compatibility for legacy CPU-staged entries.
                    addr, dtype, shape = tensor
                    assert self.pool is not None
                    tensor = self.pool.load_tensor(addr, dtype, shape, self.device)
                else:
                    # Base mode retains request-scoped staging until terminal
                    # cleanup so that the same request id can reload after
                    # preemption. Legacy accounting is kept unchanged.
                    if not self.is_base_policy:
                        self.buffer_size -= tensor.element_size() * tensor.numel()
            else:
                duration = time.time() - start_time
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, rank:%d",
                    remote_address,
                    tensor_id,
                    duration * 1000,
                    self.rank,
                )
                if self.is_base_policy:
                    raise P2pTransferError(
                        f"P2P receive returned no KV for {tensor_id}"
                    )
            return tensor

        # GET
        if remote_address is None:
            return None

        if remote_address not in self.socks:
            self.create_connect(remote_address)

        sock = self.socks[remote_address]
        comm, rank = self.comms[remote_address]

        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(msgpack.dumps(data))

        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] != 0:
            logger.warning(
                "🔴[GET]Recv From %s, tensor_id: %s, ret: %d",
                remote_address,
                tensor_id,
                data["ret"],
            )
            return None

        with torch.cuda.stream(self.recv_stream):
            tensor = torch.empty(
                data["shape"], dtype=getattr(torch, data["dtype"]), device=self.device
            )

        self.recv(comm, tensor, rank ^ 1, self.recv_stream)

        return tensor

    def _handle_base_put(self, remote_address: bytes, data: dict[str, Any]) -> None:
        """Receive one base-policy tensor with bounded request staging.

        CPU capacity is reserved before the sender is acknowledged. A rejected
        transfer is recorded under the exact tensor/request id so both the
        producer and a concurrently waiting consumer observe a hard failure
        instead of silently decoding with a missing layer.
        """
        tensor_id = data["tensor_id"]
        dtype = getattr(torch, data["dtype"])
        shape = torch.Size(data["shape"])
        tensor_size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        spill_to_cpu = self.buffer_size + tensor_size > self.buffer_size_threshold
        reserved_cpu_addr: int | None = None

        if spill_to_cpu:
            try:
                pool = self._get_or_create_cpu_pool()
                reserved_cpu_addr = pool.allocate(tensor_size)
            except (RuntimeError, ValueError, P2pTransferError) as exc:
                reason = f"CPU staging capacity unavailable: {exc}"
                self.router_socket.send_multipart([remote_address, b"2"])
                self._record_receive_failure(tensor_id, reason)
                logger.error(
                    "Base P2P rejected %s from %s: %s",
                    tensor_id,
                    remote_address.decode(),
                    reason,
                )
                return

        try:
            with torch.cuda.stream(self.recv_stream):
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
        except torch.cuda.OutOfMemoryError as exc:
            if reserved_cpu_addr is not None and self.pool is not None:
                self.pool.free(reserved_cpu_addr)
            reason = f"GPU receive staging allocation failed: {exc}"
            self.router_socket.send_multipart([remote_address, b"1"])
            self._record_receive_failure(tensor_id, reason)
            logger.error(
                "Base P2P rejected %s from %s: %s",
                tensor_id,
                remote_address.decode(),
                reason,
            )
            return

        self.router_socket.send_multipart([remote_address, b"0"])
        try:
            comm, rank = self.comms[remote_address.decode()]
            self.recv(comm, tensor, rank ^ 1, self.recv_stream)
            if reserved_cpu_addr is not None:
                pool = self._get_or_create_cpu_pool()
                pool.store_tensor_at(tensor, reserved_cpu_addr)
                staged: torch.Tensor | CpuStagedTensor = CpuStagedTensor(
                    reserved_cpu_addr, tensor.dtype, tensor.shape
                )
                logger.info(
                    "Base P2P CPU-staged tensor_id=%s size=%d rank=%d",
                    tensor_id,
                    tensor_size,
                    self.rank,
                )
            else:
                staged = tensor
                self.buffer_size += tensor_size
        except Exception as exc:
            if reserved_cpu_addr is not None and self.pool is not None:
                with suppress(ValueError):
                    self.pool.free(reserved_cpu_addr)
            reason = f"failed after accepting P2P transfer: {exc}"
            self._record_receive_failure(tensor_id, reason)
            logger.exception("Base P2P receive failed for %s", tensor_id)
            return

        with self.recv_store_cv:
            self.recv_store[tensor_id] = staged
            self.have_received_tensor_id(tensor_id)
            self.recv_store_cv.notify_all()

    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue

            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            if data["cmd"] == "NEW":
                unique_id = self.nccl.unique_id_from_bytes(bytes(data["unique_id"]))
                with torch.cuda.device(self.device):
                    rank = 1
                    with set_p2p_nccl_context(self.nccl_num_channels):
                        comm: ncclComm_t = self.nccl.ncclCommInitRank(
                            2, unique_id, rank
                        )
                    self.comms[remote_address.decode()] = (comm, rank)
                    logger.info(
                        "🤝ncclCommInitRank Success, %s👈%s, MyRank:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        rank,
                    )
            elif data["cmd"] == "PUT":
                if self.is_base_policy:
                    self._handle_base_put(remote_address, data)
                    continue
                tensor_id = data["tensor_id"]
                try:
                    with torch.cuda.stream(self.recv_stream):
                        tensor = torch.empty(
                            data["shape"],
                            dtype=getattr(torch, data["dtype"]),
                            device=self.device,
                        )
                    self.router_socket.send_multipart([remote_address, b"0"])
                    comm, rank = self.comms[remote_address.decode()]
                    self.recv(comm, tensor, rank ^ 1, self.recv_stream)
                    tensor_size = tensor.element_size() * tensor.numel()
                    if self.buffer_size + tensor_size > self.buffer_size_threshold:
                        # Store Tensor in memory pool
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Threshold, "
                            "%s👈%s, data:%s, addr:%d",
                            self.zmq_address,
                            remote_address.decode(),
                            data,
                            addr,
                        )
                    else:
                        self.buffer_size += tensor_size

                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, data:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                    )

                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()

            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                with self.send_store_cv:
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is not None:
                        data = {
                            "ret": 0,
                            "shape": tensor.shape,
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                        # LRU
                        self.send_store[tensor_id] = tensor
                        self.have_sent_tensor_id(tensor_id)
                    else:
                        data = {"ret": 1}

                self.router_socket.send_multipart([remote_address, msgpack.dumps(data)])

                if data["ret"] == 0:
                    comm, rank = self.comms[remote_address.decode()]
                    self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)
            else:
                logger.warning(
                    "🚧Unexpected, Received message from %s, data:%s",
                    remote_address,
                    data,
                )

    def have_sent_tensor_id(self, tensor_id: str):
        request_id = self._request_id_from_tensor_id(tensor_id)
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)

    def have_received_tensor_id(self, tensor_id: str):
        request_id = self._request_id_from_tensor_id(tensor_id)
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)

    def _run_send_item(self, item: SendQueueItem) -> bool:
        """Run one queued send and publish request-scoped completion state."""
        request_id = self._request_id_from_tensor_id(item.tensor_id)
        success = False
        reason = "peer rejected P2P transfer"
        try:
            success = self.send_sync(item)
        except Exception as exc:
            reason = f"P2P send raised {type(exc).__name__}: {exc}"
            logger.exception("P2P async send failed for %s", item.tensor_id)
        finally:
            with self.send_queue_cv:
                pending = self._pending_send_counts.get(request_id, 1) - 1
                if pending > 0:
                    self._pending_send_counts[request_id] = pending
                else:
                    self._pending_send_counts.pop(request_id, None)
                if not success and self.is_base_policy:
                    self.send_request_failures[request_id] = reason
                    self._unreported_send_failures.add(request_id)
                self._send_in_flight -= 1
                self.send_queue_cv.notify_all()
        return success

    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                self._send_in_flight += 1
            self._run_send_item(item)

    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue or (
                    self.is_base_policy and self._send_in_flight
                ):
                    self.send_queue_cv.wait()
                failed = sorted(self._unreported_send_failures)
                self._unreported_send_failures.difference_update(failed)
            duration = time.time() - start_time
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d",
                duration * 1000,
                self.rank,
            )
            if failed and self.is_base_policy:
                details = "; ".join(
                    f"{request_id}: {self.send_request_failures[request_id]}"
                    for request_id in failed
                )
                raise P2pTransferError(f"Base P2P send failed: {details}")

    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)

        tensor = item.tensor

        sock = self.socks[item.remote_address]
        comm, rank = self.comms[item.remote_address]
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        sock.send(msgpack.dumps(data))

        response = sock.recv()
        if response != b"0":
            logger.error(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "MyRank:%s, data:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address,
                item.remote_address,
                rank,
                data,
                tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode(),
            )
            return False

        self.send(comm, tensor.to(self.device), rank ^ 1, self.send_stream)

        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)

        return True

    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        # Clear request-scoped receive staging upon request completion. Base
        # mode deliberately has no eviction: terminal cleanup is the only
        # normal way active CPU/GPU staging is reclaimed.
        for request_id in finished_req_ids:
            for layer_name in no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id, None)
                    if isinstance(tensor, (CpuStagedTensor, tuple)):
                        self._free_cpu_staging(tensor)
                    elif isinstance(tensor, torch.Tensor) and self.is_base_policy:
                        self.buffer_size = max(
                            0,
                            self.buffer_size
                            - tensor.element_size() * tensor.numel(),
                        )
            self.recv_request_id_to_tensor_ids.pop(request_id, None)
            self.recv_request_failures.pop(request_id, None)

        finished_sending: set[str] = set()
        if self.is_base_policy:
            with self.send_queue_cv:
                self._finished_req_ids_pending.update(
                    request_id
                    for request_id in finished_req_ids
                    if request_id in self.send_request_id_to_tensor_ids
                    or request_id in self.send_request_failures
                    or request_id in self._pending_send_counts
                )
                for request_id in tuple(self._finished_req_ids_pending):
                    was_sent = request_id in self.send_request_id_to_tensor_ids
                    failed = request_id in self.send_request_failures
                    if (
                        (was_sent or failed)
                        and self._pending_send_counts.get(request_id, 0) == 0
                    ):
                        # A failed transfer is surfaced by wait_for_sent(); once
                        # terminal, report completion too so scheduler-owned
                        # source blocks cannot leak.
                        finished_sending.add(request_id)
                        self._finished_req_ids_pending.remove(request_id)
                        self.send_request_id_to_tensor_ids.pop(request_id, None)
                        self.send_request_failures.pop(request_id, None)
                        self._unreported_send_failures.discard(request_id)

        # Loading is synchronous in this connector, so requests must not enter
        # WAITING_FOR_REMOTE_KV and no finished_recving notification is needed.
        finished_recving: set[str] = set()

        return finished_sending or None, finished_recving or None

    def get_request_transfer_status(self, request_id: str) -> str:
        """Return a compact status string for diagnostics and tests."""
        if request_id in self.send_request_failures:
            return "send_failed"
        if request_id in self.recv_request_failures:
            return "receive_failed"
        if self._pending_send_counts.get(request_id, 0):
            return "sending"
        if request_id in self.send_request_id_to_tensor_ids:
            return "sent"
        if request_id in self.recv_request_id_to_tensor_ids:
            return "received"
        return "unknown"

    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)

    def send(self, comm, tensor: torch.Tensor, dst: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                dst,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
        stream.synchronize()

    def recv(self, comm, tensor: torch.Tensor, src: int, stream=None):
        assert tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {tensor.device}"
        )
        if stream is None:
            stream = current_stream()

        with torch.cuda.stream(stream):
            self.nccl.ncclRecv(
                buffer_type(tensor.data_ptr()),
                tensor.numel(),
                ncclDataTypeEnum.from_torch(tensor.dtype),
                src,
                comm,
                cudaStream_t(stream.cuda_stream),
            )
        stream.synchronize()

    def close(self) -> None:
        self._listener_thread.join()
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join()
        if self._ping_thread is not None:
            self._ping_thread.join()
