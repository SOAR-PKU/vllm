# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)


def test_connectors_default_to_no_streaming_kv_release() -> None:
    assert not KVConnectorBase_V1.supports_streaming_request_kv_release(object())


def _connector(adapter: object, *, use_native: bool = False) -> LMCacheConnectorV1:
    connector = object.__new__(LMCacheConnectorV1)
    connector._kv_transfer_config = SimpleNamespace(
        get_from_extra_config=lambda key, default: use_native,
    )
    connector._lmcache_engine = adapter
    return connector


@pytest.mark.parametrize("supported", [False, True])
def test_lmcache_delegates_streaming_kv_release_capability(supported: bool) -> None:
    capability = Mock(return_value=supported)
    connector = _connector(
        SimpleNamespace(supports_streaming_request_kv_release=capability)
    )

    assert connector.supports_streaming_request_kv_release() is supported
    capability.assert_called_once_with()


@pytest.mark.parametrize(
    "adapter",
    [
        object(),
        SimpleNamespace(
            supports_streaming_request_kv_release=True,
        ),
    ],
)
def test_lmcache_old_adapter_fails_closed(adapter: object) -> None:
    assert not _connector(adapter).supports_streaming_request_kv_release()


def test_lmcache_native_adapter_is_not_opted_in() -> None:
    capability = Mock(return_value=True)
    connector = _connector(
        SimpleNamespace(supports_streaming_request_kv_release=capability),
        use_native=True,
    )

    assert not connector.supports_streaming_request_kv_release()
    capability.assert_not_called()
