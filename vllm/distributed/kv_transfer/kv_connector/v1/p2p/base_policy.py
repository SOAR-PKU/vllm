# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

TRANSFER_POLICY_LEGACY_EAGER = "legacy_eager"
TRANSFER_POLICY_BASE = "base"

KV_TRANSFER_INTENT_LOCAL_CACHE_ONLY = "local_cache_only"
KV_TRANSFER_INTENT_QUERY_FULL = "query_full"

VALID_TRANSFER_POLICIES = {
    TRANSFER_POLICY_LEGACY_EAGER,
    TRANSFER_POLICY_BASE,
}
VALID_TRANSFER_INTENTS = {
    KV_TRANSFER_INTENT_LOCAL_CACHE_ONLY,
    KV_TRANSFER_INTENT_QUERY_FULL,
}


def normalize_transfer_intent(
    transfer_policy: str, params: dict[str, Any] | None
) -> str:
    """Resolve request transfer intent without guessing query semantics."""
    if transfer_policy != TRANSFER_POLICY_BASE:
        return KV_TRANSFER_INTENT_QUERY_FULL
    intent = (params or {}).get("kv_transfer_intent")
    if intent is None:
        return KV_TRANSFER_INTENT_LOCAL_CACHE_ONLY
    if intent not in VALID_TRANSFER_INTENTS:
        raise ValueError(f"Unsupported P2P KV transfer intent: {intent}")
    return intent


def prompt_block_ids(
    token_ids: list[int], block_ids: list[int], block_size: int
) -> list[int]:
    """Return every and only block containing the request prompt."""
    if not token_ids:
        return block_ids
    num_prompt_blocks = (len(token_ids) + block_size - 1) // block_size
    return block_ids[:num_prompt_blocks]
