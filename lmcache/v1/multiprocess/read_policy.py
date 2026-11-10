# SPDX-License-Identifier: Apache-2.0
"""Private opt-in read lifetime policy, shared by server and vLLM adapters."""
import os

LEGACY = "legacy"
TRANSFER_BORROW = "transfer_borrow"


def validate_read_policy(policy: str) -> str:
    if policy not in (LEGACY, TRANSFER_BORROW):
        raise ValueError(f"Unsupported LMCache MP read policy: {policy!r}")
    return policy


def get_read_policy() -> str:
    policy = validate_read_policy(os.environ.get("KS_LMCACHE_MP_READ_POLICY", LEGACY))
    if policy == TRANSFER_BORROW and os.environ.get("KS_LMCACHE_L1_RECHECK_ON_STATUS") == "1":
        raise ValueError("transfer_borrow cannot be combined with status_recheck")
    return policy


def validate_connector_policy(policy: str, *, stage_local: bool, failure_policy: str) -> None:
    validate_read_policy(policy)
    if policy == TRANSFER_BORROW:
        if not stage_local:
            raise ValueError("transfer_borrow requires native stage-local PP registration")
        if failure_policy != "recompute":
            raise ValueError("transfer_borrow requires kv_load_failure_policy=recompute")
