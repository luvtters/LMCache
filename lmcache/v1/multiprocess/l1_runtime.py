# SPDX-License-Identifier: Apache-2.0
"""Load an optional L1 backing/transport runtime before storage construction.

Runtime scripts launched after server startup cannot replace an already allocated
L1 arena. This entry point is deliberately early; the default path imports no
external transport and creates the usual native L1.
"""
import importlib


def create_l1_runtime(extra_config, storage_config):
    spec = extra_config.get("l1_runtime")
    if spec is None:
        return None
    if not isinstance(spec, dict) or not isinstance(spec.get("factory"), str):
        raise ValueError("l1_runtime requires a module:factory and config mapping")
    module, separator, name = spec["factory"].partition(":")
    if not separator or not module or not name:
        raise ValueError("l1_runtime factory must use module:factory syntax")
    config = spec.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("l1_runtime config must be a mapping")
    factory = getattr(importlib.import_module(module), name)
    # Failures propagate: explicit external L1 must never become an ordinary
    # CPU pool/inline transport silently.
    runtime = factory(storage_config, config)
    try:
        for method in ("create_memory_manager", "create_transfer_strategy", "close"):
            if not callable(getattr(runtime, method, None)):
                raise TypeError(f"L1 runtime is missing {method}")
    except BaseException:
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
        raise
    return runtime
