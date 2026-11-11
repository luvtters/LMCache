# SPDX-License-Identifier: Apache-2.0
"""ZMQ transport implementation for multiprocess requests."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.transport.base import RequestClient
    from lmcache.v1.multiprocess.transport.zmq_impl.client import (
        ZmqMultiprocessClient,
    )


def __getattr__(name: str) -> Any:
    """Resolve public ZMQ client types without eager implementation imports."""
    if name == "ZmqMultiprocessClient":
        # First Party
        from lmcache.v1.multiprocess.transport.zmq_impl.client import (
            ZmqMultiprocessClient,
        )

        return ZmqMultiprocessClient
    raise AttributeError(name)


def create_request_client(
    server_url: str,
    *,
    context: Any | None = None,
    outbound_batch_limit: int = 0,
) -> RequestClient:
    """Create a method-oriented request client backed by ZMQ.

    Args:
        server_url: ZMQ endpoint URL.
        context: Optional existing ``zmq.Context`` shared by the caller.
        outbound_batch_limit: Maximum queued frames per polling turn; zero
            preserves unlimited draining.

    Returns:
        A ZMQ-backed request client.
    """
    # Third Party
    import zmq

    # First Party
    from lmcache.v1.multiprocess.transport.zmq_impl.client import (
        ZmqMultiprocessClient,
    )
    from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueClient

    if context is None:
        context = zmq.Context.instance()
    return ZmqMultiprocessClient(  # type: ignore[abstract]
        MessageQueueClient(
            server_url, context, outbound_batch_limit=outbound_batch_limit
        )
    )


__all__ = ["ZmqMultiprocessClient", "create_request_client"]
