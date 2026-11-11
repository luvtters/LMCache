# SPDX-License-Identifier: Apache-2.0
"""Request client factory selected by endpoint URL scheme."""

# Standard
from typing import Any

# First Party
from lmcache.v1.multiprocess.transport.base import RequestClient

_ZMQ_SCHEMES = frozenset({"inproc", "ipc", "tcp"})
_GRPC_SCHEMES = frozenset({"grpc", "grpc+unix"})


def _normalize_server_url(server_url: str) -> tuple[str, str]:
    url = server_url.strip()
    if not url:
        raise ValueError("Request client server URL must not be empty")
    if "://" not in url:
        return "tcp", f"tcp://{url}"

    scheme, target = url.split("://", 1)
    scheme = scheme.lower()
    if not scheme or not target:
        raise ValueError(f"Invalid request client server URL: {server_url!r}")
    return scheme, f"{scheme}://{target}"


class RequestClientFactory:
    """Create a transport-specific request client from a server URL."""

    @staticmethod
    def create(
        server_url: str,
        *,
        context: Any | None = None,
        outbound_batch_limit: int = 0,
    ) -> RequestClient:
        """Create a request client selected by ``server_url`` scheme.

        Bare ``host:port`` endpoints retain the existing behavior and default
        to ``tcp://``. ZMQ handles ``tcp://``, ``ipc://``, and ``inproc://``;
        gRPC handles ``grpc://`` and ``grpc+unix://``.

        Args:
            server_url: Multiprocess server endpoint, optionally without a
                scheme for the legacy ZMQ TCP default.
            context: Optional transport context. ZMQ accepts a ``zmq.Context``;
                transports that do not need a context may ignore it.
            outbound_batch_limit: Maximum queued frames sent per ZMQ polling
                turn; zero drains the queue. Nonzero is unsupported by gRPC.

        Returns:
            A method-oriented request client for the selected transport.

        Raises:
            ValueError: If the URL is empty, malformed, unsupported, or the
                budget is negative or nonzero for gRPC.
        """
        if outbound_batch_limit < 0:
            raise ValueError("outbound_batch_limit must be nonnegative")
        scheme, normalized_url = _normalize_server_url(server_url)
        if scheme in _ZMQ_SCHEMES:
            # First Party
            from lmcache.v1.multiprocess.transport import zmq_impl

            return zmq_impl.create_request_client(
                normalized_url,
                context=context,
                outbound_batch_limit=outbound_batch_limit,
            )
        if scheme in _GRPC_SCHEMES:
            if outbound_batch_limit:
                raise ValueError("outbound_batch_limit is only supported by ZMQ")
            # First Party
            from lmcache.v1.multiprocess.transport import grpc_impl

            return grpc_impl.create_request_client(
                normalized_url,
                context=context,
            )

        supported = sorted(_ZMQ_SCHEMES | _GRPC_SCHEMES)
        raise ValueError(
            f"Unsupported request client URL scheme {scheme!r}; "
            f"supported schemes: {supported}"
        )
