# SPDX-License-Identifier: Apache-2.0
"""Retain cancelled lookup control RPCs until each endpoint acknowledges END."""
from collections.abc import Callable
from typing import Any
import logging
import threading

logger = logging.getLogger(__name__)


class _Cleanup:
    def __init__(
        self,
        request_id: str,
        clients: dict[str, Any],
        on_quiescent: Callable[[Any], None],
    ) -> None:
        self.request_id = request_id
        self.clients = dict(clients)
        self.dependencies: dict[str, list[Any]] = {url: [] for url in clients}
        self.ending = False
        self.started: set[str] = set()
        self.failed: set[str] = set()
        self.end_futures: dict[str, Any] = {}
        self.completed: set[str] = set()
        self._lock = threading.Lock()
        self._on_quiescent = on_quiescent

    def retain(self, dependencies: dict[str, list[Any]]) -> None:
        registrations = []
        with self._lock:
            for url, futures in dependencies.items():
                for future in futures:
                    if any(item is future for item in self.dependencies[url]):
                        continue
                    if url in self.started:
                        raise RuntimeError("Cannot add lookup RPC after END submission")
                    self.dependencies[url].append(future)
                    registrations.append((url, future))
        for url, future in registrations:
            future.add_done_callback(lambda done, url=url: self._advance(url))

    def end(self) -> None:
        with self._lock:
            self.ending = True
        for url in self.clients:
            self._advance(url)

    def _advance(self, url: str) -> None:
        with self._lock:
            if not self.ending or url in self.started or url in self.failed:
                return
            if not all(future.query() for future in self.dependencies[url]):
                return
            try:
                for dependency in self.dependencies[url]:
                    dependency.result(timeout=0)
            except Exception:
                # A failed transport response (for example a cancelled gRPC)
                # does not prove the remote handler stopped. Do not let END
                # acknowledge before that unknown handler and permit ID reuse.
                self.failed.add(url)
                logger.exception("Lookup outcome unknown for %s at %s; retain generation",
                                 self.request_id, url)
                return
            self.started.add(url)
        try:
            future = self.clients[url].end_session(self.request_id)
        except Exception:
            # Unknown submission outcome keeps this generation quarantined.
            logger.exception(
                "END_SESSION submission failed for %s at %s", self.request_id, url
            )
            return
        with self._lock:
            self.end_futures[url] = future
        future.add_done_callback(lambda done: self._end_complete(url, done))

    def _end_complete(self, url: str, future: Any) -> None:
        try:
            future.result(timeout=0)
        except Exception:
            logger.exception(
                "END_SESSION acknowledgement failed for %s at %s", self.request_id, url
            )
            return
        with self._lock:
            self.completed.add(url)
            quiescent = len(self.completed) == len(self.clients)
        if quiescent:
            self._on_quiescent(self)


class LookupControlRegistry:
    """Own generations independently of scheduler admission bookkeeping."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, _Cleanup] = {}

    def retain(
        self,
        request_id: str,
        clients: dict[str, Any],
        dependencies: dict[str, list[Any]],
    ) -> None:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                state = _Cleanup(request_id, clients, self._retire)
                self._states[request_id] = state
        state.retain(dependencies)

    def end(self, request_id: str) -> None:
        with self._lock:
            state = self._states[request_id]
        state.end()

    def check_reuse(self, request_id: str) -> None:
        with self._lock:
            if request_id in self._states:
                raise RuntimeError(
                    f"Lookup request {request_id!r} has an unquiescent generation"
                )

    def _retire(self, state: _Cleanup) -> None:
        with self._lock:
            if self._states.get(state.request_id) is state:
                del self._states[state.request_id]
