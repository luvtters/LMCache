# SPDX-License-Identifier: Apache-2.0
"""Opt-in host observations for the private PP candidate; no device polling."""

# Standard
import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


class PPWaitObserver:
    """Aggregate process-local host boundaries, emitting at most every ten seconds.

    No thread, future query, or synchronization is introduced. Device completion
    means first seen by the original query; native block release is unobserved.
    """

    def __init__(self, scope: str, endpoint: str) -> None:
        self.scope = scope
        self.endpoint = endpoint
        self._lock = threading.RLock()
        self._since_ns = time.monotonic_ns()
        self._metrics: dict[str, list[int]] = {}
        self._stores: dict[str, tuple[int, int | None]] = {}
        self._delayed: dict[str, tuple[int, int]] = {}
        self._delayed_blocks = 0

    def record(self, metric: str, value: int = 1) -> None:
        """Add one integer sample; log count/sum/max after ten seconds of activity."""
        with self._lock:
            values = self._metrics.setdefault(metric, [0, 0, value])
            values[0] += 1
            values[1] += value
            values[2] = max(values[2], value)
            now = time.monotonic_ns()
            if now - self._since_ns >= 10_000_000_000:
                logger.info("pp_%s %s", self.scope, json.dumps({
                    "endpoint": self.endpoint,
                    "interval_ns": now - self._since_ns,
                    "metrics": self._metrics,
                    "native_block_release": "NOT_OBSERVED",
                }, separators=(",", ":")))
                self._metrics = {}
                self._since_ns = now

    def snapshot(self) -> dict[str, tuple[int, int, int]]:
        """Return count/sum/max copies without draining live samples."""
        with self._lock:
            return {key: (v[0], v[1], v[2]) for key, v in self._metrics.items()}

    def store_submitted(self, request_id: str) -> int:
        """Start the latest accepted STORE interval and return its host timestamp."""
        with self._lock:
            submitted = time.monotonic_ns()
            self._stores[request_id] = (submitted, None)
            self.record("store_submitted")
            return submitted

    def store_rpc_ready(self, submitted_ns: int) -> None:
        """Record raw RPC completion; this is not device completion."""
        self.record("store_submit_to_rpc_ns", time.monotonic_ns() - submitted_ns)

    def store_device_seen(self, request_id: str) -> None:
        """Record readiness first seen by an existing device-future query."""
        with self._lock:
            state = self._stores.get(request_id)
            if state is not None and state[1] is None:
                now = time.monotonic_ns()
                self._stores[request_id] = (state[0], now)
                self.record("store_submit_to_device_seen_ns", now - state[0])

    def store_reported(self, request_id: str) -> None:
        """Retire host timing when get_finished actually returns the request."""
        with self._lock:
            state = self._stores.pop(request_id, None)
            if state is not None:
                now = time.monotonic_ns()
                self.record("store_submit_to_get_finished_ns", now - state[0])
                if state[1] is not None:
                    self.record("device_seen_to_get_finished_ns", now - state[1])

    def delay_free(self, request_id: str, block_count: int) -> None:
        """Track scheduler-authorized delayed blocks until its completion notice."""
        with self._lock:
            previous = self._delayed.get(request_id)
            if previous is not None:
                self._delayed_blocks -= previous[1]
            self._delayed[request_id] = (time.monotonic_ns(), block_count)
            self._delayed_blocks += block_count
            self.record("delay_free_blocks", block_count)
            self.record("delay_free_outstanding_blocks", self._delayed_blocks)

    def delay_notice(self, request_id: str) -> None:
        """Retire the scheduler interval at aggregate worker notification, not release."""
        with self._lock:
            state = self._delayed.pop(request_id, None)
            if state is not None:
                self._delayed_blocks -= state[1]
                self.record(
                    "delay_free_to_worker_notice_ns", time.monotonic_ns() - state[0]
                )


def create_pp_wait_observer(
    pp_enabled: bool, scope: str, endpoint: str
) -> PPWaitObserver | None:
    """Create observation only for explicitly selected PP processes.

    ``LMCACHE_PP_WAIT_OBSERVE=1`` is read at process construction. All other
    values disable it; no hot reload or implicit non-PP instrumentation occurs.
    """
    if pp_enabled and os.environ.get("LMCACHE_PP_WAIT_OBSERVE", "0") == "1":
        return PPWaitObserver(scope, endpoint)
    return None
