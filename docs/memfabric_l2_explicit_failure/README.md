# Explicit L2 prefetch failures (experimental)

Code: `6723d2d` plus the completed-future race fix `fa2e6ff`.
Use with the Ascend adapter candidate `137038a`.

A terminal adapter error is retained in `PrefetchResult.error`. The controller
releases its recorded L1/L2 protection, publishes the result, wakes waiters, and
retires the request. Lookup consumes the failed job before raising. ZMQ returns
an explicit error response and removes the client's pending entry; the scheduler
raises `RemoteRequestError` instead of re-querying or returning a cache miss.
An already completed error is also observed before the unhealthy fast path,
including the interval before its future callback executes.

## Selected validation, 2026-10-30

- Seven focused CPU checks passed, including real native bitmaps and ZMQ replies,
  lock release, waiter notification, job/pending removal, ordinary `None`/0/hit
  replies, and delayed-callback plus unhealthy-scheduler error propagation.
- Paired real-NPU Qwen3-8B TP1 and TP4 normal gates passed: 1,373 prompt tokens,
  1,280 external, 93 computed, zero local-prefix tokens; 180 MiB moved in each
  direction. Process exits were retained by the external supervisor.
- After a successful gate, a controlled idle pool-member departure made the next
  decoder request return HTTP 500. The decoder logged `RemoteRequestError` with
  `BM peer rank 1 left`; store/load counters stayed at 188,743,680 bytes and pins
  stayed zero. The engine failed closed with `EngineDeadError`. This is explicit
  bounded failure visibility, not request-level recovery or automatic recompute.

Client and server must both use this ZMQ candidate: the error-only wire extension
is new; successful reply frames and normal `int | None` results are unchanged.
gRPC exception mapping and a total request deadline for ordinary pending replies
are outside this validation. Native DMA hangs and performance are not validated
by these checks. Internal raw logs and deployment details remain private.
