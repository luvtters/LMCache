# Private PP query and control progress candidate

Base: B `02ffb69`. This candidate does not include C `1362d90`.

All new behavior is opt-in in the LMCache child connector's extra config:

```json
{
  "lmcache.mp.pp_query_fastpath": true,
  "lmcache.mp.pp_control_outbound_budget": 32
}
```

The query option defaults to false and requires stage-local PP>1, a distinct
server for every stage, `transfer_borrow`, and nonblocking STATUS. Deploy only
with the existing L1-only configuration. Each endpoint's STATUS follows its own
successful LOOKUP ACK. A final integer zero fixes the aggregate result at zero;
pending controls remain owned until normal request cleanup. No END is sent on
early miss, and normal local computation can still STORE. Positive results wait
for every stage and use the common minimum. Original deadlines remain unchanged.

The send budget defaults to zero (original unlimited drain). A positive budget
requires a stage-local PP adapter and ZMQ. It counts every dequeued request or
control frame, including encoding failures, preserves FIFO, and re-notifies the
shared loop after exhausting the budget. Unregistration explicitly drains all
remaining frames. The budget bounds message count, not blocking socket time.
The value 32 is a CPU test candidate, not a measured optimum.

`LMCACHE_PP_WAIT_OBSERVE=1` enables process-local observations at construction.
Set it only on the private PP scheduler/workers and the associated private CPU
control server. Scheduler and worker observation is also gated by PP identity;
the transport server has no PP topology until runtime registration, so its
explicit environment selection defines its scope. Defaults are off.

The observer records count/sum/max and logs every ten seconds of activity.
It adds no thread, device query, or synchronization. Observations include:

- Query polls, pending ACK/STATUS, and early misses.
- CPU control handler queue and handler duration, separately by operation.
- Flush and the following device synchronization, with nonoverlapping intervals.
- Retained STORE handles at entry and actual pending count from flush's original
  selection; retained handles may include already completed ones.
- Raw RPC completion, first device readiness seen by the original query, and
  `get_finished` notification. RPC completion does not prove device completion.
- Scheduler-authorized delayed block counts and time until aggregated worker
  notice, plus scheduled tokens/requests and preemptions. Block counts use the
  `request_finished` block list (first KV group for HMA).

Actual native block release is `NOT_OBSERVED`. Device readiness is a host-observed
upper bound, not a physical DMA timestamp. No cross-process time subtraction is
used. STORE frequency, flush timeout, synchronization, and free conditions are
unchanged. Per-request STORE spans are collected only for eager
`transfer_borrow`; legacy and lazy offload do not create span state. No STORE
policy optimization is included.

CPU wire tests use actual ZMQ queues/codecs/futures and controlled CPU handlers;
they substitute the device cache-key payload and exclude device/IPC encoders.
They do not validate device transfers, actual storage behavior, model quality,
cross-host PP execution, or performance. Restore B or disable both options and
the observation environment variable to compare the original behavior.
