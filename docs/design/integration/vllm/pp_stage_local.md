# External stage-local PP geometry

This candidate is based on LMCache `3e11e10812b448142c6cdb177a3058652bd826fa`.
It adds `LMCacheMPConnector.build_parallel_strategy(config, n_servers)` as an
overridable hook. The default implementation retains the existing multi-server
PP rejection and delegates supported geometry to the existing builder.

The paired LMCache-Ascend branch implements the customer-specific strategy:
TP16/PP2, DP1, one complete stage and one independent L1 server per host,
DCP/PCP=1. The scheduler and both worker stages receive the same strategy type.
Global ranks still select the local endpoint; local KV IDs select pieces inside
that server. MLA uses one piece and sixteen readers per server, with writers at
global ranks 0 and 16. Sharded KV retains sixteen pieces and one reader each.

The existing scheduler sends LOOKUP to both servers, waits for both results and
uses their minimum prefix. Its existing excess-tail lock release and request
cleanup use the strategy's corrected reader count. No server protocol, vLLM,
model scheduling or native copy kernel is changed by this candidate.

The seven-line initial STORE block fallback remains in LMCache. The local-cache
PP handshake override moves to the Ascend connector. Shared index staging is
restored to upstream behavior; the plugin selects pinned staging only in its
NPU context. This keeps the historical baseline and the customer candidate
separate rather than extending the shared pinning policy to other devices.

The policy assumes each endpoint belongs to exactly one PP stage, in front/rear
order, and independent local L1 namespaces. Shared L2, arbitrary rank-to-host
layouts, extra context-parallel token shards and DP across cache servers are
outside this candidate. Existing DSA CP is not the vLLM DCP/PCP setting.

Run `python -S -B -X utf8 -m unittest discover -s tests/ks -p 'test_*cpu.py' -v`
for the local core contracts. The plugin has companion CPU tests using actual
constructor and lookup methods with framework/network/device boundaries
replaced by fixtures. Full module imports, CANN compilation, native IPC,
quantized KV correctness, two-host inference and performance remain unverified.
