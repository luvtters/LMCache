# Native MP registration for complete pipeline stages

`LMCacheMPConnector` selects one complete pipeline stage per server when
`PP > 1` and the ordered endpoint count equals PP. This branch requires
DP=DCP=PCP=1, world size TP*PP, and `lmcache_driven` transfer. The single-server
and PP=1 strategies retain their existing geometry and registration RPC.

Workers resolve the logical layer partition through vLLM's own PP helper.
After KV group edits, registration carries the actual ordered tensor names and
engine groups. Auxiliary and MTP tensors are preserved; tensor count is not
assumed equal to the main model layer count.

`register_pp_kv_cache` carries the seven original registration payload kinds
plus `PPRegistrationInfo`. The server builds the real device context, describes
per-chunk shapes/dtypes, formats, layer/group order and attention windows, and
returns `PPRegistrationAck`. The accepted model namespace contains a versioned
digest of stage, full partition, TP geometry and actual layout. Addresses,
worker/control IDs, pointers and KV pool capacity do not enter that digest.

Pure MLA stores one piece per stage, with local TP rank zero writing and all TP
ranks reading. Sharded KV stores TP pieces, with one writer/reader per piece.
Registration does not wait for other TP ranks. After vLLM initializes worker
KV caches, the scheduler calls `get_pp_stage_status` on all endpoints using a
shared timeout budget, checks the complete TP set and distinct server process
identities, then uses each server's accepted namespace for lookup, lock release
and allocation reporting. This mapping remains fixed for its lifetime.

The server reuses `ContextEntry` and existing heartbeat/unregister/reap cleanup.
An identical instance/token/profile retry refreshes liveness; a conflicting
instance or occupied TP slot fails. A registration transaction publishes the
context and layout reference together, closing an unsuccessful import. One
active engine per stage registration space is supported initially. Worker
adapters retain their registration token across transfer-context recovery and
reject a changed profile.

The platform registration factory delegates to the ordinary device factory by
default. The Ascend plugin installs a lazy NPU hook for PP metadata, selecting
its pinned block-index context without a user-visible PP activation variable.
The original registration factory and its explicit legacy flag remain available
to the historical connector.

No vLLM or Mooncake changes are required. MP's PP peer handshake is a local-cache
no-op and neither consumes nor changes another connector's payload. New PP
clients and servers must both implement the registration schema; an unknown RPC,
timeout, incomplete stage or incompatible profile is an initialization error,
not a fallback to an older cache namespace. Existing RPC payload counts and data
transfer operations remain unchanged. Regenerate protobuf bindings after changing
the schema in a fresh interpreter before importing device frameworks.
