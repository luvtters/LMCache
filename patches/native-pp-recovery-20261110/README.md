# Native PP completion lifecycle patch

Backport of the production behavior in LMCache PR 5938, frozen at
`d74891244abbd76b47ef42d37707e1adc9bc202e`, onto native PP base
`89d93a2ea62030f6890b5bbe8f902c5b92945c8d`.
Reference: https://github.com/LMCache/LMCache/pull/5938
The PR was open and unmerged at preparation; the user approved this backport.

Only two production files change:

- `lmcache/integration/vllm/lmcache_mp_connector.py`
- `lmcache/integration/vllm/lmcache_mp_metadata.py`

The scheduler delays freeing only when a STORE was scheduled, hands that
decision to the workers, and filters send completions by those decisions.
Normal writes and completion from ranks without a local STORE remain valid.
No C++ rebuild, dependency change or vLLM source patch is involved.

In each customer container: `bash apply.sh`; undo: `bash rollback.sh`.
Both wrappers source `/opt/ks-luv-pp/env.sh`. The manager finds the installed
LMCache path without importing LMCache or device frameworks, checks the entire
patch before modifying files, and reverses only these hunks during rollback.
Repeated apply/rollback skips changes; unknown or partial states stop without
changing files. It does not start or stop services. Both PP model stages must
restart after a source change; the cache server need not restart for this patch.

CPU validation uses actual source methods with substitutes for transfer futures,
STORE metadata construction and telemetry. It covers the reproduced failure,
normal stores, consumers, preemption, lazy-offload routing, non-writer/unhealthy
ranks, and 32 independent worker control instances. It does not prove NPU data
correctness, actual PP communication, stress stability or a performance gain.

Local result: the original source failed 5 of the 11 completion contract checks;
the patched source passed all 11. All 3 patch-manager checks passed, covering
repeat apply/rollback, unrelated edits, mismatched source and partial states.
The collector condensed 1,000 repeated fixture records to 17 output lines and
retained the startup override plus source-default distinction. Both shell
wrappers passed Bash syntax checks and were checked for LF/no BOM.

`collect_lookup_logs.py` only reads existing logs and the local cache status,
printing a bounded summary. Per-request server completion/lock timestamps are
not recoverable when the original INFO logs did not record them. The current
cache snapshot describes collection time, not the incident's memory pressure.

This completion fix does not establish or repair the cause of missing cache
keys, and does not change the 300-second read lock TTL or benchmark parameters.

## Independent incident capture

`python3 watch_pp.py start --node 85` (or `--node 86`) launches a detached
standard-library-only capture process and returns only after its live PID and
ready receipt are confirmed. `summary` works from a new terminal; `stop` writes
only the collector's STOP file and waits for that collector, never sending a
signal to the model/cache. It defaults to a two-hour maximum duration.

The API node reads existing model metrics every 5 seconds; each node reads local
cache status every 15 seconds, backing off if that management probe is slow.
It records query-duration counter deltas, queues, cache locks/capacity, Linux
process CPU/RSS/wait channels, and fresh exceptions from current logs. Existing
logs begin at their current end; cache stdout regular files are discovered via
the cache process. Log floods are bounded with skipped-byte counts. All raw
samples and an atomically updated summary stay under `/opt/ks-luv-pp/watch`.

This is aggregate diagnostic capture, not per-request RPC/lock instrumentation.
It cannot establish lock acquisition age or split the lookup observation time
into exact internal phases. Periodic status calls also have some overhead;
record this as a diagnostic run rather than an uninstrumented performance run.
The standalone capture checks exercise real detached start, duplicate-start
rejection, HTTP sampling, fresh-log-only capture, flood bounds, summary/stop,
and a fixed old-log snapshot. They do not start any inference/device process.

## Model process commands

`python3 pp_model.py status|stop|start` inspects or controls the model associated
with `/opt/ks-luv-pp/startpp.sh` and private `logs/pp-*.log` descriptors. It
excludes cache, benchmark and observer processes and protects the caller's
ancestors. Stop signals the wrapper and actual model entry, lets the entry
drain its workers, checks PID start identities and refuses a new start if old
model processes remain. `--force` explicitly enables killing only verified
model remnants after the graceful timeout. Unidentified vLLM processes stop
the launch rather than being killed or duplicated.

Start preserves the original startup script, arguments and calling directory,
creates a detached process and a new private log, and reports submitted rather
than ready. It requires a healthy cache with no old GPU registrations. For
two-host PP: stop on both hosts first, start host 85 then immediately host 86,
and check health on 85. Cache and observer processes remain running. The model
control checks cover ownership, PID reuse, wrapper/entry signaling, duplicate
launch prevention and stale-registration rejection using CPU substitutes; no
customer services were started or stopped during those checks.
