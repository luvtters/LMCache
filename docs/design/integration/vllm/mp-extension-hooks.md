# MP scheduler extension hooks

The connector already allows a platform plugin to choose its worker adapter.
A transfer implementation with its own object directory also needs matching
scheduler lookup and lock-release behavior. Two hooks provide that pairing
without introducing another backend loader or changing the default transfer mode.

- `LMCacheMPConnector.create_lmcache_mp_scheduler_adapter(**adapter_kwargs)`
  constructs the scheduler adapter. Its default is `LMCacheMPSchedulerAdapter`,
  with the same arguments previously passed directly by the connector.
- `LMCacheMPSchedulerAdapter._create_lookup_state(key)` constructs a request's
  lookup operation. Its default is the existing `LookupState`. A subclass can
  supply an operation with `submit`, `on_scheduler_check`, `get_result`, `close`,
  and `drain` while keeping key alignment and scheduler bookkeeping.

The worker factory and `_create_transfer_context` remain unchanged. An external
connector can override both factories to select a matching scheduler and worker.
Ordinary connectors continue to use the original implementations.

A custom scheduler must route `free_lookup_locks`, `end_session`, `reset_cache`,
and shutdown to the same object directory used by its lookup operation. Closing
a local lookup is not proof that the service has released its pins. In particular,
cleanup can happen before end-session: retain outstanding control operations
until their replies are drained and service protections are explicitly released.
Timeouts do not establish device-transfer completion.

`tests/v1/test_mp_adapter_extension_hooks.py` executes the actual hook and lookup
submission method bodies with engine/network boundaries replaced. This is a
source-level CPU contract check; it does not validate full vLLM/native imports or
device transfers.
