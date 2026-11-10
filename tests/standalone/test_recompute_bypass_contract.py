"""Real connector/tracker methods; framework and transport use existing doubles."""
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import test_pp_completion_contract as contract

contract.execute(contract.source_nodes(contract.connector_file, 'LMCacheMPConnector',
    {'get_num_new_matched_tokens', 'update_state_after_alloc'}),
    contract.ns, contract.connector_file)
for name in ('get_num_new_matched_tokens', 'update_state_after_alloc'):
    setattr(contract.Connector, name, contract.ns[name])


class RecomputeBypassTest(unittest.TestCase):
    def test_failed_load_stays_local_until_actual_allocation(self):
        request = contract.request()
        request.num_computed_tokens = 0
        tracker = contract.Tracker(request)
        tracker.state = contract.State.READY
        tracker.num_lmcache_hit_tokens = 16
        tracker.num_stored_tokens = 16
        tracker.borrowed_restore_submitted = True
        tracker.allocated_block_ids = {0: [7]}
        connector = contract.Connector()
        connector.lazy_offload = False
        connector._read_policy = 'transfer_borrow'
        connector._mamba_relocation_window = 0
        connector.request_trackers = {request.request_id: tracker}
        connector._get_request_tracker = lambda rid: connector.request_trackers[rid]
        connector.scheduler_adapter = MagicMock()
        for _ in range(3):
            self.assertEqual(connector.get_num_new_matched_tokens(request, 0), (0, False))
        self.assertEqual(tracker.state, contract.State.BYPASS_LMCACHE)
        self.assertEqual(tracker.num_lmcache_hit_tokens, 0)
        self.assertEqual(tracker.num_stored_tokens, 0)
        self.assertFalse(tracker.borrowed_restore_submitted)
        self.assertEqual(tracker.allocated_block_ids, {})
        connector.scheduler_adapter.cleanup_lookup_result.assert_called_once_with(request.request_id)
        connector.scheduler_adapter.maybe_submit_lookup_request.assert_not_called()
        blocks = SimpleNamespace(get_block_ids=lambda: ([8],))
        connector.update_state_after_alloc(request, blocks, 0)
        self.assertEqual(tracker.state, contract.State.READY)
        self.assertEqual(tracker.allocated_block_ids, {0: [8]})


if __name__ == '__main__':
    unittest.main()
