# SPDX-License-Identifier: Apache-2.0
"""CPU contracts executing the real scheduler methods without device imports."""

import unittest
from types import SimpleNamespace

from test_lookup_deadline_contract import Reply, scheduler


def stages(enabled=True):
    obj, clock, _ = scheduler()
    obj._pp_query_fastpath = enabled
    obj._server_urls = ["early", "late"]
    obj.sent = []
    obj.replies = {"early": Reply(0), "late": Reply(2, ready=False)}

    def query(url, request_id):
        obj.sent.append((url, request_id))
        return obj.replies[url]

    obj.req_clients = {
        url: SimpleNamespace(query_prefetch_status=lambda rid, url=url: query(url, rid))
        for url in obj._server_urls
    }
    return obj, clock


class EarlyMissTest(unittest.TestCase):
    def test_final_zero_does_not_wait_for_peer(self):
        obj, _ = stages()
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertEqual(obj._finished_lookup_results["r"], 0)
        self.assertIs(obj._lookup_status["r"]["late"][0], obj.replies["late"])
        obj.replies["late"].ready = True
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertEqual(len(obj.sent), 2)

    def test_disabled_preserves_full_status_barrier(self):
        obj, _ = stages(enabled=False)
        self.assertIsNone(obj.check_lookup_result("r"))

    def test_none_is_pending_and_positive_waits_for_all(self):
        obj, _ = stages()
        obj.replies["early"] = Reply(None)
        self.assertIsNone(obj.check_lookup_result("r"))
        obj.replies["early"] = Reply(3)
        self.assertIsNone(obj.check_lookup_result("r"))
        obj.replies["late"].ready = True
        self.assertEqual(obj.check_lookup_result("r"), 8)
        self.assertEqual(len(obj.sent), 3)


if __name__ == "__main__":
    unittest.main()
