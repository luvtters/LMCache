"""Execute the actual shutdown methods with CPU-only lifecycle boundaries."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).parents[3]


def methods(relative, class_name, names, *, bases=(), namespace=None):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(body) != len(names):
        raise AssertionError('Requested shutdown method missing')
    node = ast.ClassDef(name=class_name, bases=[ast.Name(id=n, ctx=ast.Load()) for n in bases],
                        keywords=[], body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(namespace or {})
    exec(compile(module, str(ROOT / relative), 'exec'), scope)
    return scope[class_name]


class CallbackShutdownChecks(unittest.TestCase):
    def transfer(self, events, stop):
        cls = methods('lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py',
                      'LMCacheDrivenTransferModule', {'quiesce_device_callbacks', 'close'})
        obj = cls()
        obj._device_callbacks_quiesced = False
        obj._device_host_func_dispatcher = SimpleNamespace(stop=stop)
        obj._lock = threading.Lock()
        obj._cache_contexts = {1: 'owned-entry'}
        obj._release_entries = lambda entries: events.append(('release', entries))
        return obj

    def test_quiesce_is_idempotent_and_keeps_entries_until_later_close(self):
        events = []
        obj = self.transfer(events, lambda: events.append('stop-and-drain'))
        obj.quiesce_device_callbacks()
        obj.quiesce_device_callbacks()
        self.assertEqual(obj._cache_contexts, {1: 'owned-entry'})
        events.append('external-BM-wait-uninit')
        obj.close()
        self.assertEqual(events, ['stop-and-drain', 'external-BM-wait-uninit',
                                  ('release', ['owned-entry'])])

    def test_failed_quiesce_keeps_entries_and_allows_a_real_retry(self):
        events = []
        def fail():
            raise TimeoutError('dispatcher still alive')
        obj = self.transfer(events, fail)
        with self.assertRaises(TimeoutError):
            obj.close()
        self.assertFalse(obj._device_callbacks_quiesced)
        self.assertEqual(obj._cache_contexts, {1: 'owned-entry'})
        obj._device_host_func_dispatcher.stop = lambda: events.append('drained')
        obj.close()
        self.assertEqual(events, ['drained', ('release', ['owned-entry'])])

    def dispatcher(self, alive, events):
        class Base:
            def stop(self, timeout):
                events.append('join')
        registry = SimpleNamespace(unregister=lambda name: events.append('unregister'))
        cls = methods('lmcache/v1/multiprocess/native_completion.py',
            'DeviceHostFuncDispatcher', {'stop'}, bases=('Base',),
            namespace={'Base': Base, 'PeriodicThreadRegistry': SimpleNamespace(get_instance=lambda: registry)})
        obj = cls()
        obj._thread = SimpleNamespace(is_alive=lambda: alive)
        obj.name = 'test-dispatcher'
        obj._drain_once = lambda: events.append('drain')
        return obj

    def test_live_dispatcher_never_claims_quiesced_or_drains_concurrently(self):
        events = []
        with self.assertRaisesRegex(TimeoutError, 'dispatcher'):
            self.dispatcher(True, events).stop()
        self.assertEqual(events, ['join'])

    def test_stopped_dispatcher_unregisters_then_performs_final_drain(self):
        events = []
        self.dispatcher(False, events).stop()
        self.assertEqual(events, ['join', 'unregister', 'drain'])


if __name__ == '__main__':
    unittest.main()
