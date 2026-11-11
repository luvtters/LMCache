# SPDX-License-Identifier: Apache-2.0
"""CPU contracts executing actual transport methods with queue/socket substitutes."""
import ast
import itertools
import logging
from pathlib import Path
import queue
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
MQ = ROOT / 'lmcache/v1/multiprocess/transport/zmq_impl/mq.py'


def methods(owner, names):
    tree = ast.parse(MQ.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in nodes:
        n.decorator_list = []
    ns = dict(queue=queue, itertools=itertools, MessagingStream=type('Stream', (), {}),
              logger=logging.getLogger(__name__), RequestUID=int,
              msgspec_encode=lambda value, **kw: str(value).encode(),
              encode_operation=lambda value: value.encode(), decode_request_uid=int)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(MQ), 'exec'), ns)
    return type(owner, (), {name: ns[name] for name in names})


Client = methods('MessageQueueClient', ['process_outbound_task'])


def client(limit=0):
    obj = Client()
    obj.outbound_batch_limit = limit
    obj.input_queue = queue.Queue()
    obj.sent = []
    obj.notifications = []
    obj.socket = types.SimpleNamespace(send_multipart=obj.sent.append)
    obj._polling_loop = types.SimpleNamespace(notify=lambda: obj.notifications.append(True))
    obj.pending_streams = {}
    obj.pending_futures = {}
    return obj


def fill(obj, count):
    for i in range(count):
        obj.input_queue.put([str(i).encode(), b'', b'ack'])


class ProgressContracts(unittest.TestCase):
    def test_fifo_residual_and_notifications(self):
        obj = client(32)
        fill(obj, 256)
        for _ in range(8):
            obj.process_outbound_task()
        self.assertEqual([int(frame[0]) for frame in obj.sent], list(range(256)))
        self.assertEqual(len(obj.notifications), 8)
        obj.process_outbound_task()
        self.assertTrue(obj.input_queue.empty())

    def test_zero_default_and_override_drain(self):
        for limit, override in [(0, None), (32, 0)]:
            obj = client(limit)
            fill(obj, 256)
            obj.process_outbound_task(max_tasks=override)
            self.assertEqual(len(obj.sent), 256)
            self.assertFalse(obj.notifications)

    def test_negative_override(self):
        with self.assertRaises(ValueError):
            client().process_outbound_task(max_tasks=-1)

    def test_control_cancel_and_encoding_failure_consume_slots(self):
        obj = client(1)
        obj.pending_streams[0] = object()
        obj.input_queue.put([b'0', b'', b'cancel'])
        failed = []
        obj.input_queue.put(types.SimpleNamespace(
            request_uid=1, rpc_spec=types.SimpleNamespace(operation='test', payload_types=[int]),
            request_payloads=[], future=types.SimpleNamespace(set_exception=failed.append)))
        obj.input_queue.put([b'2', b'', b'ack'])
        obj.process_outbound_task()
        self.assertNotIn(0, obj.pending_streams)
        obj.process_outbound_task()
        self.assertEqual(len(failed), 1)
        self.assertEqual(obj.input_queue.qsize(), 1)
        obj.process_outbound_task()
        self.assertEqual([frame[0] for frame in obj.sent], [b'0', b'2'])

    def test_shared_loop_other_client_and_ready_reply(self):
        Loop = methods('ClientPollingLoop', ['_main_loop'])
        loop = Loop()
        first, second = client(32), client()
        fill(first, 256)
        fill(second, 1)
        events = []
        second.process_inbound = lambda: events.append(('reply', len(first.sent)))
        loop._notifier = types.SimpleNamespace(fileno=lambda: 7, consume=lambda: None)
        loop._process_ops = lambda: None
        loop._is_finished = types.SimpleNamespace(is_set=iter([False, True]).__next__)
        loop._poller = types.SimpleNamespace(poll=lambda timeout: [(7, 1), ('second', 1)])
        loop._socket_to_client = {'first': first, 'second': second}
        loop._monitor_to_client = {}
        Loop._main_loop.__globals__['zmq'] = types.SimpleNamespace(POLLIN=1)
        loop._main_loop()
        self.assertEqual(len(second.sent), 1)
        self.assertEqual(events, [('reply', 32)])

    def test_unregister_drains_before_removal(self):
        Loop = methods('ClientPollingLoop', ['_process_ops'])
        loop = Loop()
        obj = client(1)
        fill(obj, 256)
        obj.socket = types.SimpleNamespace(send_multipart=obj.sent.append)
        obj.monitor = 'monitor'
        obj.cancel_streams = lambda: None
        loop._ops_queue = queue.Queue()
        done = []
        loop._ops_queue.put(types.SimpleNamespace(kind='unregister', client=obj,
                              done=types.SimpleNamespace(set=lambda: done.append(True))))
        loop._poller = types.SimpleNamespace(unregister=lambda socket: None)
        loop._socket_to_client = {}
        loop._monitor_to_client = {}
        Loop._process_ops.__globals__['_OpKind'] = types.SimpleNamespace(REGISTER='register', UNREGISTER='unregister')
        # A socket substitute needs identity hashing, as real ZMQ sockets do.
        class Socket:
            send_multipart = lambda self, frame: obj.sent.append(frame)
        obj.socket = Socket()
        loop._process_ops()
        self.assertEqual(len(obj.sent), 256)
        self.assertEqual(done, [True])

    def test_negative_constructor_before_socket(self):
        Actual = methods('MessageQueueClient', ['__init__'])
        with self.assertRaises(ValueError):
            Actual('tcp://unused', None, outbound_batch_limit=-1)

    def test_close_order(self):
        Actual = methods('MessageQueueClient', ['close'])
        obj = Actual()
        events = []
        obj._polling_loop = types.SimpleNamespace(unregister=lambda owner: events.append('unregister'))
        obj.monitor = types.SimpleNamespace(close=lambda **kw: events.append('monitor'))
        obj.socket = types.SimpleNamespace(close=lambda: events.append('socket'))
        Actual.close.__globals__['ClientPollingLoop'] = types.SimpleNamespace(
            release_instance=lambda: events.append('release'))
        obj.close()
        self.assertEqual(events, ['unregister', 'release', 'monitor', 'socket'])

    def test_factory_forwarding_and_grpc_rejection_before_import(self):
        path = ROOT / 'lmcache/v1/multiprocess/transport/factory.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, (ast.Assign, ast.ClassDef, ast.FunctionDef))]
        calls, imports = [], []
        backend = types.SimpleNamespace(create_request_client=lambda *a, **kw: calls.append((a, kw)))
        def importer(name, *args, **kwargs):
            if name == "__future__":
                return __import__(name, *args, **kwargs)
            imports.append(name)
            return types.SimpleNamespace(zmq_impl=backend, grpc_impl=backend)
        import builtins
        ns = {'__builtins__': dict(vars(builtins), __import__=importer)}
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), 'exec'), ns)
        create = ns['RequestClientFactory'].create
        create('host:123', outbound_batch_limit=32)
        self.assertEqual(calls[-1], (('tcp://host:123',), {'context': None, 'outbound_batch_limit': 32}))
        imports.clear()
        with self.assertRaisesRegex(ValueError, 'ZMQ'):
            create('grpc://host:123', outbound_batch_limit=32)
        self.assertEqual(imports, [])
        with self.assertRaises(ValueError):
            create('host:123', outbound_batch_limit=-1)
        create('grpc://host:123')
        self.assertEqual(calls[-1], (('grpc://host:123',), {'context': None}))

    def test_budget_256_sends_32(self):
        obj = client(32)
        fill(obj, 256)
        obj.process_outbound_task()
        self.assertEqual(len(obj.sent), 32)
        self.assertEqual(obj.input_queue.qsize(), 224)


if __name__ == '__main__':
    unittest.main()
