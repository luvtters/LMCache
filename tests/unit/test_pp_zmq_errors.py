# SPDX-License-Identifier: Apache-2.0
"""Exercise production transport ASTs without optional device dependencies.

The JSON codec stand-in checks framing/control flow only; real msgspec/ZMQ
round trips are a separate CPU integration gate.
"""

# Standard
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import ast
import json
import unittest

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "lmcache/v1/multiprocess/transport/zmq_impl/mq.py"
)
PP_OPERATIONS = ("register_pp_kv_cache", "get_pp_stage_status")
MARKER = b"lmcache.pp.error.v1"


class DecodeError(ValueError):
    """Stand-in for the unavailable msgspec decoding exception."""


def encode(value: object) -> bytes:
    """Encode the dependency-free control-flow fixture."""
    return json.dumps(value).encode()


def decode(value: bytes, **kwargs: object) -> object:
    """Decode the fixture and enforce the error tuple's string schema."""
    try:
        result = json.loads(value)
    except (ValueError, UnicodeError) as exc:
        raise DecodeError(str(exc)) from exc
    if kwargs.get("type") == tuple[str, str]:
        if (
            not isinstance(result, list)
            or len(result) != 2
            or not all(isinstance(item, str) for item in result)
        ):
            raise DecodeError("Expected two strings")
        return tuple(result)
    return result


def load_transport_methods() -> dict[str, object]:
    """Compile the actual transport methods and PP constants from source."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id.startswith("_PP_")
                for target in targets
            ):
                nodes.append(node)
        if isinstance(node, ast.ClassDef):
            nodes.extend(
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef)
                and child.name in ("process_inbound", "_call_sync_handler")
            )
    namespace = {
        "msgspec": SimpleNamespace(
            msgpack=SimpleNamespace(encode=encode, decode=decode),
            DecodeError=DecodeError,
        ),
        "msgspec_decode": lambda value, cls: decode(value),
        "msgspec_encode": lambda value, cls: encode(value),
        "decode_operation": lambda value: value.decode(),
        "RequestUID": int,
        "MessagingStream": type("MessagingStream", (), {}),
        "logger": Mock(),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


class PPZmqErrorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.methods = load_transport_methods()

    def receive(self, operation: str, payloads: list[bytes]) -> Future:
        """Deliver one DEALER frame to the production inbound method."""
        future = Future()
        client = SimpleNamespace(
            socket=SimpleNamespace(
                recv_multipart=lambda: [encode(7), operation.encode(), *payloads]
            ),
            pending_streams={},
            pending_futures={
                7: (future, SimpleNamespace(operation=operation, response_type=dict))
            },
        )
        self.methods["process_inbound"](client)
        self.assertEqual(client.pending_futures, {})
        return future

    def test_pp_server_failure_returns_correlated_error_frame(self) -> None:
        for operation in PP_OPERATIONS:
            for exception_type in (ValueError, TypeError, RuntimeError, DecodeError):
                with self.subTest(operation=operation, exception_type=exception_type):
                    server = SimpleNamespace(socket=Mock())
                    handler = Mock(side_effect=exception_type("changed token"))
                    prefix = [b"identity", encode(7), operation.encode()]
                    self.methods["_call_sync_handler"](server, handler, [], prefix)
                    frames = server.socket.send_multipart.call_args.args[0]
                    self.assertEqual(frames[:4], [*prefix, MARKER])
                    self.assertEqual(
                        decode(frames[4]), [exception_type.__name__, "changed token"]
                    )

    def test_pp_client_failure_finishes_future(self) -> None:
        for operation in PP_OPERATIONS:
            with self.subTest(operation=operation):
                future = self.receive(
                    operation, [MARKER, encode(("ValueError", "changed token"))]
                )
                self.assertTrue(future.done())
                with self.assertRaisesRegex(ValueError, "changed token"):
                    future.result(timeout=0)

    def test_remote_runtime_failure_includes_operation_and_type(self) -> None:
        future = self.receive(
            PP_OPERATIONS[1], [MARKER, encode(("RuntimeError", "unavailable"))]
        )
        with self.assertRaisesRegex(
            RuntimeError, "get_pp_stage_status.*RuntimeError.*unavailable"
        ):
            future.result(timeout=0)

    def test_malformed_pp_error_frames_finish_future(self) -> None:
        for payloads in (
            [MARKER],
            [MARKER, b"broken"],
            [MARKER, encode(("ValueError", 1))],
            [MARKER, encode(("ValueError", "bad")), b"extra"],
        ):
            with self.subTest(payloads=payloads):
                future = self.receive(PP_OPERATIONS[0], payloads)
                self.assertTrue(future.done())
                with self.assertRaisesRegex(RuntimeError, "Malformed PP error"):
                    future.result(timeout=0)

    def test_success_frames_keep_existing_payload(self) -> None:
        for operation in (*PP_OPERATIONS, "register_kv_cache"):
            with self.subTest(operation=operation):
                self.assertEqual(
                    self.receive(operation, [encode({"ok": True})]).result(),
                    {"ok": True},
                )
                self.assertIsNone(self.receive(operation, []).result())

    def test_legacy_marker_keeps_existing_decoder_path(self) -> None:
        with self.assertRaises(DecodeError):
            self.receive("register_kv_cache", [MARKER, encode(("ValueError", "x"))])

    def test_legacy_sync_failure_still_raises(self) -> None:
        server = SimpleNamespace(socket=Mock())
        handler = Mock(side_effect=ValueError("legacy"))
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.methods["_call_sync_handler"](
                server, handler, [], [b"identity", encode(7), b"register_kv_cache"]
            )
        server.socket.send_multipart.assert_not_called()

    def test_server_success_frame_unchanged(self) -> None:
        for operation in (*PP_OPERATIONS, "register_kv_cache"):
            with self.subTest(operation=operation):
                server = SimpleNamespace(socket=Mock())
                handler = Mock(return_value={"ok": True})
                prefix = [b"identity", encode(7), operation.encode()]
                self.methods["_call_sync_handler"](server, handler, [], prefix)
                server.socket.send_multipart.assert_called_once_with(
                    [*prefix, encode({"ok": True})]
                )

    def test_operation_match_precedes_error_marker(self) -> None:
        future = Future()
        client = SimpleNamespace(
            socket=SimpleNamespace(
                recv_multipart=lambda: [
                    encode(7),
                    PP_OPERATIONS[1].encode(),
                    MARKER,
                    encode(("ValueError", "remote reason")),
                ]
            ),
            pending_streams={},
            pending_futures={
                7: (
                    future,
                    SimpleNamespace(operation=PP_OPERATIONS[0], response_type=dict),
                )
            },
        )
        self.methods["process_inbound"](client)
        with self.assertRaisesRegex(ValueError, "does not match"):
            future.result(timeout=0)


if __name__ == "__main__":
    unittest.main()
