# SPDX-License-Identifier: Apache-2.0
"""Immutable, stdlib-only identities for pipeline-stage registrations."""

# Standard
from dataclasses import dataclass
import hashlib
import json
import math


def _integer(value: int, name: str, minimum: int = 0) -> None:
    """Reject non-integer or out-of-range identity fields with ValueError."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _text(value: str, name: str) -> None:
    """Reject empty identity strings with ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _layers(names: tuple[str, ...]) -> tuple[str, ...]:
    """Validate ordered physical names, including auxiliary registered groups."""
    names = tuple(names)
    if not names:
        raise ValueError("layer_names must be nonempty")
    for name in names:
        _text(name, "layer name")
    if len(set(names)) != len(names):
        raise ValueError("layer_names must be unique")
    return names


def _primitive(value: object) -> None:
    """Validate finite primitive JSON recursively; raise ValueError otherwise."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _primitive(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("layout keys must be strings")
            _primitive(item)
        return
    raise ValueError("layout must contain finite JSON primitives")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys rather than silently discarding metadata."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate layout key")
        result[key] = value
    return result


@dataclass(frozen=True)
class PPStageLocator:
    """Logical topology with complete, contiguous half-open layer ranges.

    Ranges start at zero, contain one nonempty interval per PP rank, and
    preserve pipeline order. Invalid fields raise ValueError.
    """

    base_model_name: str
    tp_size: int
    pp_size: int
    pp_rank: int
    layer_ranges: tuple[tuple[int, int], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.base_model_name, "base_model_name")
        _integer(self.tp_size, "tp_size", 1)
        _integer(self.pp_size, "pp_size", 1)
        _integer(self.pp_rank, "pp_rank")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported PP schema version")
        if self.pp_rank >= self.pp_size:
            raise ValueError("pp_rank exceeds topology")
        ranges = tuple(tuple(interval) for interval in self.layer_ranges)
        if len(ranges) != self.pp_size:
            raise ValueError("layer_ranges must describe every PP stage")
        previous = 0
        for interval in ranges:
            if len(interval) != 2:
                raise ValueError("layer intervals must contain start and end")
            start, end = interval
            _integer(start, "layer start")
            _integer(end, "layer end", 1)
            if start != previous or end <= start:
                raise ValueError("layer ranges must be contiguous and nonempty")
            previous = end
        object.__setattr__(self, "layer_ranges", ranges)


@dataclass(frozen=True)
class PPRegistrationInfo:
    """Worker control identity and ordered stage layers; invalid fields fail."""

    locator: PPStageLocator
    global_rank: int
    tp_rank: int
    layer_names: tuple[str, ...]
    registration_token: str
    mla_only: bool
    read_policy: str = "legacy"

    def __post_init__(self) -> None:
        if self.read_policy not in ("legacy", "transfer_borrow"):
            raise ValueError("unsupported PP read policy")
        _integer(self.global_rank, "global_rank")
        _integer(self.tp_rank, "tp_rank")
        if self.tp_rank >= self.locator.tp_size or self.global_rank != (
            self.locator.pp_rank * self.locator.tp_size + self.tp_rank
        ):
            raise ValueError("worker rank does not match stage topology")
        _text(self.registration_token, "registration_token")
        if type(self.mla_only) is not bool:
            raise ValueError("mla_only must be boolean")
        object.__setattr__(self, "layer_names", _layers(self.layer_names))


@dataclass(frozen=True)
class PPStageProfile:
    """Content identity from logical metadata and actual canonical layout JSON.

    The server supplies layout metadata only: control IDs, addresses, host
    details and KV pool capacity must not be included in layout_json.
    Invalid JSON or stage layers raise ValueError.
    """

    locator: PPStageLocator
    layer_names: tuple[str, ...]
    mla_only: bool
    layout_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_names", _layers(self.layer_names))
        if type(self.mla_only) is not bool:
            raise ValueError("mla_only must be boolean")
        layout = json.loads(self.layout_json, object_pairs_hook=_unique_object)
        _primitive(layout)
        object.__setattr__(
            self,
            "layout_json",
            json.dumps(layout, sort_keys=True, separators=(",", ":")),
        )

    @property
    def cache_namespace(self) -> str:
        """Return the deterministic model namespace for this content format."""
        locator = self.locator
        metadata = (
            locator.schema_version,
            locator.base_model_name,
            locator.tp_size,
            locator.pp_size,
            locator.pp_rank,
            locator.layer_ranges,
            self.layer_names,
            self.mla_only,
            json.loads(self.layout_json),
        )
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        return f"{locator.base_model_name}::lmcache-pp-v1::{digest}"

    @property
    def kv_world_size(self) -> int:
        """Return the number of distinct KV pieces per cached chunk."""
        return 1 if self.mla_only else self.locator.tp_size

    @property
    def num_kv_readers(self) -> int:
        """Return the number of worker readers sharing each KV piece."""
        return self.locator.tp_size if self.mla_only else 1


@dataclass(frozen=True)
class PPRegistrationAck:
    """Server confirmation binding one worker to an accepted stage profile."""

    profile: PPStageProfile
    server_instance_id: str
    instance_id: int
    tp_rank: int
    read_policy: str = "legacy"

    def __post_init__(self) -> None:
        if self.read_policy not in ("legacy", "transfer_borrow"):
            raise ValueError("unsupported PP read policy")
        _text(self.server_instance_id, "server_instance_id")
        _integer(self.instance_id, "instance_id")
        _integer(self.tp_rank, "tp_rank")
        if self.tp_rank >= self.profile.locator.tp_size:
            raise ValueError("tp_rank exceeds topology")

    @property
    def cache_namespace(self) -> str:
        """Return the namespace accepted by the server."""
        return self.profile.cache_namespace


@dataclass(frozen=True)
class PPStageStatus:
    """Snapshot whose readiness requires an exact, duplicate-free TP set."""

    locator: PPStageLocator
    server_instance_id: str
    profile: PPStageProfile | None
    registered_tp_ranks: tuple[int, ...]
    schema_version: int = 1
    read_policy: str = "legacy"

    def __post_init__(self) -> None:
        if self.read_policy not in ("legacy", "transfer_borrow"):
            raise ValueError("unsupported PP read policy")
        _text(self.server_instance_id, "server_instance_id")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported PP schema version")
        ranks = tuple(self.registered_tp_ranks)
        for rank in ranks:
            _integer(rank, "registered TP rank")
            if rank >= self.locator.tp_size:
                raise ValueError("registered TP rank exceeds topology")
        object.__setattr__(self, "registered_tp_ranks", ranks)

    @property
    def ready(self) -> bool:
        """Return whether all TP workers agree on this exact stage locator."""
        return (
            self.profile is not None
            and self.profile.locator == self.locator
            and len(self.registered_tp_ranks) == self.locator.tp_size
            and set(self.registered_tp_ranks) == set(range(self.locator.tp_size))
        )
