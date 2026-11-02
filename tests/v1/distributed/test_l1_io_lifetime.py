# SPDX-License-Identifier: Apache-2.0
"""CPU tests for L1Manager pins held across actual I/O."""

import time

import pytest
import torch

from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager


class FakeMemoryObj:
    def __init__(self, size: int):
        self.size = size
        self.valid = True

    def get_size(self) -> int:
        return self.size

    def is_valid(self) -> bool:
        return self.valid

    def set_l1_manager(self, _manager_id: int) -> None:
        pass


class FakeMemoryManager:
    """Small host-only allocator for L1Manager lifetime tests."""

    def allocate(self, _layout, count):
        return L1Error.SUCCESS, [FakeMemoryObj(64) for _ in range(count)]

    def free(self, objects):
        for obj in objects:
            obj.valid = False
        return L1Error.SUCCESS

    def get_memory_usage(self):
        return 0, 1024

    def get_backend_type(self, _obj):
        return L1BackendType.DRAM

    def get_l1_memory_desc(self):
        return None

    def memcheck(self):
        return True

    def close(self):
        pass


@pytest.fixture
def manager():
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=16 * 1024 * 1024,
            use_lazy=True,
            init_size_in_bytes=4 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=1,
        read_ttl_seconds=1,
    )
    result = L1Manager(config, memory_manager=FakeMemoryManager())
    yield result
    # Individual tests that exercise close() close explicitly.
    try:
        result.close()
    except RuntimeError:
        pass


@pytest.fixture
def layout():
    return MemoryLayoutDesc(shapes=[torch.Size([4, 16])], dtypes=[torch.float32])


def key(n: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(n), model_name="io-lifetime", kv_rank=0
    )


def test_write_pin_survives_ttl_and_force_cleanup(manager, layout):
    item = key(1)
    result = manager.reserve_write_io([item], [False], layout, "write-1", "io")
    assert result[item][0] == L1Error.SUCCESS
    memory = result[item][1]

    time.sleep(1.1)
    assert manager.finish_write([item], "io")[item] == L1Error.KEY_IN_WRONG_STATE
    assert (
        manager.finish_write_and_delete([item], "io")[item]
        == L1Error.KEY_IN_WRONG_STATE
    )
    assert manager.delete([item], force=True)[item] == L1Error.KEY_IS_LOCKED
    manager.clear(force=True)
    assert memory.is_valid()

    assert manager.finish_write_io([item], "write-1", "io")[item] == L1Error.SUCCESS
    assert manager.get_object_state(item).memory_obj is memory
    assert (
        manager.finish_write_io([item], "write-1", "io")[item]
        == L1Error.KEY_IN_WRONG_STATE
    )


def test_write_io_pins_only_successful_subset(manager, layout):
    resident = key(5)
    staged = key(6)
    manager.reserve_write([resident], [False], layout)
    manager.finish_write([resident])

    result = manager.reserve_write_io(
        [staged, resident], [False, False], layout, "partial", "io"
    )
    assert result[staged][0] == L1Error.SUCCESS
    assert result[resident][0] == L1Error.KEY_NOT_WRITABLE
    assert manager.finish_write_io([staged, resident], "partial", "io") == {
        staged: L1Error.SUCCESS,
        resident: L1Error.KEY_IN_WRONG_STATE,
    }


def test_read_io_pins_are_independent_and_do_not_unlock_later_reader(manager, layout):
    item = key(2)
    manager.reserve_write([item], [True], layout)
    manager.finish_write([item])
    manager.reserve_read([item], read_locks=2)

    first = manager.read_prefetched_io([item], "read-1")
    second = manager.read_prefetched_io([item], "read-2")
    assert first[item][0] == second[item][0] == L1Error.SUCCESS

    assert manager.finish_read_io([item], "read-1")[item] == L1Error.SUCCESS
    assert manager.get_object_state(item) is not None
    assert manager.finish_read_io([item], "read-1")[item] == L1Error.KEY_IN_WRONG_STATE
    assert manager.delete([item], force=True)[item] == L1Error.KEY_IS_LOCKED
    assert not manager.is_key_evictable(item)
    manager.clear(force=True)
    assert manager.get_object_state(item) is not None

    # This native reader starts after the I/O pins. Completing the old I/O
    # token must not decrement its TTL lock.
    manager.reserve_read([item])
    assert manager.finish_read_io([item], "read-2")[item] == L1Error.SUCCESS
    assert manager.get_object_state(item) is not None
    assert manager.finish_read([item])[item] == L1Error.SUCCESS
    assert manager.get_object_state(item) is None


def test_close_rejects_active_pin_and_old_token_cannot_release_new_pin(
    manager, layout
):
    first = key(3)
    second = key(4)
    manager.reserve_write_io([first], [False], layout, "old", "io")
    with pytest.raises(RuntimeError, match="active I/O pins"):
        manager.close()

    assert manager.finish_write_io([first], "old", "io")[first] == L1Error.SUCCESS
    assert (
        manager.reserve_write_io([second], [False], layout, "old", "io")[second][0]
        == L1Error.KEY_IN_WRONG_STATE
    )
    assert (
        manager.reserve_write_io([second], [False], layout, "new", "io")[second][0]
        == L1Error.SUCCESS
    )
    # A late duplicate for the old token cannot affect the new allocation.
    assert (
        manager.finish_write_io([second], "old", "io")[second]
        == L1Error.KEY_IN_WRONG_STATE
    )
    assert manager.get_object_state(second) is None
    assert manager.finish_write_io([second], "new", "io")[second] == L1Error.SUCCESS
