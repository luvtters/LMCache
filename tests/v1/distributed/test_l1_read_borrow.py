# SPDX-License-Identifier: Apache-2.0
"""Real CPU allocator/native TTL-lock tests for transfer read ownership."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager


PAGE = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([PAGE])], [torch.uint8])


def key(i):
    return ObjectKey(ObjectKey.IntHash2Bytes(i), "borrow-test", 0)


@pytest.fixture
def l1():
    manager = L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(PAGE * 4, use_lazy=False, shm_name=""),
            read_ttl_seconds=1,
            write_ttl_seconds=1,
        )
    )
    yield manager
    # Keep teardown safe even when an assertion fails after a borrow.
    for token in list(getattr(manager, "_read_borrows", {})):
        manager.release_borrow(token)
    manager.close()


def publish(manager, keys, temporary=False, value=37):
    reserved = manager.reserve_write(keys, [temporary] * len(keys), LAYOUT)
    for k in keys:
        error, obj = reserved[k]
        assert error == L1Error.SUCCESS
        obj.tensor.fill_(value)
    assert all(e == L1Error.SUCCESS for e in manager.finish_write(keys).values())
    return {k: reserved[k][1] for k in keys}


def assert_contents(borrow, value=37):
    for obj in borrow.objects.values():
        assert torch.all(obj.tensor == value)


def test_borrow_outlives_native_ttl_and_blocks_real_allocator_reuse(l1):
    keys = [key(i) for i in range(4)]
    publish(l1, keys)
    l1.reserve_read(keys)
    borrowed = l1.borrow_read(keys)
    assert borrowed is not None
    addresses = {obj.tensor.data_ptr() for obj in borrowed.objects.values()}
    time.sleep(1.1)
    assert l1.finish_read(keys)[keys[0]] == L1Error.KEY_IN_WRONG_STATE
    assert all(not l1.is_key_evictable(k) for k in keys)
    assert set(l1.delete(keys, force=True).values()) == {L1Error.KEY_IS_LOCKED}
    l1.clear(force=True)
    assert l1.get_memory_usage()[0] == 4 * PAGE
    reserved = l1.reserve_write([key(99)], [False], LAYOUT)
    assert reserved[key(99)][0] == L1Error.OUT_OF_MEMORY
    assert_contents(borrowed)
    assert l1.release_borrow(borrowed.token)
    l1.clear()
    replacement = publish(l1, keys, value=91)
    assert {obj.tensor.data_ptr() for obj in replacement.values()} == addresses
    assert all(torch.all(obj.tensor == 91) for obj in replacement.values())


def test_missing_or_staging_key_leaves_no_partial_borrow(l1):
    keys = [key(i) for i in range(4)]
    publish(l1, keys[:3])
    assert l1.borrow_read(keys) is None
    assert l1.reserve_write([keys[3]], [False], LAYOUT)[keys[3]][0] == L1Error.SUCCESS
    assert l1.borrow_read(keys) is None
    assert set(l1.delete(keys[:3]).values()) == {L1Error.SUCCESS}
    assert l1.get_memory_usage()[0] == PAGE
    assert l1.finish_write_and_delete([keys[3]])[keys[3]] == L1Error.SUCCESS
    assert l1.get_memory_usage()[0] == 0


def test_sixteen_tokens_duplicate_keys_and_duplicate_or_wrong_release(l1):
    k = key(0)
    publish(l1, [k])
    with ThreadPoolExecutor(max_workers=16) as pool:
        reads = list(pool.map(lambda _: l1.borrow_read([k, k]), range(16)))
    assert len({r.token for r in reads}) == 16
    assert all(list(r.objects) == [k] for r in reads)
    assert not l1.release_borrow(replace(reads[0].token, server_instance_id="wrong"))
    assert not l1.release_borrow(replace(reads[0].token, sequence=999999))
    for read in reads[:-1]:
        assert l1.release_borrow(read.token)
        assert not l1.release_borrow(read.token)
        assert l1.delete([k], force=True)[k] == L1Error.KEY_IS_LOCKED
    assert_contents(reads[-1])
    assert l1.release_borrow(reads[-1].token)
    assert l1.delete([k])[k] == L1Error.SUCCESS
    assert l1.get_memory_usage()[0] == 0


def test_expired_legacy_reader_cannot_release_new_borrow(l1):
    k = key(0)
    publish(l1, [k], temporary=True)
    l1.reserve_read([k])
    time.sleep(1.1)
    read = l1.borrow_read([k])
    l1.finish_read([k])
    assert l1.delete([k], force=True)[k] == L1Error.KEY_IS_LOCKED
    assert_contents(read)
    assert l1.release_borrow(read.token)
    assert l1.get_memory_usage()[0] == 0


def test_old_token_cannot_release_recreated_key(l1):
    k = key(0)
    publish(l1, [k])
    old = l1.borrow_read([k])
    assert l1.release_borrow(old.token)
    assert l1.delete([k])[k] == L1Error.SUCCESS
    publish(l1, [k], value=62)
    new = l1.borrow_read([k])
    assert new.objects[k] is not old.objects[k]
    assert new.token.sequence > old.token.sequence
    assert not l1.release_borrow(old.token)
    assert l1.delete([k], force=True)[k] == L1Error.KEY_IS_LOCKED
    assert_contents(new, 62)
    assert l1.release_borrow(new.token)


def test_token_from_other_server_instance_cannot_release(l1):
    k = key(0)
    publish(l1, [k])
    old = l1.borrow_read([k])
    other = L1Manager(l1.config)
    new = None
    try:
        publish(other, [k], value=82)
        new = other.borrow_read([k])
        assert new.token.sequence == old.token.sequence
        assert new.token.server_instance_id != old.token.server_instance_id
        assert not other.release_borrow(old.token)
        assert other.delete([k], force=True)[k] == L1Error.KEY_IS_LOCKED
        assert_contents(new, 82)
    finally:
        if new is not None:
            other.release_borrow(new.token)
        other.close()
        l1.release_borrow(old.token)


def test_legacy_force_cleanup_keeps_its_original_behavior(l1):
    k = key(0)
    publish(l1, [k])
    l1.reserve_read([k])
    assert l1.delete([k])[k] == L1Error.KEY_IS_LOCKED
    assert l1.delete([k], force=True)[k] == L1Error.SUCCESS
    publish(l1, [k])
    l1.reserve_read([k])
    l1.clear()
    assert l1.get_memory_usage()[0] == PAGE
    l1.clear(force=True)
    assert l1.get_memory_usage()[0] == 0


@pytest.mark.parametrize("force", [False, True])
def test_delete_clear_keep_borrow_but_free_other_residents(l1, force):
    protected, other = key(0), key(1)
    publish(l1, [protected, other])
    read = l1.borrow_read([protected])
    assert l1.delete([protected], force=force)[protected] == L1Error.KEY_IS_LOCKED
    l1.clear(force=force)
    assert l1.get_memory_usage()[0] == PAGE
    assert_contents(read)
    assert l1.release_borrow(read.token)
    l1.clear(force=force)
    assert l1.get_memory_usage()[0] == 0


@pytest.mark.parametrize("legacy_finishes_first", [False, True])
def test_temporary_cleanup_waits_for_both_reader_kinds(l1, legacy_finishes_first):
    k = key(0)
    publish(l1, [k], temporary=True)
    l1.reserve_read([k])
    read = l1.borrow_read([k])
    if legacy_finishes_first:
        l1.finish_read([k])
    else:
        assert l1.release_borrow(read.token)
    assert l1.get_memory_usage()[0] == PAGE
    assert_contents(read)
    if legacy_finishes_first:
        assert l1.release_borrow(read.token)
    else:
        l1.finish_read([k])
    assert l1.get_memory_usage()[0] == 0


@pytest.mark.parametrize("reserve_read", [False, True])
def test_staging_discard_never_frees_borrowed_resident(l1, reserve_read):
    k = key(0)
    for tag in ("winner", "loser", "discard"):
        assert l1.reserve_write([k], [False], LAYOUT, tag=tag)[k][0] == L1Error.SUCCESS
    l1.finish_write([k], tag="winner")
    read = l1.borrow_read([k])
    read.objects[k].tensor.fill_(37)
    if reserve_read:
        l1.finish_write_and_reserve_read([k], tag="loser")
    else:
        l1.finish_write([k], tag="loser")
    l1.finish_write_and_delete([k], tag="discard")
    l1.finish_read([k])
    assert l1.get_memory_usage()[0] == PAGE
    assert_contents(read)
    assert l1.release_borrow(read.token)


@pytest.mark.parametrize("force", [False, True])
def test_clear_reclaims_staging_without_touching_borrow(l1, force):
    k = key(0)
    for tag in ("winner", "abandoned"):
        l1.reserve_write([k], [False], LAYOUT, tag=tag)
    l1.finish_write([k], tag="winner")
    read = l1.borrow_read([k])
    read.objects[k].tensor.fill_(37)
    if not force:
        time.sleep(1.1)
    l1.clear(force=force)
    assert l1.get_staging_memory_usage() == 0
    assert l1.get_memory_usage()[0] == PAGE
    assert_contents(read)
    assert l1.release_borrow(read.token)


def test_empty_borrow_and_prepare_close_stop_new_admission(l1):
    read = l1.borrow_read([])
    assert read.objects == {}
    with pytest.raises(RuntimeError, match="borrow"):
        l1.prepare_close()
    assert l1.release_borrow(read.token)
    staged = key(0)
    l1.reserve_write([staged], [False], LAYOUT)
    l1.prepare_close()
    for operation in (
        lambda: l1.borrow_read([]),
        lambda: l1.reserve_write([key(1)], [False], LAYOUT),
        lambda: l1.finish_write([staged]),
        lambda: l1.finish_write_and_reserve_read([staged]),
        lambda: l1.reserve_read([staged]),
        lambda: l1.unsafe_read([staged]),
    ):
        with pytest.raises(RuntimeError, match="clos"):
            operation()
    l1.close()


def test_close_rejects_before_dependent_context_close(l1, monkeypatch):
    publish(l1, [key(0)])
    read = l1.borrow_read([key(0)])
    close_context = Mock()
    monkeypatch.setattr(
        "lmcache.v1.distributed.l1_manager.close_l1_gds_context", close_context
    )
    with pytest.raises(RuntimeError, match="borrow"):
        l1.close()
    close_context.assert_not_called()
    assert_contents(read)
    assert l1.release_borrow(read.token)

    def drain_with_completion():
        # A device drain must allow callbacks to enter the L1 critical section.
        assert l1._lock.acquire(blocking=False)
        l1._lock.release()

    close_context.side_effect = lambda _: drain_with_completion()
    l1.close()
    with pytest.raises(RuntimeError, match="clos"):
        l1.borrow_read([key(0)])


def test_storage_manager_single_l1_api_and_close_gate(l1):
    # Exercise real routing/close preflight without spawning background controllers.
    storage = StorageManager.__new__(StorageManager)
    storage._l1_manager = l1
    storage._l1_managers_by_id = {l1.l1_manager_id: l1}
    storage._prefetch_controller = Mock()
    publish(l1, [key(0)])
    read = storage.borrow_read([key(0)])
    with pytest.raises(RuntimeError, match="borrow"):
        storage.close()
    storage._prefetch_controller.stop.assert_not_called()
    assert storage.release_borrow(read.token)
    storage._l1_managers_by_id[-1] = object()
    with pytest.raises(ValueError, match="single L1"):
        storage.borrow_read([key(0)])
    with pytest.raises(ValueError, match="single L1"):
        storage.release_borrow(read.token)
