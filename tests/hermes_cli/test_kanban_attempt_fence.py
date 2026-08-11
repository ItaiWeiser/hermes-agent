import os
import sqlite3
import sys

import pytest

from hermes_cli import kanban_db as kb
from tests.attempt_fence_helpers import (
    create_bound_attempt,
    isolated_home,
    logical_board_snapshot,
    registered_current_process,
)


darwin_only = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="libproc fence is macOS-only",
)


@darwin_only
def test_libproc_identity_has_microsecond_start_and_pgid():
    identity = kb._darwin_process_identity(os.getpid())
    assert identity is not None
    assert identity.pid == os.getpid()
    assert identity.pgid == os.getpgid(0)
    assert identity.start_tvsec > 0
    assert 0 <= identity.start_tvusec < 1_000_000
    assert identity.token == (
        f"darwin:{identity.pid}:{identity.start_tvsec}:{identity.start_tvusec}"
    )


def test_libproc_identity_mismatch_rejects_pid_reuse(monkeypatch):
    old = kb.DarwinProcessIdentity(42, 42, 100, 10)
    monkeypatch.setattr(
        kb,
        "_darwin_process_identity",
        lambda _pid: kb.DarwinProcessIdentity(42, 42, 100, 11),
    )
    assert kb._identity_matches(old) is False


@darwin_only
def test_libproc_identity_returns_none_on_binding_error(monkeypatch):
    from hermes_cli import process_bootstrap

    def fail_to_load(*_args, **_kwargs):
        raise RuntimeError("libproc unavailable")

    monkeypatch.setattr(process_bootstrap.ctypes, "CDLL", fail_to_load)
    assert process_bootstrap._darwin_process_identity(os.getpid()) is None


def test_non_darwin_dispatch_fails_before_claim_with_zero_delta(
    isolated_home,
    monkeypatch,
):
    from hermes_cli import process_bootstrap

    conn = kb.connect()
    task_id = kb.create_task(conn, title="portable", assignee="dor-coo")
    before = logical_board_snapshot(conn)
    monkeypatch.setattr(process_bootstrap.sys, "platform", "linux")
    with pytest.raises(kb.AttemptFenceCapabilityError):
        kb.dispatch_once(conn, dry_run=False)
    assert logical_board_snapshot(conn) == before
    assert kb.get_task(conn, task_id).status == "ready"


def test_fence_columns_are_durable_on_tasks_and_runs(isolated_home):
    with kb.connect() as conn:
        for table in ("tasks", "task_runs"):
            columns = {
                row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
            }
            assert {"worker_pgid", "worker_identity", "worker_fence"} <= columns
        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(tasks)")}
        assert "idx_tasks_worker_pgid" in indexes


@darwin_only
def test_registration_discovery_survives_authority_env_removal(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    provenance = kb._discover_current_worker_registration()
    assert provenance is not None
    assert provenance.task_id == fixture.task_id
    assert provenance.run_id == fixture.claimed.current_run_id
    assert provenance.claim_lock == fixture.claimed.claim_lock
    assert provenance.raw_fence == fixture.raw_fence
    assert provenance.caller_pid == os.getpid()
    assert provenance.caller_pgid == os.getpgid(0)


@darwin_only
def test_registration_discovery_rejects_multiple_rows(
    registered_current_process,
):
    fixture = registered_current_process
    identity = kb._darwin_process_identity(os.getpgid(0))
    assert identity is not None
    create_bound_attempt(fixture.conn, leader_identity=identity)
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()


@darwin_only
def test_registration_discovery_rejects_task_run_fence_mismatch(
    registered_current_process,
):
    fixture = registered_current_process
    fixture.conn.execute(
        "UPDATE task_runs SET worker_fence='{}' WHERE id=?",
        (fixture.claimed.current_run_id,),
    )
    fixture.conn.commit()
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()


@darwin_only
def test_registration_inventory_overflow_has_zero_filesystem_delta(
    isolated_home,
):
    root = kb.boards_root()
    for index in range(256):
        path = root / f"board-{index:03d}" / "kanban.db"
        path.parent.mkdir(parents=True)
        sqlite3.connect(path).close()
    paths = sorted([kb.kanban_home() / "kanban.db", *root.glob("*/kanban.db")])
    assert len(paths) == 257
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    with pytest.raises(kb.AttemptFenceInventoryOverflow):
        kb._discover_current_worker_registration()
    after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert after == before


@darwin_only
def test_registration_discovery_ignores_board_symlink_outside_root(
    isolated_home,
):
    outside = isolated_home.parent / "outside-board"
    outside.mkdir()
    sqlite3.connect(outside / "kanban.db").close()
    link = kb.boards_root() / "outside"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    assert kb._discover_current_worker_registration() is None
