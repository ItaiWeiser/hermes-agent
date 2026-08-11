import errno
import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tests.attempt_fence_helpers import (
    create_bound_attempt,
    isolated_home,
    logical_board_snapshot,
    process_tuple,
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


def test_host_identity_returns_none_on_hostname_failure(monkeypatch):
    from hermes_cli import process_bootstrap

    def fail_hostname():
        raise OSError("hostname unavailable")

    monkeypatch.setattr(process_bootstrap.socket, "gethostname", fail_hostname)
    assert process_bootstrap._host_id() is None


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


@darwin_only
def test_claim_task_rejects_fenced_retry_state_with_zero_delta(
    registered_current_process,
):
    fixture = registered_current_process
    fixture.conn.execute(
        "UPDATE tasks SET status='ready' WHERE id=?",
        (fixture.task_id,),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)

    assert (
        kb.claim_task(fixture.conn, fixture.task_id, claimer="second:claim")
        is None
    )
    assert logical_board_snapshot(fixture.conn) == before


@darwin_only
def test_claim_with_fence_precedes_dependency_demote_and_run_cleanup(
    registered_current_process,
):
    fixture = registered_current_process
    parent_id = kb.create_task(fixture.conn, title="unfinished parent")
    fixture.conn.execute(
        "UPDATE tasks SET status='todo' WHERE id=?",
        (parent_id,),
    )
    kb.link_tasks(fixture.conn, parent_id, fixture.task_id)
    fixture.conn.execute(
        "UPDATE tasks SET status='ready' WHERE id=?",
        (fixture.task_id,),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)

    assert kb.claim_task(
        fixture.conn,
        fixture.task_id,
        claimer="second:claim",
    ) is None
    assert logical_board_snapshot(fixture.conn) == before


@pytest.mark.parametrize("non_ready_status", ["todo", "blocked", "review", "done"])
@darwin_only
def test_claim_task_rejects_any_fenced_non_ready_status_with_zero_delta(
    registered_current_process,
    non_ready_status,
):
    fixture = registered_current_process
    parent_id = kb.create_task(fixture.conn, title="unfinished parent")
    fixture.conn.execute(
        "UPDATE tasks SET status='todo' WHERE id=?",
        (parent_id,),
    )
    kb.link_tasks(fixture.conn, parent_id, fixture.task_id)
    fixture.conn.execute(
        "UPDATE tasks SET status=? WHERE id=?",
        (non_ready_status, fixture.task_id),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)

    assert kb.claim_task(
        fixture.conn,
        fixture.task_id,
        claimer="second:claim",
    ) is None
    assert logical_board_snapshot(fixture.conn) == before


@darwin_only
def test_claim_review_task_rejects_fenced_retry_state_with_zero_delta(
    registered_current_process,
):
    fixture = registered_current_process
    fixture.conn.execute(
        "UPDATE tasks SET status='review', claim_lock=NULL, claim_expires=NULL "
        "WHERE id=?",
        (fixture.task_id,),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)

    assert (
        kb.claim_review_task(
            fixture.conn,
            fixture.task_id,
            claimer="second:review",
        )
        is None
    )
    assert logical_board_snapshot(fixture.conn) == before


def _run_process_tuple(conn, run_id):
    row = conn.execute(
        "SELECT claim_lock, claim_expires, worker_pid, worker_pgid, "
        "worker_identity, worker_fence FROM task_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    return tuple(row)


@pytest.mark.parametrize("operation", ["complete", "block", "gave_up"])
@darwin_only
def test_terminal_transition_preserves_fenced_process_tuple(
    registered_current_process,
    operation,
):
    fixture = registered_current_process
    run_id = fixture.claimed.current_run_id
    before_task_tuple = process_tuple(kb.get_task(fixture.conn, fixture.task_id))
    before_run_tuple = _run_process_tuple(fixture.conn, run_id)

    if operation == "complete":
        assert kb.complete_task(
            fixture.conn,
            fixture.task_id,
            result="finished",
            expected_run_id=run_id,
        )
        expected_status = "done"
        expected_outcome = "completed"
    elif operation == "block":
        assert kb.block_task(
            fixture.conn,
            fixture.task_id,
            reason="needs input",
            expected_run_id=run_id,
        )
        expected_status = "blocked"
        expected_outcome = "blocked"
    else:
        assert kb._record_task_failure(
            fixture.conn,
            fixture.task_id,
            "spawn failed",
            outcome="spawn_failed",
            force_trip=True,
            release_claim=True,
            end_run=True,
        )
        expected_status = "blocked"
        expected_outcome = "gave_up"

    task = kb.get_task(fixture.conn, fixture.task_id)
    run = fixture.conn.execute(
        "SELECT outcome FROM task_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    assert task.status == expected_status
    assert run["outcome"] == expected_outcome
    assert process_tuple(task) == before_task_tuple
    assert _run_process_tuple(fixture.conn, run_id) == before_run_tuple
    assert kb.reap_terminal_attempt_fences(fixture.conn, limit=16) == []
    before_late_call = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        kb.add_comment(fixture.conn, fixture.task_id, "dor-coo", "late")
    assert logical_board_snapshot(fixture.conn) == before_late_call


@darwin_only
def test_dependency_block_and_promotion_stay_hidden_until_exact_reap(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    run_id = fixture.claimed.current_run_id
    parent_id = kb.create_task(fixture.conn, title="dependency")
    fixture.conn.execute(
        "UPDATE tasks SET status='todo' WHERE id=?",
        (parent_id,),
    )
    kb.link_tasks(fixture.conn, parent_id, fixture.task_id)
    before_tuple = process_tuple(kb.get_task(fixture.conn, fixture.task_id))

    assert kb.block_task(
        fixture.conn,
        fixture.task_id,
        reason="waiting for dependency",
        kind="dependency",
        expected_run_id=run_id,
    )
    task = kb.get_task(fixture.conn, fixture.task_id)
    run = fixture.conn.execute(
        "SELECT outcome, metadata FROM task_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    assert task.status == "running"
    assert process_tuple(task) == before_tuple
    assert run["outcome"] == "blocked"
    assert json.loads(run["metadata"])["retry_status"] == "todo"
    assert kb.reap_terminal_attempt_fences(fixture.conn) == []

    fixture.conn.execute(
        "UPDATE tasks SET status='done' WHERE id=?",
        (parent_id,),
    )
    fixture.conn.commit()
    before_promotion = logical_board_snapshot(fixture.conn)
    assert kb.recompute_ready(fixture.conn) == 0
    assert logical_board_snapshot(fixture.conn) == before_promotion

    monkeypatch.setattr(kb, "_fenced_group_state", lambda _fence: "dead")
    assert kb.reap_terminal_attempt_fences(fixture.conn) == [
        ("task", fixture.task_id)
    ]
    assert kb.get_task(fixture.conn, fixture.task_id).status == "todo"
    assert kb.recompute_ready(fixture.conn) == 1
    assert kb.get_task(fixture.conn, fixture.task_id).status == "ready"


@pytest.mark.parametrize(
    ("kill_error", "expected"),
    [
        (None, "alive"),
        (PermissionError(), "unknown"),
        (OSError(errno.EINVAL, "uncertain"), "unknown"),
        (ProcessLookupError(), "dead"),
    ],
)
def test_fenced_group_state_is_conservative(monkeypatch, kill_error, expected):
    stored = kb.DarwinProcessIdentity(900001, 900001, 1, 2)
    fence = {
        "host": kb._host_id(),
        "leader_pid": stored.pid,
        "worker_pgid": stored.pgid,
        "worker_identity": stored.token,
    }
    monkeypatch.setattr(kb, "_darwin_process_identity", lambda _pid: None)

    def probe(_pgid, _signal):
        if kill_error is not None:
            raise kill_error

    monkeypatch.setattr(os, "killpg", probe)
    assert kb._fenced_group_state(fence) == expected


def test_matching_leader_identity_is_alive_without_group_probe(monkeypatch):
    stored = kb.DarwinProcessIdentity(900001, 900001, 1, 2)
    fence = {
        "host": kb._host_id(),
        "leader_pid": stored.pid,
        "worker_pgid": stored.pgid,
        "worker_identity": stored.token,
    }
    monkeypatch.setattr(kb, "_darwin_process_identity", lambda _pid: stored)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda *_args: pytest.fail("matching leader must short-circuit"),
    )
    assert kb._fenced_group_state(fence) == "alive"


@pytest.mark.parametrize("host_value", ["current-host", None])
def test_fenced_group_state_fails_closed_for_unknown_or_foreign_host(
    monkeypatch,
    host_value,
):
    stored = kb.DarwinProcessIdentity(900001, 900001, 1, 2)
    fence = {
        "host": "stored-host",
        "leader_pid": stored.pid,
        "worker_pgid": stored.pgid,
        "worker_identity": stored.token,
    }
    monkeypatch.setattr(kb, "_host_id", lambda: host_value)
    monkeypatch.setattr(
        os,
        "killpg",
        lambda *_args: pytest.fail("foreign/unknown host must not be probed"),
    )
    assert kb._fenced_group_state(fence) == "unknown"


@darwin_only
def test_dead_leader_with_live_group_child_is_not_reaped(isolated_home):
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess,sys,time; "
            "p=subprocess.Popen(['/bin/sleep','60']); "
            "print(p.pid, flush=True); time.sleep(60)",
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert leader.stdout is not None
    child_pid = int(leader.stdout.readline().strip())
    identity = kb._darwin_process_identity(leader.pid)
    assert identity is not None and child_pid > 0
    conn = kb.connect()
    task_id, _claimed, raw = create_bound_attempt(conn, leader_identity=identity)
    try:
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task_id,))
        conn.commit()
        leader.terminate()
        leader.wait(timeout=5)
        assert kb._fenced_group_state(json.loads(raw)) == "alive"
        assert kb.reap_terminal_attempt_fences(conn, limit=16) == []
        assert kb.get_task(conn, task_id).worker_fence == raw
    finally:
        try:
            os.killpg(identity.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@darwin_only
def test_orphan_spawn_bind_failure_reaps_only_old_run(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    old_run_id = fixture.claimed.current_run_id
    fence = json.loads(fixture.raw_fence)
    fence["reason"] = "spawn_bind_failed"
    orphan_raw = json.dumps(fence, sort_keys=True, separators=(",", ":"))
    fixture.conn.execute(
        "UPDATE task_runs SET worker_fence=? WHERE id=?",
        (orphan_raw, old_run_id),
    )
    fixture.conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
        "claim_expires=NULL, worker_pid=NULL, worker_pgid=NULL, "
        "worker_identity=NULL, worker_fence=NULL WHERE id=?",
        (fixture.task_id,),
    )
    fixture.conn.commit()
    newer = kb.claim_task(fixture.conn, fixture.task_id, claimer="new:owner")
    assert newer is not None
    before_task = process_tuple(newer)
    before_new_run = _run_process_tuple(fixture.conn, newer.current_run_id)
    monkeypatch.setattr(kb, "_fenced_group_state", lambda _fence: "dead")

    assert kb.reap_terminal_attempt_fences(fixture.conn, limit=16) == [
        ("run", old_run_id)
    ]
    assert process_tuple(kb.get_task(fixture.conn, fixture.task_id)) == before_task
    assert _run_process_tuple(fixture.conn, newer.current_run_id) == before_new_run
    assert _run_process_tuple(fixture.conn, old_run_id) == (
        None,
        None,
        None,
        None,
        None,
        None,
    )


@darwin_only
def test_dead_fenced_retry_is_materialized_once_after_atomic_reap(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    run_id = fixture.claimed.current_run_id
    assert not kb._record_task_failure(
        fixture.conn,
        fixture.task_id,
        "retry later",
        outcome="spawn_failed",
        failure_limit=3,
        release_claim=True,
        end_run=True,
    )
    task = kb.get_task(fixture.conn, fixture.task_id)
    assert task.status == "running"
    assert task.worker_fence == fixture.raw_fence
    monkeypatch.setattr(kb, "_fenced_group_state", lambda _fence: "dead")

    assert kb.reap_terminal_attempt_fences(fixture.conn, limit=16) == [
        ("task", fixture.task_id)
    ]
    task = kb.get_task(fixture.conn, fixture.task_id)
    assert task.status == "ready"
    assert process_tuple(task) == (None, None, None, None, None, None)
    assert _run_process_tuple(fixture.conn, run_id) == (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    assert kb.reap_terminal_attempt_fences(fixture.conn, limit=16) == []


@darwin_only
def test_reaper_exact_cas_rolls_back_task_clear_when_run_changed(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    run_id = fixture.claimed.current_run_id
    assert kb.complete_task(
        fixture.conn,
        fixture.task_id,
        result="done",
        expected_run_id=run_id,
    )
    before_task = process_tuple(kb.get_task(fixture.conn, fixture.task_id))

    def race_run_identity(_fence):
        fixture.conn.execute(
            "UPDATE task_runs SET worker_identity='raced:identity' WHERE id=?",
            (run_id,),
        )
        fixture.conn.commit()
        return "dead"

    monkeypatch.setattr(kb, "_fenced_group_state", race_run_identity)
    assert kb.reap_terminal_attempt_fences(fixture.conn, limit=16) == []
    assert process_tuple(kb.get_task(fixture.conn, fixture.task_id)) == before_task
    run = fixture.conn.execute(
        "SELECT worker_identity, worker_fence FROM task_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    assert run["worker_identity"] == "raced:identity"
    assert run["worker_fence"] == fixture.raw_fence


@darwin_only
def test_reaper_honors_requested_bound(registered_current_process, monkeypatch):
    fixture = registered_current_process
    identity = kb._darwin_process_identity(os.getpgid(0))
    assert identity is not None
    other_id, _claimed, _raw = create_bound_attempt(
        fixture.conn,
        leader_identity=identity,
    )
    fixture.conn.execute(
        "UPDATE tasks SET status='blocked' WHERE id IN (?, ?)",
        (fixture.task_id, other_id),
    )
    fixture.conn.commit()
    monkeypatch.setattr(kb, "_fenced_group_state", lambda _fence: "dead")

    assert len(kb.reap_terminal_attempt_fences(fixture.conn, limit=1)) == 1
    remaining = fixture.conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE worker_fence IS NOT NULL"
    ).fetchone()[0]
    assert remaining == 1


@darwin_only
def test_reaper_cursor_prevents_live_prefix_starvation_and_reaches_orphan(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    identity = kb._darwin_process_identity(os.getpgid(0))
    assert identity is not None
    attempts = [
        (fixture.task_id, fixture.claimed, fixture.raw_fence),
        create_bound_attempt(fixture.conn, leader_identity=identity),
        create_bound_attempt(fixture.conn, leader_identity=identity),
    ]
    attempts.sort(key=lambda item: item[0])

    def set_reason(task_id, claimed, raw, reason):
        fence = json.loads(raw)
        fence["reason"] = reason
        updated = json.dumps(fence, sort_keys=True, separators=(",", ":"))
        fixture.conn.execute(
            "UPDATE tasks SET status='blocked', worker_fence=? WHERE id=?",
            (updated, task_id),
        )
        fixture.conn.execute(
            "UPDATE task_runs SET worker_fence=? WHERE id=?",
            (updated, claimed.current_run_id),
        )
        return updated

    alive_id, alive_claimed, alive_raw = attempts[0]
    unknown_id, unknown_claimed, unknown_raw = attempts[1]
    dead_id, dead_claimed, dead_raw = attempts[2]
    alive_raw = set_reason(alive_id, alive_claimed, alive_raw, "fair_alive")
    unknown_raw = set_reason(
        unknown_id,
        unknown_claimed,
        unknown_raw,
        "fair_unknown",
    )
    set_reason(dead_id, dead_claimed, dead_raw, "fair_dead")

    orphan_task_id, orphan_claimed, orphan_raw = create_bound_attempt(
        fixture.conn,
        leader_identity=identity,
    )
    orphan_fence = json.loads(orphan_raw)
    orphan_fence["reason"] = "spawn_bind_failed"
    orphan_raw = json.dumps(
        orphan_fence,
        sort_keys=True,
        separators=(",", ":"),
    )
    fixture.conn.execute(
        "UPDATE task_runs SET worker_fence=? WHERE id=?",
        (orphan_raw, orphan_claimed.current_run_id),
    )
    fixture.conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
        "claim_expires=NULL, worker_pid=NULL, worker_pgid=NULL, "
        "worker_identity=NULL, worker_fence=NULL WHERE id=?",
        (orphan_task_id,),
    )
    fixture.conn.commit()
    newer = kb.claim_task(fixture.conn, orphan_task_id, claimer="new:owner")
    assert newer is not None
    newer_tuple = process_tuple(newer)

    def state_for_reason(fence):
        return {
            "fair_alive": "alive",
            "fair_unknown": "unknown",
            "fair_dead": "dead",
            "spawn_bind_failed": "dead",
        }[fence["reason"]]

    monkeypatch.setattr(kb, "_fenced_group_state", state_for_reason)
    observed = []
    for _ in range(4):
        batch = kb.reap_terminal_attempt_fences(fixture.conn, limit=2)
        assert len(batch) <= 2
        observed.extend(batch)

    assert ("task", dead_id) in observed
    assert ("run", orphan_claimed.current_run_id) in observed
    assert kb.get_task(fixture.conn, alive_id).worker_fence == alive_raw
    assert kb.get_task(fixture.conn, unknown_id).worker_fence == unknown_raw
    assert kb.get_task(fixture.conn, dead_id).worker_fence is None
    assert _run_process_tuple(
        fixture.conn,
        orphan_claimed.current_run_id,
    ) == (None, None, None, None, None, None)
    assert process_tuple(kb.get_task(fixture.conn, orphan_task_id)) == newer_tuple


def test_dispatch_tick_calls_reaper_once_and_dry_run_never_calls_it(
    isolated_home,
    monkeypatch,
):
    conn = kb.connect()
    real = kb.reap_terminal_attempt_fences
    calls = []

    def counted(conn, *, limit=kb.TERMINAL_FENCE_REAP_LIMIT):
        calls.append(limit)
        return real(conn, limit=limit)

    monkeypatch.setattr(kb, "reap_terminal_attempt_fences", counted)
    kb.dispatch_once(conn, dry_run=True)
    assert calls == []
    kb.dispatch_once(conn, dry_run=False)
    assert calls == [kb.TERMINAL_FENCE_REAP_LIMIT]


@darwin_only
def test_request_review_stays_hidden_until_fenced_group_is_dead(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    run_id = fixture.claimed.current_run_id
    before_tuple = process_tuple(kb.get_task(fixture.conn, fixture.task_id))

    assert kb.request_review(
        fixture.conn,
        fixture.task_id,
        summary="ready for review",
        expected_run_id=run_id,
    )
    task = kb.get_task(fixture.conn, fixture.task_id)
    run = fixture.conn.execute(
        "SELECT outcome, metadata FROM task_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    assert task.status == "running"
    assert process_tuple(task) == before_tuple
    assert run["outcome"] == "review_requested"
    assert json.loads(run["metadata"])["retry_status"] == "review"

    monkeypatch.setattr(kb, "_fenced_group_state", lambda _fence: "dead")
    assert kb.reap_terminal_attempt_fences(fixture.conn) == [
        ("task", fixture.task_id)
    ]
    assert kb.get_task(fixture.conn, fixture.task_id).status == "review"


@darwin_only
def test_request_changes_stays_hidden_until_fenced_group_is_dead(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    run_id = fixture.claimed.current_run_id
    claimed = fixture.conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id=? AND run_id=? "
        "AND kind='claimed'",
        (fixture.task_id, run_id),
    ).fetchone()
    payload = json.loads(claimed["payload"])
    payload["source_status"] = "review"
    fixture.conn.execute(
        "UPDATE task_events SET payload=? WHERE id=?",
        (json.dumps(payload), claimed["id"]),
    )
    kb._append_event(
        fixture.conn,
        fixture.task_id,
        "review_requested",
        {"implementer": "dor-coo", "reviewer": "yonatan"},
    )
    fixture.conn.execute(
        "UPDATE tasks SET assignee='yonatan' WHERE id=?",
        (fixture.task_id,),
    )
    fixture.conn.commit()
    before_tuple = process_tuple(kb.get_task(fixture.conn, fixture.task_id))

    ok, implementer = kb.request_changes(
        fixture.conn,
        fixture.task_id,
        reason="please revise",
        expected_run_id=run_id,
    )
    assert ok and implementer == "dor-coo"
    task = kb.get_task(fixture.conn, fixture.task_id)
    run = fixture.conn.execute(
        "SELECT outcome, metadata FROM task_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    assert task.status == "running"
    assert process_tuple(task) == before_tuple
    assert run["outcome"] == "changes_requested"
    assert json.loads(run["metadata"])["retry_status"] == "ready"

    monkeypatch.setattr(kb, "_fenced_group_state", lambda _fence: "dead")
    assert kb.reap_terminal_attempt_fences(fixture.conn) == [
        ("task", fixture.task_id)
    ]
    task = kb.get_task(fixture.conn, fixture.task_id)
    assert task.status == "ready"
    assert task.assignee == "dor-coo"


def test_claim_dependency_demote_fault_rolls_back_full_snapshot(
    isolated_home,
    monkeypatch,
):
    conn = kb.connect()
    parent_id = kb.create_task(conn, title="unfinished parent")
    child_id = kb.create_task(conn, title="ready child")
    kb.link_tasks(conn, parent_id, child_id)
    conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (parent_id,))
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child_id,))
    conn.commit()
    before = logical_board_snapshot(conn)
    real_append = kb._append_event

    def fail_after_demote(conn, task_id, kind, payload=None, *, run_id=None):
        if kind == "claim_rejected":
            raise RuntimeError("fault after dependency demotion")
        return real_append(conn, task_id, kind, payload, run_id=run_id)

    monkeypatch.setattr(kb, "_append_event", fail_after_demote)
    with pytest.raises(RuntimeError, match="fault after dependency demotion"):
        kb.claim_task(conn, child_id, claimer="fault:claim")
    assert logical_board_snapshot(conn) == before


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
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()
    assert logical_board_snapshot(fixture.conn) == before


@darwin_only
def test_registration_discovery_rejects_claim_mismatch_with_zero_delta(
    registered_current_process,
):
    fixture = registered_current_process
    fixture.conn.execute(
        "UPDATE tasks SET claim_lock='stale:claim' WHERE id=?",
        (fixture.task_id,),
    )
    fixture.conn.execute(
        "UPDATE task_runs SET claim_lock='stale:claim' WHERE id=?",
        (fixture.claimed.current_run_id,),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()
    assert logical_board_snapshot(fixture.conn) == before


@darwin_only
def test_registration_discovery_rejects_terminal_status_with_zero_delta(
    registered_current_process,
):
    fixture = registered_current_process
    fixture.conn.execute(
        "UPDATE tasks SET status='done' WHERE id=?",
        (fixture.task_id,),
    )
    fixture.conn.execute(
        "UPDATE task_runs SET status='done' WHERE id=?",
        (fixture.claimed.current_run_id,),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()
    assert logical_board_snapshot(fixture.conn) == before


@darwin_only
def test_registration_discovery_rejects_stale_leader_with_zero_delta(
    registered_current_process,
):
    fixture = registered_current_process
    fence = json.loads(fixture.raw_fence)
    stale_identity = f"darwin:{fence['leader_pid']}:0:0"
    fence["worker_identity"] = stale_identity
    raw_fence = json.dumps(fence, sort_keys=True, separators=(",", ":"))
    fixture.conn.execute(
        "UPDATE tasks SET worker_identity=?, worker_fence=? WHERE id=?",
        (stale_identity, raw_fence, fixture.task_id),
    )
    fixture.conn.execute(
        "UPDATE task_runs SET worker_identity=?, worker_fence=? WHERE id=?",
        (stale_identity, raw_fence, fixture.claimed.current_run_id),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()
    assert logical_board_snapshot(fixture.conn) == before


@darwin_only
def test_missing_host_identity_cannot_match_unknown_fence(
    registered_current_process,
    monkeypatch,
):
    from hermes_cli import process_bootstrap

    fixture = registered_current_process
    fence = json.loads(fixture.raw_fence)
    fence["host"] = "unknown"
    raw_fence = json.dumps(fence, sort_keys=True, separators=(",", ":"))
    fixture.conn.execute(
        "UPDATE tasks SET worker_fence=? WHERE id=?",
        (raw_fence, fixture.task_id),
    )
    fixture.conn.execute(
        "UPDATE task_runs SET worker_fence=? WHERE id=?",
        (raw_fence, fixture.claimed.current_run_id),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)

    def fail_hostname():
        raise OSError("hostname unavailable")

    monkeypatch.setattr(process_bootstrap.socket, "gethostname", fail_hostname)
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()
    assert logical_board_snapshot(fixture.conn) == before


@darwin_only
def test_registration_discovery_rejects_foreign_host_with_zero_delta(
    registered_current_process,
):
    fixture = registered_current_process
    fence = json.loads(fixture.raw_fence)
    fence["host"] = "foreign-host"
    raw_fence = json.dumps(fence, sort_keys=True, separators=(",", ":"))
    fixture.conn.execute(
        "UPDATE tasks SET worker_fence=? WHERE id=?",
        (raw_fence, fixture.task_id),
    )
    fixture.conn.execute(
        "UPDATE task_runs SET worker_fence=? WHERE id=?",
        (raw_fence, fixture.claimed.current_run_id),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()
    assert logical_board_snapshot(fixture.conn) == before


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        pytest.param("run_id", True, id="run-id-bool"),
        pytest.param("run_id", 0, id="run-id-zero"),
        pytest.param("claim_lock", "", id="claim-lock-empty"),
        pytest.param("host", "", id="host-empty"),
        pytest.param("leader_pid", True, id="leader-pid-bool"),
        pytest.param("leader_pid", 0, id="leader-pid-zero"),
        pytest.param("worker_pgid", True, id="worker-pgid-bool"),
        pytest.param("worker_pgid", -1, id="worker-pgid-negative"),
        pytest.param("worker_identity", "", id="worker-identity-empty"),
        pytest.param("reason", "", id="reason-empty"),
        pytest.param("created_at", True, id="created-at-bool"),
        pytest.param("created_at", "1", id="created-at-string"),
    ],
)
@darwin_only
def test_provenance_rejects_malformed_fence_fields_with_zero_delta(
    registered_current_process,
    field,
    malformed,
):
    from hermes_cli import process_bootstrap

    fixture = registered_current_process
    caller_identity = kb._darwin_process_identity(os.getpid())
    assert caller_identity is not None
    row = dict(
        process_bootstrap._read_registration_rows(
            fixture.board_path,
            os.getpgid(0),
        )[0]
    )
    fence = json.loads(row["task_worker_fence"])
    fence[field] = malformed
    raw_fence = json.dumps(fence, sort_keys=True, separators=(",", ":"))
    row["task_worker_fence"] = raw_fence
    row["run_worker_fence"] = raw_fence
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        process_bootstrap._validated_provenance(
            fixture.board_path,
            row,
            caller_identity,
        )
    assert logical_board_snapshot(fixture.conn) == before


@pytest.mark.parametrize(
    "case",
    [
        "run-id-bool",
        "run-id-string",
        "claim-lock-int",
        "leader-pid-string",
        "leader-pid-invalid-string",
        "worker-pgid-string",
        "worker-identity-bytes",
        "task-id-bytes",
    ],
)
@darwin_only
def test_provenance_rejects_malformed_row_fields_with_zero_delta(
    registered_current_process,
    case,
):
    from hermes_cli import process_bootstrap

    fixture = registered_current_process
    caller_identity = kb._darwin_process_identity(os.getpid())
    assert caller_identity is not None
    row = dict(
        process_bootstrap._read_registration_rows(
            fixture.board_path,
            os.getpgid(0),
        )[0]
    )
    fence = json.loads(row["task_worker_fence"])
    if case == "run-id-bool":
        malformed = True
        row.update(task_run_id=malformed, run_id=malformed)
        fence["run_id"] = malformed
    elif case == "run-id-string":
        malformed = str(row["task_run_id"])
        row.update(task_run_id=malformed, run_id=malformed)
        fence["run_id"] = malformed
    elif case == "claim-lock-int":
        malformed = 7
        row.update(task_claim_lock=malformed, run_claim_lock=malformed)
        fence["claim_lock"] = malformed
    elif case in {"leader-pid-string", "leader-pid-invalid-string"}:
        malformed = (
            str(row["task_worker_pid"]) if case == "leader-pid-string" else "not-an-int"
        )
        row.update(task_worker_pid=malformed, run_worker_pid=malformed)
        fence["leader_pid"] = malformed
    elif case == "worker-pgid-string":
        malformed = str(row["task_worker_pgid"])
        row.update(task_worker_pgid=malformed, run_worker_pgid=malformed)
        fence["worker_pgid"] = malformed
    elif case == "worker-identity-bytes":
        malformed = row["task_worker_identity"].encode()
        row.update(
            task_worker_identity=malformed,
            run_worker_identity=malformed,
        )
        fence["worker_identity"] = "not-the-row-bytes"
    else:
        malformed = row["task_id"].encode()
        row.update(task_id=malformed, run_task_id=malformed)
    raw_fence = json.dumps(fence, sort_keys=True, separators=(",", ":"))
    row["task_worker_fence"] = raw_fence
    row["run_worker_fence"] = raw_fence
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        process_bootstrap._validated_provenance(
            fixture.board_path,
            row,
            caller_identity,
        )
    assert logical_board_snapshot(fixture.conn) == before


@darwin_only
def test_registration_discovery_rejects_cross_task_run_with_zero_delta(
    registered_current_process,
):
    fixture = registered_current_process
    identity = kb._darwin_process_identity(os.getpgid(0))
    assert identity is not None
    other_task_id, other_claimed, other_fence = create_bound_attempt(
        fixture.conn,
        leader_identity=identity,
    )
    fixture.conn.execute(
        "UPDATE tasks SET current_run_id=?, claim_lock=?, worker_pid=?, "
        "worker_pgid=?, worker_identity=?, worker_fence=? WHERE id=?",
        (
            other_claimed.current_run_id,
            other_claimed.claim_lock,
            identity.pid,
            identity.pgid,
            identity.token,
            other_fence,
            fixture.task_id,
        ),
    )
    fixture.conn.execute(
        "UPDATE tasks SET worker_pgid=NULL, worker_fence=NULL WHERE id=?",
        (other_task_id,),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.StaleAttemptError):
        kb._discover_current_worker_registration()
    assert logical_board_snapshot(fixture.conn) == before


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


def test_board_inventory_fails_closed_when_board_root_stat_is_unreadable(
    isolated_home,
    monkeypatch,
):
    from hermes_cli import process_bootstrap

    target = kb.boards_root()
    real_stat = Path.stat

    def guarded_stat(path, *args, **kwargs):
        if path == target:
            raise OSError("board root unreadable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(kb.StaleAttemptError):
        process_bootstrap._canonical_board_db_paths()


def test_board_inventory_fails_closed_when_board_db_stat_is_unreadable(
    isolated_home,
    monkeypatch,
):
    from hermes_cli import process_bootstrap

    target = kb.kanban_home() / "kanban.db"
    real_stat = Path.stat

    def guarded_stat(path, *args, **kwargs):
        if path == target:
            raise OSError("board DB unreadable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    with pytest.raises(kb.StaleAttemptError):
        process_bootstrap._canonical_board_db_paths()


def test_board_inventory_fails_closed_when_directory_listing_fails(
    isolated_home,
    monkeypatch,
):
    from hermes_cli import process_bootstrap

    root = kb.boards_root()
    root.mkdir(parents=True, exist_ok=True)
    real_iterdir = Path.iterdir

    def guarded_iterdir(path):
        if path == root:
            raise OSError("board listing unreadable")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    with pytest.raises(kb.StaleAttemptError):
        process_bootstrap._canonical_board_db_paths()
