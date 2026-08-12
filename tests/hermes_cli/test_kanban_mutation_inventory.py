import ast
import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tests.attempt_fence_helpers import (
    isolated_home,
    logical_board_snapshot,
    process_tuple,
    registered_current_process,
)


CLASSIFIED_INTERNAL_EFFECTS = frozenset(
    {
        "_validate_current_worker_attempt_locked",
        "_inherit_notify_subs",
        "_append_event",
        "_end_run",
        "_synthesize_ended_run",
        "_insert_completion_attachment",
        "_reclaim_proven_dead_fenced_attempt",
        "_reclaim_dangling_run",
        "_bind_pending_worker",
        "_record_and_abort_failed_bind",
        "_record_post_bind_release_failure",
        "_defer_reclaim_for_live_worker",
        "_set_worker_pid",
        "_clear_failure_counter",
        "_retag_legacy_worker_sessions",
        "_terminate_reclaimed_worker",
        "_reserve_terminal_fence_candidates",
        "_persist_scratch_completion_artifacts",
        "_cleanup_workspace",
        "_try_cleanup_parent_workspaces",
        "_mark_scratch_tip_shown",
        "_maybe_emit_scratch_tip",
        "_ensure_git_worktree",
        "_rotate_worker_log",
        "_cleanup_worker_tmux",
        "_default_spawn",
        "_spawn_behind_bootstrap",
        "_dispatch_once_locked",
        "_control_plane_write_txn",
        "_authorized_write_txn",
        "resolve_workspace",
        "build_worker_context",
    }
)

CLASSIFIED_INFRASTRUCTURE_EFFECTS = frozenset(
    {
        "connect",
        "connect_closing",
        "write_txn",
        "_cross_process_init_lock",
        "_dispatch_tick_lock",
        "_maybe_checkpoint_wal",
        "_prune_corrupt_backups",
        "_backup_corrupt_db",
        "_attempt_index_reindex_repair",
        "_migrate_add_optional_columns",
        "_rebuild_drifted_tables",
    }
)

CLASSIFIED_PURE_FALSE_POSITIVES = frozenset(
    {
        "_default_board_display_name",
        "_safe_attachment_name",
        "_fenced_group_state",
        "_to_epoch",
        "_check_file_length_invariant",
        "_guard_existing_db_is_healthy",
        "count_notify_subs",
        "_git_branch_exists",
        "_git_current_branch",
    }
)

EXPLICIT_PRIVATE_ENTRY_EFFECTS = frozenset({"_record_task_failure"})

CLASSIFIED_EXTERNAL_INTERNAL_EFFECT_EDGES = frozenset(
    {
        "tools/kanban_tools.py:build_worker_context",
        "hermes_cli/kanban_swarm.py:_synthesize_ended_run",
        "hermes_cli/kanban_swarm.py:_append_event",
        "hermes_cli/kanban.py:resolve_workspace",
        "hermes_cli/kanban.py:build_worker_context",
        "plugins/kanban/dashboard/plugin_api.py:_safe_attachment_name",
        "plugins/kanban/dashboard/plugin_api.py:_terminate_reclaimed_worker",
        "plugins/kanban/dashboard/plugin_api.py:_end_run",
        "tui_gateway/server.py:count_notify_subs",
        "gateway/kanban_watchers.py:count_notify_subs",
    }
)

CLASSIFIED_PROCESS_FD_EFFECTS = frozenset({
    "hermes_cli/kanban_db.py:PendingWorkerProcess._close_gate",
    "hermes_cli/kanban_db.py:PendingWorkerProcess._group_has_live_members",
    "hermes_cli/kanban_db.py:PendingWorkerProcess.abort",
    "hermes_cli/kanban_db.py:PendingWorkerProcess.release",
    "hermes_cli/kanban_db.py:_cleanup_worker_tmux",
    "hermes_cli/kanban_db.py:_ensure_git_worktree",
    "hermes_cli/kanban_db.py:_fenced_group_state",
    "hermes_cli/kanban_db.py:_git_branch_exists",
    "hermes_cli/kanban_db.py:_git_common_dir",
    "hermes_cli/kanban_db.py:_git_current_branch",
    "hermes_cli/kanban_db.py:_git_dir",
    "hermes_cli/kanban_db.py:_git_toplevel",
    "hermes_cli/kanban_db.py:_pid_alive",
    "hermes_cli/kanban_db.py:_spawn_behind_bootstrap",
    "hermes_cli/kanban_db.py:reap_worker_zombies",
    "hermes_cli/process_bootstrap.py:main",
})

TRANSACTIONAL_CONTROL_PLANE_SURFACES = frozenset(
    {
        "claim_task",
        "claim_review_task",
        "_reserve_terminal_fence_candidates",
        "_reclaim_proven_dead_fenced_attempt",
        "_defer_reclaim_for_live_worker",
        "_set_worker_pid",
        "release_stale_claims",
        "reap_terminal_attempt_fences",
        "reclaim_task",
        "enforce_max_runtime",
        "detect_stale_running",
        "reconcile_orphaned_running",
        "detect_crashed_workers",
        "_dispatch_once_locked",
        "gc_events",
    }
)


def _derived_effect_functions(source: str) -> set[str]:
    module = ast.parse(source)
    filesystem_calls = {
        "mkdir",
        "unlink",
        "replace",
        "rename",
        "rmdir",
        "touch",
        "write_bytes",
        "write_text",
        "move",
        "rmtree",
        "copy",
        "copy2",
        "copytree",
        "kill",
        "killpg",
        "terminate",
        "send_signal",
        "close",
    }

    def is_state_write(call):
        if not isinstance(call, ast.Call):
            return False
        name = (
            call.func.attr
            if isinstance(call.func, ast.Attribute)
            else call.func.id
            if isinstance(call.func, ast.Name)
            else ""
        )
        if name in filesystem_calls or name in {"write_txn", "Popen"}:
            return True
        if name in {"execute", "executemany", "executescript"} and call.args:
            sql = call.args[0]
            if isinstance(sql, ast.Constant) and isinstance(sql.value, str):
                head = sql.value.lstrip().upper()
                return head.startswith(
                    (
                        "INSERT",
                        "UPDATE",
                        "DELETE",
                        "REPLACE",
                        "CREATE",
                        "ALTER",
                        "DROP",
                        "VACUUM",
                        "PRAGMA JOURNAL_MODE",
                        "PRAGMA WAL_CHECKPOINT",
                        "BEGIN",
                        "COMMIT",
                        "ROLLBACK",
                    )
                )
        if name in {"run", "check_call", "check_output"}:
            text = ast.get_source_segment(source, call) or ""
            return any(
                token in text
                for token in (
                    "worktree",
                    "checkout",
                    "switch",
                    "branch",
                    "reset",
                    "tmux",
                )
            )
        return False

    def has_indirect_sql_write(node):
        has_execute = any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in {"execute", "executemany", "executescript"}
            for child in ast.walk(node)
        )
        constants = "\n".join(
            child.value
            for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        ).upper()
        return has_execute and any(
            token in constants
            for token in (
                "INSERT ",
                "UPDATE ",
                "DELETE ",
                "REPLACE ",
                "CREATE ",
                "ALTER ",
                "DROP ",
                "VACUUM",
                "PRAGMA WAL_CHECKPOINT",
                "BEGIN ",
                "COMMIT",
                "ROLLBACK",
            )
        )

    return {
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (
            any(is_state_write(child) for child in ast.walk(node))
            or has_indirect_sql_write(node)
        )
    }


def _derived_bootstrap_process_effects(source: str) -> set[str]:
    """Return class-qualified functions that own OS process/FD effects."""
    module = ast.parse(source)
    process_calls = {
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.posix_spawnp",
        "os.open",
        "os.pipe",
        "os.pipe2",
        "os.dup",
        "os.dup2",
        "os.fdopen",
        "os.read",
        "os.write",
        "os.close",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.kill",
        "os.killpg",
        "os.wait",
        "os.waitid",
        "os.waitpid",
        "subprocess.Popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
    }

    def dotted_name(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    discovered = set()

    def visit_function(node, prefix=""):
        if any(
            isinstance(child, ast.Call)
            and (
                dotted_name(child.func) in process_calls
                or (
                    isinstance(child.func, ast.Attribute)
                    and child.func.attr in {
                        "communicate",
                        "kill",
                        "poll",
                        "send_signal",
                        "terminate",
                        "wait",
                    }
                    and any(
                        part in (ast.get_source_segment(source, child.func.value) or "")
                        for part in ("proc", "process")
                    )
                )
            )
            for child in ast.walk(node)
        ):
            discovered.add(f"{prefix}{node.name}")

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_function(node)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit_function(child, f"{node.name}.")
    return discovered


def test_derived_mutation_inventory_has_no_unknown_surface():
    source = Path(kb.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    discovered = _derived_effect_functions(source)
    classified = (
        set(kb.WORKER_CALLABLE_MUTATION_SURFACES)
        | set(kb.REGISTERED_WORKER_DENIED_SURFACES)
        | set(CLASSIFIED_INTERNAL_EFFECTS)
        | set(CLASSIFIED_INFRASTRUCTURE_EFFECTS)
        | set(CLASSIFIED_PURE_FALSE_POSITIVES)
        | set(EXPLICIT_PRIVATE_ENTRY_EFFECTS)
    )
    unknown = discovered - classified
    assert unknown == set(), f"unclassified Hermes mutation surfaces: {sorted(unknown)}"
    derived_classifications = (
        set(CLASSIFIED_INTERNAL_EFFECTS)
        | set(CLASSIFIED_INFRASTRUCTURE_EFFECTS)
        | set(CLASSIFIED_PURE_FALSE_POSITIVES)
        | set(EXPLICIT_PRIVATE_ENTRY_EFFECTS)
    )
    stale = derived_classifications - discovered
    assert stale == set(), f"stale Hermes effect classifications: {sorted(stale)}"

    gate_calls = {
        "_authorize_mutation_locked",
        "_deny_registered_worker_control_plane_locked",
        "_authorized_write_txn",
        "_control_plane_write_txn",
        "_deny_registered_worker_filesystem_control_plane",
        "_authorize_filesystem_mutation",
    }
    for name in (
        set(kb.WORKER_CALLABLE_MUTATION_SURFACES)
        | set(kb.REGISTERED_WORKER_DENIED_SURFACES)
        | set(EXPLICIT_PRIVATE_ENTRY_EFFECTS)
    ):
        node = functions[name]
        calls = {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert calls & gate_calls, f"ungated Hermes mutation surface: {name}"


def test_process_fd_inventory_is_qualified_and_exhaustive():
    bootstrap_path = Path(kb.__file__).with_name("process_bootstrap.py")
    sources = {
        "hermes_cli/kanban_db.py": Path(kb.__file__).read_text(encoding="utf-8"),
        "hermes_cli/process_bootstrap.py": bootstrap_path.read_text(encoding="utf-8"),
    }
    discovered = {
        f"{path}:{name}"
        for path, source in sources.items()
        for name in _derived_bootstrap_process_effects(source)
    }
    assert "hermes_cli/kanban_db.py:PendingWorkerProcess._close_gate" in discovered
    assert "hermes_cli/kanban_db.py:PendingWorkerProcess.release" in discovered
    assert "hermes_cli/kanban_db.py:PendingWorkerProcess.abort" in discovered
    assert "hermes_cli/kanban_db.py:_spawn_behind_bootstrap" in discovered
    assert "hermes_cli/process_bootstrap.py:main" in discovered
    assert discovered == set(CLASSIFIED_PROCESS_FD_EFFECTS)

    sample = """
import os
import subprocess
class PendingWorkerProcess:
    def _close_gate(self): os.close(self.fd)
    def release(self): os.write(self.fd, b'x')
    def abort(self): os.kill(self.pid, 9)
def main(): os.execv('/bin/true', ['true'])
"""
    assert _derived_bootstrap_process_effects(sample) == {
        "PendingWorkerProcess._close_gate",
        "PendingWorkerProcess.release",
        "PendingWorkerProcess.abort",
        "main",
    }


def test_control_plane_inventory_rejects_split_transaction_gates():
    source = Path(kb.__file__).read_text(encoding="utf-8")
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in TRANSACTIONAL_CONTROL_PLANE_SURFACES:
        calls = {
            child.func.id
            for child in ast.walk(functions[name])
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert "_deny_registered_worker_control_plane" not in calls, (
            f"{name} authorizes in a transaction that ends before its writes"
        )
        assert "_control_plane_write_txn" in calls, (
            f"{name} lacks a same-transaction control-plane gate"
        )
        assert "write_txn" not in calls, (
            f"{name} contains an ungated direct write transaction"
        )

    for name in {"reassign_task", "dispatch_once"}:
        calls = {
            child.func.id
            for child in ast.walk(functions[name])
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert "_deny_registered_worker_control_plane" not in calls
        assert "_deny_registered_worker_filesystem_control_plane" in calls


def test_nested_ready_continuations_are_statically_target_bounded():
    source = Path(kb.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    violations = []
    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "recompute_ready"
        ):
            continue
        keyword_names = {keyword.arg for keyword in node.keywords}
        if "_mutation_auth" in keyword_names and "target_task_ids" not in keyword_names:
            violations.append(getattr(node, "lineno", None))
    assert violations == [], (
        f"nested recompute_ready calls lack bounded targets at lines {violations}"
    )


def test_external_call_edges_use_only_public_or_explicit_private_effects():
    root = Path(kb.__file__).resolve().parents[1]
    source = Path(kb.__file__).read_text(encoding="utf-8")
    effect_names = _derived_effect_functions(source)
    public_effects = (
        set(kb.WORKER_CALLABLE_MUTATION_SURFACES)
        | set(kb.REGISTERED_WORKER_DENIED_SURFACES)
        | set(CLASSIFIED_INFRASTRUCTURE_EFFECTS)
    )
    allowed_private = set(EXPLICIT_PRIVATE_ENTRY_EFFECTS)
    violations = []
    consumers = []

    for path in root.rglob("*.py"):
        if path == Path(kb.__file__).resolve() or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "kanban_db" not in text:
            continue
        module = ast.parse(text)
        module_aliases = set()
        direct_aliases = {}
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name == "hermes_cli.kanban_db":
                        module_aliases.add(imported.asname or "kanban_db")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "hermes_cli" and any(
                    imported.name == "kanban_db" for imported in node.names
                ):
                    module_aliases.update(
                        imported.asname or imported.name
                        for imported in node.names
                        if imported.name == "kanban_db"
                    )
                elif node.module == "hermes_cli.kanban_db":
                    direct_aliases.update(
                        {
                            imported.asname or imported.name: imported.name
                            for imported in node.names
                        }
                    )
        if not module_aliases and not direct_aliases:
            continue
        consumers.append(path.relative_to(root).as_posix())
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            target = None
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_aliases
            ):
                target = node.func.attr
            elif isinstance(node.func, ast.Name):
                target = direct_aliases.get(node.func.id)
            if target not in effect_names:
                continue
            if target not in public_effects and target not in allowed_private:
                violations.append(
                    f"{path.relative_to(root).as_posix()}:{target}"
                )

    assert "tools/kanban_tools.py" in consumers
    assert "agent/turn_finalizer.py" in consumers
    assert "gateway/kanban_watchers.py" in consumers
    assert set(violations) == set(CLASSIFIED_EXTERNAL_INTERNAL_EFFECT_EDGES), (
        f"private/unknown mutation call edges changed: {violations}"
    )


def test_write_txn_token_resets_after_commit(isolated_home):
    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            token = kb._active_write_txn_token(conn)
            assert token
        with pytest.raises(RuntimeError, match="active write_txn"):
            kb._active_write_txn_token(conn)
    finally:
        conn.close()


def test_write_txn_token_resets_after_rollback(isolated_home):
    conn = kb.connect()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            with kb.write_txn(conn):
                assert kb._active_write_txn_token(conn)
                raise RuntimeError("boom")
        with pytest.raises(RuntimeError, match="active write_txn"):
            kb._active_write_txn_token(conn)
    finally:
        conn.close()


def test_nested_same_connection_inherits_token_without_inner_commit(isolated_home):
    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            outer = kb._active_write_txn_token(conn)
            with kb.write_txn(conn, allow_nested=True):
                assert kb._active_write_txn_token(conn) == outer
            assert conn.in_transaction
            assert kb._active_write_txn_token(conn) == outer
    finally:
        conn.close()


@pytest.mark.macos_only
def test_authorization_cannot_be_reused_in_later_transaction(
    registered_current_process,
):
    fixture = registered_current_process
    prepared = kb._discover_current_worker_registration()
    with kb.write_txn(fixture.conn):
        auth = kb._authorize_mutation_locked(
            fixture.conn,
            target_task_ids=(fixture.task_id,),
            prepared_provenance=prepared,
        )
    before = logical_board_snapshot(fixture.conn)
    with kb.write_txn(fixture.conn):
        with pytest.raises(kb.StaleAttemptError):
            kb._authorize_mutation_locked(
                fixture.conn,
                target_task_ids=(fixture.task_id,),
                prepared_provenance=prepared,
                existing=auth,
            )
    assert logical_board_snapshot(fixture.conn) == before


@pytest.fixture
def outside_registered_group(isolated_home):
    leader = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
    identity = kb._darwin_process_identity(leader.pid)
    assert identity is not None
    conn = kb.connect()
    from tests.attempt_fence_helpers import create_bound_attempt

    task_id, claimed, raw_fence = create_bound_attempt(
        conn,
        leader_identity=identity,
    )
    try:
        yield conn, task_id, claimed, raw_fence
    finally:
        conn.close()
        try:
            os.killpg(identity.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        leader.wait(timeout=5)


@pytest.mark.macos_only
@pytest.mark.parametrize(
    "operation",
    ["comment", "complete", "block", "heartbeat", "link", "unlink"],
)
def test_outside_group_public_mutation_has_zero_delta(
    outside_registered_group,
    operation,
):
    conn, task_id, claimed, _raw_fence = outside_registered_group
    other_id = kb.create_task(conn, title="other")
    if operation == "unlink":
        conn.execute(
            "INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)",
            (task_id, other_id),
        )
        conn.commit()
    before = logical_board_snapshot(conn)

    with pytest.raises(kb.FencedTargetError):
        if operation == "comment":
            kb.add_comment(conn, task_id, "outside", "late")
        elif operation == "complete":
            kb.complete_task(
                conn,
                task_id,
                result="late",
                expected_run_id=claimed.current_run_id,
            )
        elif operation == "block":
            kb.block_task(
                conn,
                task_id,
                reason="late",
                expected_run_id=claimed.current_run_id,
            )
        elif operation == "heartbeat":
            kb.heartbeat_claim(conn, task_id)
        elif operation == "link":
            kb.link_tasks(conn, task_id, other_id)
        else:
            kb.unlink_tasks(conn, task_id, other_id)
    assert logical_board_snapshot(conn) == before


@pytest.mark.macos_only
@pytest.mark.parametrize(
    "operation",
    ["assign", "model", "reasoning", "archive", "schedule", "notify"],
)
def test_outside_group_lifecycle_mutation_has_zero_delta(
    outside_registered_group,
    operation,
):
    conn, task_id, _claimed, _raw_fence = outside_registered_group
    before = logical_board_snapshot(conn)
    with pytest.raises(kb.FencedTargetError):
        if operation == "assign":
            kb.assign_task(conn, task_id, "yonatan")
        elif operation == "model":
            kb.set_model_override(conn, task_id, "gpt-5")
        elif operation == "reasoning":
            kb.set_reasoning_effort(conn, task_id, "high")
        elif operation == "archive":
            kb.archive_task(conn, task_id)
        elif operation == "schedule":
            kb.schedule_task(conn, task_id, reason="late")
        else:
            kb.add_notify_sub(
                conn,
                task_id=task_id,
                platform="test",
                chat_id="foreign",
            )
    assert logical_board_snapshot(conn) == before


@pytest.mark.macos_only
def test_outside_group_attachment_fails_before_filesystem_or_db_delta(
    outside_registered_group,
):
    conn, task_id, _claimed, _raw_fence = outside_registered_group
    attachment_dir = kb.task_attachments_dir(task_id)
    assert not attachment_dir.exists()
    before = logical_board_snapshot(conn)
    with pytest.raises(kb.FencedTargetError):
        kb.store_attachment_bytes(conn, task_id, "proof.txt", b"must not land")
    assert not attachment_dir.exists()
    assert logical_board_snapshot(conn) == before


@pytest.mark.macos_only
def test_registered_worker_control_plane_denied_after_env_removal(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    other_id = kb.create_task(fixture.conn, title="claim target", assignee="dor-coo")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.RegisteredWorkerControlPlaneError):
        kb.claim_task(fixture.conn, other_id, claimer="late-worker")
    assert logical_board_snapshot(fixture.conn) == before


@pytest.mark.macos_only
def test_registered_worker_cannot_delete_its_registration_row(
    registered_current_process,
):
    fixture = registered_current_process
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.RegisteredWorkerControlPlaneError):
        kb.delete_task(fixture.conn, fixture.task_id)
    assert logical_board_snapshot(fixture.conn) == before


@pytest.mark.macos_only
def test_registered_worker_cannot_mutate_another_unfenced_task(
    registered_current_process,
):
    fixture = registered_current_process
    other_id = kb.create_task(fixture.conn, title="worker-created child")
    before = logical_board_snapshot(fixture.conn)
    with pytest.raises(kb.FencedTargetError):
        kb.add_comment(fixture.conn, other_id, "dor-coo", "cross-task")
    assert logical_board_snapshot(fixture.conn) == before


@pytest.mark.macos_only
def test_registered_worker_filesystem_control_plane_has_zero_delta(
    registered_current_process,
    monkeypatch,
):
    fixture = registered_current_process
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    current_path = kb.current_board_path()
    before_current = current_path.read_bytes() if current_path.exists() else None
    metadata_path = kb.board_metadata_path("blocked-by-fence")
    assert not metadata_path.exists()

    with pytest.raises(kb.RegisteredWorkerControlPlaneError):
        kb.set_current_board("blocked-by-fence")
    with pytest.raises(kb.RegisteredWorkerControlPlaneError):
        kb.write_board_metadata("blocked-by-fence", name="must not land")

    after_current = current_path.read_bytes() if current_path.exists() else None
    assert after_current == before_current
    assert not metadata_path.exists()


@pytest.mark.macos_only
def test_registered_worker_cannot_open_other_board_without_env(
    isolated_home,
    monkeypatch,
):
    other_path = isolated_home / "kanban" / "boards" / "other" / "kanban.db"
    kb.init_db(other_path)
    own_conn = kb.connect()
    identity = kb._darwin_process_identity(os.getpgid(0))
    assert identity is not None
    from tests.attempt_fence_helpers import create_bound_attempt

    create_bound_attempt(own_conn, leader_identity=identity)
    own_conn.close()
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    before = other_path.read_bytes()
    with pytest.raises(kb.CrossBoardWorkerMutationError):
        kb.connect(other_path)
    assert other_path.read_bytes() == before


@pytest.mark.macos_only
def test_registered_group_connects_existing_own_board_without_migration(
    registered_current_process,
):
    fixture = registered_current_process
    fixture.conn.close()
    conn = kb.connect(fixture.board_path)
    try:
        comment_id = kb.add_comment(conn, fixture.task_id, "dor-coo", "authorized")
        assert comment_id > 0
    finally:
        conn.close()


@pytest.mark.macos_only
def test_worker_graph_mutation_requires_own_attempt_as_an_endpoint(
    registered_current_process,
):
    fixture = registered_current_process
    parent_id = kb.create_task(fixture.conn, title="unrelated parent")
    child_id = kb.create_task(fixture.conn, title="unrelated child")
    fixture.conn.execute(
        "INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)",
        (parent_id, child_id),
    )
    fixture.conn.commit()
    before = logical_board_snapshot(fixture.conn)

    with pytest.raises(kb.FencedTargetError):
        kb.unlink_tasks(fixture.conn, parent_id, child_id)
    with pytest.raises(kb.FencedTargetError):
        kb.link_tasks(fixture.conn, child_id, parent_id)

    assert logical_board_snapshot(fixture.conn) == before


@pytest.mark.macos_only
@pytest.mark.parametrize("own_is_parent", [True, False])
@pytest.mark.parametrize("operation", ["link", "unlink"])
def test_worker_graph_mutation_allows_own_attempt_endpoint_only(
    registered_current_process,
    own_is_parent,
    operation,
):
    fixture = registered_current_process
    other_id = kb.create_task(fixture.conn, title="attempt-related node")
    parent_id, child_id = (
        (fixture.task_id, other_id)
        if own_is_parent
        else (other_id, fixture.task_id)
    )
    if operation == "unlink":
        fixture.conn.execute(
            "INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )
        fixture.conn.commit()
        assert kb.unlink_tasks(fixture.conn, parent_id, child_id) is True
        assert fixture.conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?",
            (parent_id, child_id),
        ).fetchone() is None
    else:
        kb.link_tasks(fixture.conn, parent_id, child_id)
        assert fixture.conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?",
            (parent_id, child_id),
        ).fetchone() is not None


def test_unlink_and_ready_recompute_roll_back_together(isolated_home, monkeypatch):
    conn = kb.connect()
    try:
        parent_id = kb.create_task(conn, title="parent")
        child_id = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent_id, child_id)
        before = logical_board_snapshot(conn)

        def fail_recompute(*_args, **_kwargs):
            raise RuntimeError("forced recompute failure")

        monkeypatch.setattr(kb, "recompute_ready", fail_recompute)
        with pytest.raises(RuntimeError, match="forced recompute failure"):
            kb.unlink_tasks(conn, parent_id, child_id)
        assert logical_board_snapshot(conn) == before
    finally:
        conn.close()


@pytest.mark.macos_only
@pytest.mark.parametrize("operation", ["archive", "schedule"])
def test_worker_terminal_lifecycle_preserves_exact_attempt_tuple(
    registered_current_process,
    operation,
):
    fixture = registered_current_process
    before_task = process_tuple(kb.get_task(fixture.conn, fixture.task_id))
    before_run = tuple(
        fixture.conn.execute(
            "SELECT claim_lock, worker_pid, worker_pgid, worker_identity, worker_fence "
            "FROM task_runs WHERE id=?",
            (fixture.claimed.current_run_id,),
        ).fetchone()
    )

    if operation == "archive":
        assert kb.archive_task(fixture.conn, fixture.task_id) is True
        expected_status = "archived"
    else:
        assert kb.schedule_task(
            fixture.conn,
            fixture.task_id,
            reason="authorized terminal park",
            expected_run_id=fixture.claimed.current_run_id,
        ) is True
        expected_status = "scheduled"

    task = kb.get_task(fixture.conn, fixture.task_id)
    assert task.status == expected_status
    assert process_tuple(task) == before_task
    run = fixture.conn.execute(
        "SELECT claim_lock, worker_pid, worker_pgid, worker_identity, worker_fence, "
        "ended_at FROM task_runs WHERE id=?",
        (fixture.claimed.current_run_id,),
    ).fetchone()
    assert tuple(run)[:5] == before_run
    assert run["ended_at"] is not None


def test_claim_review_authorization_and_mutation_use_one_transaction(isolated_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="review target")
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
        conn.commit()
        statements = []
        conn.set_trace_callback(statements.append)
        assert kb.claim_review_task(conn, task_id, claimer="reviewer") is not None
        begins = [sql for sql in statements if sql.strip().upper() == "BEGIN IMMEDIATE"]
        assert len(begins) == 1
    finally:
        conn.set_trace_callback(None)
        conn.close()


@pytest.mark.macos_only
def test_claim_review_has_no_registration_race_between_gate_and_write(
    isolated_home,
    monkeypatch,
):
    conn = kb.connect()
    try:
        review_id = kb.create_task(conn, title="review target")
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (review_id,))
        worker_id = kb.create_task(conn, title="worker registration")
        claimed = kb.claim_task(conn, worker_id, claimer="fixture:claim")
        conn.commit()
        identity = kb._darwin_process_identity(os.getpgid(0))
        assert identity is not None
        raw_fence = json.dumps(
            {
                "run_id": claimed.current_run_id,
                "claim_lock": claimed.claim_lock,
                "host": kb._host_id(),
                "leader_pid": identity.pid,
                "worker_pgid": identity.pgid,
                "worker_identity": identity.token,
                "reason": "race",
                "created_at": int(time.time()),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        real_write_txn = kb.write_txn
        calls = 0
        injected = False

        @contextlib.contextmanager
        def adversarial_write_txn(target_conn, *args, **kwargs):
            nonlocal calls, injected
            calls += 1
            if calls == 2:
                injected = True
                target_conn.execute(
                    "UPDATE tasks SET worker_pid=?, worker_pgid=?, worker_identity=?, "
                    "worker_fence=? WHERE id=?",
                    (identity.pid, identity.pgid, identity.token, raw_fence, worker_id),
                )
                target_conn.execute(
                    "UPDATE task_runs SET worker_pid=?, worker_pgid=?, worker_identity=?, "
                    "worker_fence=? WHERE id=?",
                    (
                        identity.pid,
                        identity.pgid,
                        identity.token,
                        raw_fence,
                        claimed.current_run_id,
                    ),
                )
                target_conn.commit()
            with real_write_txn(target_conn, *args, **kwargs):
                yield

        monkeypatch.setattr(kb, "write_txn", adversarial_write_txn)
        try:
            result = kb.claim_review_task(conn, review_id, claimer="reviewer")
        except kb.RegisteredWorkerControlPlaneError:
            result = None
        if injected:
            assert result is None
            assert kb.get_task(conn, review_id).status == "review"
        else:
            assert calls == 1
            assert result is not None
    finally:
        conn.close()


@pytest.mark.macos_only
def test_registered_worker_connect_rejects_missing_terminal_reaper_schema(
    registered_current_process,
):
    fixture = registered_current_process
    board_path = fixture.board_path
    fixture.conn.execute("DROP TABLE terminal_fence_reap_state")
    fixture.conn.commit()
    fixture.conn.close()
    before = board_path.read_bytes()
    with pytest.raises(kb.AttemptFenceCapabilityError):
        kb.connect(board_path)
    assert board_path.read_bytes() == before


def _task_event_snapshot(conn, task_id):
    task = tuple(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
    events = [
        tuple(row)
        for row in conn.execute(
            "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,)
        )
    ]
    return task, events


@pytest.mark.macos_only
@pytest.mark.parametrize("operation", ["unlink", "complete", "archive"])
def test_worker_nested_ready_continuation_is_bounded_to_affected_child(
    registered_current_process,
    operation,
):
    fixture = registered_current_process
    child_id = kb.create_task(fixture.conn, title="affected child")
    kb.link_tasks(fixture.conn, fixture.task_id, child_id)
    unrelated_id = kb.create_task(fixture.conn, title="unrelated eligible")
    fixture.conn.execute(
        "UPDATE tasks SET status='todo' WHERE id=?", (unrelated_id,)
    )
    fixture.conn.commit()
    unrelated_before = _task_event_snapshot(fixture.conn, unrelated_id)

    if operation == "unlink":
        assert kb.unlink_tasks(fixture.conn, fixture.task_id, child_id) is True
    elif operation == "complete":
        assert kb.complete_task(
            fixture.conn,
            fixture.task_id,
            result="done",
            expected_run_id=fixture.claimed.current_run_id,
        ) is True
    else:
        assert kb.archive_task(fixture.conn, fixture.task_id) is True

    assert kb.get_task(fixture.conn, child_id).status == "ready"
    assert _task_event_snapshot(fixture.conn, unrelated_id) == unrelated_before


def test_decompose_nested_ready_continuation_is_bounded_to_created_graph(
    isolated_home,
):
    conn = kb.connect()
    try:
        root_id = kb.create_task(conn, title="triage root")
        unrelated_id = kb.create_task(conn, title="unrelated eligible")
        conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (root_id,))
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (unrelated_id,))
        conn.commit()
        unrelated_before = _task_event_snapshot(conn, unrelated_id)

        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="dor-coo",
            children=[{"title": "bounded child", "assignee": "yonatan"}],
        )

        assert child_ids and kb.get_task(conn, child_ids[0]).status == "ready"
        assert _task_event_snapshot(conn, unrelated_id) == unrelated_before
    finally:
        conn.close()


def test_operator_recompute_ready_remains_board_wide(isolated_home):
    conn = kb.connect()
    try:
        first_id = kb.create_task(conn, title="first eligible")
        second_id = kb.create_task(conn, title="second eligible")
        conn.execute(
            "UPDATE tasks SET status='todo' WHERE id IN (?, ?)",
            (first_id, second_id),
        )
        conn.commit()

        assert kb.recompute_ready(conn) == 2
        assert kb.get_task(conn, first_id).status == "ready"
        assert kb.get_task(conn, second_id).status == "ready"
    finally:
        conn.close()
