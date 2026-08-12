import ast
import os
import signal
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tests.attempt_fence_helpers import (
    isolated_home,
    logical_board_snapshot,
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
        "_dispatch_once_locked",
        "_deny_registered_worker_control_plane",
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

    gate_calls = {
        "_authorize_mutation_locked",
            "_deny_registered_worker_control_plane_locked",
            "_deny_registered_worker_control_plane",
            "_authorized_write_txn",
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
