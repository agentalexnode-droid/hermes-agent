"""Security-boundary tests for the cross-process prompt-queue approval bridge.

A response file is a real authority path into the gateway's dangerous-command
gate. These tests assert the contract, not the source shape: only a challenge
the gateway itself issued for a genuinely pending request can resolve it, and
the file itself must be a plain owner-only file.

Real imports, real filesystem, temp HERMES_HOME -- no mocks of the gate.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


@pytest.fixture
def queue_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "prompt-queue" / "responses").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home / "prompt-queue"


@pytest.fixture
def approval_state():
    """A real pending approval in tools.approval, with cleanup."""
    from tools import approval as ap
    from tools.approval_gateway_wait import _ApprovalEntry

    session_key = "test:security:session"
    entry = _ApprovalEntry({"command": "rm -rf /important", "description": "test"})
    with ap._lock:
        ap._gateway_queues.setdefault(session_key, []).append(entry)
    try:
        yield session_key, entry
    finally:
        with ap._lock:
            ap._gateway_queues.pop(session_key, None)


def _write(queue_home: Path, name: str, payload: dict) -> Path:
    dest = queue_home / "responses" / name
    dest.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(dest, 0o600)
    return dest


# --- challenge minting --------------------------------------------------------


def test_challenge_is_bound_to_kind_and_id(queue_home):
    from tools.prompt_queue import challenge_for

    a = challenge_for("approval", "abc123")
    assert len(a) == 64 and a == challenge_for("approval", "abc123")
    assert a != challenge_for("clarify", "abc123")
    assert a != challenge_for("approval", "abc124")
    assert not challenge_for("approval", "")
    assert not challenge_for("", "abc123")


def test_challenge_differs_per_process(queue_home):
    """A captured token from a previous gateway lifetime must not replay."""
    import importlib

    from tools import prompt_queue

    first = prompt_queue.challenge_for("approval", "abc123")
    reloaded = importlib.reload(prompt_queue)
    try:
        assert reloaded.challenge_for("approval", "abc123") != first
    finally:
        importlib.reload(prompt_queue)


def test_snapshot_publishes_challenge_for_each_pending_item(queue_home, approval_state):
    session_key, entry = approval_state
    from tools.prompt_queue import snapshot_from_live

    snap = snapshot_from_live()
    assert snap["count"] == 1
    item = snap["approvals"][0]
    assert item["request_id"] == entry.data["request_id"]
    assert item["session_key"] == session_key
    assert item["challenge"] == entry.data.get("challenge") or item["challenge"]
    assert len(item["challenge"]) == 64


# --- rejection: missing / wrong / replayed challenge --------------------------


def test_missing_challenge_does_not_approve(queue_home, approval_state):
    session_key, entry = approval_state
    from tools.prompt_queue import drain_responses

    _write(queue_home, "approval-1.json", {
        "kind": "approval",
        "request_id": entry.data["request_id"],
        "session_key": session_key,
        "choice": "always",
    })
    drain_responses()

    assert not entry.event.is_set()
    assert entry.result is None
    assert list((queue_home / "responses").iterdir()) == []


def test_wrong_challenge_does_not_approve(queue_home, approval_state):
    session_key, entry = approval_state
    from tools.prompt_queue import drain_responses

    _write(queue_home, "approval-1.json", {
        "kind": "approval",
        "request_id": entry.data["request_id"],
        "session_key": session_key,
        "choice": "always",
        "challenge": "f" * 64,
    })
    drain_responses()

    assert not entry.event.is_set()
    assert entry.result is None


def test_challenge_from_a_different_id_does_not_approve(queue_home, approval_state):
    """The real token, but for a different request -- must not transfer."""
    session_key, entry = approval_state
    from tools.prompt_queue import challenge_for, drain_responses

    _write(queue_home, "approval-1.json", {
        "kind": "approval",
        "request_id": entry.data["request_id"],
        "session_key": session_key,
        "choice": "always",
        "challenge": challenge_for("approval", "some-other-id"),
    })
    drain_responses()

    assert not entry.event.is_set()


def test_valid_challenge_approves(queue_home, approval_state):
    """The happy path still works end-to-end through the real file bridge."""
    session_key, entry = approval_state
    from tools.prompt_queue import challenge_for, drain_responses

    _write(queue_home, "approval-1.json", {
        "kind": "approval",
        "request_id": entry.data["request_id"],
        "session_key": session_key,
        "choice": "once",
        "challenge": challenge_for("approval", entry.data["request_id"]),
    })
    drain_responses()

    assert entry.event.is_set()
    assert entry.result == "once"


def test_valid_challenge_for_non_pending_request_does_not_approve(queue_home, approval_state):
    """A minted-for-now token for an already-resolved id is not authority."""
    session_key, entry = approval_state
    from tools.prompt_queue import challenge_for, drain_responses

    ghost_id = "already-gone-id"
    _write(queue_home, "approval-2.json", {
        "kind": "approval",
        "request_id": ghost_id,
        "session_key": session_key,
        "choice": "always",
        "challenge": challenge_for("approval", ghost_id),
    })
    drain_responses()

    assert not entry.event.is_set()
    assert entry.result is None


# --- rejection: file shape ----------------------------------------------------


def test_group_writable_file_is_rejected(queue_home, approval_state):
    session_key, entry = approval_state
    from tools.prompt_queue import challenge_for, drain_responses

    dest = _write(queue_home, "approval-1.json", {
        "kind": "approval",
        "request_id": entry.data["request_id"],
        "session_key": session_key,
        "choice": "always",
        "challenge": challenge_for("approval", entry.data["request_id"]),
    })
    os.chmod(dest, 0o666)  # someone else could rewrite the answer in flight
    drain_responses()

    assert not entry.event.is_set()
    assert not dest.exists()


def test_symlink_is_rejected(queue_home, approval_state):
    session_key, entry = approval_state
    from tools.prompt_queue import challenge_for, drain_responses

    # Target lives OUTSIDE responses/ so the only in-queue name is the symlink:
    # the rejection must be about file shape, not about a second valid file
    # being drained alongside it.
    target = queue_home / "planted-target.json"
    target.write_text(json.dumps({
        "kind": "approval",
        "request_id": entry.data["request_id"],
        "session_key": session_key,
        "choice": "always",
        "challenge": challenge_for("approval", entry.data["request_id"]),
    }), encoding="utf-8")
    os.chmod(target, 0o600)
    link = queue_home / "responses" / "approval-1.json"
    os.symlink(target, link)
    drain_responses()

    assert not entry.event.is_set()
    assert entry.result is None
    assert not link.exists()
    assert target.is_file()


def test_stale_response_is_dropped(queue_home, approval_state):
    session_key, entry = approval_state
    from tools.prompt_queue import challenge_for, drain_responses

    dest = _write(queue_home, "approval-1.json", {
        "kind": "approval",
        "request_id": entry.data["request_id"],
        "session_key": session_key,
        "choice": "always",
        "challenge": challenge_for("approval", entry.data["request_id"]),
    })
    old = time.time() - 60 * 60
    os.utime(dest, (old, old))
    drain_responses()

    assert not entry.event.is_set()
    assert entry.result is None


def test_non_object_payload_is_dropped(queue_home, approval_state):
    session_key, entry = approval_state
    from tools.prompt_queue import drain_responses

    dest = queue_home / "responses" / "approval-1.json"
    dest.write_text('["not", "an", "object"]', encoding="utf-8")
    os.chmod(dest, 0o600)
    drain_responses()

    assert not entry.event.is_set()
    assert entry.result is None


# --- clarify parity ----------------------------------------------------------


def test_clarify_without_challenge_is_rejected(queue_home):
    from tools import clarify_gateway as cg

    session_key = "test:security:clarify"
    entry = cg.register("cid-sec-1", session_key, "Install skill?", ["Yes", "No"])
    try:
        from tools.prompt_queue import drain_responses

        _write(queue_home, "clarify-1.json", {
            "kind": "clarify",
            "clarify_id": "cid-sec-1",
            "response": "Yes",
        })
        drain_responses()

        assert entry.response is None
    finally:
        with cg._lock:
            cg._entries.pop("cid-sec-1", None)
            cg._session_index.pop(session_key, None)


def test_clarify_with_valid_challenge_resolves(queue_home):
    from tools import clarify_gateway as cg

    session_key = "test:security:clarify2"
    entry = cg.register("cid-sec-2", session_key, "Install skill?", ["Yes", "No"])
    try:
        from tools.prompt_queue import challenge_for, drain_responses

        _write(queue_home, "clarify-1.json", {
            "kind": "clarify",
            "clarify_id": "cid-sec-2",
            "response": "Yes",
            "challenge": challenge_for("clarify", "cid-sec-2"),
        })
        drain_responses()

        assert entry.response == "Yes"
    finally:
        with cg._lock:
            cg._entries.pop("cid-sec-2", None)
            cg._session_index.pop(session_key, None)


# --- queue hardening ---------------------------------------------------------


def test_queue_dirs_are_owner_only(queue_home):
    from tools.prompt_queue import pending_path, responses_dir

    assert oct(pending_path().parent.stat().st_mode)[-3:] == "700"
    assert oct(responses_dir().stat().st_mode)[-3:] == "700"


def test_pending_file_is_owner_only(queue_home, approval_state):
    from tools.prompt_queue import pending_path, sync

    sync()
    assert oct(pending_path().stat().st_mode)[-3:] == "600"
