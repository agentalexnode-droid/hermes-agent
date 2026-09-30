"""Cross-process pending-prompt queue so etdash can answer gateway prompts.

Clarify and dangerous-command approval state is in-process in the *gateway*
interpreter. The dashboard process cannot see it. This module mirrors the
live queues to ``<hermes_home>/prompt-queue/pending.json`` and lets another
process (AgentOS / etdash) drop a response file that the gateway wait loops
drain on their existing 1s poll.

No new env var: path is always ``get_hermes_home() / "prompt-queue"``.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import stat
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# A response file is a second authority path into the gateway's human-approval
# gate, so file presence alone must never be enough to approve a dangerous
# command. Every challenge is an HMAC under a secret that exists only inside
# this process: a reader can echo the token it was shown, but cannot mint one
# for an id it has not seen, and cannot replay a token captured before a
# gateway restart. The token is derived rather than stored, so it is stable for
# the whole lifetime of the pending request no matter how often sync() runs.
_CHALLENGE_SECRET = secrets.token_bytes(32)

# Human answers land seconds after the prompt appears. Anything older is a
# stale write or a replay of a captured file, not a current decision.
_RESPONSE_MAX_AGE_S = 15 * 60


def challenge_for(kind: str, item_id: str) -> str:
    """Return the process-local challenge token bound to one pending request."""
    if not kind or not item_id:
        return ""
    msg = f"{kind}\x00{item_id}".encode("utf-8")
    return hmac.new(_CHALLENGE_SECRET, msg, sha256).hexdigest()


def _challenge_matches(kind: str, item_id: str, presented: Any) -> bool:
    expected = challenge_for(kind, item_id)
    if not expected or not isinstance(presented, str):
        return False
    return hmac.compare_digest(presented, expected)


def _queue_dir() -> Path:
    from hermes_constants import get_hermes_home

    d = Path(get_hermes_home()) / "prompt-queue"
    d.mkdir(parents=True, exist_ok=True)
    (d / "responses").mkdir(exist_ok=True)
    # The queue carries live approval authority: keep the directory tree
    # owner-only even if the parent home is ever relaxed.
    for path in (d, d / "responses"):
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
    return d


def pending_path() -> Path:
    return _queue_dir() / "pending.json"


def responses_dir() -> Path:
    return _queue_dir() / "responses"


def write_pending(payload: dict[str, Any]) -> None:
    path = pending_path()
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".pending.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def snapshot_from_live() -> dict[str, Any]:
    """Build pending.json from in-process approval + clarify state."""
    approvals: list[dict[str, Any]] = []
    clarifies: list[dict[str, Any]] = []
    try:
        from tools import approval as ap

        with ap._lock:
            for session_key, queue in ap._gateway_queues.items():
                for entry in queue:
                    item = dict(entry.data)
                    item["session_key"] = session_key
                    item["kind"] = "approval"
                    item["challenge"] = challenge_for("approval", str(item.get("request_id") or ""))
                    approvals.append(item)
    except Exception:
        logger.debug("prompt_queue: approval snapshot failed", exc_info=True)
    try:
        from tools import clarify_gateway as cg

        with cg._lock:
            for _session_key, cids in list(cg._session_index.items()):
                for cid in cids:
                    entry = cg._entries.get(cid)
                    if entry is not None and not entry.event.is_set():
                        clarifies.append(
                            {
                                "kind": "clarify",
                                "clarify_id": entry.clarify_id,
                                "session_key": entry.session_key,
                                "question": entry.question,
                                "choices": entry.choices,
                                "multi_select": entry.multi_select,
                                "awaiting_text": entry.awaiting_text,
                                "challenge": challenge_for("clarify", entry.clarify_id),
                            }
                        )
    except Exception:
        logger.debug("prompt_queue: clarify snapshot failed", exc_info=True)
    return {
        "approvals": approvals,
        "clarifies": clarifies,
        "count": len(approvals) + len(clarifies),
    }


def sync() -> None:
    try:
        write_pending(snapshot_from_live())
    except Exception:
        logger.debug("prompt_queue: sync failed", exc_info=True)


def _apply_payload(payload: dict[str, Any]) -> bool:
    kind = payload.get("kind")
    if kind == "approval":
        from tools.approval import list_gateway_approvals, resolve_gateway_approval

        rid = str(payload.get("request_id") or "")
        session_key = str(payload.get("session_key") or "")
        choice = str(payload.get("choice") or "deny")
        reason = payload.get("reason")
        if not rid or not session_key:
            return False
        # Bind the answer to a challenge this process issued for THIS request,
        # and to a request that is still genuinely pending. A well-formed JSON
        # file naming a request_id is not authorization.
        if not _challenge_matches("approval", rid, payload.get("challenge")):
            logger.warning(
                "prompt_queue: rejected approval response for %s (missing or invalid challenge)",
                rid[:12] or "?",
            )
            return False
        if not any(
            str(a.get("request_id") or "") == rid
            for a in list_gateway_approvals(session_key)
        ):
            logger.warning(
                "prompt_queue: rejected approval response for %s (not pending in that session)",
                rid[:12] or "?",
            )
            return False
        return resolve_gateway_approval(
            session_key, choice, reason=reason, request_id=rid
        ) > 0
    if kind == "clarify":
        from tools.clarify_gateway import resolve_gateway_clarify

        cid = str(payload.get("clarify_id") or "")
        if not cid or "response" not in payload:
            return False
        if not _challenge_matches("clarify", cid, payload.get("challenge")):
            logger.warning(
                "prompt_queue: rejected clarify response for %s (missing or invalid challenge)",
                cid[:12] or "?",
            )
            return False
        return bool(resolve_gateway_clarify(cid, str(payload.get("response") or "")))
    return False


def _response_file_is_trustworthy(src: Path) -> bool:
    """Reject anything a same-uid stranger could plant to buy approval authority.

    lstat (not stat) so a symlink can never redirect the read; require a plain
    file owned by the gateway user, and refuse group/other-writable files. The
    challenge token is the real authentication — these checks only close the
    cheap substitution and tampering paths, and keep one process from being able
    to write another process's queue.
    """
    try:
        st = os.lstat(src)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        logger.warning("prompt_queue: rejected non-regular response file %s", src.name)
        return False
    if st.st_uid != os.geteuid():
        logger.warning(
            "prompt_queue: rejected response file %s owned by uid %s (gateway uid %s)",
            src.name, st.st_uid, os.geteuid(),
        )
        return False
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        logger.warning("prompt_queue: rejected group/world-writable response file %s", src.name)
        return False
    return True


def drain_responses() -> None:
    """Apply response files written by etdash; claim via rename so waiters don't double-apply."""
    try:
        rdir = responses_dir()
        now = time.time()
        names = sorted(
            p.name
            for p in rdir.iterdir()
            if p.suffix == ".json" and not p.name.startswith(".")
        )
    except OSError:
        return
    applied_any = False
    for name in names:
        src = rdir / name
        if not _response_file_is_trustworthy(src):
            try:
                src.unlink()
            except OSError:
                pass
            continue
        claimed = rdir / f"{name}.claimed"
        try:
            os.rename(src, claimed)
        except OSError:
            continue
        try:
            payload = json.loads(claimed.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("response payload is not an object")
            # A file that sat in the queue longer than the human-decision
            # window is a stale write or a captured replay, never a live answer.
            if now - claimed.stat().st_mtime > _RESPONSE_MAX_AGE_S:
                logger.warning("prompt_queue: dropped stale response file %s", name)
                continue
            if _apply_payload(payload):
                applied_any = True
        except Exception:
            logger.debug("prompt_queue: apply failed for %s", name, exc_info=True)
        try:
            claimed.unlink()
        except OSError:
            pass
    if applied_any:
        sync()
