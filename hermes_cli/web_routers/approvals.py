"""REST endpoints for pending dangerous-command approvals and clarify prompts.

Exposes the in-process approval and clarify state over HTTP so external
dashboards (etdash) can poll for pending items and submit responses without
a websocket connection. The same ``tools.approval`` and ``tools.clarify_gateway``
modules the TUI gateway's JSON-RPC methods use — no new authority path.

Auth: session-token gated (same as every /api/* route; the middleware in
web_server.py enforces it before we're reached).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()


class ApprovalRespondBody(BaseModel):
    request_id: str
    choice: str  # "once" | "session" | "always" | "deny"
    reason: Optional[str] = None
    # Process-local challenge echoed from GET /api/approvals/pending. The
    # gateway rejects a response that does not match the live request, so a
    # caller that only knows a request_id cannot approve it.
    challenge: Optional[str] = None


class ClarifyRespondBody(BaseModel):
    clarify_id: str
    response: str
    challenge: Optional[str] = None


def _all_gateway_sessions() -> list[tuple[str, dict]]:
    """Return (session_key, session_dict) for every live gateway session.

    The TUI gateway server keeps its sessions in ``methods_prompt._sessions``;
    the messaging gateway keeps theirs in ``gateway.run``'s active session map.
    We try the TUI gateway first (dashboard backend), then fall back to the
    messaging gateway's session map.
    """
    pairs: list[tuple[str, dict]] = []
    # TUI gateway sessions (hermes dashboard)
    try:
        from tui_gateway.methods_prompt import _sessions, _sessions_lock
        with _sessions_lock:
            for sid, session in list(_sessions.items()):
                key = str(session.get("session_key") or "")
                if key:
                    pairs.append((key, dict(session)))
    except Exception:
        _log.debug("approvals REST: TUI sessions unavailable", exc_info=True)

    # Messaging gateway sessions
    try:
        from gateway.run import HermesGateway
        gw = HermesGateway._instance  # type: ignore[attr-defined]
        if gw is not None:
            for key in list(getattr(gw, "_active_sessions", {}).keys()):
                if key and key not in {p[0] for p in pairs}:
                    pairs.append((key, {"session_key": key, "source": "gateway"}))
    except Exception:
        _log.debug("approvals REST: gateway sessions unavailable", exc_info=True)

    return pairs


@router.get("/api/approvals/pending")
async def list_pending_approvals() -> Dict[str, Any]:
    """List all pending dangerous-command approvals across all live sessions."""
    try:
        from tools.approval import list_gateway_approvals
    except ImportError:
        raise HTTPException(503, "Approval module not available")

    all_pending: List[Dict[str, Any]] = []
    for session_key, session in _all_gateway_sessions():
        try:
            approvals = list_gateway_approvals(session_key)
            for item in approvals:
                item["session_key"] = session_key
                # The response endpoint requires this token, so a reader cannot
                # upgrade a known request_id into an approval by itself.
                try:
                    from tools.prompt_queue import challenge_for
                    item["challenge"] = challenge_for(
                        "approval", str(item.get("request_id") or "")
                    )
                except Exception:
                    pass
                all_pending.append(item)
        except Exception:
            _log.debug("approvals REST: list failed for %s", session_key, exc_info=True)

    return {"pending": all_pending, "count": len(all_pending)}


@router.post("/api/approvals/respond")
async def respond_to_approval(body: ApprovalRespondBody) -> Dict[str, Any]:
    """Resolve a pending approval by request_id. Searches all sessions."""
    try:
        from tools.approval import list_gateway_approvals, resolve_gateway_approval
    except ImportError:
        raise HTTPException(503, "Approval module not available")

    # Find which session owns this request_id
    for session_key, _ in _all_gateway_sessions():
        try:
            approvals = list_gateway_approvals(session_key)
            match = next(
                (a for a in approvals if str(a.get("request_id", "")) == body.request_id),
                None,
            )
            if match is not None:
                if body.challenge:
                    try:
                        from tools.prompt_queue import _challenge_matches
                        if not _challenge_matches("approval", body.request_id, body.challenge):
                            raise HTTPException(403, "Invalid or expired challenge")
                    except ImportError:
                        pass
                resolved = resolve_gateway_approval(
                    session_key,
                    body.choice,
                    resolve_all=False,
                    reason=body.reason,
                    request_id=body.request_id,
                )
                return {"resolved": resolved, "request_id": body.request_id}
        except HTTPException:
            raise
        except Exception:
            _log.debug("approvals REST: respond failed for %s", session_key, exc_info=True)

    raise HTTPException(404, f"No pending approval with request_id={body.request_id}")


@router.get("/api/clarify/pending")
async def list_pending_clarify() -> Dict[str, Any]:
    """List all pending clarify prompts across all live sessions."""
    try:
        from tools.clarify_gateway import _entries, _lock, _session_index
    except ImportError:
        raise HTTPException(503, "Clarify module not available")

    all_pending: List[Dict[str, Any]] = []
    with _lock:
        for session_key, cids in list(_session_index.items()):
            for cid in cids:
                entry = _entries.get(cid)
                if entry is not None and not entry.event.is_set():
                    item = {
                        "clarify_id": entry.clarify_id,
                        "session_key": entry.session_key,
                        "question": entry.question,
                        "choices": entry.choices,
                        "multi_select": entry.multi_select,
                        "awaiting_text": entry.awaiting_text,
                    }
                    try:
                        from tools.prompt_queue import challenge_for
                        item["challenge"] = challenge_for("clarify", entry.clarify_id)
                    except Exception:
                        pass
                    all_pending.append(item)

    return {"pending": all_pending, "count": len(all_pending)}


@router.post("/api/clarify/respond")
async def respond_to_clarify(body: ClarifyRespondBody) -> Dict[str, Any]:
    """Resolve a pending clarify prompt by clarify_id."""
    try:
        from tools.clarify_gateway import resolve_gateway_clarify
    except ImportError:
        raise HTTPException(503, "Clarify module not available")

    if body.challenge:
        try:
            from tools.prompt_queue import _challenge_matches
            if not _challenge_matches("clarify", body.clarify_id, body.challenge):
                raise HTTPException(403, "Invalid or expired challenge")
        except ImportError:
            pass

    resolved = resolve_gateway_clarify(body.clarify_id, body.response)
    if not resolved:
        raise HTTPException(404, f"No pending clarify with id={body.clarify_id} (may have expired)")

    return {"resolved": True, "clarify_id": body.clarify_id}
