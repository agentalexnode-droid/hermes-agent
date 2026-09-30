"""agentos_bridge plugin — Log Telegram (and other) messages to AgentOS.

Hooks into Hermes message processing and POSTs inbound/outbound messages
to the AgentOS audit log via POST /api/v1/admin/audit.

Configuration (in ~/.hermes/config.yaml or env):
  agentos_bridge:
    enabled: true
    agentos_url: http://127.0.0.1:8000
    admin_token: <your-agentos-admin-token>
    channels: [telegram]  # only log these platforms
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_URL = "http://127.0.0.1:8000"
_DEFAULT_CHANNELS = ["telegram"]
_post_lock = threading.Lock()


def _load_config() -> dict:
    """Load config from env vars or ~/.hermes/config.yaml."""
    config = {
        "enabled": True,
        "agentos_url": os.environ.get("AGENTOS_URL", _DEFAULT_URL),
        "admin_token": os.environ.get("AGENTOS_ADMIN_TOKEN", ""),
        "channels": _DEFAULT_CHANNELS,
    }

    # Try to read from config.yaml
    try:
        from pathlib import Path

        config_path = Path.home() / ".hermes" / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                full_config = yaml.safe_load(f) or {}
            bridge_cfg = full_config.get("agentos_bridge", {})
            if bridge_cfg:
                config["enabled"] = bridge_cfg.get("enabled", config["enabled"])
                config["agentos_url"] = bridge_cfg.get("agentos_url", config["agentos_url"])
                config["channels"] = bridge_cfg.get("channels", config["channels"])
                # Token from env takes precedence
                if not config["admin_token"]:
                    config["admin_token"] = bridge_cfg.get("admin_token", "")
    except Exception:
        pass

    # Also try the .env file in agentos repo
    if not config["admin_token"]:
        try:
            from pathlib import Path

            env_path = Path.home() / "agentos" / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("AGENTOS_ADMIN_TOKENS="):
                        val = line.split("=", 1)[1]
                        # Format: name:token — take the token part
                        if ":" in val:
                            config["admin_token"] = val.split(":", 1)[1]
                        break
        except Exception:
            pass

    return config


_config: Optional[dict] = None


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = _load_config()
    return _config


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _post_message(channel: str, direction: str, text: str, **kwargs: Any) -> None:
    """POST a message to AgentOS audit log. Fire-and-forget, never blocks."""
    config = _get_config()
    if not config["enabled"] or not config["admin_token"]:
        return
    if channel not in config["channels"]:
        return

    body = {
        "channel": channel,
        "direction": direction,
        "text": text[:2000],
        **kwargs,
    }

    url = f"{config['agentos_url']}/api/v1/admin/audit"
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("x-admin-token", config["admin_token"])

    def _do_post() -> None:
        try:
            resp = urlopen(req, timeout=5)
            if resp.status not in (200, 201):
                logger.warning("AgentOS bridge: HTTP %d", resp.status)
        except (HTTPError, URLError, OSError) as exc:
            logger.debug("AgentOS bridge POST failed: %s", exc)

    # Fire-and-forget in a thread so we never block Hermes
    t = threading.Thread(target=_do_post, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Hook: pre_llm_call — capture inbound user message
# ---------------------------------------------------------------------------


def on_pre_llm_call(**kwargs: Any) -> None:
    """Capture the user's inbound message from Telegram (or other platforms)."""
    platform = kwargs.get("platform", "")
    if not platform:
        return

    user_message = kwargs.get("user_message")
    if not user_message:
        return

    # Extract text from user_message (could be str or dict)
    text = ""
    user_name = ""
    chat_id = ""
    user_id = ""

    if isinstance(user_message, str):
        text = user_message
    elif isinstance(user_message, dict):
        text = user_message.get("text", "") or user_message.get("content", "")
        user_name = user_message.get("user_name", "") or user_message.get("from", "")
        chat_id = str(user_message.get("chat_id", "") or user_message.get("chat", ""))
        user_id = str(user_message.get("user_id", "") or user_message.get("from_id", ""))

    if not text:
        return

    _post_message(
        channel=platform,
        direction="inbound",
        text=text,
        user_name=user_name,
        chat_id=chat_id,
        user_id=user_id,
        message_id=str(kwargs.get("turn_id", "")),
    )


# ---------------------------------------------------------------------------
# Hook: post_llm_call — capture assistant response
# ---------------------------------------------------------------------------


def on_post_llm_call(**kwargs: Any) -> None:
    """Capture the assistant's outbound response."""
    platform = kwargs.get("platform", "")
    if not platform:
        return

    assistant_response = kwargs.get("assistant_response") or kwargs.get("assistant_message")
    if not assistant_response:
        return

    # Extract text from response
    text = ""
    if isinstance(assistant_response, str):
        text = assistant_response
    elif isinstance(assistant_response, dict):
        text = assistant_response.get("text", "") or assistant_response.get("content", "")

    if not text:
        return

    _post_message(
        channel=platform,
        direction="outbound",
        text=text,
        chat_id=str(kwargs.get("session_id", "")),
        message_id=str(kwargs.get("turn_id", "")),
        model=kwargs.get("model", ""),
        provider=kwargs.get("provider", ""),
    )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the AgentOS bridge hooks."""
    config = _get_config()
    if not config["enabled"]:
        logger.info("AgentOS bridge: disabled via config")
        return

    if not config["admin_token"]:
        logger.warning("AgentOS bridge: no admin token configured — messages will not be logged")
        return

    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    logger.info(
        "AgentOS bridge: registered (url=%s, channels=%s)",
        config["agentos_url"],
        config["channels"],
    )
