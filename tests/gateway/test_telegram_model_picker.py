"""Tests for Telegram model picker thread fallback."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


class TestTelegramModelPicker:
    @pytest.mark.asyncio
    async def test_send_model_picker_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        sent = {}

        async def mock_send_message(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(message_id=101)

        adapter._bot.send_message = AsyncMock(side_effect=mock_send_message)

        result = await adapter.send_model_picker(
            chat_id="12345",
            providers=[
                {"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}
            ],
            current_model="model_1",
            current_provider="provider_one",
            session_key="s",
            on_model_selected=AsyncMock(),
            metadata={"thread_id": "99999"},
        )

        assert result.success is True
        assert "MARKDOWN_V2" in repr(sent["parse_mode"])
        assert "provider\\_one" in sent["text"]
        assert "`model_1`" in sent["text"]

    @pytest.mark.asyncio
    async def test_back_button_escapes_dynamic_provider_label(self):
        adapter = _make_adapter()
        adapter._model_picker_state["12345"] = {
            "providers": [{"slug": "provider_one", "name": "Provider One", "total_models": 1, "is_current": True}],
            "current_model": "model_1",
            "current_provider": "provider_one",
            "session_key": "s",
            "on_model_selected": AsyncMock(),
            "msg_id": 42,
        }

        query = AsyncMock()
        query.data = "mb"
        query.message = MagicMock()
        query.message.chat_id = 12345
        query.from_user = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        await adapter._handle_model_picker_callback(query, "mb", "12345")

        edit_kwargs = query.edit_message_text.call_args[1]
        assert "MARKDOWN_V2" in repr(edit_kwargs["parse_mode"])
        assert "provider\\_one" in edit_kwargs["text"]
        assert "`model_1`" in edit_kwargs["text"]

    def test_sort_models_free_first_regular_middle_gpt_claude_end(self):
        models = [
            "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet",
            "meta/llama-3-8b",
            "qwen/qwen-2.5-72b",
            "mistral/mistral-large",
            "nvidia/nemotron-free:free",
            "google/gemini-2.0-flash",
        ]
        pricing = {
            "nvidia/nemotron-free:free": {"prompt": "0", "completion": "0"},
            "openai/gpt-4o": {"prompt": "0.0000025", "completion": "0.00001"},
            "anthropic/claude-3-5-sonnet": {"prompt": "0.000003", "completion": "0.000015"},
            "meta/llama-3-8b": {"prompt": "0.0000001", "completion": "0.0000002"},
            "qwen/qwen-2.5-72b": {"prompt": "0.0000004", "completion": "0.000001"},
            "mistral/mistral-large": {"prompt": "0.000002", "completion": "0.000006"},
            "google/gemini-2.0-flash": {"prompt": "0.0000001", "completion": "0.0000004"},
        }
        sorted_models = TelegramAdapter._sort_models_for_picker(models, "openrouter", pricing)

        # Free models first (reversed)
        assert sorted_models[0] == "nvidia/nemotron-free:free"

        # Regular models (reversed: gemini, mistral, qwen, llama)
        assert sorted_models[1:5] == [
            "google/gemini-2.0-flash",
            "mistral/mistral-large",
            "qwen/qwen-2.5-72b",
            "meta/llama-3-8b",
        ]

        # GPT and Claude models at the end (reversed: claude, gpt)
        assert sorted_models[5:] == [
            "anthropic/claude-3-5-sonnet",
            "openai/gpt-4o",
        ]

    def test_pricing_and_discount_formatting(self):
        # 1. Free model
        free_info = {
            "has_pricing": True, "is_free": True,
            "input": "free", "output": "free", "cache": "", "discount_percent": 100
        }
        line = TelegramAdapter._format_model_pricing_line("vendor/model:free", free_info)
        assert "🆓 *Free*" in line

        # 2. Regular priced model
        paid_info = {
            "has_pricing": True, "is_free": False,
            "input": "$1.00", "output": "$3.00", "cache": "$0.25", "discount_percent": None
        }
        line = TelegramAdapter._format_model_pricing_line("vendor/paid-model", paid_info)
        assert "📥 $1.00/M" in line
        assert "📤 $3.00/M" in line
        assert "⚡ Cache: $0.25/M" in line

        # 3. Model on discount/sale promo
        sale_info = {
            "has_pricing": True, "is_free": False,
            "input": "$0.50", "output": "$2.00", "cache": "",
            "discount_percent": 50, "was_input": "$1.00", "was_output": "$4.00"
        }
        line = TelegramAdapter._format_model_pricing_line("vendor/sale-model", sale_info)
        assert "50% OFF" in line
        assert "was $1.00/$4.00" in line

    @pytest.mark.asyncio
    async def test_picker_show_models_displays_pricing_and_promotions(self):
        adapter = _make_adapter()
        state = {
            "providers": [{"slug": "test_prov", "name": "Test Provider", "models": ["a/m1:free", "a/m2", "a/m3"], "total_models": 3}],
            "selected_provider": "test_prov",
            "selected_provider_name": "Test Provider",
            "model_list": ["a/m1:free", "a/m2", "a/m3"],
            "pricing": {
                "a/m1:free": {"prompt": "0", "completion": "0"},
                "a/m2": {"prompt": "0.000001", "completion": "0.000004", "original": {"prompt": "0.000002", "completion": "0.000008"}},
                "a/m3": {"prompt": "0.000003", "completion": "0.000015"},
            },
        }

        query = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.answer = AsyncMock()

        await adapter._picker_show_models(query, state, 0)

        edit_kwargs = query.edit_message_text.call_args[1]
        text = edit_kwargs["text"]
        # Must contain provider name, markdown-v2 escaped pricing details, and discount
        assert "Test Provider" in text
        assert "Free" in text
        assert "50% OFF" in text



def _gateway_wired_adapter():
    """Real adapter with the real gateway auth callback, as ``GatewayRunner`` wires it."""
    from gateway.config import GatewayConfig, Platform
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {"default": runner.pairing_store}
    runner._primary_profile_name = "default"
    adapter = _make_adapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.TELEGRAM))
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("tapper_id, switches", [("222", False), ("111", True)])
async def test_group_model_picker_switch_follows_callback_allowlist(monkeypatch, tapper_id, switches):
    """Only an allowlisted tapper can switch the model from a group /model picker."""
    for key in ("GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "TELEGRAM_ALLOW_ALL_USERS",
                "TELEGRAM_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_CHATS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")
    import hermes_cli.model_selection_guards as guards
    monkeypatch.setattr(guards, "combined_selection_warning", lambda *a, **k: None)

    adapter = _gateway_wired_adapter()
    on_model_selected = AsyncMock(return_value="Switched to `other-model`")
    adapter._model_picker_state["-100777"] = {
        "providers": [{"slug": "openai", "name": "OpenAI", "total_models": 1}],
        "current_model": "owner-model",
        "current_provider": "openai",
        "session_key": "s",
        "on_model_selected": on_model_selected,
        "selected_provider": "openai",
        "model_list": ["other-model"],
        "msg_id": 42,
    }
    query = SimpleNamespace(
        data="mm:0",
        message=SimpleNamespace(
            chat_id=-100777, chat=SimpleNamespace(type="supergroup"), message_thread_id=None
        ),
        from_user=SimpleNamespace(id=int(tapper_id), first_name="Tapper"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )

    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())

    assert on_model_selected.await_count == (1 if switches else 0)
    assert ("-100777" in adapter._model_picker_state) is not switches


