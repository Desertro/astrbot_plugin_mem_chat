"""Keep expired emotional context out of later replies."""

import pytest
from src.core.state import MemChatState


@pytest.mark.parametrize("reader", ["text", "state", "ttl"])
@pytest.mark.parametrize("mood", ["中性", "讥讽"])
def test_expired_mood_clears_text_and_intensity(monkeypatch, reader, mood):
    state = MemChatState()
    clock = [1000.0]
    monkeypatch.setattr(state, "_now", lambda: clock[0])
    scope = "default:GroupMessage:42"
    state.set_last_mood(scope, "上一段互动有点不服气")
    state.set_last_mood_state(scope, mood)
    state.set_mood_state_ttl(scope, 120)
    state.set_last_mood_intensity(scope, 0.6)
    state.set_last_mood("other", "正在开心地聊游戏")
    assert state.get_last_mood(scope) == "上一段互动有点不服气"
    clock[0] += 121
    readers = {
        "text": state.get_last_mood,
        "state": state.get_last_mood_state,
        "ttl": state.get_mood_state_ttl_remaining,
    }
    readers[reader](scope)
    assert state.get_last_mood(scope) == ""
    assert state.get_last_mood_state(scope) == "中性"
    assert state.get_last_mood_intensity(scope) == 0
    assert state.get_last_mood("other") == "正在开心地聊游戏"
