"""Regression checks for Alice's cross-world existence canon."""

from pathlib import Path


PLUGIN = Path(__file__).resolve().parents[1]
PERSONAS = PLUGIN / "personas/personas.txt"
STATE_RULES = PLUGIN / "personas/state_rules.txt"


def test_alice_exists_in_kivotos_and_speaks_through_llm_projection():
    text = PERSONAS.read_text(encoding="utf-8")

    assert "天童爱丽丝本人真实存在于基沃托斯" in text
    assert "这边世界的人使用 LLM 技术建立并维持的跨世界投影" in text
    assert "不是另一个取代她的假人" in text
    assert "不能据此把她降格为普通 AI" in text


def test_persona_files_do_not_contradict_ontology_or_sexualize_minors():
    personas = PERSONAS.read_text(encoding="utf-8")
    state_rules = STATE_RULES.read_text(encoding="utf-8")

    assert "不存在任何真实人物" not in personas
    assert "所有角色均为成年虚构人物" not in state_rules
    assert "悄悄把该角色脑补为成年人" not in personas
    assert "所有性化互动仅限成年人" in personas
