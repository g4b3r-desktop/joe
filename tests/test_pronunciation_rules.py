from joe.pronunciation_rules import (
    apply_pronunciation_rules,
    load_pronunciation_rules,
    serialize_pronunciation_rules,
)


def make_rule(**overrides):
    rule = {
        "id": "rule-1",
        "term": "Joe",
        "pronunciation": "Jô",
        "language": "all",
        "match_mode": "word",
        "case_sensitive": False,
        "enabled": True,
        "priority": 0,
    }
    rule.update(overrides)
    return rule


def test_word_rule_is_case_insensitive_and_respects_word_boundaries():
    result = apply_pronunciation_rules("JOE e JoeStudio.", [make_rule()], "portuguese")
    assert result == "Jô e JoeStudio."


def test_contains_rule_can_replace_part_of_a_word():
    rule = make_rule(term="AI", pronunciation="êi ái", match_mode="contains")
    assert apply_pronunciation_rules("OpenAI", [rule], "portuguese") == "Openêi ái"


def test_phrase_rule_matches_variable_whitespace():
    rule = make_rule(term="Open AI", pronunciation="Ôupen êi ái", match_mode="phrase")
    assert apply_pronunciation_rules("Open   AI", [rule], "english") == "Ôupen êi ái"


def test_language_specific_rule_only_runs_for_selected_language():
    rule = make_rule(language="portuguese")
    assert apply_pronunciation_rules("Joe", [rule], "english") == "Joe"
    assert apply_pronunciation_rules("Joe", [rule], "portuguese") == "Jô"


def test_case_sensitive_rule_preserves_other_casing():
    rule = make_rule(term="GPU", pronunciation="gê pê u", case_sensitive=True)
    assert apply_pronunciation_rules("GPU gpu", [rule], "portuguese") == "gê pê u gpu"


def test_disabled_rules_are_ignored():
    rule = make_rule(enabled=False)
    assert apply_pronunciation_rules("Joe", [rule], "portuguese") == "Joe"


def test_higher_priority_rule_wins_over_lower_priority_overlap():
    high = make_rule(term="Joe", pronunciation="Jô avançado", priority=10)
    low = make_rule(id="rule-2", term="Joe", pronunciation="Jô antigo", priority=0)
    assert apply_pronunciation_rules("Joe", [low, high], "portuguese") == "Jô avançado"


def test_load_and_serialize_rules_keep_metadata_and_ignore_invalid_entries():
    raw = [
        make_rule(language="english", match_mode="phrase", priority=12),
        {"id": "bad", "term": "", "pronunciation": "x"},
    ]
    loaded = load_pronunciation_rules(serialize_pronunciation_rules(raw))
    assert len(loaded) == 1
    assert loaded[0]["language"] == "english"
    assert loaded[0]["match_mode"] == "phrase"
    assert loaded[0]["priority"] == 12


def test_duplicate_rule_ids_are_ignored_when_loading():
    rule = make_rule()
    assert len(load_pronunciation_rules([rule, dict(rule, term="other")])) == 1
