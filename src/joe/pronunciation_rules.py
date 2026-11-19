from __future__ import annotations

import json
from typing import Any

SUPPORTED_LANGUAGES = {
    "portuguese", "english", "spanish", "french", "italian", "german", "dutch",
}
MATCH_MODES = {"word", "phrase", "contains"}


def normalize_pronunciation_rule(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    rule_id = str(raw.get("id", "")).strip()
    term = str(raw.get("term", "")).strip()
    pronunciation = str(raw.get("pronunciation", "")).strip()
    language = str(raw.get("language", "all")).strip()
    match_mode = str(raw.get("match_mode", "word")).strip()
    if (
        not rule_id or not term or not pronunciation
        or language != "all" and language not in SUPPORTED_LANGUAGES
        or match_mode not in MATCH_MODES
    ):
        return None
    try:
        priority = max(-1000, min(1000, int(raw.get("priority", 0))))
    except (TypeError, ValueError):
        priority = 0
    return {
        "id": rule_id,
        "term": term,
        "pronunciation": pronunciation,
        "language": language,
        "match_mode": match_mode,
        "case_sensitive": bool(raw.get("case_sensitive", False)),
        "enabled": bool(raw.get("enabled", True)),
        "priority": priority,
    }


def load_pronunciation_rules(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    rules = []
    seen_ids = set()
    for item in raw:
        rule = normalize_pronunciation_rule(item)
        if rule is not None and rule["id"] not in seen_ids:
            rules.append(rule)
            seen_ids.add(rule["id"])
    return rules


def serialize_pronunciation_rules(rules: list[dict[str, Any]]) -> str:
    normalized = [
        rule for raw in rules
        if (rule := normalize_pronunciation_rule(raw)) is not None
    ]
    normalized.sort(key=lambda item: (-item["priority"], -len(item["term"]), item["term"].casefold()))
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def protect_pronunciation_rules(
    text: str,
    rules: list[dict[str, Any]] | None,
    language: str,
) -> tuple[str, dict[str, str]]:
    """Replace matching source terms with placeholders for safe later restoration."""
    import re

    normalized = text
    protected: dict[str, str] = {}
    valid_rules = [
        rule for raw in (rules or [])
        if (rule := normalize_pronunciation_rule(raw)) is not None
        and rule["enabled"]
        and rule["language"] in {"all", language}
    ]
    valid_rules.sort(
        key=lambda item: (-item["priority"], -len(item["term"]), item["term"].casefold())
    )
    for index, rule in enumerate(valid_rules):
        term = rule["term"]
        flags = 0 if rule["case_sensitive"] else re.IGNORECASE
        escaped = re.escape(term)
        if rule["match_mode"] == "word":
            pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags)
        elif rule["match_mode"] == "phrase":
            phrase_pattern = r"\s+".join(re.escape(part) for part in term.split())
            pattern = re.compile(rf"(?<!\w){phrase_pattern}(?!\w)", flags)
        else:
            pattern = re.compile(escaped, flags)
        placeholder = f"\uE000ADVPRON{index}TOKEN\uE001"
        if pattern.search(normalized):
            normalized = pattern.sub(lambda _match, key=placeholder: key, normalized)
            protected[placeholder] = rule["pronunciation"]
    return normalized, protected


def apply_pronunciation_rules(
    text: str,
    rules: list[dict[str, Any]] | None,
    language: str,
) -> str:
    """Apply enabled custom rules, with higher-priority and longer terms first."""
    normalized, protected = protect_pronunciation_rules(text, rules, language)
    for placeholder, pronunciation in protected.items():
        normalized = normalized.replace(placeholder, pronunciation)
    return normalized
