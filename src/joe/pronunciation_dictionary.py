from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_pairs(raw: Any) -> tuple[dict[str, str], int]:
    """Read an object of term/pronunciation pairs or a list of entry objects."""
    if isinstance(raw, dict):
        if "pronunciation_dictionary" in raw:
            raw = raw["pronunciation_dictionary"]
        elif "entries" in raw and isinstance(raw["entries"], (dict, list)):
            raw = raw["entries"]

    if isinstance(raw, dict):
        pairs = list(raw.items())
    elif isinstance(raw, list):
        pairs = []
        skipped = 0
        for item in raw:
            if isinstance(item, dict):
                pairs.append((item.get("term"), item.get("pronunciation")))
            else:
                skipped += 1
    else:
        raise ValueError(
            'O JSON deve ser um objeto {"termo": "pronúncia"} ou uma lista '
            "de objetos com term e pronunciation."
        )

    entries: dict[str, str] = {}
    keys_by_casefold: dict[str, str] = {}
    if isinstance(raw, dict):
        skipped = 0
    for raw_term, raw_pronunciation in pairs:
        if not isinstance(raw_term, str) or not isinstance(raw_pronunciation, str):
            skipped += 1
            continue
        term = raw_term.strip()
        pronunciation = raw_pronunciation.strip()
        if not term or not pronunciation:
            skipped += 1
            continue
        normalized_key = term.casefold()
        existing_key = keys_by_casefold.get(normalized_key)
        if existing_key is not None:
            entries[existing_key] = pronunciation
        else:
            entries[term] = pronunciation
            keys_by_casefold[normalized_key] = term

    if not entries:
        raise ValueError("O arquivo não contém entradas válidas de termo e pronúncia.")
    return entries, skipped


def load_pronunciation_dictionary_json(
    path: str | Path,
) -> tuple[dict[str, str], int]:
    """Load a UTF-8 pronunciation dictionary JSON file, with optional BOM."""
    source = Path(path).expanduser()
    content = source.read_text(encoding="utf-8-sig")
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON inválido na linha {exc.lineno}, coluna {exc.colno}: {exc.msg}"
        ) from exc
    return _normalize_pairs(raw)


def merge_pronunciation_entries(
    current: dict[str, str],
    imported: dict[str, str],
) -> tuple[dict[str, str], int, int]:
    """Merge imported terms case-insensitively, preserving existing term spelling."""
    merged = dict(current)
    keys_by_casefold = {term.casefold(): term for term in merged}
    added = 0
    replaced = 0

    for raw_term, raw_pronunciation in imported.items():
        term = str(raw_term).strip()
        pronunciation = str(raw_pronunciation).strip()
        if not term or not pronunciation:
            continue
        existing_key = keys_by_casefold.get(term.casefold())
        if existing_key is None:
            merged[term] = pronunciation
            keys_by_casefold[term.casefold()] = term
            added += 1
        elif merged[existing_key] != pronunciation:
            merged[existing_key] = pronunciation
            replaced += 1

    return merged, added, replaced
