from __future__ import annotations

import json

import pytest

from joe.pronunciation_dictionary import (
    load_pronunciation_dictionary_json,
    merge_pronunciation_entries,
)


def test_loads_simple_term_to_pronunciation_object(tmp_path):
    source = tmp_path / "pronunciation.json"
    source.write_text(
        json.dumps({"cpu": "cê pê u", "gpu": "gê pê u"}, ensure_ascii=False),
        encoding="utf-8",
    )

    entries, skipped = load_pronunciation_dictionary_json(source)

    assert entries == {"cpu": "cê pê u", "gpu": "gê pê u"}
    assert skipped == 0


def test_loads_utf8_bom_and_nested_entries_object(tmp_path):
    source = tmp_path / "pronunciation.json"
    source.write_text(
        '\ufeff{"entries": {"Three.js": "thri dji és"}}',
        encoding="utf-8",
    )

    entries, skipped = load_pronunciation_dictionary_json(source)

    assert entries == {"Three.js": "thri dji és"}
    assert skipped == 0


def test_loads_list_of_term_and_pronunciation_items(tmp_path):
    source = tmp_path / "pronunciation.json"
    source.write_text(
        json.dumps(
            [
                {"term": "CPU", "pronunciation": "cê pê u"},
                {"term": "GPU", "pronunciation": "gê pê u"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entries, skipped = load_pronunciation_dictionary_json(source)

    assert entries == {"CPU": "cê pê u", "GPU": "gê pê u"}
    assert skipped == 0


def test_skips_invalid_items_but_keeps_valid_entries(tmp_path):
    source = tmp_path / "pronunciation.json"
    source.write_text(
        json.dumps(
            {
                "cpu": "cê pê u",
                "": "pronúncia vazia",
                "gpu": " ",
                "number": 42,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entries, skipped = load_pronunciation_dictionary_json(source)

    assert entries == {"cpu": "cê pê u"}
    assert skipped == 3


def test_invalid_list_items_are_counted(tmp_path):
    source = tmp_path / "pronunciation.json"
    source.write_text(
        json.dumps([
            {"term": "API", "pronunciation": "a pê i"},
            "bad entry",
            {"term": "GPU", "pronunciation": ""},
        ]),
        encoding="utf-8",
    )

    entries, skipped = load_pronunciation_dictionary_json(source)

    assert entries == {"API": "a pê i"}
    assert skipped == 2


def test_reports_invalid_json_and_files_without_valid_entries(tmp_path):
    malformed = tmp_path / "bad.json"
    malformed.write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON inválido"):
        load_pronunciation_dictionary_json(malformed)

    empty = tmp_path / "empty.json"
    empty.write_text('{"cpu": ""}', encoding="utf-8")
    with pytest.raises(ValueError, match="não contém entradas válidas"):
        load_pronunciation_dictionary_json(empty)


def test_merge_adds_new_terms_and_overwrites_duplicates_case_insensitively():
    merged, added, replaced = merge_pronunciation_entries(
        {"CPU": "c p u", "GPU": "g p u"},
        {"cpu": "cê pê u", "Three.js": "thri dji és"},
    )

    assert merged == {
        "CPU": "cê pê u",
        "GPU": "g p u",
        "Three.js": "thri dji és",
    }
    assert added == 1
    assert replaced == 1


def test_large_dictionary_import_keeps_all_entries(tmp_path):
    source = tmp_path / "many-entries.json"
    dictionary = {
        f"term-{index:04d}": f"pronunciation {index}"
        for index in range(2500)
    }
    source.write_text(json.dumps(dictionary), encoding="utf-8")

    entries, skipped = load_pronunciation_dictionary_json(source)

    assert len(entries) == 2500
    assert skipped == 0
