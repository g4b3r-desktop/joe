from joe.text_normalization import normalize_text_for_tts


def test_portuguese_normalizes_numbers_percent_dates_and_times():
    result = normalize_text_for_tts(
        "Tenho 2 gatos, 25%, em 18/11/2026 às 10:30."
    )

    assert "dois gatos" in result
    assert "vinte e cinco por cento" in result
    assert "dezoito de novembro de dois mil e vinte e seis" in result
    assert "dez horas e trinta minutos" in result


def test_portuguese_normalizes_currency_and_decimal_numbers():
    result = normalize_text_for_tts("Custa R$ 1.234,50 ou 2,5.")

    assert "mil duzentos e trinta e quatro reais e cinquenta centavos" in result
    assert "dois vírgula cinco" in result


def test_portuguese_expands_common_abbreviations_and_acronyms():
    result = normalize_text_for_tts("Dr. Silva usa GPU, CPU, API e TTS.")

    assert "doutor Silva" in result
    assert "gê pê u" in result
    assert "cê pê u" in result
    assert "a pê i" in result
    assert "tê tê ésse" in result


def test_custom_pronunciation_dictionary_overrides_normalization():
    result = normalize_text_for_tts(
        "O Qwen3.5 usa GPU.",
        pronunciation_dictionary={
            "Qwen3.5": "cuên três ponto cinco",
            "GPU": "gê pê u",
        },
    )

    assert result == "O cuên três ponto cinco usa gê pê u."


def test_custom_dictionary_is_case_insensitive_and_respects_word_boundaries():
    result = normalize_text_for_tts(
        "API e APIs.",
        pronunciation_dictionary={"api": "a pê i"},
    )

    assert result == "a pê i e APIs."


def test_iso_date_is_normalized():
    result = normalize_text_for_tts("A data é 2026-11-18.")

    assert "dezoito de novembro de dois mil e vinte e seis" in result


def test_invalid_date_is_left_unchanged():
    result = normalize_text_for_tts("Data inválida: 31/02/2026.")

    assert "31/02/2026" in result


def test_normalizer_does_not_change_source_string():
    original = "Tenho 2 gatos e uso GPU."
    normalize_text_for_tts(original)

    assert original == "Tenho 2 gatos e uso GPU."



def test_english_decimal_point_is_not_treated_as_thousands_separator():
    result = normalize_text_for_tts("The value is 1.234.", language="english")

    assert "one point two three four" in result


def test_portuguese_accepts_mixed_thousands_and_decimal_separators():
    result = normalize_text_for_tts("O valor é 1,234.56.")

    assert "mil duzentos e trinta e quatro vírgula cinco seis" in result


def test_invalid_numeric_grouping_is_preserved():
    result = normalize_text_for_tts("Código 1.234.5.")

    assert "1.234.5" in result



def test_english_slash_date_uses_month_day_year_order():
    result = normalize_text_for_tts("The meeting is on 11/18/2026.", language="english")

    assert "November eighteenth" in result



def test_custom_dictionary_remains_active_when_automatic_normalization_is_disabled():
    result = normalize_text_for_tts(
        "GPU custa 25%.",
        pronunciation_dictionary={"GPU": "gê pê u"},
        apply_automatic=False,
    )

    assert result == "gê pê u custa 25%."



def test_decimal_numbers_are_normalized_before_sentence_punctuation():
    result = normalize_text_for_tts("O valor é 2,5.")

    assert result == "O valor é dois vírgula cinco."


def test_currency_is_normalized_before_sentence_punctuation():
    result = normalize_text_for_tts("O preço é R$ 1.234,50.")

    assert result == (
        "O preço é mil duzentos e trinta e quatro reais "
        "e cinquenta centavos."
    )


def test_dotted_version_numbers_are_not_partially_normalized():
    result = normalize_text_for_tts("A versão é 1.2.3.")

    assert result == "A versão é 1.2.3."



def test_advanced_pronunciation_rules_are_applied_before_tts_normalization():
    result = normalize_text_for_tts(
        "Uso Joe e GPU.",
        language="portuguese",
        pronunciation_rules=[
            {
                "id": "joe-pt",
                "term": "Joe",
                "pronunciation": "Jô",
                "language": "portuguese",
                "match_mode": "word",
                "case_sensitive": False,
                "enabled": True,
                "priority": 5,
            }
        ],
        apply_automatic=False,
    )

    assert result == "Uso Jô e GPU."



def test_advanced_pronunciation_is_not_modified_by_automatic_rules():
    result = normalize_text_for_tts(
        "Qwen3.5 usa GPU.",
        pronunciation_rules=[
            {
                "id": "qwen",
                "term": "Qwen3.5",
                "pronunciation": "Qwen três ponto cinco",
                "language": "all",
                "match_mode": "word",
                "case_sensitive": False,
                "enabled": True,
                "priority": 10,
            }
        ],
    )

    assert result == "Qwen três ponto cinco usa gê pê u."



def test_japanese_text_is_preserved_and_custom_dictionary_still_applies():
    original = "今日は。2026年11月22日、価格は25%です。"
    result = normalize_text_for_tts(
        original,
        language="japanese",
        pronunciation_dictionary={"今日は": "こんにちは"},
        apply_automatic=True,
    )

    assert result == "こんにちは。2026年11月22日、価格は25%です。"
