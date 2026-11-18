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
