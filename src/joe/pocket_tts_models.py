from __future__ import annotations

from typing import Final

# Tuples contain: Pocket TTS model identifier, UI label, and an explanatory tooltip.
MODEL_VARIANTS_BY_LANGUAGE: Final[dict[str, tuple[tuple[str, str, str], ...]]] = {
    "portuguese": (
        (
            "portuguese",
            "Padrão — Português",
            "Versão padrão oficial do Pocket TTS para português.",
        ),
        (
            "portuguese_24l",
            "24 camadas — Português (preview)",
            "Modelo maior de 24 camadas. A documentação oficial o classifica como preview; "
            "pode consumir mais memória e ser mais lento.",
        ),
    ),
    "english": (
        (
            "english",
            "Padrão — English 2026-10",
            "Versão padrão atual; o identificador english corresponde ao modelo English 2026-10.",
        ),
        ("english_2026-01", "English 2026-01", "Versão inglesa anterior de janeiro de 2026."),
        ("english_2026-04", "English 2026-04", "Versão inglesa anterior de abril de 2026."),
        ("english_2026-09", "English 2026-09", "Versão inglesa anterior de setembro de 2026."),
        (
            "english_2026-09_2",
            "English 2026-09 (revisão 2)",
            "Revisão 2 da versão de setembro de 2026.",
        ),
        (
            "english_drifting_26-09",
            "English — sampler Drifting",
            "Variante inglesa treinada com o sampler Drifting em vez do LSD padrão.",
        ),
        (
            "english_18m_2026-10",
            "English compacto — 18,2 M",
            "Modelo compacto de aproximadamente 18,2 milhões de parâmetros; prioriza velocidade.",
        ),
        (
            "english_18m_refl_2026-10",
            "English compacto — 18,2 M REFL",
            "Variante compacta de 18,2 M com ajuste adicional REFL.",
        ),
        (
            "english_7m_2026-10",
            "English compacto — 6,9 M",
            "Modelo compacto de aproximadamente 6,9 milhões de parâmetros; prioriza velocidade.",
        ),
        (
            "english_7m_refl_2026-10",
            "English compacto — 6,9 M REFL",
            "Variante compacta de 6,9 M com ajuste adicional REFL.",
        ),
        (
            "english_2m_2026-10",
            "English ultracompacto — 1,9 M",
            "Modelo ultracompacto de aproximadamente 1,9 milhão de parâmetros.",
        ),
        (
            "english_2m_refl_2026-10",
            "English ultracompacto — 1,9 M REFL",
            "Variante ultracompacta de 1,9 M com ajuste adicional REFL.",
        ),
    ),
    "spanish": (
        ("spanish", "Padrão — Espanhol", "Versão padrão oficial do Pocket TTS para espanhol."),
        (
            "spanish_24l",
            "24 camadas — Espanhol (preview)",
            "Modelo maior de 24 camadas, ainda em preview segundo a documentação oficial.",
        ),
    ),
    "french": (
        ("french", "Padrão — Francês", "Versão padrão oficial do Pocket TTS para francês."),
        (
            "french_24l",
            "24 camadas — Francês (preview)",
            "Modelo maior de 24 camadas, ainda em preview segundo a documentação oficial.",
        ),
    ),
    "italian": (
        ("italian", "Padrão — Italiano", "Versão padrão oficial do Pocket TTS para italiano."),
        (
            "italian_24l",
            "24 camadas — Italiano (preview)",
            "Modelo maior de 24 camadas, ainda em preview segundo a documentação oficial.",
        ),
    ),
    "german": (
        ("german", "Padrão — Alemão", "Versão padrão oficial do Pocket TTS para alemão."),
        (
            "german_24l",
            "24 camadas — Alemão (preview)",
            "Modelo maior de 24 camadas, ainda em preview segundo a documentação oficial.",
        ),
    ),
    "dutch": (
        ("dutch", "Padrão — Neerlandês", "Versão padrão oficial do Pocket TTS para neerlandês."),
        (
            "dutch_24l",
            "24 camadas — Neerlandês (preview)",
            "Modelo maior de 24 camadas, ainda em preview segundo a documentação oficial.",
        ),
    ),
}


def default_model_variant(language: str) -> str:
    """Return the standard Pocket TTS model identifier for a supported language."""
    options = MODEL_VARIANTS_BY_LANGUAGE.get(str(language))
    return options[0][0] if options else "portuguese"


def model_variants_for_language(
    language: str,
) -> tuple[tuple[str, str, str], ...]:
    """Return only the official model variants available for a language."""
    return MODEL_VARIANTS_BY_LANGUAGE.get(str(language), ())


def is_model_variant_for_language(language: str, model_variant: str) -> bool:
    """Validate that a model identifier is documented for the selected language."""
    return any(
        variant == str(model_variant)
        for variant, _label, _description in model_variants_for_language(language)
    )
