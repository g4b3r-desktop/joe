from __future__ import annotations

import re
from datetime import date
from typing import Mapping

LANGUAGE_LOCALES = {
    "portuguese": "pt_BR",
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "dutch": "nl",
}

_MONTHS = {
    "portuguese": (
        "janeiro", "fevereiro", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
    ),
    "english": (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ),
    "spanish": (
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ),
    "french": (
        "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre",
    ),
    "german": (
        "Januar", "Februar", "März", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember",
    ),
    "italian": (
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    ),
    "dutch": (
        "januari", "februari", "maart", "april", "mei", "juni",
        "juli", "augustus", "september", "oktober", "november", "december",
    ),
}

_DECIMAL_WORDS = {
    "portuguese": "vírgula",
    "english": "point",
    "spanish": "coma",
    "french": "virgule",
    "german": "Komma",
    "italian": "virgola",
    "dutch": "komma",
}
_PERCENT_WORDS = {
    "portuguese": "por cento",
    "english": "percent",
    "spanish": "por ciento",
    "french": "pour cent",
    "german": "Prozent",
    "italian": "per cento",
    "dutch": "procent",
}

_PT_ABBREVIATIONS = {
    "dr": "doutor",
    "dra": "doutora",
    "sr": "senhor",
    "sra": "senhora",
    "srta": "senhorita",
    "prof": "professor",
    "profa": "professora",
    "etc": "etcetera",
}

_PT_LETTERS = {
    "A": "a", "B": "bê", "C": "cê", "D": "dê", "E": "e", "F": "éfe",
    "G": "gê", "H": "agá", "I": "i", "J": "jota", "K": "cá", "L": "éle",
    "M": "ême", "N": "êne", "O": "ó", "P": "pê", "Q": "quê", "R": "érre",
    "S": "ésse", "T": "tê", "U": "u", "V": "vê", "W": "dáblio",
    "X": "xis", "Y": "ípsilon", "Z": "zê",
}
_PT_ACRONYMS = {
    "API": "a pê i",
    "ASR": "a ésse érre",
    "CPU": "cê pê u",
    "CSS": "cê ésse ésse",
    "GPU": "gê pê u",
    "HTML": "agá tê eme éle",
    "HTTP": "agá tê tê pê",
    "HTTPS": "agá tê tê pê ésse",
    "IA": "i a",
    "LLM": "éle éle ême",
    "OCR": "ó cê érre",
    "PDF": "pê dê éfe",
    "RAM": "rãm",
    "RGB": "érre gê bê",
    "ROM": "rôm",
    "RTX": "érre tê xis",
    "SSD": "ésse ésse dê",
    "STT": "ésse tê tê",
    "TTS": "tê tê ésse",
    "URL": "u érre éle",
    "USB": "u ésse bê",
    "VRAM": "vê rãm",
}
_PT_WORD_LIKE_BRANDS = {
    "BLENDER", "BOM", "COLAB", "FIM", "JOE", "LINUX", "NAO", "NÃO",
    "LASER", "NASA", "NVIDIA", "OK", "ONU", "OPENAI", "POCKET",
    "PYTHON", "QWEN", "SIM", "STEAM", "UNESCO",
}

_DATE_PATTERN = re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)")
_ISO_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_TIME_PATTERN = re.compile(r"(?<!\w)([01]?\d|2[0-3]):([0-5]\d)(?!\w)")
_PERCENT_PATTERN = re.compile(r"(?<![\w./-])([+-]?\d+(?:[.,]\d+)*)(\s*%)(?!\w)")
_CURRENCY_PATTERN = re.compile(r"(?<!\w)R\$\s*([0-9][0-9.,]*)(?!\w)", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"(?<![\w./-])([+-]?\d+(?:[.,]\d+)*)(?![\w/-]|\.\d)")
_ACRONYM_PATTERN = re.compile(r"(?<!\w)([A-Z]{2,6})(?!\w)")


def _number_to_words(value: int, language: str) -> str:
    try:
        from num2words import num2words
    except ImportError as exc:
        raise RuntimeError(
            "A normalização de números precisa do pacote num2words. "
            "Atualize as dependências do Joe com pip install -e ."
        ) from exc

    # The comma is formatting punctuation in num2words output, not speech punctuation.
    return str(num2words(value, lang=LANGUAGE_LOCALES.get(language, "pt_BR"))).replace(",", "")


def _split_number(raw: str, language: str) -> tuple[str, str | None, bool]:
    """Return integer digits, optional decimal digits and sign."""
    value = raw.strip()
    negative = value.startswith("-")
    value = value.lstrip("+-")

    decimal_separator = "," if language in {
        "portuguese", "spanish", "german", "italian", "dutch"
    } else "."
    group_separator = "." if decimal_separator == "," else ","

    integer_part = value
    fractional_part = None
    if decimal_separator in value and group_separator in value:
        last_decimal = max(value.rfind(decimal_separator), value.rfind(group_separator))
        integer_part = value[:last_decimal].replace(decimal_separator, "").replace(
            group_separator, ""
        )
        fractional_part = value[last_decimal + 1:]
    elif decimal_separator in value:
        if value.count(decimal_separator) == 1:
            left, right = value.split(decimal_separator, 1)
            if decimal_separator == "." or len(right) != 3:
                integer_part, fractional_part = left, right
            else:
                integer_part = value.replace(decimal_separator, "")
        else:
            integer_part = value.replace(decimal_separator, "")
    elif group_separator in value:
        pieces = value.split(group_separator)
        if all(len(piece) == 3 for piece in pieces[1:]):
            integer_part = "".join(pieces)
        elif len(pieces) == 2 and len(pieces[1]) != 3:
            integer_part, fractional_part = pieces
        else:
            raise ValueError(f"Separadores numéricos inválidos: {raw}")

    if not integer_part.isdigit() or (
        fractional_part is not None and not fractional_part.isdigit()
    ):
        raise ValueError(f"Número inválido: {raw}")
    return integer_part, fractional_part, negative


def _numeric_words(raw: str, language: str) -> str:
    integer, fractional, negative = _split_number(raw, language)
    words = _number_to_words(int(integer or "0"), language)
    if negative:
        minus_words = {
            "portuguese": "menos", "spanish": "menos", "french": "moins",
            "german": "minus", "italian": "meno", "dutch": "min",
        }
        words = f"{minus_words.get(language, 'minus')} {words}"
    if fractional:
        decimal_word = _DECIMAL_WORDS.get(language, "vírgula")
        digit_words = " ".join(_number_to_words(int(digit), language) for digit in fractional)
        words = f"{words} {decimal_word} {digit_words}"
    return words


def _replace_dates(text: str, language: str) -> str:
    months = _MONTHS.get(language, _MONTHS["portuguese"])

    def format_date(day: int, month: int, year: int) -> str | None:
        try:
            date(year, month, day)
        except ValueError:
            return None

        day_words = _number_to_words(day, language)
        year_words = _number_to_words(year, language)
        month_name = months[month - 1]
        if language == "english":
            try:
                from num2words import num2words
                ordinal = str(num2words(day, lang="en", to="ordinal"))
            except (ImportError, NotImplementedError):
                ordinal = day_words
            return f"{month_name} {ordinal}, {year_words}"
        if language == "german":
            return f"{day_words} {month_name} {year_words}"
        if language in {"portuguese", "spanish"}:
            return f"{day_words} de {month_name} de {year_words}"
        return f"{day_words} {month_name} {year_words}"

    def replace_dmy(match: re.Match[str]) -> str:
        first, second, year = (int(part) for part in match.groups())
        if language == "english":
            month, day = first, second
        else:
            day, month = first, second
        return format_date(day, month, year) or match.group(0)

    def replace_iso(match: re.Match[str]) -> str:
        year, month, day = (int(part) for part in match.groups())
        return format_date(day, month, year) or match.group(0)

    text = _DATE_PATTERN.sub(replace_dmy, text)
    return _ISO_DATE_PATTERN.sub(replace_iso, text)


def _replace_times(text: str, language: str) -> str:
    hour_units = {
        "portuguese": ("hora", "horas"), "english": ("hour", "hours"),
        "spanish": ("hora", "horas"), "french": ("heure", "heures"),
        "german": ("Uhr", "Uhr"), "italian": ("ora", "ore"),
        "dutch": ("uur", "uur"),
    }
    minute_units = {
        "portuguese": ("minuto", "minutos"), "english": ("minute", "minutes"),
        "spanish": ("minuto", "minutos"), "french": ("minute", "minutes"),
        "german": ("Minute", "Minuten"), "italian": ("minuto", "minuti"),
        "dutch": ("minuut", "minuten"),
    }
    conjunctions = {
        "portuguese": "e", "english": "and", "spanish": "y",
        "french": "et", "german": "und", "italian": "e", "dutch": "en",
    }
    one_hour_words = {
        "portuguese": "uma", "english": "one", "spanish": "una",
        "french": "une", "german": "ein", "italian": "una", "dutch": "één",
    }
    one_minute_words = {
        "portuguese": "um", "english": "one", "spanish": "un",
        "french": "une", "german": "eine", "italian": "un", "dutch": "één",
    }

    def replace(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        hour_words = one_hour_words.get(language, "um") if hour == 1 else _number_to_words(
            hour, language
        )
        hour_unit = hour_units.get(language, ("", ""))[0 if hour == 1 else 1]
        if minute == 0:
            return f"{hour_words} {hour_unit}".strip()

        minute_words = (
            one_minute_words.get(language, "um")
            if minute == 1
            else _number_to_words(minute, language)
        )
        minute_unit = minute_units.get(language, ("", ""))[0 if minute == 1 else 1]
        conjunction = conjunctions.get(language, "e")
        return (
            f"{hour_words} {hour_unit} {conjunction} {minute_words} {minute_unit}"
        ).strip()

    return _TIME_PATTERN.sub(replace, text)


def _replace_portuguese_abbreviations(text: str) -> str:
    for abbreviation, spoken in _PT_ABBREVIATIONS.items():
        pattern = re.compile(rf"(?<!\w){re.escape(abbreviation)}\.?(?!\w)", re.IGNORECASE)
        text = pattern.sub(spoken, text)
    return text


def _replace_portuguese_acronyms(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        acronym = match.group(1)
        if acronym in _PT_WORD_LIKE_BRANDS:
            return acronym
        if acronym in _PT_ACRONYMS:
            return _PT_ACRONYMS[acronym]
        return " ".join(_PT_LETTERS.get(letter, letter.lower()) for letter in acronym)

    return _ACRONYM_PATTERN.sub(replace, text)


def normalize_text_for_tts(
    text: str,
    language: str = "portuguese",
    pronunciation_dictionary: Mapping[str, str] | None = None,
    apply_automatic: bool = True,
) -> str:
    """Prepare a TTS-only copy of text without modifying the source shown in the UI.

    Custom pronunciation entries are applied first and protected from subsequent
    normalization, so their phonetic spelling is sent to the model unchanged.
    """
    if not text:
        return text

    normalized = text
    protected: dict[str, str] = {}
    entries = pronunciation_dictionary or {}
    for index, (term, pronunciation) in enumerate(
        sorted(entries.items(), key=lambda item: len(item[0]), reverse=True)
    ):
        term = str(term).strip()
        pronunciation = str(pronunciation).strip()
        if not term or not pronunciation:
            continue
        placeholder = f"\uE000PRON{index}TOKEN\uE001"
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        if pattern.search(normalized):
            normalized = pattern.sub(lambda _match, key=placeholder: key, normalized)
            protected[placeholder] = pronunciation

    if not apply_automatic:
        for placeholder, pronunciation in protected.items():
            normalized = normalized.replace(placeholder, pronunciation)
        return normalized

    normalized = _replace_dates(normalized, language)
    normalized = _replace_times(normalized, language)

    if language == "portuguese":
        normalized = _replace_portuguese_abbreviations(normalized)
        normalized = _replace_portuguese_acronyms(normalized)

    def replace_percent(match: re.Match[str]) -> str:
        number_words = _numeric_words(match.group(1), language)
        return f"{number_words} {_PERCENT_WORDS.get(language, 'por cento')}"

    def safe_percent(match: re.Match[str]) -> str:
        try:
            return replace_percent(match)
        except ValueError:
            return match.group(0)

    normalized = _PERCENT_PATTERN.sub(safe_percent, normalized)

    def replace_currency(match: re.Match[str]) -> str:
        integer, fractional, negative = _split_number(match.group(1), "portuguese")
        amount = int(integer or "0")
        result = _number_to_words(amount, "portuguese")
        if negative:
            result = f"menos {result}"
        result += " real" if amount == 1 else " reais"
        if fractional:
            cents = int(fractional.ljust(2, "0")[:2])
            if cents:
                cent_label = "centavo" if cents == 1 else "centavos"
                result += f" e {_number_to_words(cents, 'portuguese')} {cent_label}"
        return result

    def safe_currency(match: re.Match[str]) -> str:
        try:
            return replace_currency(match)
        except ValueError:
            return match.group(0)

    normalized = _CURRENCY_PATTERN.sub(safe_currency, normalized)

    def replace_number(match: re.Match[str]) -> str:
        try:
            return _numeric_words(match.group(1), language)
        except ValueError:
            return match.group(0)

    normalized = _NUMBER_PATTERN.sub(replace_number, normalized)

    for placeholder, pronunciation in protected.items():
        normalized = normalized.replace(placeholder, pronunciation)

    return normalized
