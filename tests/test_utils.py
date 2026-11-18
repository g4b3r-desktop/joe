from joe.utils import build_srt, format_srt_time


def test_format_srt_time():
    assert format_srt_time(0.0) == "00:00:00,000"
    assert format_srt_time(61.234) == "00:01:01,234"
    assert format_srt_time(3661.999) == "01:01:02,000"


def test_build_srt():
    result = build_srt([
        {"start": 0.2, "end": 1.4, "text": "Olá mundo."},
        {"start": 1.5, "end": 3.0, "text": "Tudo bem?"},
    ])
    assert "1\n00:00:00,200 --> 00:00:01,400\nOlá mundo." in result
    assert "2\n00:00:01,500 --> 00:00:03,000\nTudo bem?" in result


def test_build_srt_skips_empty_segments():
    assert build_srt([{"start": 0, "end": 1, "text": " " }]) == ""
