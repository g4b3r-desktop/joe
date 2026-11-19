from joe.transcript_sync import active_segment_index, normalize_timed_segments


def test_normalize_timed_segments_discards_invalid_and_sorts():
    result = normalize_timed_segments([
        {"start": 2, "end": 3, "text": " segundo "},
        {"start": 1, "end": 1, "text": "inválido"},
        {"start": -1, "end": 1, "text": "negativo"},
        {"start": 0, "end": 1, "text": "primeiro", "speaker": "A"},
        {"start": "x", "end": 4, "text": "sem tempo"},
        {"start": 3, "end": 4, "text": " "},
    ])
    assert [item["text"] for item in result] == ["primeiro", "segundo"]
    assert result[0]["speaker"] == "A"
    assert result[0]["start"] == 0.0


def test_active_segment_index_tracks_position_and_gaps():
    segments = normalize_timed_segments([
        {"start": 0.5, "end": 1.5, "text": "um"},
        {"start": 2.0, "end": 3.0, "text": "dois"},
    ])
    assert active_segment_index(segments, 0) is None
    assert active_segment_index(segments, 500) == 0
    assert active_segment_index(segments, 1499) == 0
    assert active_segment_index(segments, 1500) is None
    assert active_segment_index(segments, 2500) == 1
    assert active_segment_index(segments, 3000) is None


def test_active_segment_index_handles_empty_segments():
    assert active_segment_index([], 1000) is None
