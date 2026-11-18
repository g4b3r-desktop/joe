from __future__ import annotations

from joe.playback import decide_playback_action


def test_playback_switches_from_an_unrelated_audio(tmp_path):
    current = tmp_path / "voice-preview.wav"
    target = tmp_path / "generated.wav"

    assert decide_playback_action(
        str(current), target, is_playing=True, is_paused=False
    ) == "play"


def test_playback_stops_only_the_same_audio_when_already_playing(tmp_path):
    target = tmp_path / "generated.wav"

    assert decide_playback_action(
        str(target), target, is_playing=True, is_paused=False
    ) == "stop"


def test_playback_resumes_the_same_paused_audio(tmp_path):
    target = tmp_path / "generated.wav"

    assert decide_playback_action(
        str(target), target, is_playing=False, is_paused=True
    ) == "resume"
