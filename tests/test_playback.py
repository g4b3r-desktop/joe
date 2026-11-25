from __future__ import annotations

from joe.playback import (
    decide_clipboard_shortcut_action,
    decide_playback_action,
)


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



def test_clipboard_shortcut_focuses_existing_dialog_while_generation_is_active():
    assert decide_clipboard_shortcut_action(
        dialog_open=True, generation_active=True
    ) == "focus"


def test_clipboard_shortcut_replaces_completed_dialog_with_new_clipboard_text():
    assert decide_clipboard_shortcut_action(
        dialog_open=True, generation_active=False
    ) == "replace"


def test_clipboard_shortcut_starts_a_new_dialog_when_none_is_open():
    assert decide_clipboard_shortcut_action(
        dialog_open=False, generation_active=False
    ) == "start"


def test_clipboard_shortcut_waits_when_worker_has_no_dialog():
    assert decide_clipboard_shortcut_action(
        dialog_open=False, generation_active=True
    ) == "busy"
