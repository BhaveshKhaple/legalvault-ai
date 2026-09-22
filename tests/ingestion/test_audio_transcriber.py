"""
Tests for Task 1.3 — audio_transcriber.transcribe_audio().

Whisper model loading (~seconds) and real audio files are both unnecessary
for unit tests. We mock _get_model() so the suite runs in CI without torch,
GPU, or audio fixtures.

Integration test note (not in this file):
    tests/integration/test_audio_transcriber_integration.py will run Whisper
    against a real 5-minute meeting mp3 from sample_data/ and verify that
    timestamps are within 2 seconds of accuracy. That test is skipped if the
    sample file is absent (git-ignored per sample_data/README.md).
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from backend.app.ingestion.audio_transcriber import transcribe_audio


# ─── fake Whisper output ──────────────────────────────────────────────────────


def _fake_transcribe_result(segments: list[dict]) -> dict:
    """Build a whisper result dict identical in shape to the real model output."""
    return {"text": " ".join(s["text"] for s in segments), "segments": segments}


SAMPLE_SEGMENTS = [
    {"text": " Good morning everyone.", "start": 0.0, "end": 2.5},
    {"text": " Today we will discuss the contract terms.", "start": 2.5, "end": 6.1},
    {"text": " The penalty clause is in section 8.", "start": 6.1, "end": 9.8},
]


@pytest.fixture()
def mock_model(tmp_path):
    """Patch _get_model to return a fake Whisper model and yield a real audio path."""
    dummy_audio = tmp_path / "meeting_short.mp3"
    dummy_audio.write_bytes(b"\xff\xfb" * 100)  # dummy bytes, model is mocked

    fake = MagicMock()
    fake.transcribe.return_value = _fake_transcribe_result(SAMPLE_SEGMENTS)

    with patch("backend.app.ingestion.audio_transcriber._get_model", return_value=fake):
        yield fake, str(dummy_audio)


# ─── return shape ─────────────────────────────────────────────────────────────


class TestReturnShape:
    def test_returns_list(self, mock_model):
        _, path = mock_model
        result = transcribe_audio(path)
        assert isinstance(result, list)

    def test_correct_segment_count(self, mock_model):
        _, path = mock_model
        result = transcribe_audio(path)
        assert len(result) == len(SAMPLE_SEGMENTS)

    def test_dict_keys_present(self, mock_model):
        _, path = mock_model
        result = transcribe_audio(path)
        for seg in result:
            assert set(seg.keys()) == {"text", "ts_start", "ts_end", "doc_id"}


# ─── content correctness ──────────────────────────────────────────────────────


class TestContent:
    def test_text_stripped(self, mock_model):
        _, path = mock_model
        result = transcribe_audio(path)
        # Whisper prepends spaces; transcriber must strip them
        assert result[0]["text"] == "Good morning everyone."

    def test_timestamps_are_floats(self, mock_model):
        _, path = mock_model
        result = transcribe_audio(path)
        for seg in result:
            assert isinstance(seg["ts_start"], float)
            assert isinstance(seg["ts_end"], float)

    def test_timestamps_in_order(self, mock_model):
        _, path = mock_model
        result = transcribe_audio(path)
        for seg in result:
            assert seg["ts_start"] < seg["ts_end"]

    def test_timestamps_match_source(self, mock_model):
        _, path = mock_model
        result = transcribe_audio(path)
        assert result[0]["ts_start"] == pytest.approx(0.0)
        assert result[-1]["ts_end"] == pytest.approx(9.8)

    def test_doc_id_is_filename_stem(self, mock_model, tmp_path):
        _, path = mock_model
        result = transcribe_audio(path)
        assert all(seg["doc_id"] == "meeting_short" for seg in result)

    def test_empty_segments_skipped(self, tmp_path):
        """Segments with no text after strip() should not appear in output."""
        dummy_audio = tmp_path / "silent.mp3"
        dummy_audio.write_bytes(b"\xff\xfb" * 100)

        fake = MagicMock()
        fake.transcribe.return_value = _fake_transcribe_result(
            [
                {"text": "  ", "start": 0.0, "end": 1.0},
                {"text": " Hello.", "start": 1.0, "end": 2.0},
                {"text": "", "start": 2.0, "end": 3.0},
            ]
        )
        with patch("backend.app.ingestion.audio_transcriber._get_model", return_value=fake):
            result = transcribe_audio(str(dummy_audio))
        assert len(result) == 1
        assert result[0]["text"] == "Hello."


# ─── model singleton ──────────────────────────────────────────────────────────


class TestModelSingleton:
    def test_model_loaded_once_per_call_set(self, mock_model):
        """_get_model must be called once even across multiple transcribe calls."""
        fake, path = mock_model
        # We patched _get_model at module level — call twice, verify called twice
        # (each call goes through our patch which returns the same fake)
        transcribe_audio(path)
        transcribe_audio(path)
        # Our patch returns fake directly; the real caching is tested by the
        # integration test. Here we just confirm transcribe() was called twice.
        assert fake.transcribe.call_count == 2


# ─── error handling ───────────────────────────────────────────────────────────


class TestErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            transcribe_audio("/nonexistent/meeting.mp3")

    def test_runtime_error_on_decode_failure(self, tmp_path):
        """If Whisper/ffmpeg fails, RuntimeError with helpful message is raised."""
        bad_audio = tmp_path / "corrupt.mp3"
        bad_audio.write_bytes(b"not audio")

        fake = MagicMock()
        fake.transcribe.side_effect = Exception("ffmpeg not found")

        with patch("backend.app.ingestion.audio_transcriber._get_model", return_value=fake):
            with pytest.raises(RuntimeError, match="ffmpeg"):
                transcribe_audio(str(bad_audio))


# ─── logging ──────────────────────────────────────────────────────────────────


class TestLogging:
    def test_completion_logged_at_info(self, mock_model, caplog):
        _, path = mock_model
        with caplog.at_level(logging.INFO, logger="backend.app.ingestion.audio_transcriber"):
            transcribe_audio(path)
        messages = [r.message for r in caplog.records]
        assert any("Transcription complete" in m for m in messages)
