"""
Task 1.3 — Audio transcriber via OpenAI Whisper (local, no cloud).

Transcribes audio files with per-segment timestamps so the AI can later
cite "said at 12:34 in Meeting.mp3".

The Whisper model is loaded once (global singleton) to avoid ~3–10s reload
cost on every call. CPU inference only — no GPU required.

Prerequisites (system-level):
    ffmpeg must be installed and on PATH. Verify with: ffmpeg -version
    openai-whisper Python package: pip install openai-whisper

Model size guidance (from tracker Task 1.3):
    "base"   — fast, good for dev/CI, ~74M params
    "small"  — balanced, ~244M params
    "medium" — best accuracy, ~769M params, use for production runs
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level singleton. Keyed by model_size so switching tiers triggers a
# fresh load rather than reusing a mismatched model.
_loaded_models: dict[str, Any] = {}


def _get_model(model_size: str) -> Any:
    """Load Whisper model once per size; return cached instance thereafter."""
    if model_size not in _loaded_models:
        # Import here so the module is importable even when whisper is absent
        # (unit tests mock this function).
        import whisper  # noqa: PLC0415

        logger.info("Loading Whisper model '%s' — this takes a few seconds the first time.", model_size)
        _loaded_models[model_size] = whisper.load_model(model_size)
        logger.info("Whisper model '%s' ready.", model_size)
    return _loaded_models[model_size]


def transcribe_audio(path: str, model_size: str = "base") -> list[dict]:
    """Transcribe an audio file using local Whisper.

    Args:
        path: Absolute or relative path to the audio file
              (mp3, wav, m4a, mp4, ogg, etc. — anything ffmpeg handles).
        model_size: Whisper model variant. See module docstring for guidance.
                    Defaults to "base" for dev speed.

    Returns:
        List of segment dicts::

            [
                {
                    "text":     str,   # transcribed text for this segment
                    "ts_start": float, # start timestamp in seconds
                    "ts_end":   float, # end timestamp in seconds
                    "doc_id":   str,   # derived from filename stem
                },
                ...
            ]

        Empty list if the audio contains no speech.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If ffmpeg is not installed or the file cannot be decoded.
    """
    audio_path = Path(path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    doc_id = audio_path.stem
    model = _get_model(model_size)

    logger.info("Transcribing '%s' with model '%s'...", audio_path.name, model_size)

    try:
        result = model.transcribe(str(audio_path), verbose=False)
    except Exception as exc:
        # ffmpeg missing, corrupt audio, unsupported codec, etc.
        raise RuntimeError(
            f"Failed to transcribe '{audio_path.name}': {exc}. "
            "Make sure ffmpeg is installed and on PATH."
        ) from exc

    segments = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue  # skip empty / silence segments
        segments.append(
            {
                "text": text,
                "ts_start": float(seg["start"]),
                "ts_end": float(seg["end"]),
                "doc_id": doc_id,
            }
        )

    logger.info(
        "Transcription complete: %d segments from '%s'.",
        len(segments),
        audio_path.name,
    )
    return segments
