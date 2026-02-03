"""
Audio transcription module using OpenAI Whisper API.

Provides server-side speech-to-text for push-to-talk feature.
"""
import tempfile
import os
from pathlib import Path

from openai import OpenAI

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

client = OpenAI(api_key=settings.openai_api_key)

# Supported audio formats by Whisper
SUPPORTED_FORMATS = {".webm", ".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".flac"}


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    language: str | None = None,
) -> dict:
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_bytes: Raw audio data
        filename: Original filename (used to determine format)
        language: Optional ISO-639-1 language code (e.g., "en", "es")
                  If None, Whisper auto-detects language.

    Returns:
        Dict with keys:
        - text: Transcribed text
        - language: Detected/specified language
        - duration_seconds: Audio duration (if available)

    Raises:
        ValueError: If audio format not supported
        Exception: If transcription fails
    """
    # Validate format
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {suffix}. Supported: {SUPPORTED_FORMATS}")

    # Write to temp file (Whisper API requires file)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        logger.debug(f"Transcribing audio | size={len(audio_bytes)} | format={suffix}")

        # Call Whisper API
        with open(tmp_path, "rb") as audio_file:
            kwargs = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "verbose_json",
            }
            if language:
                kwargs["language"] = language

            response = client.audio.transcriptions.create(**kwargs)

        text = response.text.strip()
        detected_language = getattr(response, "language", language or "unknown")
        duration = getattr(response, "duration", None)

        logger.info(
            f"Transcription complete | chars={len(text)} | "
            f"language={detected_language} | duration={duration}"
        )

        return {
            "text": text,
            "language": detected_language,
            "duration_seconds": duration,
        }

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
