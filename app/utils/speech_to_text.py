from typing import BinaryIO

from faster_whisper import WhisperModel


def transcribe_audio(model: WhisperModel, audio_file: BinaryIO, lang: str):
    segments, info = model.transcribe(
        audio_file,
        language=lang,
    )

    text = "".join(segment.text for segment in segments)

    return text
