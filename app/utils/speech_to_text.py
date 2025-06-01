from typing import BinaryIO

from faster_whisper import WhisperModel

from app.core.config import settings


class SttModel:
    def __init__(self, model_path: str):
        self.model = WhisperModel(
            model_path,
            device=settings.device,
        )

    def stt(self, audio_file: BinaryIO, lang: str):
        segments, info = self.model.transcribe(
            audio_file,
            language=lang,
        )

        text = "".join(segment.text for segment in segments)

        return text


model = SttModel(settings.model_path)
