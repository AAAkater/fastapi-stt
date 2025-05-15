from faster_whisper import WhisperModel

from app.utils.speech_to_text import transcribe_audio

# model_path = "./models/faster-distil-whisper-large-v2"
# model_path = "./models/distil-large-v3.5-ct2"
model_path = "./models/faster-whisper-large-v3"
# audio_path = "./assets/hotwords.mp3"
audio_path = "./assets/[丁真].mp3"

model_large_v3 = WhisperModel(
    model_path,
    device="cuda",
    compute_type="float16",
    local_files_only=True,
)


with open(audio_path, "rb") as audio_file:
    text = transcribe_audio(model_large_v3, audio_file, "zh")

    print(text)
