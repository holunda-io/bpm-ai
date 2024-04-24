from bpm_ai_core.speech_recognition.openai_whisper import OpenAIWhisperASR


async def test_asr():
    asr = OpenAIWhisperASR()
    result = await asr.transcribe(audio_or_path="files/example.mp3")

    assert "half-fantastic curiosity" in result.text.lower()