from bpm_ai_core.speech_recognition.faster_whisper import FasterWhisperASR


async def test_faster_whisper():
    fw = FasterWhisperASR()
    text = await fw.transcribe("test.mp3")
    assert text.lower().strip() == "looking with a half-fantastic curiosity to see whether the tender grass of early spring"
