from bpm_ai_core.question_answering.transformers_qa import TransformersExtractiveQA
from bpm_ai_core.ocr.tesseract import TesseractOCR
from bpm_ai_core.util.image import load_images


def test_qa():
    context = "My name is John and I live in Hawaii"
    question = "Where does John live?"
    expected = "Hawaii"

    qa = TransformersExtractiveQA()
    actual = qa.answer(context, question)

    assert actual.strip() == expected


async def test_qa_ocr():
    ocr = TesseractOCR()
    images = load_images("dummy.pdf")
    text = await ocr.images_to_text(images)

    question = "What kind of PDF is this?"

    qa = TransformersExtractiveQA()
    actual = qa.answer(text, question)

    assert actual.strip() == "Dummy"


def test_qa_unanswerable():
    context = "My name is John and I live in Hawaii"
    question = "How much is the fish?"

    qa = TransformersExtractiveQA()
    actual = qa.answer(context, question, confidence_threshold=0.1)

    assert actual is None
