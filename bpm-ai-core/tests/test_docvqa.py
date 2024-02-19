from bpm_ai_core.question_answering.transformers_docvqa import TransformersDocVQA
from bpm_ai_core.util.image import load_images


async def test_docvqa():
    model = TransformersDocVQA()

    image = load_images("sample-invoice.webp")
    question = "What is the total?"

    text = model.answer(context=image[0], question=question)

    assert "300" in text
