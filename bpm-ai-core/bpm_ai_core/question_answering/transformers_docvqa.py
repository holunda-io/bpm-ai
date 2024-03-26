import logging
from typing_extensions import override

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.question_answering.question_answering import QuestionAnswering, QAResult
from bpm_ai_core.util.image import blob_as_images

try:
    from transformers import pipeline, AutoTokenizer, DocumentQuestionAnsweringPipeline

    has_transformers = True
except ImportError:
    has_transformers = False

logger = logging.getLogger(__name__)

IMAGE_FORMATS = ["png", "jpeg"]


class TransformersDocVQA(QuestionAnswering):
    """
    Local visual document question answering model based on Huggingface transformers library.

    To use, you should have the ``transformers`` python package and the ``tesseract`` or ``tesseract-ocr`` package installed.
    """

    def __init__(self, model: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
        if not has_transformers:
            raise ImportError('transformers is not installed')
        self.model = model

    @override
    async def _do_answer(
            self,
            context_str_or_blob: str | Blob,
            question: str
    ) -> QAResult:
        if isinstance(context_str_or_blob, str) or not (context_str_or_blob.is_image() or context_str_or_blob.is_pdf()):
            raise Exception('TransformersExtractiveDocVQA only supports image or PDF input')
        images = await blob_as_images(context_str_or_blob, accept_formats=IMAGE_FORMATS)

        if len(images) > 1:
            logger.warning('Multiple images provided, using only first image.')

        qa_model = pipeline("document-question-answering", model=self.model)

        prediction = qa_model(
            question=question,
            image=images[0]
        )[0]
        logger.debug(f"prediction: {prediction}")

        return QAResult(
            answer=prediction['answer'],
            score=prediction.get('score', 1.0),
            start_index=prediction.get('start', None),
            end_index=prediction.get('end', None),
        )



