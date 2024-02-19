import logging

from PIL.Image import Image

from bpm_ai_core.question_answering.question_answering import QuestionAnswering, QAResult

try:
    from transformers import pipeline, AutoTokenizer, DocumentQuestionAnsweringPipeline

    has_transformers = True
except ImportError:
    has_transformers = False

logger = logging.getLogger(__name__)


class TransformersDocVQA(QuestionAnswering):
    """
    Local visual document question answering model based on Huggingface transformers library.

    To use, you should have the ``transformers`` python package and the ``tesseract`` or ``tesseract-ocr`` package installed.
    """

    def __init__(self, model: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
        if not has_transformers:
            raise ImportError('transformers is not installed')
        self.model = model

    def answer_with_metadata(
            self,
            context: str | Image,
            question: str
    ) -> QAResult:
        if not isinstance(context, Image):
            raise Exception('TransformersExtractiveDocVQA only supports image input')

        qa_model = pipeline("document-question-answering", model=self.model)

        prediction = qa_model(
            question=question,
            image=context
        )[0]
        logger.debug(f"prediction: {prediction}")

        return QAResult(
            answer=prediction['answer'],
            score=prediction.get('score', 1.0),
            start_index=prediction.get('start', None),
            end_index=prediction.get('end', None),
        )



