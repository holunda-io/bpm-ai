import logging

from typing_extensions import override

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.question_answering.question_answering import QuestionAnswering, QAResult
from bpm_ai_core.util.image import blob_as_images

try:
    from aiobotocore.session import get_session

    has_textract = True
except ImportError:
    has_textract = False

logger = logging.getLogger(__name__)

IMAGE_FORMATS = ["png", "jpeg", "tiff"]


class AmazonTextractDocVQA(QuestionAnswering):
    """

    """
    def __init__(self, region_name: str = None):
        if not has_textract:
            raise ImportError('aiobotocore is not installed')
        self.region_name = region_name

    @override
    async def _do_answer(
            self,
            context_str_or_blob: str | Blob,
            question: str
    ) -> QAResult:
        if isinstance(context_str_or_blob, str) or not (context_str_or_blob.is_image() or context_str_or_blob.is_pdf()):
            raise Exception('AmazonTextractDocVQA only supports image or PDF input')
        if context_str_or_blob.is_pdf():
            _bytes = await context_str_or_blob.as_bytes()
        else:
            _bytes = (await blob_as_images(context_str_or_blob, accept_formats=IMAGE_FORMATS, return_bytes=True))[0]

        async with get_session().create_client("textract", region_name=self.region_name) as client:
            response = await client.analyze_document(
                Document={"Bytes": _bytes},
                FeatureTypes=["QUERIES"],
                QueriesConfig={'Queries': [
                    {'Text': question}
                ]}
            )

        prediction = next((b for b in response['Blocks'] if b["BlockType"] == "QUERY_RESULT"), {})

        logger.debug(f"prediction: {prediction}")

        if prediction is None:
            raise Exception('AmazonTextractDocVQA failed to extract information.')

        return QAResult(
            answer=prediction['Text'],
            score=prediction['Confidence'] / 100.0,
            start_index=None,
            end_index=None,
        )



