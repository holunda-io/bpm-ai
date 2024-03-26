import logging
import os
import re
from io import BytesIO

from typing_extensions import override

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.question_answering.question_answering import QuestionAnswering, QAResult
from bpm_ai_core.util.image import blob_as_images
from bpm_ai_core.util.linguistics import stopwords

try:
    from azure.ai.documentintelligence.aio import DocumentIntelligenceClient as AsyncDocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentAnalysisFeature
    from azure.core.credentials import AzureKeyCredential

    has_azure_doc = True
except ImportError:
    has_azure_doc = False

azure_logger = logging.getLogger('azure')
azure_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

IMAGE_FORMATS = ["png", "jpeg", "tiff"]


class AzureDocVQA(QuestionAnswering):
    def __init__(self, endpoint: str = None):
        if not has_azure_doc:
            raise ImportError('azure-ai-documentintelligence is not installed')
        self.endpoint = endpoint or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

    @override
    async def _do_answer(
            self,
            context_str_or_blob: str | Blob,
            question: str
    ) -> QAResult:
        if isinstance(context_str_or_blob, str) or not (context_str_or_blob.is_image() or context_str_or_blob.is_pdf()):
            raise ValueError("Blob must be a PDF or an image")
        if context_str_or_blob.is_pdf():
            bytes_io = await context_str_or_blob.as_bytes_io()
        else:
            bytes_io = BytesIO((await blob_as_images(context_str_or_blob, accept_formats=IMAGE_FORMATS, return_bytes=True))[0])

        question = self.to_camel_case(question)
        logger.info(f"Modified query: '{question}'")

        async with AsyncDocumentIntelligenceClient(
            self.endpoint,
            AzureKeyCredential(
                os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
            )
        ) as client:
            document = await client.begin_analyze_document(
                model_id="prebuilt-layout",
                analyze_request=bytes_io,
                content_type="application/octet-stream",
                features=[DocumentAnalysisFeature.QUERY_FIELDS],
                query_fields=[question]
            )

            # Wait for the extraction to complete asynchronously
            result = await document.result()
            prediction = result['documents'][0]['fields'][question]

        return QAResult(
            answer=prediction['content'],
            score=prediction['confidence'],
            start_index=None,
            end_index=None,
        )

    @staticmethod
    def to_camel_case(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = [word for word in text.split() if word not in stopwords]
        camel_case_words = [words[0]] + [word.capitalize() for word in words[1:]]
        return ''.join(camel_case_words)
