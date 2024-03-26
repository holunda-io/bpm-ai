import os

import pytest

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.question_answering.amazon_textract_docvqa import AmazonTextractDocVQA
from bpm_ai_core.question_answering.azure_doc_intelligence_docvqa import AzureDocVQA
from bpm_ai_core.question_answering.transformers_docvqa import TransformersDocVQA


async def test_docvqa():
    model = TransformersDocVQA()

    question = "What is the total?"

    result = await model.answer(
        context_str_or_blob=Blob.from_path_or_url("sample-invoice.webp"),
        question=question
    )

    assert "300" in result.answer


async def test_docvqa_textract():
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        pytest.skip("no API key provided")

    model = AmazonTextractDocVQA()

    question = "total"

    result = await model.answer(
        context_str_or_blob=Blob.from_path_or_url("sample-invoice.webp"),
        question=question
    )

    assert "300" in result.answer


async def test_docvqa_azure():
    if not os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY"):
        pytest.skip("no API key provided")

    model = AzureDocVQA()

    question = "What is the total on the invoice?"

    result = await model.answer(
        context_str_or_blob=Blob.from_path_or_url("sample-invoice.webp"),
        question=question
    )

    assert "300" in result.answer
