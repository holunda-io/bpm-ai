import logging
from typing_extensions import override

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.question_answering.question_answering import QuestionAnswering, QAResult
from bpm_ai_core.util.image import blob_as_images

try:
    from transformers import pipeline, AutoTokenizer, DocumentQuestionAnsweringPipeline, \
    Pix2StructForConditionalGeneration, Pix2StructProcessor
    import torch

    has_transformers = True
except ImportError:
    has_transformers = False

logger = logging.getLogger(__name__)

IMAGE_FORMATS = ["png", "jpeg"]


class Pix2StructVQA(QuestionAnswering):
    """
    Local visual question answering model based on Pix2Struct and Huggingface transformers library.

    To use, you should have the ``transformers`` python package installed.
    """

    def __init__(
            self,
            model: str = "google/pix2struct-docvqa-base"
    ):
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
            raise Exception('Pix2StructVQA only supports image or PDF input')
        images = await blob_as_images(context_str_or_blob, accept_formats=IMAGE_FORMATS)

        pix2Struct = Pix2StructForConditionalGeneration.from_pretrained(self.model)
        pix2Struct.config.vocab_size = 50244
        processor = Pix2StructProcessor.from_pretrained(self.model)

        inputs = processor(images=images, text=question, return_tensors="pt")
        predictions = pix2Struct.generate(**inputs, return_dict_in_generate=True, output_scores=True)
        prediction = processor.decode(predictions.sequences[0], skip_special_tokens=True)

        logger.debug(f"prediction: {prediction}")

        return QAResult(
            answer=prediction,
            score=1.0,
            start_index=None,
            end_index=None,
        )



