import logging

from PIL.Image import Image

from bpm_ai_core.question_answering.question_answering import QuestionAnswering, QAResult

try:
    from transformers import pipeline, AutoTokenizer, DocumentQuestionAnsweringPipeline, \
    Pix2StructForConditionalGeneration, Pix2StructProcessor
    import torch

    has_transformers = True
except ImportError:
    has_transformers = False

logger = logging.getLogger(__name__)


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

    def answer_with_metadata(
            self,
            context: str | Image,
            question: str
    ) -> QAResult:
        if not isinstance(context, Image):
            raise Exception('Pix2StructVQA only supports image input')

        pix2Struct = Pix2StructForConditionalGeneration.from_pretrained(self.model)
        pix2Struct.config.vocab_size = 50244
        processor = Pix2StructProcessor.from_pretrained(self.model)

        inputs = processor(images=context, text=question, return_tensors="pt")
        predictions = pix2Struct.generate(**inputs, return_dict_in_generate=True, output_scores=True)
        prediction = processor.decode(predictions.sequences[0], skip_special_tokens=True)

        transition_scores = pix2Struct.compute_transition_scores(predictions.sequences, predictions.scores, normalize_logits=True)
        transition_proba = torch.exp(transition_scores)[0]

        logger.debug(f"prediction: {prediction}")

        return QAResult(
            answer=prediction,
            score=1.0,
            start_index=None,
            end_index=None,
        )



