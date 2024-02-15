import logging

from bpm_ai_core.classification.zero_shot_classifier import ZeroShotClassifier, ClassificationResult

try:
    from transformers import pipeline, AutoTokenizer
    has_transformers = True
except ImportError:
    has_transformers = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL_EN = "MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33"
DEFAULT_MODEL_MULTI = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


class TransformersClassifier(ZeroShotClassifier):
    """
    Local zero-shot classification model based on Huggingface transformers library.

    To use, you should have the ``transformers`` python package installed.
    """

    def __init__(self, model: str = DEFAULT_MODEL_EN):
        if not has_transformers:
            raise ImportError('transformers is not installed')
        self.model = model

    def classify_with_metadata(
            self,
            text: str,
            classes: list[str],
            hypothesis_template: str | None = None
    ) -> ClassificationResult:
        zeroshot_classifier = pipeline("zero-shot-classification", model=self.model)

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        input_tokens = len(tokenizer.encode(text))
        max_tokens = tokenizer.model_max_length
        logger.debug(f"Input tokens: {input_tokens}")
        if input_tokens > max_tokens:
            logger.warning(
                f"Input tokens exceed max model context size: {input_tokens} > {max_tokens}. Input will be truncated."
            )

        prediction = zeroshot_classifier(
            text,
            classes,
            hypothesis_template=hypothesis_template or "This example is about {}",
            multi_label=False
        )
        # Zip the labels and scores together and find the label with the max score
        labels_scores = list(zip(prediction['labels'], prediction['scores']))
        max_label, max_score = max(labels_scores, key=lambda x: x[1])

        return ClassificationResult(
            max_label=max_label,
            max_score=max_score,
            labels_scores=labels_scores
        )



