from bpm_ai_core.classification.transformers_classifier import TransformersClassifier


def test_classify():
    text = "I am so sleepy today."
    classes = ["tired", "energized", "unknown"]
    expected = "tired"

    classifier = TransformersClassifier()
    actual = classifier.classify(text, classes, confidence_threshold=0.8)

    assert actual == expected


def test_classify_threshold():
    text = "I am ok."
    classes = ["tired", "energized"]
    expected = None

    classifier = TransformersClassifier()
    actual = classifier.classify(text, classes, confidence_threshold=0.9)

    assert actual == expected
