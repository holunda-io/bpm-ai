from bpm_ai_core.translation.easy_nmt.easy_nmt import EasyNMT


def test_translate():
    given_text = "Das ist ein Test"
    target_language = "en"
    expected_translated = "This is a test"

    model = EasyNMT()
    translated = model.translate(given_text, target_language)

    assert translated == expected_translated


def test_translate_same():
    given_text = "Das ist ein Test"
    target_language = "de"
    expected_translated = "Das ist ein Test"

    model = EasyNMT()
    translated = model.translate(given_text, target_language)

    assert translated == expected_translated


def test_translate_multiple():
    given_texts = ["Das ist ein Test", "Un altro test"]
    target_language = "en"
    expected_translated = ["This is a test", "Another test"]

    model = EasyNMT()
    translated = model.translate(given_texts, target_language)

    assert translated == expected_translated
