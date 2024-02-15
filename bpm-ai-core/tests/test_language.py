from bpm_ai_core.util.language import indentify_language


def test_language():
    given_text = "Das ist ein Test"
    expected_language = "de"

    language = indentify_language(given_text)

    assert language == expected_language
