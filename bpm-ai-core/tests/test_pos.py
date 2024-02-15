from bpm_ai_core.pos.spacy_pos_tagger import SpacyPOSTagger


def test_pos():
    text = "It's me, John Meier."

    tagger = SpacyPOSTagger()
    tags = tagger.tag(text)

    assert tags == [
        ('It', 'PRON'), ("'s ", 'AUX'), ('me', 'PRON'), (', ', 'PUNCT'), ('John ', 'PROPN'), ('Meier', 'PROPN'), ('.', 'PUNCT')
    ]
