from bpm_ai_core.ocr.tesseract import TesseractOCR
from bpm_ai_core.question_answering.transformers_qa import TransformersExtractiveQA
from bpm_ai_core.llm.openai_chat import ChatOpenAI
from bpm_ai_core.speech_recognition.faster_whisper import FasterWhisperASR
from bpm_ai_core.testing.fake_llm import FakeLLM, tool_response

from bpm_ai.extract.extract import extract_llm, extract_qa


async def test_extract(use_real_llm=False):
    llm = FakeLLM(
        name="openai",
        real_llm_delegate=ChatOpenAI() if use_real_llm else None,
        responses=[
            tool_response(
                name="information_extraction",
                payload='{"firstname": "John", "lastname": "Meier", "age": 30, "language": "de"}'
            )
        ]
    )
    result = await extract_llm(
        llm=llm,
        input_data={"email": "Hey ich bins, der John Meier. Mein 30. Geburtstag war mega!"},
        output_schema={
            "firstname": "the firstname",
            "lastname": "the lastname",
            "age": {"type": "integer", "description": "age in years"},
            "language": "the language the email is written in, as two-letter ISO code"
        }
    )
    llm.assert_last_request_contains("John Meier")
    llm.assert_last_request_defined_tool("information_extraction", is_fixed_tool_choice=True)

    assert result["firstname"] == "John"
    assert result["lastname"] == "Meier"
    assert result["age"] == 30
    assert result["language"] == "de"


async def test_extract_multiple(use_real_llm=False):
    llm = FakeLLM(
        name="openai",
        real_llm_delegate=ChatOpenAI() if use_real_llm else None,
        responses=[
            tool_response(
                name="information_extraction",
                payload='{"entities": [{"firstname": "Jörg"}, {"firstname": "Mike"}, {"firstname": "Sepp"}]}'
            )
        ]
    )
    result = await extract_llm(
        llm=llm,
        input_data={"email": "Hey ich wollte nur sagen, Jörg, Mike und Sepp kommen alle mit!"},
        output_schema={
            "firstname": "the firstname",
        },
        multiple=True,
        multiple_description="People"
    )
    llm.assert_last_request_contains("Jörg, Mike und Sepp")
    llm.assert_last_request_defined_tool("information_extraction", is_fixed_tool_choice=True)

    assert result[0]["firstname"] == "Jörg"
    assert result[1]["firstname"] == "Mike"
    assert result[2]["firstname"] == "Sepp"


async def test_extract_none():
    input_data = {
        "email": None,
        "subject": None
    }
    llm = FakeLLM(name="openai")
    result = await extract_llm(
        llm=llm,
        input_data=input_data,
        output_schema={
            "firstname": "the firstname",
        }
    )

    # LLM should not be used if input is all None
    llm.assert_no_request()

    assert result["email"] is None
    assert result["subject"] is None


async def test_extract_no_output_schema():
    input_data = {
        "doc": "example.png",
        "doc2": "test.mp3",
        "subject": "Test"
    }
    output_schema = {}

    llm = FakeLLM(name="openai")
    result = await extract_llm(
        llm=llm,
        input_data=input_data,
        output_schema=output_schema,
        ocr=TesseractOCR(),
        asr=FasterWhisperASR()
    )

    # LLM should not be used if output_schema is empty
    llm.assert_no_request()

    # ocr and asr should still be applied if output_schema is empty
    assert result["doc"].strip() == "example image"
    assert "half-fantastic" in result["doc2"]
    assert result["subject"] == "Test"


async def test_extract_qa():
    qa = TransformersExtractiveQA()
    actual = await extract_qa(
        qa=qa,
        input_data={"email": "Hey it's me, John Meier. I live in Hamburg and I am 30 years old."},
        output_schema={
            "lastname": "What is the family name (not forename)?",
            "firstname": "What is the forename of the person named {lastname}?",
            "age": {
                "type": "integer",
                "description": "What is the age in years of {firstname} {lastname}?"
            },
            "hometown": "What is the home town of {firstname} {lastname}?"
        }
    )
    assert actual == {'lastname': 'Meier', 'firstname': 'John', 'age': 30, 'hometown': 'Hamburg'}


async def test_extract_qa_none():
    input_data = {
        "email": None,
        "subject": None
    }
    qa = TransformersExtractiveQA()
    result = await extract_qa(
        qa=qa,
        input_data=input_data,
        output_schema={
            "firstname": "the firstname",
        }
    )

    assert result["email"] is None
    assert result["subject"] is None


async def test_extract_qa_no_output_schema():
    input_data = {
        "doc": "example.png",
        "doc2": "test.mp3",
        "subject": "Test"
    }
    output_schema = {}

    qa = TransformersExtractiveQA()
    result = await extract_qa(
        qa=qa,
        input_data=input_data,
        output_schema=output_schema,
        ocr=TesseractOCR(),
        asr=FasterWhisperASR()
    )

    # ocr and asr should still be applied if output_schema is empty
    assert result["doc"].strip() == "example image"
    assert "half-fantastic" in result["doc2"]
    assert result["subject"] == "Test"


async def test_extract_ocr():
    qa = TransformersExtractiveQA()
    ocr = TesseractOCR()
    actual = await extract_qa(
        qa=qa,
        ocr=ocr,
        input_data={
            "email": "Hey it's me, John Meier. I attached the invoice. Have a good one.",
            "invoice": "sample-invoice.webp"
        },
        output_schema={
            "invoice_number": "What is the invoice number?",
            "total": {
                "type": "number",
                "description": "What is the total?"
            },
        }
    )
    assert actual == {'invoice_number': '102', 'total': 300.0}


async def test_extract_qa_multiple():
    text = "We received the following orders: Pizza (10.99€) and Steak (28.89€)."
    schema = {
        "product": "What is the name of the product?",
        "price_eur": {
            "type": "number",
            "description": "What is price in Euro for the product?"
        },
    }

    qa = TransformersExtractiveQA()
    actual = await extract_qa(
        qa=qa,
        input_data={"email": text},
        output_schema=schema,
        multiple=True,
        multiple_description="Meal Order"
    )
    assert actual == [{'product': 'Pizza', 'price_eur': 10.99}, {'product': 'Steak', 'price_eur': 28.89}]