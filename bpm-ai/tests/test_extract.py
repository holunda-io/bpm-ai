from bpm_ai_core.extractive_qa.transformers_qa import TransformersExtractiveQA
from bpm_ai_core.llm.openai_chat import ChatOpenAI
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


async def test_extract_qa():
    qa = TransformersExtractiveQA()
    actual = await extract_qa(
        extractive_qa=qa,
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


async def test_extract_qa_multiple():
    text = "We received the following orders: Pizza (10.99€) and Steak (28.89€)."
    schema = {
        "product": "What is the name of the product?",
        "price_eur": {
            "type": "float",
            "description": "What is price in Euro for the product?"
        },
    }

    qa = TransformersExtractiveQA()
    actual = await extract_qa(
        extractive_qa=qa,
        input_data={"email": text},
        output_schema=schema,
        multiple=True,
        multiple_description="Meal Order"
    )
    assert actual == [{'product': 'Pizza', 'price_eur': 10.99}, {'product': 'Steak', 'price_eur': 28.89}]