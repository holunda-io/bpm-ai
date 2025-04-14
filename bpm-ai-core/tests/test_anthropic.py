import pytest

from bpm_ai_core.llm.anthropic_chat.anthropic_chat import ChatAnthropic
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.llm.openai_chat.openai_chat import ChatOpenAI
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.tracing.decorators import trace

@pytest.mark.skip
async def test_anthropic_tools():
    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string("""\
    [# user #]
    Extract all orders from the input in <input></input> tags.
    <input>
    I would like to order the #3 and the #12. For my friend I would like to get the #2.
    </input>""")
    tool = Tool.create(
        name="store_orders",
        description="accepts a list of order numbers (without leading #)",
        args_schema={"orders": ["an order number"]}
    )
    result = await llm.generate_message(prompt, tools=[tool])

    assert result.has_tool_calls()
    assert result.tool_calls[0].payload_dict() == {"orders": ["3", "12", "2"]}


@pytest.mark.skip
@pytest.mark.parametrize("filename", ["files/invoice.png", "files/invoice.pdf"])
async def test_anthropic_document(filename):
    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string(f"""\
    [# system #]
    Your task is to extract the information as specified by the function schema from given images.
    
    [# user #]
    [# blob {filename} #]
    Extract the information requested by the function schema. Pay attention to the descriptions and data types in the schema!
    """)
    result = await llm.generate_message(prompt, output_schema={
        "total": {"type": "number", "description": "the total"},
        "duedate": "the due date",
    })

    assert result.content == {
        "total": 93.50,
        "duedate": "January 31, 2016",
    }


@pytest.mark.skip
@pytest.mark.parametrize("filename,info", [("files/example.jpg", "labrador"), ("files/invoice-simple.webp", "300")])
async def test_anthropic_image(filename, info):
    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string(f"""\
    [# system #]
    Your task is to describe the most central object information present in the given image in a single sentence.

    [# user #]
    [# blob {filename} #]
    Describe in a single sentence!
    """)
    result = await llm.generate_message(prompt)

    assert info in result.content.lower()


@pytest.mark.skip
@pytest.mark.parametrize("filename,info", [("files/test.txt", "jim")])
async def test_anthropic_text_file(filename, info):
    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string(f"""\
    [# system #]
    You are a helpful assistant.
    [# user #]
    [# blob {filename} #]
    Very briefly, what is the file about?
    """)
    result = await llm.generate_message(prompt)

    assert info in result.content.lower()
