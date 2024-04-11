import pytest

from bpm_ai_core.llm.anthropic_chat.anthropic_chat import ChatAnthropic
from bpm_ai_core.llm.common.tool import Tool
from bpm_ai_core.llm.openai_chat.openai_chat import ChatOpenAI
from bpm_ai_core.prompt.prompt import Prompt


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


@pytest.mark.parametrize("filename", ["invoice.png", "invoice.pdf"])
async def test_anthropic_image(filename):
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
        "tax": {"type": "number", "description": "the tax amount"},
        "duedate": "the due date",
        "invoiceNumber": {"type": "integer", "description": "the invoice number"},
        "senderEmail": "the email address of the sender",
    })

    assert result.content == {
        "total": 93.50,
        "tax": 8.5,
        "duedate": "January 31, 2016",
        "invoiceNumber": 3337,
        "senderEmail": "admin@slicedinvoices.com",
    }
