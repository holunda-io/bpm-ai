from pathlib import Path

from PIL.Image import Image

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.llm.common.message import ToolResultMessage, AssistantMessage, SystemMessage, UserMessage


def test_prompt_format():
    template_vars = {
        "image_url": Path(__file__).parent.absolute() / "example.jpg",
        "task": "Do the task"
    }
    prompt = Prompt.from_file("test", **template_vars)
    messages = prompt.format()

    assert len(messages) == 9

    # [# system #]
    # You are a smart assistant.
    # [# blob {{image_url}} #]
    # Go!
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].role == "system"
    assert isinstance(messages[0].content, list)
    assert messages[0].content[0] == "You are a smart assistant."
    assert isinstance(messages[0].content[1], Blob)
    assert messages[0].content[1].is_image()
    assert messages[0].content[2] == "Go!"

    # [# user #]
    # What is one plus one?
    assert isinstance(messages[1], UserMessage)
    assert messages[1].role == "user"
    assert messages[1].content == "What is one plus one?"

    # [# assistant #]
    # I will call some tools.
    # [# tool_call: foo (foo_id) #]
    # x
    # [# tool_call: bar #]
    # y
    assert isinstance(messages[2], AssistantMessage)
    assert messages[2].role == "assistant"
    assert messages[2].content == "I will call some tools."
    assert len(messages[2].tool_calls) == 2
    assert messages[2].tool_calls[0].id == "foo_id"
    assert messages[2].tool_calls[0].name == "foo"
    assert messages[2].tool_calls[0].payload == "x"
    assert messages[2].tool_calls[1].name == "bar"
    assert messages[2].tool_calls[1].payload == "y"

    # [# tool_result: foo_id #]
    # the result
    assert isinstance(messages[3], ToolResultMessage)
    assert messages[3].role == "tool"
    assert messages[3].id == "foo_id"
    assert messages[3].content == "the result"

    # [# assistant #]
    # Looks good, now another one:
    # [# tool_call: other (other_id) #]
    # z
    assert isinstance(messages[4], AssistantMessage)
    assert messages[4].role == "assistant"
    assert messages[4].content == "Looks good, now another one:"
    assert len(messages[4].tool_calls) == 1
    assert messages[4].tool_calls[0].id == "other_id"
    assert messages[4].tool_calls[0].name == "other"
    assert messages[4].tool_calls[0].payload == "z"

    # [# tool_result: other_id #]
    # the result 2
    assert isinstance(messages[5], ToolResultMessage)
    assert messages[5].role == "tool"
    assert messages[5].id == "other_id"
    assert messages[5].content == "the result 2"

    # [# assistant #]
    # [# tool_call: another #]
    # 123
    assert isinstance(messages[6], AssistantMessage)
    assert messages[6].role == "assistant"
    assert len(messages[6].tool_calls) == 1
    assert messages[6].tool_calls[0].name == "another"
    assert messages[6].tool_calls[0].payload == "123"

    # [# assistant #]
    # That's that.
    assert isinstance(messages[7], AssistantMessage)
    assert messages[7].role == "assistant"
    assert messages[7].content == "That's that."

    # [# user #]
    # Here is an image:
    # [# blob {{image_url}} #]
    # {{task}}
    assert isinstance(messages[8], UserMessage)
    assert messages[8].role == "user"
    assert isinstance(messages[8].content, list)
    assert messages[8].content[0] == "Here is an image:"
    assert isinstance(messages[0].content[1], Blob)
    assert messages[0].content[1].is_image()
    assert messages[8].content[2] == "Do the task"


def test_prompt_filter():
    input_dict = {
        "text": "What is your name?",
        "x": 42,
        "address": {
            "street": "Street",
            "city": "City"
        }
    }
    prompt = Prompt.from_file("test_filter", input=input_dict)
    messages = prompt.format()

    assert messages[0].content == """\
{
  "text": "What is your name?",
  "x": 42,
  "address": {
    "street": "Street",
    "city": "City"
  }
}"""

    assert messages[1].content == """\
text: What is your name?
x: 42
## address
street: Street
city: City"""

    assert messages[2].content.replace('\t', '    ') == """\
<input>
    <text>What is your name?</text>
    <x>42</x>
    <address>
        <street>Street</street>
        <city>City</city>
    </address>
</input>"""
