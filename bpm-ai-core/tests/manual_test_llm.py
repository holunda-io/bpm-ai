import os

from bpm_ai_core.llm.anthropic_chat.anthropic_chat import ChatAnthropic
from bpm_ai_core.llm.common.message import ToolResultMessage
from bpm_ai_core.llm.common.tool import Tool
#from bpm_ai_core.llm.mistral_chat import ChatMistral
from bpm_ai_core.llm.openai_chat.openai_chat import ChatOpenAI
from bpm_ai_core.ocr.azure_doc_intelligence import AzureOCR
from bpm_ai_core.prompt.prompt import Prompt
from bpm_ai_core.util.rpc import remote_object


async def test_openai_functionary():
    llm = ChatOpenAI.for_openai_compatible(
        endpoint="http://ai-test-arm.holisticon.de/v1",
        model="functionary"
    )
    prompt = Prompt.from_string("""\
    [# system #]
    Extract all orders from the user input.
    [# user #]
    Hello it's Mike. I would like to order the #3 and the #12. For my friend Jeff I would like to get the #2.
    """)
    tool = Tool.create(name="store_orders", description="accepts a list of orders", args_schema={"orders": {
        "type": "array",
        "description": "the orders",
        "items": {
            "type": "object",
              "properties": {
                "number": { "type": "number", "description": "the order number" },
                "customer_name": "the name of the customer that this order is for"
              }
        }
    }})
    res = await llm.generate_message(prompt, tools=[tool])
    print(res)
    if res.has_tool_calls():
        print(res.tool_calls[0].payload_dict())


async def test_openai_hermes_pro():
    llm = ChatOpenAI.for_openai_compatible(
        endpoint="http://localhost:8000/v1",
        model="hermespro"
    )
    prompt = Prompt.from_string("""\
    [# system #]
    Extract all orders from the user input as a JSON list.
    [# user #]
    Hello it's Mike. I would like to order the #3 and the #12. For my friend Jeff I would like to get the #2.
    """)
    tool = Tool.create(name="store_orders", description="accepts a list of orders", args_schema={"orders": {
        "type": "array",
        "description": "the orders",
        "items": {
            "type": "object",
            "properties": {
                "number": {"type": "number", "description": "the order number"},
                "customer_name": "the name of the customer that this order is for"
            }
        }
    }})
    res = await llm.generate_message(prompt, tools=[tool])
    print(res)

async def test_azure_openai():
    llm = ChatOpenAI.for_azure(
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
    )
    prompt = Prompt.from_string("""\
    [# system #]
    Extract all orders from the user input.
    [# user #]
    I would like to order the #3 and the #12. For my friend I would like to get the #2.""")
    tool = Tool.create(name="store_orders", description="accepts a list of order numbers", args_schema={"orders": ["an order number"]})
    res = await llm.generate_message(prompt, tools=[tool])
    print(res)
    if res.has_tool_calls():
        print(res.tool_calls[0].payload_dict())


async def test_anthropic():
    llm = ChatAnthropic.for_anthropic()
    prompt = Prompt.from_string("""\
    [# system #]
    Extract all orders from the user input as a list of integers in <orders></orders> tags. Explain your result in the end.
    [# user #]
    I would like to order the #3 and the #12. For my friend I would like to get the #2.
    [# assistant #]
    <orders>""")
    res = await llm.generate_message(prompt, stop=["</orders>"], tools=[]) #, tools=[tool])
    print(res)


async def test_anthropic_image():
    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string("""\
    [# system #]
    You are a genius data extraction AI.
    Your task is to extract the information that the user asks for from given images.
    Never make up anything, set fields to null if you can't extract them from the given input!
    [# user #]
    [# blob invoice.png #]
    How much is the total and what is the due date?""")
    res = await llm.generate_message(prompt, output_schema={
        "total": {"type": "number", "description": "the total"},
        "tax": {"type": "number", "description": "the tax amount"},
        "duedate": "the due date in format dd.mm.yyyy",
        "invoiceNumber": {"type": "integer", "description": "the invoice number"},
        "senderEmail": "the email address of the sender",
    })
    print(res)


async def test_openai_image():
    llm = ChatOpenAI()
    prompt = Prompt.from_string("""\
    [# system #]
    You are a genius data extraction AI.
    Your task is to extract the information that the user asks for from given images.
    Never make up anything, set fields to null if you can't extract them from the given input!
    [# user #]
    [# blob files/invoice.png #]
    How much is the total and what is the due date?""")
    res = await llm.generate_message(prompt, output_schema={
        "total": {"type": "number", "description": "the total"},
        "tax": {"type": "number", "description": "the tax amount"},
        "duedate": "the due date in format dd.mm.yyyy",
        "invoiceNumber": {"type": "integer", "description": "the invoice number"},
        "senderEmail": "the email address of the sender",
    })
    print(res)


async def test_anthropic_tools():
    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string("""\
    [# user #]
    Extract all orders from the input in <input></input> tags.
    <input>
    I would like to order the #3 and the #12. For my friend I would like to get the #2.
    </input>""")
    tool = Tool.create(name="store_orders", description="accepts a list of order numbers", args_schema={"orders": ["an order number"]})
    res = await llm.generate_message(prompt, tools=[tool])
    print(res)
    if res.has_tool_calls():
        print(res.tool_calls[0].payload_dict())


async def test_anthropic_tools_multiturn():
    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string("""\
    [# user #]
    What is 263 + 100 - 50?
    """)
    add = Tool.create(name="add", description="adds two numbers", args_schema={
        "a": {"type": "number", "description": "first number"},
        "b": {"type": "number", "description": "second number"}
    })
    sub = Tool.create(name="subtract", description="subtracts two numbers", args_schema={
        "a": {"type": "number", "description": "first number"},
        "b": {"type": "number", "description": "second number"}
    })
    messages = prompt.format()

    message = await llm.generate_message(prompt, tools=[add, sub])
    messages.append(message)
    print(message)
    if message.has_tool_calls():
        print(message.tool_calls[0].payload_dict())

    messages.append(
        ToolResultMessage(id=message.tool_calls[0].id, content="363")
    )

    message = await llm.generate_message(messages, tools=[add, sub])
    messages.append(message)
    print(message)
    if message.has_tool_calls():
        print(message.tool_calls[0].payload_dict())



async def test_anthropic_output_schema():
    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string("""\
    [# system #]
    Extract all orders from the user input.
    [# user #]
    I would like to order the #3 and the #12. For my friend I would like to get the #2.""")

    res = await llm.generate_message(prompt, output_schema={"orders": ["an order number"]})
    print(res.content)


async def test_tools2():
    addition_tool = Tool.create(
        name="addition",
        description="Add two numbers",
        args_schema= {
            "a": {"type": "number", "description": "The first number to add, such as 5"},
            "b": {"type": "number", "description": "The second number to add, such as 4.6"},
            "note": {"type": "string", "description": "A note why this calculation is done"}
        }
    )

    llm = ChatAnthropic.for_anthropic(model="claude-3-haiku-20240307")
    prompt = Prompt.from_string("""\
        [# user #]
        Sally has 17 apples. Later that day, Peter gives 6 Bananas to Sally. How many pieces of fruit does Sally have at the end of the day?
        """)

    res = await llm.generate_message(prompt, tools=[addition_tool])
    print(res)
    if res.has_tool_calls():
        print(res.tool_calls[0].payload_dict())


# async def test_mistral():
#     llm = ChatMistral.for_le_plateforme()
#     prompt = Prompt.from_string("""\
#     [# system #]
#     Extract all orders from the user input as a list of integers.
#     [# user #]
#     I would like to order the #3 and the #12. For my friend I would like to get the #2.""")
#     tool = Tool.create(name="store_orders", description="accepts a list of order numbers",
#                        args_schema={"orders": ["an order number"]})
#     res = await llm.generate_message(prompt, tools=[tool])
#     print(res)


# async def test_mistral_azure():
#     llm = ChatMistral.for_azure(endpoint="https://Mistral-large-holi-serverless.francecentral.inference.ai.azure.com")
#     prompt = Prompt.from_string("""\
#     Write a paragraph about Tesla.""")
#     res = await llm.generate_message(prompt)
#     print(res)


async def test_groq():
    llm = ChatOpenAI.for_groq()
    prompt = Prompt.from_string("""\
    [# system #]
    Extract all orders from the user input as a list of integers in <orders></orders> tags. Explain your result in the end.
    [# user #]
    I would like to order the #3 and the #12. For my friend I would like to get the #2.
    [# assistant #]
    <orders>""")
    res = await llm.generate_message(prompt, stop=["</orders>"])
    print(res)


async def test_groq_extract():
    #doc = (await AzureOCR().process(blob_or_path="files/invoice.pdf")).full_text
    llm = ChatOpenAI.for_groq()
    prompt = Prompt.from_string(f"""\
    [# system #]
    You are a genius data extraction AI.
    Your task is to extract the information that the user asks for from given document OCR text.
    Never make up anything, set fields to null if you can't extract them from the given input!
    [# user #]
<document>
<figure>
![](figures/0)
<!-- FigureContent="Sliced Invoices" -->
</figure>
Invoice
===
|||
| - | - |
| Invoice Number | INV-3337 |
| Order Number | 12345 |
| Invoice Date | January 25, 2016 |
| Due Date | January 31, 2016 |
| Total Due | $93.50 |
To:
Test Business
123 Somewhere St
Melbourne, VIC 3000
test@test.com
| Hrs/Qty | Service | Rate/Price | Adjust | Sub Total |
| - | - | - | - | - |
| 1.00 | Web Design This is a sample description ...| $85.00 | 0.00% | $85.00 |
|||
| - | - |
| Sub Total | $85.00 |
| Tax | $8.50 |
| Total | $93.50 |
ANZ Bank ACC # 1234 1234 BSB # 4321 432
 Pa 
Payment is due within 30 days from date of invoice. Late payment is subject to fees of 5% per month. Thanks for choosing DEMO - Sliced Invoices | admin@slicedinvoices.com Page 1/1
From:
DEMO - Sliced Invoices
Suite 5A-1204
123 Somewhere Street
Your City AZ 12345
admin@slicedinvoices.com
</document>

How much is the total and what is the due date?""")
    res = await llm.generate_message(prompt, output_schema={
        "total": {"type": "number", "description": "the total"},
        "tax": {"type": "number", "description": "the tax amount"},
        "duedate": "the due date in format dd.mm.yyyy",
        "invoiceNumber": {"type": "integer", "description": "the invoice number"},
        "senderEmail": "the email address of the sender",
    })
    print(res)


async def test_groq_tools():
    llm = ChatOpenAI.for_groq()
    prompt = Prompt.from_string("""\
    [# user #]
    Extract all orders from the input in <input></input> tags.
    <input>
    I would like to order the #3 and the #12. For my friend I would like to get the #2.
    </input>""")
    tool = Tool.create(name="store_orders", description="accepts a list of order numbers", args_schema={"orders": ["an order number"]})
    res = await llm.generate_message(prompt, tools=[tool])
    print(res)
    if res.has_tool_calls():
        print(res.tool_calls[0].payload_dict())


async def test_groq_output_schema():
    llm = ChatOpenAI.for_groq()
    prompt = Prompt.from_string("""\
    [# system #]
    Extract all orders from the user input.
    [# user #]
    I would like to order the #3 and the #12. For my friend I would like to get the #2.""")

    res = await llm.generate_message(prompt, output_schema={"orders": ["an order number"]})
    print(res.content)

#####################################################

async def test_llama_output_schema():
    llm = remote_object("ChatLlamaCpp", host="localhost", port=6666, model="QuantFactory/Phi-3-mini-4k-instruct-GGUF", filename="*Q8_0.gguf")
    prompt = Prompt.from_string(f"""\
        [# system #]
        You are a genius data extraction AI.
        Your task is to extract the information that the user asks for from given document OCR text.
        Never make up anything, set fields to null if you can't extract them from the given input!
        [# user #]
    <document>
    <figure>
    ![](figures/0)
    <!-- FigureContent="Sliced Invoices" -->
    </figure>
    Invoice
    ===
    |||
    | - | - |
    | Invoice Number | INV-3337 |
    | Order Number | 12345 |
    | Invoice Date | January 25, 2016 |
    | Due Date | January 31, 2016 |
    | Total Due | $93.50 |
    To:
    Test Business
    123 Somewhere St
    Melbourne, VIC 3000
    test@test.com
    | Hrs/Qty | Service | Rate/Price | Adjust | Sub Total |
    | - | - | - | - | - |
    | 1.00 | Web Design This is a sample description ...| $85.00 | 0.00% | $85.00 |
    |||
    | - | - |
    | Sub Total | $85.00 |
    | Tax | $8.50 |
    | Total | $93.50 |
    ANZ Bank ACC # 1234 1234 BSB # 4321 432
     Pa 
    Payment is due within 30 days from date of invoice. Late payment is subject to fees of 5% per month. Thanks for choosing DEMO - Sliced Invoices | admin@slicedinvoices.com Page 1/1
    From:
    DEMO - Sliced Invoices
    Suite 5A-1204
    123 Somewhere Street
    Your City AZ 12345
    admin@slicedinvoices.com
    </document>

    How much is the total and what is the due date?""")
    res = await llm.generate_message(prompt, output_schema={
        "total": {"type": "number", "description": "the total"},
        "tax": {"type": "number", "description": "the tax amount"},
        "duedate": "the due date in format dd.mm.yyyy",
        "invoiceNumber": {"type": "integer", "description": "the invoice number"},
        "senderEmail": "the email address of the sender",
    })
    print(res)
