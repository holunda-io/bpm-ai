import inspect
import os
import re
from typing import List, Dict, Any

from jinja2 import Template, environment

from bpm_ai_core.llm.common.blob import Blob
from bpm_ai_core.llm.common.message import ChatMessage, AssistantMessage, ToolResultMessage, ToolCallMessage, \
    UserMessage, SystemMessage
from bpm_ai_core.prompt.filters import dict_to_markdown, dict_to_xml, dict_to_json

environment.DEFAULT_FILTERS['markdown'] = dict_to_markdown
environment.DEFAULT_FILTERS['json'] = dict_to_json
environment.DEFAULT_FILTERS['xml'] = dict_to_xml


class Prompt:

    def __init__(self, kwargs: Dict[str, Any], path: str | None = None, template_str: str | None = None) -> None:
        self.path = path
        self.template_str = template_str
        self.template_vars = kwargs

    @classmethod
    def from_file(cls, path: str, **kwargs):
        # Get the frame of the caller of the function that called this function
        caller_frame = inspect.stack()[1]
        caller_filename = caller_frame.filename

        # Get the directory of the caller's file
        current_dir = os.path.dirname(os.path.abspath(caller_filename))
        file_path = os.path.join(current_dir, path)

        return cls(kwargs, path=file_path)

    @classmethod
    def from_string(cls, template: str, **kwargs):
        return cls(kwargs, template_str=template)

    def format(self, llm_name: str = "") -> List[ChatMessage]:
        template = self.load_template(self.path, llm_name) if self.path else Template(self.template_str)
        full_prompt = template.render(self.template_vars)

        regex = r'\[#\s*(user|assistant|system|tool_result:.*|)\s*#\]'
        blob_regex = r'\[#\s*blob\s*(.*?)\s*#\]'
        tool_call_regex = r'\[#\s*tool_call:\s*(.*?)\s*#\]'

        # Check if the raw template contains message prompt sections
        if re.search(regex, full_prompt):
            # Split the full prompt into sections
            sections = re.split(regex, full_prompt)[1:]
            messages = []
            for i in range(0, len(sections), 2):
                role = sections[i].strip()
                content = sections[i + 1].strip()

                # Check if tool result
                if role.startswith('tool_result:'):
                    id = role.split(':')[1].strip()
                    message = ToolResultMessage(content=content.strip(), role='tool', id=id)
                    messages.append(message)
                    continue

                content_parts = []
                tool_calls = []

                # todo can't use images and call tools at the same time right now
                # Check for blobs (images, videos, ...) in the content and replace with Blob objects
                if re.search(blob_regex, content):
                    start = 0
                    for match in re.finditer(blob_regex, content):
                        # Add the text before the blob
                        before_text = content[start:match.start()].strip()
                        if before_text:
                            content_parts.append(before_text)
                        # Add the blob
                        blob_url = match.group(1)
                        content_parts.append(
                            Blob.from_path_or_url(blob_url)
                        )
                        start = match.end()

                    # Add the remaining text after the last blob
                    content_parts.append(content[start:].strip())

                # Check for tool calls in the content
                elif re.search(tool_call_regex, content):
                    # Split the content into tool call sections
                    tool_call_sections = re.split(tool_call_regex, content)
                    # The content before the first tool call is the message content
                    content_parts.append(tool_call_sections[0].strip())
                    tool_call_sections = tool_call_sections[1:]

                    for j in range(0, len(tool_call_sections), 2):
                        tool_call_info = tool_call_sections[j]
                        tool_call_content = tool_call_sections[j + 1].strip()

                        tool_call_pattern = r"^(.+?)\s*\((.+)\)$"
                        match = re.search(tool_call_pattern, tool_call_info)
                        if match:
                            tool_name = match.group(1)
                            tool_call_id = match.group(2)
                        else:
                            tool_name = tool_call_info
                            tool_call_id = None

                        tool_calls.append(ToolCallMessage(
                            id=(tool_call_id or tool_name).strip(),
                            name=tool_name.strip(),
                            payload=tool_call_content.strip())
                        )
                else:
                    # If there are no tool calls or images, the entire content is the message content
                    content_parts.append(content.strip())

                # Create the message object
                content = content_parts[0] if len(content_parts) == 1 else (content_parts if len(content_parts) > 0 else None)
                if role == "assistant":
                    message = AssistantMessage(content=content, tool_calls=tool_calls if tool_calls else None)
                elif role == "system":
                    message = SystemMessage(content=content)
                else:
                    message = UserMessage(content=content)

                messages.append(message)
        else:
            # If the raw template doesn't contain any sections, treat whole file as 'user' message
            messages = [UserMessage(content=full_prompt)]

        return [m for m in messages if m]

    @staticmethod
    def load_template(path: str, llm_name: str) -> Template:
        default_path = f"{path}.prompt"
        llm_specific_path = f"{path}.{llm_name}.prompt"
        filename = llm_specific_path if os.path.exists(llm_specific_path) else default_path
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No prompt file found at {filename}")
        with open(filename, 'r') as f:
            return Template(f.read())
