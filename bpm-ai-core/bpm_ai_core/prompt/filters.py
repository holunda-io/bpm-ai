import json

from bpm_ai_core.util.markdown import dict_to_md
from bpm_ai_core.util.xml_convert import dict_to_xml as _dict_to_xml


def dict_to_markdown(d: dict) -> str:
    return dict_to_md(d).strip()


def dict_to_json(d: dict) -> str:
    return json.dumps(d, indent=2)


def dict_to_xml(d: dict, root: str = "root") -> str:
    return _dict_to_xml(d, root_tag=root)
