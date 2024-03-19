from .base_tool import BaseTool


class AnthropicTool(BaseTool):

    def __init__(self, name, description, args_schema):
        super().__init__(
            name,
            description,
            tool_parameters_from_json_schema(args_schema)
        )

    def use_tool(self):
        pass


def tool_parameters_from_json_schema(schema: dict):
    """
    Converts a JSON schema into tool parameter definitions.

    Args:
        schema (dict): A dictionary representing the JSON schema.

    Returns:
        list: A list of dictionaries representing the tool parameter definitions.
    """
    type_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict"
    }

    # Extract the parameter definitions from the JSON schema
    parameters = []
    for name, prop in schema.get("properties", {}).items():
        json_type = prop.get("type")
        description = prop.get("description", "")

        # Check for nested objects
        if json_type == "object" or "properties" in prop:
            raise ValueError("Nested types or objects are not supported in Anthropic tool parameters.")

        # Convert the JSON schema type to Python type
        param_type = type_mapping.get(json_type, "str")

        param_def = {
            "name": name,
            "type": param_type,
            "description": description
        }
        parameters.append(param_def)
    return parameters
