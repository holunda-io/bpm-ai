
def get_decision_output_schema(
    output_type: str,
    possible_values: list | None = None,
    multiple_decision_values: bool = False
):
    return {
        "0_reasoning": "Concise description of the reasoning behind the decision",
        "1_decision": {
            "description": f"The final decision value{'s' if multiple_decision_values else ''}, may be null if no decision was possible",
            **({"type": output_type, **({"enum": possible_values} if possible_values is not None else {})}
               if not multiple_decision_values else _get_multi_value_schema(output_type, possible_values))
        }
    }


def get_cot_decision_output_schema(
    output_type: str,
    possible_values: list | None = None,
    multiple_decision_values: bool = False
):
    return {
        "0_reasoning": {
            "type": "object",
            "properties": {
                "00_relevantFacts": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A discrete fact"
                    }
                },
                "01_deducedInformation": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Additional information that can be deduced from the relevantFacts"
                    }
                },
                "02_reasoningSteps": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A discrete reasoning step. Do not perform multiple steps in one. Be very fine-grained and use discrete steps/items."
                    }
                },
                "03_finalReasoning": {
                    "type": "string",
                    "description": "Concise description of the final reasoning behind the decision"
                }
            }
        },
        "1_decision": {
            "description": f"The final decision value{'s' if multiple_decision_values else ''}, may be null if no decision was possible",
            **({"type": output_type, **({"enum": possible_values} if possible_values is not None else {})}
               if not multiple_decision_values else _get_multi_value_schema(output_type, possible_values))
        }
    }


def _get_multi_value_schema(output_type, possible_values):
    multi_value_schema = {
        "type": "array",
        "items": {
            "type": output_type,
            "description": "A single decision value",
            **({"enum": possible_values} if possible_values is not None else {})
        }
    }
    return multi_value_schema


def remove_order_prefix_from_keys(data):
    if isinstance(data, dict):
        return {key.lstrip('0123456789_'): remove_order_prefix_from_keys(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [remove_order_prefix_from_keys(item) for item in data]
    else:
        return data
