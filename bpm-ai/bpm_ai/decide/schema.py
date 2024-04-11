
def get_decision_output_schema(
    output_type: str,
    possible_values: list | None = None
):
    return {
        "0_reasoning": "concise description of the reasoning behind the decision",
        "1_decision": {
            "description": "the final decision value, may be null if no decision was possible",
            "type": output_type,
            **({"enum": possible_values} if possible_values is not None else {})
        }
    }


def get_cot_decision_output_schema(
    output_type: str,
    possible_values: list | None = None
):
    return {
        "0_reasoning": {
            "type": "object",
            "properties": {
                "0_relevantFacts": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A discrete fact"
                    }
                },
                "1_deducedInformation": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Additional information that can be deduced from the relevantFacts"
                    }
                },
                "2_reasoningSteps": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A discrete reasoning step. Do not perform multiple steps in one. Be very fine-grained and use discrete steps/items."
                    }
                },
                "3_finalReasoning": {
                    "type": "string",
                    "description": "Concise description of the final reasoning behind the decision"
                }
            }
        },
        "1_decision": {
            "description": "The final decision value, may be null if no decision was possible",
            "type": output_type,
            **({"enum": possible_values} if possible_values is not None else {})
        }
    }


def remove_order_prefix_from_keys(data):
    if isinstance(data, dict):
        return {key.lstrip('0123456789_'): remove_order_prefix_from_keys(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [remove_order_prefix_from_keys(item) for item in data]
    else:
        return data
