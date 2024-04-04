from typing import Callable


def merge_dicts(dicts, precedence_dicts):
    """
    Merge multiple dictionaries with identical keys and structure into one dictionary,
    giving precedence to a list of dictionaries when a value is present or not None.

    Args:
        dicts (list): A list of dictionaries to merge.
        precedence_dicts (list): A list of dictionaries to give precedence to when a value is present or not None.

    Returns:
        dict: The merged dictionary.
    """
    if not dicts:
        return {}

    merged_dict = {}

    for key in dicts[0].keys():
        values = [d[key] for d in dicts if key in d]
        precedence_values = [d[key] for d in precedence_dicts if key in d and d[key] is not None]

        if isinstance(values[0], dict):
            # Recursive call for nested dictionaries
            merged_dict[key] = merge_dicts(
                [v for v in values if v is not None],
                [v for v in precedence_values if isinstance(v, dict)]
            )
        else:
            # Find the first non-None value from precedence dictionaries
            if precedence_values:
                merged_dict[key] = precedence_values[0]
            else:
                # Find the first non-None value from non-precedence dictionaries
                for value in values:
                    if value is not None:
                        merged_dict[key] = value
                        break

    return merged_dict


async def create_json_object(target: str, schema, get_value: Callable, current_obj=None, root_obj=None, parent_key='', prefix=''):
    if current_obj is None:
        current_obj = {}
        root_obj = {}

    for name, properties in schema.items():
        full_key = f'{parent_key}.{name}' if parent_key else name
        if properties['type'] == 'object':
            current_obj[name] = await create_json_object(target, properties['properties'], get_value, {}, root_obj, full_key)
        else:
            description = properties.get('description')
            enum = properties.get('enum', None)
            if prefix:
                description = prefix + (description[:1].lower() + description[1:])
            value = await get_value(target, full_key, properties['type'], description, enum, root_obj) if target else None
            current_obj[name] = value
            root_obj[full_key] = value

    return current_obj


def strip_non_numeric_chars(s):
    while len(s) > 0 and not s[0].isdigit():
        s = s[1:]
    while len(s) > 0 and not s[-1].isdigit():
        s = s[:-1]
    return s