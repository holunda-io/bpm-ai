import xmltodict


def dict_to_xml(data, root_tag='root'):
    """
    Serializes a dict to XML.

    Args:
        data (dict): The dict to be serialized to XML.
        root_tag (str): The name of the root tag in the XML output. Default is 'root'.

    Returns:
        str: The serialized XML string.
    """
    return xmltodict.unparse({root_tag: data}, pretty=True, full_document=False)


def xml_to_dict(xml_data):
    """
    Deserializes XML to a dict.

    Args:
        xml_data (str): The XML string to be deserialized.

    Returns:
        dict: The deserialized dict.
    """
    data_dict = xmltodict.parse(xml_data)
    return next(iter(data_dict.values()))
