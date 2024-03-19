import hashlib
import logging
from contextvars import ContextVar

from bpm_ai_core.llm.openai_chat._constants import default_client_kwargs, AZURE_API_KEY_ENV_VAR

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    from openai.lib.azure import AsyncAzureOpenAI

    _clients: ContextVar[dict[str, AsyncOpenAI]] = ContextVar('openai_clients', default={})
except ImportError:
    pass


def get_openai_client(endpoint: str = None, api_key: str = None) -> AsyncOpenAI:
    client_map = _clients.get()
    hash_key = hashlib.sha256(((endpoint or "default") + (api_key or "default")).encode()).hexdigest()
    if hash_key in client_map.keys():
        return client_map[hash_key]
    else:
        client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            **default_client_kwargs
        )
        client_map[hash_key] = client
        return client


def get_azure_openai_client(azure_endpoint: str, api_version: str, api_key: str) -> AsyncOpenAI:
    client_map = _clients.get()
    hash_key = hashlib.sha256((azure_endpoint + api_key).encode()).hexdigest()
    if hash_key in client_map.keys():
        return client_map[hash_key]
    else:
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            **default_client_kwargs
        )
        client_map[hash_key] = client
        return client
