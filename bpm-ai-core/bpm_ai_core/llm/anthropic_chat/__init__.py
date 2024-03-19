import hashlib
import logging
from contextvars import ContextVar

from bpm_ai_core.llm.anthropic_chat._constants import default_client_kwargs

logger = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic

    _clients: ContextVar[dict[str, AsyncAnthropic]] = ContextVar('anthropic_clients', default={})
except ImportError:
    pass


def get_anthropic_client(endpoint: str = None, api_key: str = None) -> AsyncAnthropic:
    client_map = _clients.get()
    hash_key = hashlib.sha256(((endpoint or "default") + (api_key or "default")).encode()).hexdigest()
    if hash_key in client_map.keys():
        return client_map[hash_key]
    else:
        client = AsyncAnthropic(
            base_url=endpoint,
            api_key=api_key,
            **default_client_kwargs
        )
        client_map[hash_key] = client
        return client
