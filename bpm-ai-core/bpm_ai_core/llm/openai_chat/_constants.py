import httpx

DEFAULT_MODEL = "gpt-4-turbo-preview"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
DEFAULT_MAX_RETRIES = 8
OPENAI_COMPATIBLE_API_KEY_ENV_VAR = "LLM_API_KEY"
AZURE_API_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"

default_client_kwargs = {
    "http_client": httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    ),
    "timeout": 600,
    "max_retries": 0  # we use own retry logic
}