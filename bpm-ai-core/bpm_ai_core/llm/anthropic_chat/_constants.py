import httpx

DEFAULT_MODEL = "claude-3-opus-20240229"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 8

default_client_kwargs = {
    "http_client": httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    ),
    "max_retries": 0  # we use own retry logic
}