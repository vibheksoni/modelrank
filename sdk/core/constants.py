"""Shared constants for the OpenRouter API."""

# ── endpoints ──────────────────────────────────────────────────────────

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_ENDPOINT_URL = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"
FIND_URL = "https://openrouter.ai/api/frontend/v1/models/find"
PERF_URL = "https://openrouter.ai/api/frontend/v1/rankings/performance"
TIMEOUT = 30

# Host used for DNS pre-resolution + CurlOpt.RESOLVE pinning.
OPENROUTER_HOST = "openrouter.ai"
OPENROUTER_PORT = 443

# NOTE: no User-Agent here on purpose. curl_cffi's impersonation layer
# sends a real browser UA automatically (e.g. Chrome/142.0.0.0) and a custom
# UA would break the TLS/HTTP2 fingerprint match.

BASE_HEADERS = {
    "Accept": "application/json",
}

# ── frontend /find mapping ─────────────────────────────────────────────
# Key: CLI mode -> output modality value the site uses.

MODALITY_MAP = {
    "embed": "embeddings",
    "llm": "text",
    "image": "image",
}

ORDER_CHOICES = ["most-popular", "newest", "cheapest", "name"]