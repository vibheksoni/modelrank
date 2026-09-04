"""OpenRouter API fetchers.

Every function takes an :class:`OpenRouterClient`; nothing here constructs
its own HTTP session. All pricing is normalized to USD per 1M tokens.
"""

from __future__ import annotations

import sys
from typing import Optional

from sdk.core.client import OpenRouterClient
from sdk.core.constants import (
    FIND_URL,
    MODALITY_MAP,
    OPENROUTER_ENDPOINT_URL,
    OPENROUTER_MODELS_URL,
    PERF_URL,
)
from sdk.core.types import (
    BenchmarkSnapshot,
    ModelCard,
    ModelEntry,
    PerformanceEntry,
    Pricing,
    ProviderEndpoint,
    _to_bool,
    _to_float,
    _to_int,
    _to_opt_float,
    _to_str,
)


# ── shared endpoint parsing ────────────────────────────────────────────


def _parse_endpoint_meta(e: dict, *, endpoints_detail: bool = False) -> ProviderEndpoint:
    """Parse a provider endpoint object into the full typed structure."""
    return ProviderEndpoint.from_raw(e)


# ── public /v1/models ──────────────────────────────────────────────────


def fetch_models_v1(client: OpenRouterClient) -> list[ModelEntry]:
    """Fetch all models from the public /api/v1/models endpoint."""
    data = client.get_json(OPENROUTER_MODELS_URL)

    models: list[ModelEntry] = []
    for m in data.get("data", []):
        pricing = m.get("pricing", {}) or {}
        input_price = max(_to_float(pricing.get("prompt")) * 1_000_000, 0.0)
        output_price = max(_to_float(pricing.get("completion")) * 1_000_000, 0.0)
        is_free = input_price == 0.0 and output_price == 0.0

        model_id = m.get("id", "")
        parts = model_id.split("/", 1)
        provider = parts[0] if len(parts) == 2 else "unknown"

        architecture = m.get("architecture", {}) or {}
        in_mods = architecture.get("input_modalities", ["text"])
        out_mods = architecture.get("output_modalities", ["text"])
        in_mods_str = ",".join(in_mods) if isinstance(in_mods, list) else str(in_mods)
        out_mods_str = ",".join(out_mods) if isinstance(out_mods, list) else str(out_mods)

        models.append(ModelEntry(
            id=model_id,
            permaslug=model_id,
            name=m.get("name", model_id),
            provider=provider,
            input_price=round(input_price, 6),
            output_price=round(output_price, 6),
            context_length=_to_int(m.get("context_length")),
            is_free=is_free,
            input_modalities=in_mods_str,
            output_modalities=out_mods_str,
            description=m.get("description", "") or "",
            supports_reasoning="reasoning" in str(m.get("supported_parameters", [])),
            is_alias=model_id.startswith("~"),
            author=_to_str(m.get("author", "unknown")) ,
            knowledge_cutoff=_to_str(m.get("knowledge_cutoff")),
            warning_message=_to_str(m.get("warning_message")),
            promotion_message=_to_str(m.get("promotion_message")),
        ))

    return models


# ── frontend /find ─────────────────────────────────────────────────────


def fetch_models_find(
    client: OpenRouterClient,
    modality: Optional[str] = None,
    query: str = "",
    order: str = "most-popular",
    max_price: Optional[float] = None,
) -> list[ModelEntry]:
    """Fetch models from the frontend /find endpoint, optionally by modality.

    The v1 find endpoint returns every active model regardless of the
    output_modalities query param, so modality filtering happens client-side.
    Pass modality=None to fetch every model (used by compare).
    """
    params = {
        "active": "true",
        "fmt": "cards",
        "order": order,
    }
    if query:
        params["q"] = query
    if max_price is not None:
        # The frontend API takes raw USD per 1M tokens (not per-token).
        params["max_price"] = str(max_price)

    data = client.get_json(FIND_URL, params=params)
    raw_models = data.get("data", {}).get("models", [])
    target = MODALITY_MAP.get(modality, modality)
    parsed = [_parse_find_model(m) for m in raw_models]
    if target is None:
        return parsed
    return [m for m in parsed if target in m.output_modalities.lower()]


def _parse_find_model(raw: dict) -> ModelEntry:
    """Parse a raw model object from the /find API into a ModelEntry."""
    endpoint = raw.get("endpoint", {}) or {}
    pricing = endpoint.get("pricing", {}) or {}

    input_price = max(_to_float(pricing.get("prompt")) * 1_000_000, 0.0)
    output_price = max(_to_float(pricing.get("completion")) * 1_000_000, 0.0)

    in_mods = raw.get("input_modalities", ["text"])
    out_mods = raw.get("output_modalities", ["text"])
    in_mods_str = ",".join(in_mods) if isinstance(in_mods, list) else str(in_mods)
    out_mods_str = ",".join(out_mods) if isinstance(out_mods, list) else str(out_mods)

    # The /find response has a separate endpoint metadata object we keep per model.
    endpoint_entry = _parse_endpoint_meta(endpoint) if endpoint else None

    card = ModelCard.from_raw(raw)
    return ModelEntry(
        id=card.slug,
        permaslug=card.permaslug or card.slug,
        name=card.name,
        provider=(
            endpoint.get("provider_display_name")
            or endpoint.get("provider_name")
            or raw.get("author")
            or raw.get("slug", "").split("/", 1)[0]
        ),
        input_price=round(input_price, 6),
        output_price=round(output_price, 6),
        context_length=card.context_length,
        is_free=endpoint.get("is_free", False) or (endpoint.get("pricing", {}) or {}).get("prompt") in ("0", "0.0"),
        input_modalities=in_mods_str,
        output_modalities=out_mods_str,
        description=raw.get("short_description") or card.description or "",
        supports_reasoning=card.supports_reasoning,
        quantization=endpoint.get("quantization", ""),
        model_version_group_id=card.model_version_group_id,
        is_alias=card.slug.startswith("~"),
        author=card.author,
        knowledge_cutoff=card.knowledge_cutoff,
        limit_rpm=card.limit_rpm,
        limit_rpd=card.limit_rpd,
        warning_message=card.warning_message,
        promotion_message=card.promotion_message,
        card=card,
    )


# ── performance ranks (TPS) ────────────────────────────────────────────


def fetch_performance(client: OpenRouterClient) -> dict[str, PerformanceEntry]:
    """Fetch the bulk throughput/latency rankings, keyed by model slug."""
    try:
        entries = client.get_json(PERF_URL).get("data", [])
    except Exception:
        return {}

    out: dict[str, PerformanceEntry] = {}
    for e in entries:
        entry = PerformanceEntry.from_raw(e)
        if entry.slug:
            out[entry.slug] = entry
    return out


def enrich_with_tps(models: list[ModelEntry], client: OpenRouterClient) -> None:
    """Attach bulk performance stats to each model in-place (single request)."""
    print("Fetching TPS stats...", file=sys.stderr)

    slug_stats = fetch_performance(client)

    for m in models:
        tps = slug_stats.get(m.permaslug or m.id)
        if tps is None:
            continue
        m.tps_p50 = int(tps.p50_throughput)
        m.tps_p90 = 0
        m.tps_latency = tps.p50_latency
        m.tps_provider = tps.best_throughput_provider
        m.tps_requests = tps.request_count


# ── benchmark rankings ─────────────────────────────────────────────────


def fetch_benchmarks(client: OpenRouterClient) -> BenchmarkSnapshot:
    """Fetch the full benchmark rankings payload (design-arena + adaptive-arena)."""
    return BenchmarkSnapshot.from_raw(client.get_json("https://openrouter.ai/api/frontend/v1/rankings/benchmarks"))


# ── model endpoint details ─────────────────────────────────────────────


def fetch_endpoints(client: OpenRouterClient, model_id: str) -> list[ProviderEndpoint]:
    """Fetch all upstream provider endpoints for a single model."""
    url = OPENROUTER_ENDPOINT_URL.format(model_id=model_id)
    body = client.get_json(url)
    raw_endpoints = body.get("data", {}).get("endpoints", [])
    return [_parse_endpoint_meta(e, endpoints_detail=True) for e in raw_endpoints]


def enrich_with_endpoints(
    models: list[ModelEntry],
    client: OpenRouterClient,
    max_workers: int = 8,
) -> None:
    """Fetch endpoints for every model in-place using a thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch(m: ModelEntry) -> tuple[ModelEntry, list[ProviderEndpoint]]:
        try:
            eps = fetch_endpoints(client, m.id)
        except Exception:
            eps = []
        return m, eps

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, m): m for m in models}
        for fut in as_completed(futures):
            m, eps = fut.result()
            m.endpoints = eps