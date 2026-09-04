"""Model comparison mode.

`compare` fetches a set of models by id (aliases, short slugs, and canonical
ids all work), enriches them with TPS + provider endpoints when asked, and
renders an attribute-by-attribute comparison table.

Comparison attributes are grouped conceptually:

- identity      - id, provider, family / aliases / related versions
- price         - input, output, total per 1M tokens, free
- capacity      - context length, max prompt/completion tokens
- interface     - input/output modalities, reasoning, quantization
- performance   - TPS p50, latency, requests, best provider
- supply        - provider count, BYOK, regions, deranked, disabled
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Optional

from sdk.core.client import OpenRouterClient
from sdk.openrouter.api import (
    enrich_with_endpoints,
    enrich_with_tps,
    fetch_endpoints,
    fetch_models_find,
    fetch_models_v1,
)
from sdk.core.types import ModelEntry
from sdk.openrouter.relations import base_id, is_alias, resolve_alias_target, resolve_relations


def _score_match(m: ModelEntry, key: str) -> int:
    """Rank how well a model matches a requested id (higher = better)."""
    mid = m.id
    if mid == key:
        return 3
    if base_id(mid) == key:
        return 3
    if mid.endswith("/" + key):
        return 2
    if key in mid:
        return 1
    return 0


def fetch_models_by_id(
    client: OpenRouterClient,
    model_ids: list[str],
    query: str = "",
    order: str = "most-popular",
) -> list[ModelEntry]:
    """Fetch exactly the requested models.

    Two-pass strategy:
    1. Frontend /find search (handles aliases like ``~openai/gpt-5-latest``
       and embedding models). Query by each id's base with no modality filter.
    2. If some ids are still missing, fall back to the public /v1 catalog.
    """
    found: dict[str, ModelEntry] = {}

    for mid in model_ids:
        key = base_id(mid)
        try:
            hits = fetch_models_find(client, None, query=key.lower(), order=order)
        except Exception:
            hits = []
        best = max(hits, key=lambda m: _score_match(m, key), default=None)
        if best is not None and _score_match(best, key) > 0:
            found[best.id] = best

    found_keys = list(found.keys())

    def _found_for(requested: str) -> bool:
        key = base_id(requested)
        return any(
            k == key or base_id(k) == key or k.endswith("/" + key) or key in k
            for k in found_keys
        )

    missing = [m for m in model_ids if not _found_for(m)]
    if missing:
        try:
            catalog = fetch_models_v1(client)
        except Exception:
            catalog = []
        for m in catalog:
            if any(_score_match(m, base_id(x)) > 0 for x in missing):
                found[m.id] = m
                break

    models = list(found.values())
    resolve_relations(models)
    _inherit_alias_data(models)
    return models


def _inherit_alias_data(models: list[ModelEntry]) -> None:
    """Copy canonical pricing/capacity/interface into alias rows.

    Alias rows in the catalog often carry no pricing of their own; after
    relation resolution they know their canonical id, so we make them
    comparable by inheriting the canonical's real values.
    """
    by_id = {m.id: m for m in models}
    for m in models:
        if not m.is_alias or not m.canonical_id:
            continue
        canon = by_id.get(m.canonical_id)
        if canon is None:
            continue
        if m.input_price <= 0 and canon.input_price > 0:
            m.input_price = canon.input_price
            m.output_price = canon.output_price
            m.is_free = canon.is_free
        if not m.context_length:
            m.context_length = canon.context_length
        if not m.quantization:
            m.quantization = canon.quantization
        if not m.input_modalities or m.input_modalities == "text":
            m.input_modalities = canon.input_modalities
        if not m.output_modalities or m.output_modalities == "text":
            m.output_modalities = canon.output_modalities
        if not m.supports_reasoning:
            m.supports_reasoning = canon.supports_reasoning
        if not m.quantization:
            m.quantization = canon.quantization
        # Same family => same perf ranking slug; inherit so TPS lookup matches
        # the dated permaslug the performance endpoint uses.
        if hasattr(m, "permaslug") and (not m.permaslug or m.permaslug == m.id):
            m.permaslug = canon.permaslug


def compare_attributes(models: list[ModelEntry]) -> list[tuple[str, list[str]]]:
    """Build the attribute row list for a comparison table.

    Returns [(label, [formatted value per model, ...]), ...].
    """
    rows: list[tuple[str, list[str]]] = []

    def add(label: str, fn) -> None:
        rows.append((label, [fn(m) for m in models]))

    def money(v: float) -> str:
        return "-" if v <= 0 else f"${v:.4f}"

    def first_endpoint(m: ModelEntry, attr: str):
        if not m.endpoints:
            return None
        return getattr(m.endpoints[0], attr)

    add("ID", lambda m: m.id)
    add("Provider", lambda m: m.provider)
    add("Version", lambda m: m.permaslug or "-")
    add("Alias of", lambda m: m.canonical_id or "no")
    add("Family", lambda m: m.model_version_group_id[:12] or "-")
    add("Related", lambda m: ", ".join(m.related_ids[:4]) or "-")
    add("Input $/1M", lambda m: money(m.input_price))
    add("Output $/1M", lambda m: money(m.output_price))
    add("Total $/1M", lambda m: money(m.total_price))
    add("Free", lambda m: "yes" if m.is_free else "no")
    add("Context", lambda m: f"{m.context_length:,}" if m.context_length else "-")
    add("Max prompt tok", lambda m: f"{first_endpoint(m, 'max_prompt_tokens'):,}" if first_endpoint(m, 'max_prompt_tokens') else "-")
    add("Max output tok", lambda m: f"{first_endpoint(m, 'max_completion_tokens'):,}" if first_endpoint(m, 'max_completion_tokens') else "-")
    add("Input modes", lambda m: m.input_modalities or "-")
    add("Output modes", lambda m: m.output_modalities or "-")
    add("Reasoning", lambda m: "yes" if m.supports_reasoning else "no")
    add("Quant", lambda m: m.quantization or "-")
    add("TPS p50", lambda m: f"{m.tps_p50:,}" if m.tps_p50 else "-")
    add("Latency p50", lambda m: f"{m.tps_latency:,.0f} ms" if m.tps_latency else "-")
    add("Requests 30m", lambda m: f"{m.tps_requests:,}" if m.tps_requests else "-")
    add("Best TPS provider", lambda m: m.tps_provider or "-")
    add("Providers", lambda m: f"{len(m.endpoints)}" if m.endpoints else "-")
    add("BYOK", lambda m: "yes" if any(e.is_byok for e in m.endpoints) else "no")
    add("Regions", lambda m: ", ".join(sorted({e.provider_region for e in m.endpoints if e.provider_region})) or "-")
    add("Deranked", lambda m: "yes" if any(e.is_deranked for e in m.endpoints) else "no")
    add("Disabled", lambda m: "yes" if any(e.is_disabled for e in m.endpoints) else "no")

    return rows


def prepare_comparison(
    client: OpenRouterClient,
    model_ids: list[str],
    *,
    tps: bool = False,
    show_providers: bool = False,
) -> list[ModelEntry]:
    """Fetch models and enrich with TPS + endpoint metadata."""
    models = fetch_models_by_id(client, model_ids)

    if tps:
        enrich_with_tps(models, client)
    if show_providers:
        enrich_with_endpoints(models, client)
    else:
        # Light endpoint fetch always supplies provider flags (BYOK/regions/deranked).
        for m in models:
            try:
                m.endpoints = fetch_endpoints(client, m.id)
            except Exception:
                m.endpoints = []

    return models

# ── compare machine formats ────────────────────────────────────────────


def compare_csv_rows(models, rows) -> list[list[str]]:
    """CSV-ready rows: header [Attribute, model ids...] then one row per attribute."""
    header = ["Attribute"] + [m.id for m in models]
    body = [[label] + [str(v) for v in values] for label, values in rows]
    return [header] + body


def _write_compare(models, rows, delimiter: str) -> None:
    import csv
    writer = csv.writer(sys.stdout, delimiter=delimiter, lineterminator=chr(10))
    writer.writerows(compare_csv_rows(models, rows))


def print_compare_csv(models, rows) -> None:
    """Comparison matrix as CSV (Attribute x models)."""
    _write_compare(models, rows, ",")


def print_compare_tsv(models, rows) -> None:
    """Comparison matrix as TSV (Attribute x models)."""
    _write_compare(models, rows, chr(9))


def print_compare_jsonl(models, rows) -> None:
    """NDJSON: each line = {"model": {...}, "attributes": {...}} per model."""
    import json
    for i, m in enumerate(models):
        attrs = {label: values[i] if i < len(values) else "-" for label, values in rows}
        json.dump({"model": asdict(m), "attributes": attrs}, sys.stdout, separators=(",", ":"))
        sys.stdout.write(chr(10))
