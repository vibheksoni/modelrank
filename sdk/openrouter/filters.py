"""Client-side filtering and deduplication."""

from __future__ import annotations

from typing import Optional

from sdk.core.types import ModelEntry


def filter_models(
    models: list[ModelEntry],
    *,
    min_input: Optional[float] = None,
    max_input: Optional[float] = None,
    min_output: Optional[float] = None,
    max_output: Optional[float] = None,
    min_total: Optional[float] = None,
    max_total: Optional[float] = None,
    min_context: Optional[int] = None,
    max_context: Optional[int] = None,
    provider: Optional[str] = None,
    search: Optional[str] = None,
    free_only: bool = False,
    paid_only: bool = False,
    no_free: bool = False,
    reasoning_only: bool = False,
    input_modality: Optional[str] = None,
    output_modality: Optional[str] = None,
    min_tps: Optional[int] = None,
) -> list[ModelEntry]:

    result = []
    for m in models:
        if min_input is not None and m.input_price < min_input:
            continue
        if max_input is not None and m.input_price > max_input:
            continue
        if min_output is not None and m.output_price < min_output:
            continue
        if max_output is not None and m.output_price > max_output:
            continue
        if min_total is not None and m.total_price < min_total:
            continue
        if max_total is not None and m.total_price > max_total:
            continue
        if min_context is not None and m.context_length < min_context:
            continue
        if max_context is not None and m.context_length > max_context:
            continue
        if provider and provider.lower() not in m.provider.lower():
            continue
        if search and search.lower() not in m.id.lower() and search.lower() not in m.name.lower():
            continue
        if free_only and not m.is_free:
            continue
        if paid_only and m.is_free:
            continue
        if no_free and (m.is_free or m.input_price <= 0):
            continue
        if reasoning_only and not m.supports_reasoning:
            continue
        if input_modality:
            wanted = [x.strip().lower() for x in input_modality.split(",") if x.strip()]
            have = m.input_modalities.lower()
            if not all(w in have for w in wanted):
                continue
        if output_modality:
            wanted = [x.strip().lower() for x in output_modality.split(",") if x.strip()]
            have = m.output_modalities.lower()
            if not all(w in have for w in wanted):
                continue
        if min_tps is not None and m.tps_p50 < min_tps:
            continue
        result.append(m)

    return result


def dedupe_models(models: list[ModelEntry]) -> list[ModelEntry]:
    """Collapse duplicate model rows (one per provider endpoint) to one per id.

    Keeps the cheapest variant by input price, preferring free. Run AFTER
    client-side filters so flags like --no-free still see paid variants.
    """
    best: dict[str, ModelEntry] = {}
    for m in models:
        cur = best.get(m.id)
        if cur is None:
            best[m.id] = m
            continue
        cur_free = cur.is_free or cur.input_price <= 0
        new_free = m.is_free or m.input_price <= 0
        if new_free and not cur_free:
            best[m.id] = m
        elif new_free == cur_free and m.input_price < cur.input_price:
            best[m.id] = m
    return list(best.values())