"""Sorting helpers."""

from __future__ import annotations

from sdk.core.types import ModelEntry

SORT_FIELDS = {
    "input": lambda m: m.input_price,
    "output": lambda m: m.output_price,
    "total": lambda m: m.total_price,
    "context": lambda m: m.context_length,
    "name": lambda m: m.id,
    "provider": lambda m: m.provider,
    "tps": lambda m: -(m.tps_p50 or 0),
}


def sort_models(models: list[ModelEntry], sort_by: str, descending: bool = False) -> list[ModelEntry]:
    key = SORT_FIELDS.get(sort_by, SORT_FIELDS["input"])
    return sorted(models, key=key, reverse=descending)