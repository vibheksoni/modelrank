"""Model relationship resolution.

OpenRouter exposes two cross-model relations that matter:

- **Aliases**: ids starting with ``~`` (e.g. ``~openai/gpt-5-latest``) point at a
  canonical model and share its ``model_version_group_id``. Alias rows in the
  /v1/models list often carry no pricing; resolving to the canonical entry gives
  real pricing, endpoints and TPS.
- **Version families**: every dated revision (``deepseek-v4-flash-20260731``),
  human alias (``deepseek-v4-flash``) and ``~latest`` alias of the same model
  share a ``model_version_group_id``. Family members are "related models".
"""

from __future__ import annotations

from collections import defaultdict

from sdk.core.types import ModelEntry


def is_alias(model_id: str) -> bool:
    return model_id.startswith("~")


def base_id(model_id: str) -> str:
    """Strip a leading `~` alias marker."""
    return model_id.lstrip("~")


def resolve_relations(models: list[ModelEntry]) -> None:
    """Populate alias/canonical and family-sibling fields in-place.

    - Marks ``is_alias`` and resolves ``canonical_id`` for every ``~`` model.
    - Groups models by ``model_version_group_id`` (fallback: exact id) and
      fills ``related_ids`` with same-family siblings.
    """
    by_id: dict[str, ModelEntry] = {}
    for m in models:
        by_id[m.id] = m
        m.is_alias = is_alias(m.id)
        m.canonical_id = ""
        m.related_ids = []

    # 1. canonical targets: aliases -> non-alias member of the same family
    families: dict[str, list[ModelEntry]] = defaultdict(list)
    for m in models:
        families[m.family_key].append(m)

    group_of: dict[str, str] = {}
    for key, members in families.items():
        for m in members:
            group_of[m.id] = key
        canon = next((m for m in members if not m.is_alias), None)
        if canon is None and members:
            # family made only of aliases: pick the first as canonical
            canon = members[0]
        if canon is not None:
            for m in members:
                if m.is_alias:
                    m.canonical_id = canon.id

    # 2. sibling lists per family
    for key, members in families.items():
        sibling_ids = sorted(m.id for m in members)
        for m in members:
            m.related_ids = [i for i in sibling_ids if i != m.id]

    # 3. alias-only fallback: if an alias never matched a family, try base id
    for m in models:
        if m.is_alias and not m.canonical_id:
            target = by_id.get(base_id(m.id))
            if target is not None:
                m.canonical_id = target.id


def resolve_alias_target(model: ModelEntry, all_models: list[ModelEntry]) -> ModelEntry:
    """Return the canonical entry an alias row points at (or the row itself)."""
    if not model.is_alias or not model.canonical_id:
        return model
    for other in all_models:
        if other.id == model.canonical_id:
            return other
    return model