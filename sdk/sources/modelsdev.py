"""models.dev integration.

models.dev is the AI-SDK model catalog (Vercel's models.dev) - the same
dataset LiteLLM consumes. The live endpoint ``https://models.dev/api.json``
returns a dict of ~200 providers, each with connection metadata plus a
``models`` map of typed model entries (cost, limits, modalities, reasoning
options, release dates).

The repo clone lives at ``.references/models.dev`` (source of truth for
schema); this module talks to the live API so pricing stays current.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sdk.core.client import OpenRouterClient
from sdk.core.types import _to_bool, _to_dict, _to_float, _to_int, _to_str

MODELS_DEV_URL = "https://models.dev/api.json"


# ── typed data structures ──────────────────────────────────────────────


@dataclass
class MDReasoningOption:
    type: str = ""
    values: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Any) -> "MDReasoningOption":
        raw = _to_dict(raw)
        return cls(type=_to_str(raw.get("type")), values=[_to_str(v) for v in (raw.get("values") or [])])


@dataclass
class MDModalities:
    input: list[str] = field(default_factory=list)
    output: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Any) -> "MDModalities":
        raw = _to_dict(raw)
        return cls(
            input=[_to_str(v) for v in (raw.get("input") or [])],
            output=[_to_str(v) for v in (raw.get("output") or [])],
        )


@dataclass
class MDCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: Optional[float] = None
    cache_write: Optional[float] = None

    @classmethod
    def from_raw(cls, raw: Any) -> "MDCost":
        raw = _to_dict(raw)
        return cls(
            input=_to_float(raw.get("input")),
            output=_to_float(raw.get("output")),
            cache_read=raw.get("cache_read"),
            cache_write=raw.get("cache_write"),
        )


@dataclass
class MDLimit:
    context: int = 0
    output: int = 0

    @classmethod
    def from_raw(cls, raw: Any) -> "MDLimit":
        raw = _to_dict(raw)
        return cls(context=_to_int(raw.get("context")), output=_to_int(raw.get("output")))


@dataclass
class MDInterleaved:
    field: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> "MDInterleaved":
        raw = _to_dict(raw)
        return cls(field=_to_str(raw.get("field")))


@dataclass
class MDModel:
    """A models.dev model entry (per provider, custom id)."""

    id: str = ""
    name: str = ""
    description: str = ""
    family: str = ""
    attachment: bool = False
    reasoning: bool = False
    reasoning_options: list[MDReasoningOption] = field(default_factory=list)
    tool_call: bool = False
    interleaved: Optional[MDInterleaved] = None
    structured_output: bool = False
    temperature: bool = False
    knowledge: str = ""
    release_date: str = ""
    last_updated: str = ""
    modalities: Optional[MDModalities] = None
    open_weights: bool = False
    limit: Optional[MDLimit] = None
    cost: Optional[MDCost] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "MDModel":
        raw = _to_dict(raw)
        return cls(
            id=_to_str(raw.get("id")),
            name=_to_str(raw.get("name")),
            description=_to_str(raw.get("description")),
            family=_to_str(raw.get("family")),
            attachment=_to_bool(raw.get("attachment")),
            reasoning=_to_bool(raw.get("reasoning")),
            reasoning_options=[
                MDReasoningOption.from_raw(o) for o in (raw.get("reasoning_options") or []) if isinstance(o, dict)
            ],
            tool_call=_to_bool(raw.get("tool_call")),
            interleaved=MDInterleaved.from_raw(raw.get("interleaved")) if raw.get("interleaved") else None,
            structured_output=_to_bool(raw.get("structured_output")),
            temperature=_to_bool(raw.get("temperature")),
            knowledge=_to_str(raw.get("knowledge")),
            release_date=_to_str(raw.get("release_date")),
            last_updated=_to_str(raw.get("last_updated")),
            modalities=MDModalities.from_raw(raw.get("modalities")) if raw.get("modalities") else None,
            open_weights=_to_bool(raw.get("open_weights")),
            limit=MDLimit.from_raw(raw.get("limit")) if raw.get("limit") else None,
            cost=MDCost.from_raw(raw.get("cost")) if raw.get("cost") else None,
            raw=raw,
        )


@dataclass
class MDProvider:
    """A models.dev provider: connection info + model map."""

    id: str = ""
    name: str = ""
    api: str = ""
    env: list[str] = field(default_factory=list)
    npm: str = ""
    doc: str = ""
    models: dict[str, MDModel] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, prof_id: str, raw: Any) -> "MDProvider":
        raw = _to_dict(raw)
        models = {
            mid: MDModel.from_raw(m)
            for mid, m in (_to_dict(raw.get("models")).items())
            if isinstance(m, dict)
        }
        return cls(
            id=_to_str(raw.get("id") or prof_id),
            name=_to_str(raw.get("name")),
            api=_to_str(raw.get("api")),
            env=[_to_str(v) for v in (raw.get("env") or [])],
            npm=_to_str(raw.get("npm")),
            doc=_to_str(raw.get("doc")),
            models=models,
            raw=raw,
        )


@dataclass
class ModelsDevDataset:
    """The whole models.dev catalog."""

    providers: dict[str, MDProvider] = field(default_factory=dict)

    @property
    def all_models(self) -> list[MDModel]:
        out: list[MDModel] = []
        for p in self.providers.values():
            out.extend(p.models.values())
        return out


# ── fetcher ────────────────────────────────────────────────────────────


def fetch_models_dev(client: OpenRouterClient) -> ModelsDevDataset:
    """Fetch and parse the full models.dev catalog (live api.json)."""
    data = client.get_json(MODELS_DEV_URL, timeout=60)
    return ModelsDevDataset(
        providers={pid: MDProvider.from_raw(pid, p) for pid, p in data.items() if isinstance(p, dict)}
    )