"""Typed data models for every structure the OpenRouter API returns.

All classes are plain dataclasses with full type hints. Parsing is complete:
every field the endpoints expose is captured into a typed structure, even
when the CLI does not currently render it, so the data is available for
future use cases (routing, cost estimation, provider vetting, etc.).

Structure mirror of the API:

- :class:`Pricing` / :class:`DisplayPrice` - per-token raw + per-M display pricing
- :class:`ReasoningConfig`              - reasoning tokens/effort config
- :class:`ModelFeatures`                - feature flags (reasoning, tooling)
- :class:`ProviderDataPolicy`           - training/retention privacy policy
- :class:`ProviderInfo`                 - upstream provider identity + policy
- :class:`EndpointCapabilities`         - endpoint flags (tools, multipart, etc.)
- :class:`ProviderEndpoint`             - a single upstream host for a model
- :class:`ModelCard`                    - full model object (find + endpoint.model)
- :class:`ModelEntry`                   - flattened row used by list/compare output
- :class:`PerformanceEntry`             - one row from the TPS rankings endpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ── scalar coercion helpers ────────────────────────────────────────────


def _to_float(value) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_opt_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).lower() in ("1", "true", "yes", "y")


def _to_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _to_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _to_dict(value) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {}


# ── pricing ────────────────────────────────────────────────────────────


@dataclass
class DisplayPrice:
    """One display-pricing row (per-M token, per-image, per-video, etc.)."""

    kind: str = ""
    sku_label: str = ""
    price: str = ""
    display_multiplier: int = 0
    unit_label: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> "DisplayPrice":
        raw = _to_dict(raw)
        return cls(
            kind=_to_str(raw.get("kind")),
            sku_label=_to_str(raw.get("sku_label")),
            price=_to_str(raw.get("price")),
            display_multiplier=_to_int(raw.get("displayMultiplier")),
            unit_label=_to_str(raw.get("unitLabel")),
        )


@dataclass
class Pricing:
    """Raw per-token pricing plus the display-price breakdown.

    ``prompt`` / ``completion`` are per-token USD strings from the API;
    the convenience properties expose USD per 1M tokens as floats.
    """

    prompt: str = ""
    completion: str = ""
    discount: float = 0.0
    display_pricing: list[DisplayPrice] = field(default_factory=list)

    @property
    def prompt_per_million(self) -> float:
        return max(_to_float(self.prompt) * 1_000_000, 0.0)

    @property
    def completion_per_million(self) -> float:
        return max(_to_float(self.completion) * 1_000_000, 0.0)

    @property
    def is_free(self) -> bool:
        return self.prompt_per_million == 0.0 and self.completion_per_million == 0.0

    @classmethod
    def from_raw(cls, raw: Any) -> "Pricing":
        raw = _to_dict(raw)
        return cls(
            prompt=_to_str(raw.get("prompt")),
            completion=_to_str(raw.get("completion")),
            discount=_to_float(raw.get("discount")),
            display_pricing=[
                DisplayPrice.from_raw(d) for d in (raw.get("display_pricing") or []) if isinstance(d, dict)
            ],
        )


# ── reasoning / features ───────────────────────────────────────────────


@dataclass
class ReasoningConfig:
    """Reasoning token configuration for a model."""

    start_token: str = ""
    end_token: str = ""
    is_mandatory_reasoning: bool = False
    supports_reasoning_effort: bool = False
    supported_reasoning_efforts: list[str] = field(default_factory=list)
    default_reasoning_effort: str = ""
    default_reasoning_enabled: bool = False
    reasoning_return_mechanism: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> "ReasoningConfig":
        raw = _to_dict(raw)
        return cls(
            start_token=_to_str(raw.get("start_token")),
            end_token=_to_str(raw.get("end_token")),
            is_mandatory_reasoning=_to_bool(raw.get("is_mandatory_reasoning")),
            supports_reasoning_effort=_to_bool(raw.get("supports_reasoning_effort")),
            supported_reasoning_efforts=_to_str_list(raw.get("supported_reasoning_efforts")),
            default_reasoning_effort=_to_str(raw.get("default_reasoning_effort")),
            default_reasoning_enabled=_to_bool(raw.get("default_reasoning_enabled")),
            reasoning_return_mechanism=_to_str(raw.get("reasoning_return_mechanism")),
        )


@dataclass
class ModelFeatures:
    """Feature flags attached to a model card (keys vary by model).

    ``raw`` keeps the complete payload so unknown feature keys survive.
    """

    reasoning_config: Optional[ReasoningConfig] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "ModelFeatures":
        raw = _to_dict(raw)
        return cls(
            reasoning_config=ReasoningConfig.from_raw(raw.get("reasoning_config"))
            if raw.get("reasoning_config")
            else None,
            raw=raw,
        )


# ── provider policy / identity ─────────────────────────────────────────


@dataclass
class ProviderDataPolicy:
    """Data retention/privacy policy of an upstream provider."""

    training: bool = False
    training_open_router: bool = False
    retains_prompts: bool = False
    can_publish: bool = False
    terms_of_service_url: str = ""
    privacy_policy_url: str = ""
    requires_user_ids: bool = False

    @classmethod
    def from_raw(cls, raw: Any) -> "ProviderDataPolicy":
        raw = _to_dict(raw)
        return cls(
            training=_to_bool(raw.get("training")),
            training_open_router=_to_bool(raw.get("trainingOpenRouter")),
            retains_prompts=_to_bool(raw.get("retainsPrompts")),
            can_publish=_to_bool(raw.get("canPublish")),
            terms_of_service_url=_to_str(raw.get("termsOfServiceURL")),
            privacy_policy_url=_to_str(raw.get("privacyPolicyURL")),
            requires_user_ids=_to_bool(raw.get("requiresUserIDs")),
        )


@dataclass
class ProviderInfo:
    """Identity, policy and capabilities of an upstream provider."""

    name: str = ""
    display_name: str = ""
    slug: str = ""
    base_url: str = ""
    data_policy: Optional[ProviderDataPolicy] = None
    has_chat_completions: bool = False
    has_completions: bool = False
    is_abortable: bool = False
    moderation_required: bool = False
    adapter_name: str = ""
    status_page_url: str = ""
    byok_enabled: bool = False
    icon_url: str = ""
    send_client_ip: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "ProviderInfo":
        raw = _to_dict(raw)
        icon = _to_dict(raw.get("icon"))
        return cls(
            name=_to_str(raw.get("name")),
            display_name=_to_str(raw.get("displayName")),
            slug=_to_str(raw.get("slug")),
            base_url=_to_str(raw.get("baseUrl")),
            data_policy=ProviderDataPolicy.from_raw(raw.get("dataPolicy"))
            if raw.get("dataPolicy")
            else None,
            has_chat_completions=_to_bool(raw.get("hasChatCompletions")),
            has_completions=_to_bool(raw.get("hasCompletions")),
            is_abortable=_to_bool(raw.get("isAbortable")),
            moderation_required=_to_bool(raw.get("moderationRequired")),
            adapter_name=_to_str(raw.get("adapterName")),
            status_page_url=_to_str(raw.get("statusPageUrl")),
            byok_enabled=_to_bool(raw.get("byokEnabled")),
            icon_url=_to_str(icon.get("url")),
            send_client_ip=_to_bool(raw.get("sendClientIp")),
            raw=raw,
        )


@dataclass
class EndpointCapabilities:
    """Capability flags on an endpoint."""

    supports_tool_parameters: bool = False
    supports_reasoning: bool = False
    supports_multipart: bool = False
    has_completions: bool = False
    has_chat_completions: bool = False
    can_abort: bool = False
    moderation_required: bool = False
    supported_parameters: list[str] = field(default_factory=list)
    excluded_parameters: list[str] = field(default_factory=list)
    allowed_passthrough_parameters: list[str] = field(default_factory=list)
    supported_video_parameters: list[str] = field(default_factory=list)
    supported_image_parameters: list[str] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: Any) -> "EndpointCapabilities":
        raw = _to_dict(raw)
        return cls(
            supports_tool_parameters=_to_bool(raw.get("supports_tool_parameters")),
            supports_reasoning=_to_bool(raw.get("supports_reasoning")),
            supports_multipart=_to_bool(raw.get("supports_multipart")),
            has_completions=_to_bool(raw.get("has_completions")),
            has_chat_completions=_to_bool(raw.get("has_chat_completions")),
            can_abort=_to_bool(raw.get("can_abort")),
            moderation_required=_to_bool(raw.get("moderation_required")),
            supported_parameters=_to_str_list(raw.get("supported_parameters")),
            excluded_parameters=_to_str_list(raw.get("excluded_parameters")),
            allowed_passthrough_parameters=_to_str_list(raw.get("allowed_passthrough_parameters")),
            supported_video_parameters=_to_str_list(raw.get("supported_video_parameters")),
            supported_image_parameters=_to_str_list(raw.get("supported_image_parameters")),
        )


# ── endpoint ───────────────────────────────────────────────────────────


@dataclass
class ProviderEndpoint:
    """A single upstream provider host for a model.

    Captures every field from the frontend ``endpoint`` object and the
    ``/v1/models/{id}/endpoints`` detail entries, including nested typed
    structures (:class:`Pricing`, :class:`ProviderInfo`, capabilities).
    """

    provider_name: str
    tag: str
    input_price: float
    output_price: float
    context_length: int
    uptime_30m: Optional[float]
    latency_30m: Optional[float]
    throughput_30m: Optional[float]

    # extended metadata (defaults when source omits them)
    provider_slug: str = ""
    provider_region: str = ""
    quantization: str = ""
    is_free: bool = False
    status: str = ""
    is_byok: bool = False
    is_byok_only: bool = False
    is_deranked: bool = False
    is_disabled: bool = False
    is_hipaa_eligible: bool = False
    is_private: bool = False
    limit_rpm: int = 0
    limit_rpd: int = 0
    max_prompt_tokens: int = 0
    max_completion_tokens: int = 0
    max_tokens_per_image: int = 0
    capacity_tpm: int = 0
    data_policy: str = ""
    deprecation_date: str = ""
    variant: str = ""
    adapter_name: str = ""
    model_variant_slug: str = ""
    model_variant_permaslug: str = ""
    created_at: str = ""
    endpoint_id: str = ""
    # /endpoints detail API extras
    uptime_last_1d: Optional[float] = None
    uptime_last_5m: Optional[float] = None
    supports_implicit_caching: bool = False
    supports_voice_cloning: bool = False
    model_id: str = ""
    model_name: str = ""

    # nested typed structures
    pricing: Optional[Pricing] = None
    provider_info: Optional[ProviderInfo] = None
    capabilities: Optional[EndpointCapabilities] = None
    model_card: Optional["ModelCard"] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_price(self) -> float:
        return self.input_price + self.output_price

    @classmethod
    def from_raw(cls, raw: Any) -> "ProviderEndpoint":
        raw = _to_dict(raw)
        pricing_raw = raw.get("pricing")
        pricing = Pricing.from_raw(pricing_raw) if isinstance(pricing_raw, dict) else None
        model_card = ModelCard.from_raw(raw.get("model")) if isinstance(raw.get("model"), dict) else None
        return cls(
            provider_name=_to_str(raw.get("provider_name") or raw.get("provider_display_name") or "?"),
            tag=_to_str(raw.get("tag")),
            input_price=pricing.prompt_per_million if pricing else _to_float(raw.get("input_price", 0)) ,
            output_price=pricing.completion_per_million if pricing else _to_float(raw.get("output_price", 0)),
            context_length=_to_int(raw.get("context_length")),
            uptime_30m=_to_opt_float(raw.get("uptime_last_30m")),
            latency_30m=_to_opt_float(raw.get("latency_last_30m")),
            throughput_30m=_to_opt_float(raw.get("throughput_last_30m")),
            provider_slug=_to_str(raw.get("provider_slug")),
            provider_region=_to_str(raw.get("provider_region")),
            quantization=_to_str(raw.get("quantization")),
            is_free=_to_bool(raw.get("is_free")) or (pricing.is_free if pricing else False),
            status=_to_str(raw.get("status")),
            is_byok=_to_bool(raw.get("is_byok")),
            is_byok_only=_to_bool(raw.get("is_byok_only")),
            is_deranked=_to_bool(raw.get("is_deranked")),
            is_disabled=_to_bool(raw.get("is_disabled")),
            is_hipaa_eligible=_to_bool(raw.get("is_hipaa_eligible")),
            is_private=_to_bool(raw.get("is_private")),
            limit_rpm=_to_int(raw.get("limit_rpm")),
            limit_rpd=_to_int(raw.get("limit_rpd")),
            max_prompt_tokens=_to_int(raw.get("max_prompt_tokens")),
            max_completion_tokens=_to_int(raw.get("max_completion_tokens")),
            max_tokens_per_image=_to_int(raw.get("max_tokens_per_image")),
            capacity_tpm=_to_int(raw.get("capacity_tpm")),
            data_policy=_to_str(raw.get("data_policy")),
            deprecation_date=_to_str(raw.get("deprecation_date")),
            variant=_to_str(raw.get("variant")),
            adapter_name=_to_str(raw.get("adapter_name")),
            model_variant_slug=_to_str(raw.get("model_variant_slug")),
            model_variant_permaslug=_to_str(raw.get("model_variant_permaslug")),
            created_at=_to_str(raw.get("created_at")),
            endpoint_id=_to_str(raw.get("id")),
            uptime_last_1d=_to_opt_float(raw.get("uptime_last_1d")),
            uptime_last_5m=_to_opt_float(raw.get("uptime_last_5m")),
            supports_implicit_caching=_to_bool(raw.get("supports_implicit_caching")),
            supports_voice_cloning=_to_bool(raw.get("supports_voice_cloning")),
            model_id=_to_str(raw.get("model_id")),
            model_name=_to_str(raw.get("model_name")),
            pricing=pricing,
            provider_info=ProviderInfo.from_raw(raw.get("provider_info")) if raw.get("provider_info") else None,
            capabilities=EndpointCapabilities.from_raw(raw) if raw.get("supported_parameters") is not None or raw.get("supports_tool_parameters") is not None else None,
            model_card=model_card,
            raw=raw,
        )


# ── model card (find / endpoint.model / catalog) ───────────────────────


@dataclass
class ModelCard:
    """Full model object from the frontend find/catalog endpoints.

    This is the complete card; :class:`ModelEntry` is the flattened working
    row the CLI uses, while this keeps everything the API sent.
    """

    slug: str = ""
    permaslug: str = ""
    hf_slug: str = ""
    name: str = ""
    short_name: str = ""
    author: str = ""
    author_display_name: str = ""
    author_icon_uri: str = ""
    description: str = ""
    model_version_group_id: str = ""
    group: str = ""
    context_length: int = 0
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    has_text_output: bool = False
    instruct_type: str = ""
    default_system: str = ""
    default_stops: list[str] = field(default_factory=list)
    hidden: bool = False
    router: Optional[Any] = None
    warning_message: str = ""
    promotion_message: str = ""
    routing_error_message: str = ""
    required_attestation_types: list[str] = field(default_factory=list)
    is_private: bool = False
    supports_reasoning: bool = False
    reasoning_config: Optional[ReasoningConfig] = None
    features: Optional[ModelFeatures] = None
    default_parameters: dict[str, Any] = field(default_factory=dict)
    default_order: list[str] = field(default_factory=list)
    quick_start_example_type: str = ""
    previews_by_modality: dict[str, Any] = field(default_factory=dict)
    preview_thumbnail_url: str = ""
    preview_audio: Optional[Any] = None
    author_flagship_modalities: list[str] = field(default_factory=list)
    is_trainable_text: bool = False
    is_trainable_image: Optional[bool] = None
    knowledge_cutoff: str = ""
    limit_rpm: int = 0
    limit_rpd: int = 0
    supported_tts_voices: list[str] = field(default_factory=list)
    updated_at: str = ""
    created_at: str = ""
    hf_updated_at: Optional[str] = None
    endpoint: Optional[ProviderEndpoint] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "ModelCard":
        raw = _to_dict(raw)
        card = cls(
            slug=_to_str(raw.get("slug")),
            permaslug=_to_str(raw.get("permaslug")),
            hf_slug=_to_str(raw.get("hf_slug")),
            name=_to_str(raw.get("name")),
            short_name=_to_str(raw.get("short_name")),
            author=_to_str(raw.get("author")),
            author_display_name=_to_str(raw.get("author_display_name")),
            author_icon_uri=_to_str(raw.get("author_icon_uri")),
            description=_to_str(raw.get("description")),
            model_version_group_id=_to_str(raw.get("model_version_group_id")),
            group=_to_str(raw.get("group")),
            context_length=_to_int(raw.get("context_length")),
            input_modalities=_to_str_list(raw.get("input_modalities")),
            output_modalities=_to_str_list(raw.get("output_modalities")),
            has_text_output=_to_bool(raw.get("has_text_output")),
            instruct_type=_to_str(raw.get("instruct_type")),
            default_system=_to_str(raw.get("default_system")),
            default_stops=_to_str_list(raw.get("default_stops")),
            hidden=_to_bool(raw.get("hidden")),
            router=raw.get("router"),
            warning_message=_to_str(raw.get("warning_message")),
            promotion_message=_to_str(raw.get("promotion_message")),
            routing_error_message=_to_str(raw.get("routing_error_message")),
            required_attestation_types=_to_str_list(raw.get("required_attestation_types")),
            is_private=_to_bool(raw.get("is_private")),
            supports_reasoning=_to_bool(raw.get("supports_reasoning")),
            reasoning_config=ReasoningConfig.from_raw(raw.get("reasoning_config"))
            if raw.get("reasoning_config")
            else None,
            features=ModelFeatures.from_raw(raw.get("features")) if raw.get("features") else None,
            default_parameters=_to_dict(raw.get("default_parameters")),
            default_order=_to_str_list(raw.get("default_order")),
            quick_start_example_type=_to_str(raw.get("quick_start_example_type")),
            previews_by_modality=_to_dict(raw.get("previews_by_modality")),
            preview_thumbnail_url=_to_str(raw.get("preview_thumbnail_url")),
            preview_audio=raw.get("preview_audio"),
            author_flagship_modalities=_to_str_list(raw.get("author_flagship_modalities")),
            is_trainable_text=_to_bool(raw.get("is_trainable_text")),
            is_trainable_image=raw.get("is_trainable_image"),
            knowledge_cutoff=_to_str(raw.get("knowledge_cutoff")),
            limit_rpm=_to_int(raw.get("limit_rpm")),
            limit_rpd=_to_int(raw.get("limit_rpd")),
            supported_tts_voices=_to_str_list(raw.get("supported_tts_voices")),
            updated_at=_to_str(raw.get("updated_at")),
            created_at=_to_str(raw.get("created_at")),
            hf_updated_at=raw.get("hf_updated_at"),
            raw=raw,
        )
        if isinstance(raw.get("endpoint"), dict):
            card.endpoint = ProviderEndpoint.from_raw(raw["endpoint"])
        return card


# ── flattened working row ──────────────────────────────────────────────


@dataclass
class ModelEntry:
    """Flattened model row used by list/compare output.

    Keeps the frequently used fields at the top level while retaining the
    full typed structures (:class:`ModelCard`) in ``card`` / ``endpoints``.
    """

    id: str
    name: str
    provider: str
    input_price: float
    output_price: float
    context_length: int
    is_free: bool
    input_modalities: str
    output_modalities: str
    description: str
    permaslug: str = ""
    supports_reasoning: bool = False
    quantization: str = ""
    tps_p50: int = 0
    tps_p90: int = 0
    tps_latency: float = 0.0
    tps_provider: str = ""
    tps_requests: int = 0
    endpoints: list[ProviderEndpoint] = field(default_factory=list)

    # relationship / identity metadata
    model_version_group_id: str = ""
    is_alias: bool = False
    canonical_id: str = ""
    related_ids: list[str] = field(default_factory=list)
    author: str = ""
    knowledge_cutoff: str = ""
    limit_rpm: int = 0
    limit_rpd: int = 0
    warning_message: str = ""
    promotion_message: str = ""

    # full typed card for future use cases
    card: Optional[ModelCard] = None

    @property
    def total_price(self) -> float:
        return self.input_price + self.output_price

    @property
    def family_key(self) -> str:
        """Stable family grouping: version group id, else alias-resolved base."""
        if self.model_version_group_id:
            return self.model_version_group_id
        return self.id.lstrip("~").rsplit("-", 1)[0] if self.is_alias else self.id


# ── performance rankings ───────────────────────────────────────────────


@dataclass
class PerformanceEntry:
    """One row from the frontend performance rankings endpoint."""

    id: str = ""
    slug: str = ""
    name: str = ""
    author: str = ""
    request_count: int = 0
    p50_latency: float = 0.0
    p50_throughput: float = 0.0
    best_latency_provider: str = ""
    best_latency_price: float = 0.0
    best_throughput_provider: str = ""
    best_throughput_price: float = 0.0
    provider_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "PerformanceEntry":
        raw = _to_dict(raw)
        return cls(
            id=_to_str(raw.get("id")),
            slug=_to_str(raw.get("slug")),
            name=_to_str(raw.get("name")),
            author=_to_str(raw.get("author")),
            request_count=_to_int(raw.get("request_count")),
            p50_latency=_to_float(raw.get("p50_latency")),
            p50_throughput=_to_float(raw.get("p50_throughput")),
            best_latency_provider=_to_str(raw.get("best_latency_provider")),
            best_latency_price=_to_float(raw.get("best_latency_price")),
            best_throughput_provider=_to_str(raw.get("best_throughput_provider")),
            best_throughput_price=_to_float(raw.get("best_throughput_price")),
            provider_count=_to_int(raw.get("provider_count")),
            raw=raw,
        )

# ── benchmark rankings ─────────────────────────────────────────────────


@dataclass
class BenchmarkEntry:
    """One ranked model row inside a benchmark category.

    daData rows (design-arena style): score, win_rate, generation time.
    aaData rows (adaptive-arena style): reasoning-effort-tagged scores.
    """

    da_model_id: str = ""
    permaslug: str = ""
    openrouter_id: str = ""
    display_name: str = ""
    score: float = 0.0
    win_rate: float = 0.0
    avg_generation_time_ms: float = 0.0
    # aaData-only fields
    uid: str = ""
    openrouter_slug: str = ""
    heuristic_openrouter_slug: str = ""
    aa_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "BenchmarkEntry":
        raw = _to_dict(raw)
        return cls(
            da_model_id=_to_str(raw.get("da_model_id")),
            permaslug=_to_str(raw.get("permaslug")),
            openrouter_id=_to_str(raw.get("openrouter_id")),
            display_name=_to_str(raw.get("display_name") or raw.get("aa_name")),
            score=_to_float(raw.get("score")),
            win_rate=_to_float(raw.get("win_rate")),
            avg_generation_time_ms=_to_float(raw.get("avg_generation_time_ms")),
            uid=_to_str(raw.get("uid")),
            openrouter_slug=_to_str(raw.get("openrouter_slug")),
            heuristic_openrouter_slug=_to_str(raw.get("heuristic_openrouter_slug")),
            aa_name=_to_str(raw.get("aa_name")),
            raw=raw,
        )


@dataclass
class BenchmarkSnapshot:
    """Full /rankings/benchmarks payload.

    ``da_categories`` maps category id (e.g. ``models-codecategories``) to
    ranked entries; ``aa_categories`` holds the adaptive-arena groups
    (``intelligence``, ``coding``, ``agentic``) and per-slug percentiles.
    """

    da_categories: dict[str, list[BenchmarkEntry]] = field(default_factory=dict)
    aa_categories: dict[str, list[BenchmarkEntry]] = field(default_factory=dict)
    percentiles_by_slug: dict[str, Any] = field(default_factory=dict)
    weighted_input_prices: dict[str, float] = field(default_factory=dict)
    cost_per_request: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "BenchmarkSnapshot":
        raw = _to_dict(raw)
        data = _to_dict(raw.get("data"))
        da = _to_dict(data.get("daData"))
        aa = _to_dict(data.get("aaData"))
        return cls(
            da_categories={
                k: [BenchmarkEntry.from_raw(r) for r in v if isinstance(r, dict)]
                for k, v in da.items()
                if isinstance(v, list)
            },
            aa_categories={
                k: [BenchmarkEntry.from_raw(r) for r in v if isinstance(r, dict)]
                for k, v in aa.items()
                if isinstance(v, list)
            },
            percentiles_by_slug=_to_dict(aa.get("percentilesBySlug")),
            weighted_input_prices={
                k: _to_float(v) for k, v in _to_dict(data.get("weightedInputPrices")).items()
            },
            cost_per_request={
                k: _to_float(v) for k, v in _to_dict(data.get("costPerRequest")).items()
            },
            raw=data,
        )

    @property
    def da_category_ids(self) -> list[str]:
        return sorted(self.da_categories.keys())

    @property
    def aa_category_ids(self) -> list[str]:
        return [k for k in ("intelligence", "coding", "agentic") if k in self.aa_categories]
