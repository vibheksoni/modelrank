"""ArtificialAnalysis.ai integration.

Data flow (replicated from the site's client-side JS):
1. The model leaderboard page embeds a manifest in its RSC payload:
   ``{"path": "/data/<hash>.txt", "key": "<64-hex AES key>"}``.
2. Fetch the data file - it is a binary blob: AES-GCM(plaintext gzip json),
   where the passphrase is the manifest key, the raw key bytes are the AES
   key, and the IV is ``sha256(key)[:12]``.
3. Decrypt, gunzip, JSON.parse -> the full model dataset.

The endpoint returns ~900 models with intelligence/agentic/omniscience
indices, per-benchmark scores, pricing variants (blended ratios, cache,
images), and performance distributions by prompt type.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from sdk.core.client import OpenRouterClient
from sdk.core.types import _to_bool, _to_dict, _to_float, _to_int, _to_str

AA_BASE = "https://artificialanalysis.ai"


# ── typed data structures ──────────────────────────────────────────────


@dataclass
class AAEffort:
    slug: str = ""
    label: str = ""
    level: int = 0

    @classmethod
    def from_raw(cls, raw: Any) -> "AAEffort":
        raw = _to_dict(raw)
        return cls(slug=_to_str(raw.get("slug")), label=_to_str(raw.get("label")), level=_to_int(raw.get("level")))


@dataclass
class AARelease:
    slug: str = ""
    name: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> "AARelease":
        raw = _to_dict(raw)
        return cls(slug=_to_str(raw.get("slug")), name=_to_str(raw.get("name")))


@dataclass
class AAOmniscienceBreakdown:
    accuracy: float = 0.0
    hallucination_rate: float = 0.0

    @classmethod
    def from_raw(cls, raw: Any) -> "AAOmniscienceBreakdown":
        raw = _to_dict(raw)
        return cls(accuracy=_to_float(raw.get("accuracy")), hallucination_rate=_to_float(raw.get("hallucinationRate")))


@dataclass
class AAIntelligenceCost:
    total: float = 0.0
    input: float = 0.0
    non_cache_input: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    output: float = 0.0
    reasoning: float = 0.0
    answer: float = 0.0

    @classmethod
    def from_raw(cls, raw: Any) -> "AAIntelligenceCost":
        raw = _to_dict(raw)
        return cls(
            total=_to_float(raw.get("total")),
            input=_to_float(raw.get("input")),
            non_cache_input=_to_float(raw.get("nonCacheInput")),
            cache_read=_to_float(raw.get("cacheRead")),
            cache_write=_to_float(raw.get("cacheWrite")),
            output=_to_float(raw.get("output")),
            reasoning=_to_float(raw.get("reasoning")),
            answer=_to_float(raw.get("answer")),
        )


@dataclass
class AAVariance:
    p05: float = 0.0
    q25: float = 0.0
    median: float = 0.0
    q75: float = 0.0
    p95: float = 0.0

    @classmethod
    def from_raw(cls, raw: Any) -> "AAVariance":
        raw = _to_dict(raw)
        return cls(
            p05=_to_float(raw.get("p05")),
            q25=_to_float(raw.get("q25")),
            median=_to_float(raw.get("median")),
            q75=_to_float(raw.get("q75")),
            p95=_to_float(raw.get("p95")),
        )


@dataclass
class AATimescale:
    median_output_speed: float = 0.0
    median_time_to_first_chunk: float = 0.0

    @classmethod
    def from_raw(cls, raw: Any) -> "AATimescale":
        raw = _to_dict(raw)
        return cls(
            median_output_speed=_to_float(raw.get("medianOutputSpeed")),
            median_time_to_first_chunk=_to_float(raw.get("medianTimeToFirstChunk")),
        )


@dataclass
class AAResponseTiming:
    input: float = 0.0
    reasoning: float = 0.0
    answer: float = 0.0
    total: float = 0.0

    @classmethod
    def from_raw(cls, raw: Any) -> "AAResponseTiming":
        raw = _to_dict(raw)
        return cls(
            input=_to_float(raw.get("input")),
            reasoning=_to_float(raw.get("reasoning")),
            answer=_to_float(raw.get("answer")),
            total=_to_float(raw.get("total")),
        )


@dataclass
class AABenchmarks:
    """Per-model benchmark indices from the flagship leaderboard rows.

    Every field maps 1:1 to an artificialanalysis.ai benchmark. Values are
    None when the benchmark was not run for this model.
    """

    intelligence_index: Optional[float] = None
    intelligence_index_is_estimated: bool = False
    intelligence_index_token_count: Optional[float] = None
    intelligence_index_output_tokens_per_task: Optional[float] = None
    agentic_index: Optional[float] = None
    omniscience: Optional[float] = None
    omniscience_breakdown: Optional[AAOmniscienceBreakdown] = None
    mlcr_overall: Optional[float] = None
    gdpval: Optional[float] = None
    gdpval_normalized: Optional[float] = None
    it_bench_sre: Optional[float] = None
    tau2: Optional[float] = None
    tau_banking: Optional[float] = None
    terminalbench_hard: Optional[float] = None
    terminalbench_v21: Optional[float] = None
    scicode: Optional[float] = None
    lcr: Optional[float] = None
    ifbench: Optional[float] = None
    hle: Optional[float] = None
    gpqa: Optional[float] = None
    critpt: Optional[float] = None
    apex_agents: Optional[float] = None
    mmmu_pro: Optional[float] = None
    livecodebench: Optional[float] = None
    aime25: Optional[float] = None
    analyst_agent: Optional[float] = None
    microevals_enabled: bool = False
    performance_data_source: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "AABenchmarks":
        raw = _to_dict(raw)
        def f(key):
            v = raw.get(key)
            return float(v) if isinstance(v, (int, float)) else None
        return cls(
            intelligence_index=f("intelligenceIndex"),
            intelligence_index_is_estimated=_to_bool(raw.get("intelligenceIndexIsEstimated")),
            intelligence_index_token_count=f("canonicalIntelligenceIndexTokenCount"),
            intelligence_index_output_tokens_per_task=f("intelligenceIndexOutputTokensPerTask"),
            agentic_index=f("agenticIndex"),
            omniscience=f("omniscience"),
            omniscience_breakdown=AAOmniscienceBreakdown.from_raw(raw.get("omniscienceBreakdown"))
            if raw.get("omniscienceBreakdown")
            else None,
            mlcr_overall=f("mlcrOverall"),
            gdpval=f("gdpval"),
            gdpval_normalized=f("gdpvalNormalized"),
            it_bench_sre=f("itBenchSre"),
            tau2=f("tau2"),
            tau_banking=f("tauBanking"),
            terminalbench_hard=f("terminalbenchHard"),
            terminalbench_v21=f("terminalbenchV21"),
            scicode=f("scicode"),
            lcr=f("lcr"),
            ifbench=f("ifbench"),
            hle=f("hle"),
            gpqa=f("gpqa"),
            critpt=f("critpt"),
            apex_agents=f("apexAgents"),
            mmmu_pro=f("mmmuPro"),
            livecodebench=f("livecodebench"),
            aime25=f("aime25"),
            analyst_agent=f("analystAgent"),
            microevals_enabled=_to_bool(raw.get("microevalsEnabled")),
            performance_data_source=_to_str(raw.get("performanceDataSource")),
            raw=raw,
        )


@dataclass
class AAModel:
    """One entry from the artificialanalysis.ai model leaderboard."""

    id: str = ""
    slug: str = ""
    model_id: str = ""
    model_slug: str = ""
    host_id: str = ""
    host: str = ""
    name: str = ""
    short_name: str = ""
    context_window_tokens: int = 0
    price_1m_input_tokens: float = 0.0
    price_1m_output_tokens: float = 0.0
    cache_hit_price: float = 0.0
    cache_hit_discount_percent: float = 0.0
    cache_write_price: Optional[float] = None
    price_per_1k_1m_images: Optional[float] = None
    price_1m_blended_0_3_1: float = 0.0
    price_1m_blended_7_2_1: float = 0.0
    price_1m_blended_0_1_1: float = 0.0
    price_1m_blended_100_1_1: float = 0.0
    price_1m_blended_0_100_1: float = 0.0
    intelligence_index_cost: Optional[AAIntelligenceCost] = None
    intelligence_index_cost_per_task: dict[str, Any] = field(default_factory=dict)
    briefcase_total_cost: Optional[float] = None
    output_speed_variance: Optional[AAVariance] = None
    time_to_first_chunk_variance: Optional[AAVariance] = None
    performance_by_prompt_type: dict[str, Any] = field(default_factory=dict)
    end_to_end_response_time: Optional[AAResponseTiming] = None
    time_to_first_answer_token: Optional[AAResponseTiming] = None
    timescale_data: Optional[AATimescale] = None
    benchmarks: Optional[AABenchmarks] = None
    effort: Optional[AAEffort] = None
    reasoning: bool = False
    reasoning_tokens: int = 0
    input_modality_text: bool = False
    input_modality_image: bool = False
    input_modality_speech: bool = False
    input_modality_video: bool = False
    output_modality_text: bool = False
    output_modality_image: bool = False
    output_modality_speech: bool = False
    output_modality_video: bool = False
    creator: str = ""
    host_model_count: int = 0
    parameters: Optional[float] = None
    size_class: str = ""
    open_source_categorization: str = ""
    deprecated: bool = False
    deprecated_to: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Any) -> "AAModel":
        raw = _to_dict(raw)
        host = _to_dict(raw.get("host"))
        return cls(
            id=_to_str(raw.get("id")),
            slug=_to_str(raw.get("slug")),
            model_id=_to_str(raw.get("modelId")),
            model_slug=_to_str(raw.get("modelSlug")),
            host_id=_to_str(raw.get("hostId")),
            host=_to_str(host.get("name") or raw.get("host")),
            name=_to_str(raw.get("name")),
            short_name=_to_str(raw.get("shortName")),
            context_window_tokens=_to_int(raw.get("contextWindowTokens")),
            price_1m_input_tokens=_to_float(raw.get("price1mInputTokens")),
            price_1m_output_tokens=_to_float(raw.get("price1mOutputTokens")),
            cache_hit_price=_to_float(raw.get("cacheHitPrice")),
            cache_hit_discount_percent=_to_float(raw.get("cacheHitDiscountPercent")),
            cache_write_price=raw.get("cacheWritePrice"),
            price_per_1k_1m_images=raw.get("pricePer1k1mpImages"),
            price_1m_blended_0_3_1=_to_float(raw.get("price1mBlended0To3To1")),
            price_1m_blended_7_2_1=_to_float(raw.get("price1mBlended7To2To1")),
            price_1m_blended_0_1_1=_to_float(raw.get("price1mBlended0To1To1")),
            price_1m_blended_100_1_1=_to_float(raw.get("price1mBlended100To1To1")),
            price_1m_blended_0_100_1=_to_float(raw.get("price1mBlended0To100To1")),
            intelligence_index_cost=AAIntelligenceCost.from_raw(raw.get("intelligenceIndexCost"))
            if raw.get("intelligenceIndexCost")
            else None,
            intelligence_index_cost_per_task=_to_dict(raw.get("intelligenceIndexCostPerTask")),
            briefcase_total_cost=raw.get("briefcaseTotalCost"),
            output_speed_variance=AAVariance.from_raw(raw.get("outputSpeedVariance"))
            if raw.get("outputSpeedVariance")
            else None,
            time_to_first_chunk_variance=AAVariance.from_raw(raw.get("timeToFirstChunkVariance"))
            if raw.get("timeToFirstChunkVariance")
            else None,
            performance_by_prompt_type=_to_dict(raw.get("performanceByPromptType")),
            end_to_end_response_time=AAResponseTiming.from_raw(raw.get("endToEndResponseTime"))
            if raw.get("endToEndResponseTime")
            else None,
            time_to_first_answer_token=AAResponseTiming.from_raw(raw.get("timeToFirstAnswerToken"))
            if raw.get("timeToFirstAnswerToken")
            else None,
            timescale_data=AATimescale.from_raw(raw.get("timescaleData"))
            if raw.get("timescaleData")
            else None,
            benchmarks=AABenchmarks.from_raw(raw) if isinstance(raw.get("intelligenceIndex"), (int, float)) else None,
            effort=AAEffort.from_raw(raw.get("effort")) if raw.get("effort") else None,
            reasoning=_to_bool(raw.get("isReasoning")),
            reasoning_tokens=_to_int(raw.get("reasoningTokens")),
            input_modality_text=_to_bool(raw.get("inputModalityText")),
            input_modality_image=_to_bool(raw.get("inputModalityImage")),
            input_modality_speech=_to_bool(raw.get("inputModalitySpeech")),
            input_modality_video=_to_bool(raw.get("inputModalityVideo")),
            output_modality_text=_to_bool(raw.get("outputModalityText")),
            output_modality_image=_to_bool(raw.get("outputModalityImage")),
            output_modality_speech=_to_bool(raw.get("outputModalitySpeech")),
            output_modality_video=_to_bool(raw.get("outputModalityVideo")),
            creator=_to_str((_to_dict(raw.get("creator"))).get("name")),
            host_model_count=_to_int(raw.get("hostModelCount")),
            parameters=raw.get("parameters"),
            size_class=_to_str(raw.get("sizeClass")),
            open_source_categorization=_to_str(raw.get("openSourceCategorization")),
            deprecated=_to_bool(raw.get("deprecated")),
            deprecated_to=raw.get("deprecatedTo"),
            raw=raw,
        )

# ── fetcher ────────────────────────────────────────────────────────────

# The site embeds a manifest in its RSC flight payload. The raw HTML keeps
# the flight escaped ("\\" before every quote) AND the payload is split across
# multiple script tags, so detection has to handle several shapes:
#   - {"manifest":{"path":"/data/<hex>.txt","key":"<64-hex>"}}
#   - the escaped form  {\\"manifest\\":{\\"path\\":\\"...\\",\\"key\\":\\"...\\"}}
#   - key order swapped, extra fields, unicode escapes, line breaks
_MANIFEST_PATH_RE = re.compile(
    r'"(?:path|file|url)"\s*:\s*"(/data/[a-f0-9]{16,64}\.txt)"'
)
_MANIFEST_KEY_RE = re.compile(
    r'"(?:key|decryptKey|encryptionKey)"\s*:\s*"([a-f0-9]{64})"'
)

# escaped variant: every quote is preceded by a backslash inside the flight string
_ESCAPED_PATH_RE = re.compile(
    r'\\"(?:path|file|url)\\":\\"(/data/[a-f0-9]{16,64}\.txt)\\"'
)
_ESCAPED_KEY_RE = re.compile(
    r'\\"(?:key|decryptKey|encryptionKey)\\":\\"([a-f0-9]{64})\\"'
)


def _find_manifests_in_text(text: str) -> list[tuple[str, str]]:
    """Extract (path, key) pairs from a payload, order preserved.

    Tries both plain and escaped regexes so key order or extra fields do
    not matter. Only pairs where both halves match are returned.
    """
    out: list[tuple[str, str]] = []
    for path_re, key_re in ((_MANIFEST_PATH_RE, _MANIFEST_KEY_RE), (_ESCAPED_PATH_RE, _ESCAPED_KEY_RE)):
        paths = path_re.findall(text)
        keys = key_re.findall(text)
        n = max(len(paths), len(keys))
        for i in range(n):
            p = paths[i] if i < len(paths) else None
            k = keys[i] if i < len(keys) else None
            if p and k:
                out.append((p, k))
    return out


def fetch_aa_manifests(client: OpenRouterClient, page: str = "/models") -> list[tuple[str, str]]:
    """Fetch an AA page and pull every data manifest out of its payload.

    Returns a list of (path, key) pairs in page order. Handles both the
    escaped flight form and plain JSON by scanning every script tag body
    (not just __next_f.push) and the unescaped HTML as fallback.
    """
    body = client.get(AA_BASE + page, timeout=60).text

    manifests: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    candidates: list[str] = []
    # 1. every __next_f.push([N,"..."]) script: parse the array arg to raw string
    for m in re.finditer(r"self\.__next_f\.push\((.*?)\)</script>", body, re.S):
        candidates.append(m.group(1))
    # 2. any script containing manifest keys (escaped or not)
    for m in re.finditer(r"<script[^>]*>([^<]{200,}?)</script>", body, re.S):
        candidates.append(m.group(1))
    # 3. the whole HTML body itself (last resort)
    candidates.append(body)

    for raw in candidates:
        payloads: list[str] = [raw]
        stripped = raw.strip()
        if stripped.startswith("[") and "]" in stripped:
            try:
                arr = json.loads(stripped)
                if len(arr) > 1 and isinstance(arr[1], str):
                    payloads.insert(0, arr[1])
            except Exception:
                pass
        for payload in payloads:
            for path, key in _find_manifests_in_text(payload):
                if (path, key) not in seen:
                    seen.add((path, key))
                    manifests.append((path, key))

    return manifests


def _decrypt_dataset(client: OpenRouterClient, path: str, keyhex: str):
    """Fetch + decrypt + gunzip + JSON.parse one manifest. Validates each step."""
    if not re.fullmatch(r"[a-f0-9]{64}", keyhex):
        raise ValueError(f"manifest key is not 64 hex chars: {keyhex[:12]}...")
    blob = client.get(AA_BASE + path, timeout=60).content
    if not blob:
        raise ValueError(f"empty data blob for {path}")

    key = bytes.fromhex(keyhex)
    iv = hashlib.sha256(key).digest()[:12]
    try:
        plain = AESGCM(key).decrypt(iv, blob, None)
    except Exception as e:
        # fall back: some blobs may be plain gzip (no AES wrapping)
        try:
            data = json.loads(gzip.decompress(blob).decode("utf-8", errors="replace"))
            return data.get("models", data) if isinstance(data, dict) else data
        except Exception:
            raise ValueError(
                f"AES-GCM auth failed for {path} (wrong key/rotated manifest): {e}"
            ) from e

    try:
        data = json.loads(gzip.decompress(plain).decode("utf-8", errors="replace"))
    except Exception as e:
        raise ValueError(f"data blob for {path} is not gzip json: {e}") from e

    if isinstance(data, dict):
        data = data.get("models", [])
    if not isinstance(data, list):
        raise ValueError(f"decrypted data for {path} is not a list (got {type(data).__name__})")
    if data and not isinstance(data[0], dict):
        raise ValueError(f"decrypted data for {path} is not a model list")
    return data


def fetch_aa_dataset(client: OpenRouterClient, page: str = "/models") -> list[AAModel]:
    """Decrypt and parse the artificialanalysis.ai model leaderboard.

    Tries every manifest found on the page until one decrypts and parses.
    Raises a descriptive error when nothing works.
    """
    manifests = fetch_aa_manifests(client, page=page)
    if not manifests:
        raise RuntimeError(
            f"No data manifest found on {page}. The site may have changed its "
            "payload format - check src/aa.py _MANIFEST_PATH_RE/_MANIFEST_KEY_RE "
            "against the current /data/*.txt + key in the page source."
        )
    errors = []
    for path, key in manifests:
        try:
            data = _decrypt_dataset(client, path, key)
            return [AAModel.from_raw(m) for m in data if isinstance(m, dict)]
        except Exception as e:
            errors.append(f"{path}: {e}")
            continue
    raise RuntimeError(
        "Failed to decrypt any AA manifest on "
        + page
        + ". Attempted "
        + str(len(manifests))
        + " manifest(s):\n"
        + "\n".join(errors)
        )


def fetch_aa_merged(client: OpenRouterClient) -> list[AAModel]:
    """Fetch all AA manifests and merge by model slug, preferring the richer row.

    The /models page carries two datasets: the full leaderboard (907) and an
    openness-weighted view (621). Rows keyed by ``model_slug`` are merged,
    keeping the entry with the widest ``performance_by_prompt_type`` map.
    """
    merged: dict[str, AAModel] = {}
    for m in fetch_aa_dataset(client, page="/models"):
        key = m.model_slug or m.slug
        if not key:
            continue
        cur = merged.get(key)
        if cur is None or len(m.performance_by_prompt_type) > len(cur.performance_by_prompt_type):
            merged[key] = m
    return list(merged.values())

# ── flagship benchmark rows (initialModels in the RSC flight) ──────────


def _walk_for_key(obj, key: str):
    """Depth-first walk of a parsed RSC segment looking for `key`."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _walk_for_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _walk_for_key(v, key)
            if r is not None:
                return r
    return None


def fetch_aa_initial_models(client: OpenRouterClient, page: str = "/models") -> list[AAModel]:
    """Extract the flagship rows (initialModels) directly from the RSC flight.

    These rows carry the full benchmark suite (intelligence/agentic indices,
    HLE, GPQA, AIME25, scicode, ...) that the /data/*.txt manifest lacks.
    """
    body = client.get(AA_BASE + page, timeout=60).text
    scripts = re.findall(r"self\.__next_f\.push\((.*?)\)</script>", body, re.S)
    rows: list[AAModel] = []
    for s_ in scripts:
        try:
            arr = json.loads(s_)
            payload = arr[1] if len(arr) > 1 and isinstance(arr[1], str) else ""
        except Exception:
            continue
        if "initialModels" not in payload:
            continue
        decoded = payload.encode("utf-8", errors="replace").decode("unicode_escape", errors="replace")
        for line in decoded.split(chr(10)):
            m = re.match(r"\d+:(.*)", line, re.S)
            if not m:
                continue
            try:
                val = json.loads(m.group(1))
            except Exception:
                continue
            found = _walk_for_key(val, "initialModels")
            if isinstance(found, list):
                rows.extend(AAModel.from_raw(r) for r in found if isinstance(r, dict))
    seen = set()
    out = []
    for m in rows:
        k = m.model_slug or m.slug
        if k and k not in seen:
            seen.add(k)
            out.append(m)
    return out


def fetch_aa_full(client: OpenRouterClient) -> list[AAModel]:
    """Merge flagship benchmark rows + broad manifest dataset by model slug.

    Prefers the flagship row when it exists (benchmarks present); otherwise
    keeps the manifest row (broad coverage). Per-source perf fields are kept
    from whichever row carries them.
    """
    merged: dict[str, AAModel] = {}
    for m in fetch_aa_initial_models(client, page="/models"):
        key = m.model_slug or m.slug
        if key:
            merged[key] = m
    for m in fetch_aa_dataset(client, page="/models"):
        key = m.model_slug or m.slug
        if not key:
            continue
        cur = merged.get(key)
        if cur is None:
            merged[key] = m
        else:
            for attr in (
                "host",
                "output_speed_variance",
                "time_to_first_chunk_variance",
                "performance_by_prompt_type",
                "end_to_end_response_time",
                "time_to_first_answer_token",
                "timescale_data",
            ):
                if getattr(cur, attr) in (None, "", {}) and getattr(m, attr) not in (None, "", {}):
                    setattr(cur, attr, getattr(m, attr))
    return list(merged.values())