"""Cross-source model analysis.

Brings together every dataset integrated in this tool for one model id:

- OpenRouter catalog (pricing, providers, TPS, relationships)
- artificialanalysis.ai (intelligence/agentic indices, benchmark scores,
  performance distributions, blended pricing)
- models.dev (AI-SDK catalog: costs, limits, capabilities, release dates)
- OpenRouter benchmarks (design-arena score, win rate)

Usage:
    python cli.py analyze deepseek-v4-flash
    python cli.py analyze gpt-5 --format json
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Optional

from sdk.sources.artificialanalysis import fetch_aa_dataset, fetch_aa_full
from sdk.openrouter.benchmarks import fetch_benchmarks
from sdk.core.client import OpenRouterClient
from sdk.openrouter.compare import _score_match, fetch_models_by_id
from sdk.openrouter.api import fetch_models_v1
from sdk.sources.modelsdev import fetch_models_dev
from sdk.openrouter.relations import base_id


def _best_match(models, key: str):
    best = None
    best_score = 0
    for m in models:
        sc = _score_match(m, key) if hasattr(m, "id") else 0
        if sc > best_score:
            best_score = sc
            best = m
    return best if best_score > 0 else None


def analyze_model(
    client: OpenRouterClient,
    model_id: str,
    *,
    tps: bool = True,
    show_providers: bool = False,
) -> dict:
    """Fetch a model id from every source and merge into one analysis dict."""
    key = base_id(model_id).lower()

    # 1. OpenRouter (compare path gives aliases/family/endpoints)
    or_model = None
    try:
        from sdk.openrouter.api import enrich_with_endpoints, enrich_with_tps

        or_models = fetch_models_by_id(client, [model_id])
        if tps:
            enrich_with_tps(or_models, client)
        if show_providers:
            enrich_with_endpoints(or_models, client)
        or_model = or_models[0] if or_models else None
    except Exception:
        pass

    # 2. artificialanalysis.ai (prefer exact slug/id, fall back to substring)
    aa = None
    try:
        try:
            aa_rows = fetch_aa_full(client)
        except Exception:
            aa_rows = fetch_aa_dataset(client, page="/models")
        for m in aa_rows:
            if key == m.slug.lower() or key == (m.model_slug or "").lower():
                aa = m
                break
        if aa is None:
            for m in aa_rows:
                if key in m.slug.lower() or key in m.name.lower() or key in (m.model_slug or "").lower():
                    aa = m
                    break
    except Exception:
        pass

    # 3. models.dev (prefer exact id, fall back to substring)
    md = None
    try:
        md_rows = fetch_models_dev(client).all_models
        for m in md_rows:
            if key == m.id.lower():
                md = m
                break
        if md is None:
            for m in md_rows:
                if key in m.id.lower() or key in m.name.lower():
                    md = m
                    break
    except Exception:
        pass

    # 4. benchmarks (design-arena + adaptive-arena)
    bench = None
    try:
        snap = fetch_benchmarks(client)
        for rows in list(snap.da_categories.values()):
            for r in rows:
                if key in (r.openrouter_id or "").lower() or key in (r.permaslug or "").lower():
                    bench = r
                    break
            if bench:
                break
        if bench is None:
            for rows in list(snap.aa_categories.values()):
                for r in rows:
                    if key in (r.permaslug or "").lower():
                        bench = r
                        break
                if bench:
                    break
    except Exception:
        pass

    return {
        "requested_id": model_id,
        "openrouter": asdict(or_model) if or_model else None,
        "artificialanalysis": asdict(aa) if aa else None,
        "modelsdev": asdict(md) if md else None,
        "benchmark": asdict(bench) if bench else None,
    }


# ── renderers ──────────────────────────────────────────────────────────


def print_analysis_plain(analysis: dict) -> None:
    rid = analysis["requested_id"]
    print(f"# Analysis: {rid}")
    print("=" * 70)

    orm = analysis.get("openrouter")
    aa = analysis.get("artificialanalysis")
    md = analysis.get("modelsdev")
    bench = analysis.get("benchmark")

    if orm:
        print(f"\n[OpenRouter] {orm.get('name')} ({orm.get('id')})")
        print(f"  provider: {orm.get('provider')}")
        print(f"  price in/out: ${orm.get('input_price', 0):.3f} / ${orm.get('output_price', 0):.3f} per 1M")
        print(f"  context: {orm.get('context_length', 0):,} | reasoning: {orm.get('supports_reasoning')}")
        if orm.get("tps_p50"):
            print(f"  TPS p50: {orm.get('tps_p50'):,} ({orm.get('tps_provider')})")
        if orm.get("related_ids"):
            print(f"  related: {', '.join(orm['related_ids'][:6])}")
    else:
        print("\n[OpenRouter] not found")

    if aa:
        print(f"\n[ArtificialAnalysis] {aa.get('name')} @ {aa.get('host')}")
        print(f"  context: {aa.get('context_window_tokens', 0):,} | price in/out: ${aa.get('price_1m_input_tokens', 0):.2f}/${aa.get('price_1m_output_tokens', 0):.2f}")
        bench = aa.get("benchmarks") or {}
        if bench.get("intelligence_index") is not None:
            print(f"  intelligence: {bench['intelligence_index']:.2f} | agentic: {bench.get('agentic_index')}")
            print(f"  benchmarks: HLE {bench.get('hle')} | GPQA {bench.get('gpqa')} | scicode {bench.get('scicode')} | TerminalBench {bench.get('terminalbench_v21')}")
        if aa.get("intelligence_index_cost"):
            c = aa["intelligence_index_cost"]
            print(f"  intelligence cost: ${c.get('total', 0):,.2f} (in ${c.get('input', 0):,.2f} + reason ${c.get('reasoning', 0):,.2f})")
        ts = aa.get("timescale_data") or {}
        if ts.get("median_output_speed"):
            print(f"  median output speed: {ts['median_output_speed']:.1f} tok/s")
        rs = aa.get("end_to_end_response_time") or {}
        if rs.get("total"):
            print(f"  end-to-end response: {rs['total']:.0f} ms")
    else:
        print("\n[ArtificialAnalysis] not found")

    if md:
        cost = md.get("cost") or {}
        lim = md.get("limit") or {}
        print(f"\n[models.dev] {md.get('name')} ({md.get('id')})")
        print(f"  family: {md.get('family')} | reasoning: {md.get('reasoning')} | tool_call: {md.get('tool_call')} | open_weights: {md.get('open_weights')}")
        if cost:
            print(f"  cost in/out: ${cost.get('input', 0):.3f}/${cost.get('output', 0):.3f} | cache_read: ${cost.get('cache_read') or '-'}")
        if lim:
            print(f"  limit ctx/out: {lim.get('context', 0):,}/{lim.get('output', 0):,}")
        if md.get("release_date"):
            print(f"  release: {md.get('release_date')} | knowledge: {md.get('knowledge')}")
        if md.get("reasoning_options"):
            print(f"  reasoning options: {md['reasoning_options']}")
    else:
        print("\n[models.dev] not found")

    if bench:
        print(f"\n[Benchmark] {bench.get('display_name') or bench.get('da_model_id')}")
        print(f"  score: {bench.get('score')} | win_rate: {bench.get('win_rate')}%")
        if bench.get("avg_generation_time_ms"):
            print(f"  avg gen: {bench.get('avg_generation_time_ms'):,.0f} ms")
    else:
        print("\n[Benchmark] not found")


def print_analysis_json(analysis: dict) -> None:
    json.dump(analysis, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def print_analysis_jsonl(analysis: dict) -> None:
    json.dump(analysis, sys.stdout, separators=(",", ":"), ensure_ascii=False, default=str)
    sys.stdout.write("\n")