#!/usr/bin/env python3
"""OpenRouter model filter CLI entry point.

Usable in three modes:

- No positional arg: public /api/v1/models (full model list).
- `embed` / `llm`: frontend /find API with modality-specific views.
- `--tps` adds live throughput stats; `--show-providers` expands endpoints.

Example:
    python cli.py llm -r --no-free --min-tps 100 --sort tps
"""

from __future__ import annotations

import argparse
import sys

from curl_cffi.requests.exceptions import RequestException

from sdk.core.client import OpenRouterClient
from sdk.core.constants import ORDER_CHOICES
from sdk.openrouter.api import (
    enrich_with_endpoints,
    enrich_with_tps,
    fetch_benchmarks,
    fetch_models_find,
    fetch_models_v1,
)
from sdk.sources.artificialanalysis import fetch_aa_dataset, fetch_aa_full, fetch_aa_merged
from sdk.analysis import (
    analyze_model,
    print_analysis_json,
    print_analysis_jsonl,
    print_analysis_plain,
)
from sdk.sources.modelsdev import fetch_models_dev
from sdk.openrouter.benchmarks import (
    filter_rows,
    get_rows,
    list_categories,
    print_benchmarks_json,
    print_benchmarks_plain,
    print_benchmarks_rich,
    print_categories_plain,
)
from sdk.openrouter.compare import (
    compare_attributes,
    prepare_comparison,
    print_compare_csv,
    print_compare_jsonl,
    print_compare_tsv,
)
from sdk.openrouter.filters import dedupe_models, filter_models
from sdk.core.types import ModelEntry
from sdk.openrouter.output import (
    OUTPUT_FORMATS,
    print_compare_json,
    print_compare_plain,
    print_csv,
    print_detail_plain,
    print_json,
    print_jsonl,
    print_table_plain,
    print_tsv,
)
from sdk.openrouter.relations import resolve_relations
from sdk.openrouter.rich import HAS_RICH, print_compare_rich, print_detail_rich, print_table_rich
from sdk.openrouter.benchmarks import (
    enrich_with_models,
    print_benchmarks_csv,
    print_benchmarks_jsonl,
    print_benchmarks_tsv,
)
from sdk.openrouter.sorting import SORT_FIELDS, sort_models


# ── CLI ────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Filter and sort OpenRouter models by price, provider, context length, "
            "modality, throughput, and more. Supports both the public /v1/models API "
            "and the internal frontend /find API with live TPS stats."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Cheapest 20 models by input price (public API)
  python cli.py --sort input --top 20

  # Embedding models under $0.03/1M with live TPS stats (frontend API)
  python cli.py embed --no-free --max-price 0.03 --tps --sort tps --limit 12

  # LLM models from deepseek with detail view
  python cli.py llm -s deepseek -d

  # LLM models sorted by throughput
  python cli.py llm --tps --sort tps -n 10

  # Reasoning models with TPS, no free tier
  python cli.py llm -r --no-free --tps

  # 100K+ context LLMs sorted by context length
  python cli.py llm -c 100000 --sort context

  # Free models only (public API)
  python cli.py --free-only

  # Anthropic models under $1/M input, show all upstream providers
  python cli.py --provider anthropic --max-input 1.0 --sort output --show-providers

  # Search "claude" and output as JSON
  python cli.py --search claude --json

  # Models that accept audio input and output text
  python cli.py --input-modality audio --output-modality text --sort input

  # Models that can generate images
  python cli.py --output-modality image --sort input

  # Models that accept text+image+video input (must support ALL three)
  python cli.py --input-modality text,image,video --sort input
""",
    )

    p.add_argument(
        "mode",
        nargs="?",
        default=None,
        choices=["embed", "llm", "compare", "benchmarks", "aa", "modelsdev", "analyze"],
        help="Model type: 'embed' for embeddings, 'llm' for text generation, "
             "'compare' for side-by-side model comparison, "
             "'benchmarks' for benchmark leaderboards, "
             "'aa' for artificialanalysis.ai (multi-analysis), "
             "'modelsdev' for the models.dev catalog, "
             "'analyze' for cross-source analysis of one model id. "
             "If omitted, uses the public /v1/models API with full filter set.",
    )
    p.add_argument(
        "model_ids",
        nargs="*",
        default=[],
        help="Compare mode: model ids to compare (deepseek/deepseek-v4-flash "
             "openai/gpt-5 ~openai/gpt-5-latest ...). Benchmarks mode: a single "
             "category id (models-codecategories, intelligence, ...).",
    )

    p.add_argument("--api-key", default=None, help="OpenRouter API key (optional, increases rate limits)")

    p.add_argument(
        "--search", "-s",
        default=None,
        help="Search query (e.g. 'deepseek', 'qwen', 'code'). Sent to API for server-side filtering.",
    )
    p.add_argument(
        "--order", "-o",
        default="most-popular",
        choices=ORDER_CHOICES,
        help="Server-side sort order for frontend API (default: most-popular).",
    )
    p.add_argument(
        "--max-price", "-p",
        type=float,
        default=None,
        help="Max input price per 1M tokens (server-side filter for frontend API).",
    )

    p.add_argument("--min-input", type=float, default=None, help="Min input price (USD per 1M tokens)")
    p.add_argument("--max-input", type=float, default=None, help="Max input price (USD per 1M tokens)")
    p.add_argument("--min-output", type=float, default=None, help="Min output price (USD per 1M tokens)")
    p.add_argument("--max-output", type=float, default=None, help="Max output price (USD per 1M tokens)")
    p.add_argument("--min-total", type=float, default=None, help="Min total price (in+out, USD per 1M tokens)")
    p.add_argument("--max-total", type=float, default=None, help="Max total price (in+out, USD per 1M tokens)")
    p.add_argument("--max-output-price", type=float, default=None,
                   help="Alias for --max-output (client-side filter)")

    p.add_argument("--min-context", "-c", type=int, default=None, help="Min context length in tokens")
    p.add_argument("--max-context", type=int, default=None, help="Max context length in tokens")

    p.add_argument("--provider", default=None, help="Filter by model provider prefix (e.g. anthropic, openai)")
    p.add_argument("--input-modality", default=None,
                   help="Filter by input modality - model must support ALL listed (comma-separated)")
    p.add_argument("--output-modality", default=None,
                   help="Filter by output modality - model must support ALL listed (comma-separated)")

    p.add_argument("--free-only", action="store_true", help="Only show free models ($0 input + $0 output)")
    p.add_argument("--paid-only", action="store_true", help="Only show paid models")
    p.add_argument("--no-free", action="store_true", help="Exclude free-tier models")
    p.add_argument("--reasoning", "-r", action="store_true", help="Only show models with reasoning support")

    p.add_argument("--tps", action="store_true",
                   help="Fetch live throughput stats (TPS) for each model. Slower but shows speed.")
    p.add_argument("--min-tps", type=int, default=None,
                   help="Minimum p50 throughput (tok/s). Implies --tps.")

    p.add_argument("--sort", default="input", choices=list(SORT_FIELDS.keys()),
                   help="Sort field (default: input)")
    p.add_argument("--desc", action="store_true", help="Sort descending")

    p.add_argument("--top", type=int, default=None, help="Limit to top N results")
    p.add_argument("--limit", "-n", type=int, default=None, help="Alias for --top")
    p.add_argument("--detail", "-d", action="store_true", help="Show detailed info for each model")
    p.add_argument("--format", default=None, choices=list(OUTPUT_FORMATS),
                   help="Output format: table (rich), plain (ascii), json, jsonl (ndjson), csv, tsv. "
                        "Default: table when rich is available, plain otherwise.")
    p.add_argument("--json", action="store_true", help="Alias for --format json")
    p.add_argument("--show-providers", action="store_true",
                   help="Fetch and display all upstream provider endpoints per model "
                        "with per-provider pricing, uptime, and latency")
    p.add_argument("--prices", action="store_true",
                   help="Benchmarks mode: join entries with live catalog pricing "
                        "(cost + capacity alongside benchmark scores).")

    return p


# ── main ───────────────────────────────────────────────────────────────


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "compare":
        run_compare(args)
        return

    if args.mode == "benchmarks":
        run_benchmarks(args)
        return

    if args.mode == "aa":
        run_aa(args)
        return

    if args.mode == "modelsdev":
        run_modelsdev(args)
        return

    if args.mode == "analyze":
        run_analyze(args)
        return

    use_find_api = args.mode is not None
    is_embed = args.mode == "embed"
    limit = args.top or args.limit
    fetch_tps = args.tps or args.min_tps is not None or args.sort == "tps"

    with OpenRouterClient(api_key=args.api_key) as client:
        # ── fetch ──
        # The frontend /find API may be unavailable (404). Fall back to /v1/models
        # with client-side modality filtering when that happens.
        models: list[ModelEntry] | None = None
        if use_find_api:
            try:
                models = fetch_models_find(
                    client,
                    modality=args.mode,
                    query=args.search or "",
                    order=args.order,
                    max_price=args.max_price,
                )
            except RequestException:
                models = None

        if models is None:
            try:
                models = fetch_models_v1(client)
            except RequestException as e:
                print(f"Error fetching models from OpenRouter: {e}", file=sys.stderr)
                sys.exit(1)

            if use_find_api:
                # v1 API fallback: filter by output modality in code.
                target_modality = "embeddings" if is_embed else "text"
                models = [m for m in models if target_modality in m.output_modalities.lower()]

        resolve_relations(models)

        # ── filter ──
        max_output = args.max_output or args.max_output_price

        models = filter_models(
            models,
            min_input=args.min_input,
            max_input=args.max_input or args.max_price,
            min_output=args.min_output,
            max_output=max_output,
            min_total=args.min_total,
            max_total=args.max_total,
            min_context=args.min_context,
            max_context=args.max_context,
            provider=args.provider,
            search=args.search,
            free_only=args.free_only,
            paid_only=args.paid_only,
            no_free=args.no_free,
            reasoning_only=args.reasoning,
            input_modality=args.input_modality,
            output_modality=args.output_modality,
            min_tps=args.min_tps if not fetch_tps else None,
        )

        models = dedupe_models(models)

        # ── sort (pre-TPS) ──
        if args.sort and args.sort != "tps":
            models = sort_models(models, args.sort, descending=args.desc)

        # ── limit before TPS fetch ──
        if limit and not fetch_tps:
            models = models[:limit]

        if not models:
            print("No models found matching your criteria.")
            return

        # ── fetch TPS if needed ──
        if fetch_tps:
            target = models[:limit] if limit else models
            enrich_with_tps(target, client)
            models = target

            if args.min_tps is not None:
                models = filter_models(models, min_tps=args.min_tps)

            if args.sort == "tps":
                models = sort_models(models, "tps", descending=args.desc)

            if limit:
                models = models[:limit]

        if not models:
            print("No models found matching your criteria.")
            return

        # ── fetch provider endpoints ──
        if args.show_providers:
            print(f"Fetching provider endpoints for {len(models)} model(s)...", file=sys.stderr)
            enrich_with_endpoints(models, client)

        # ── output ──
        fmt = _resolve_format(args)

        if fmt == "json":
            print_json(models)
        elif fmt == "jsonl":
            print_jsonl(models)
        elif fmt == "csv":
            print_csv(models)
        elif fmt == "tsv":
            print_tsv(models)
        elif args.detail:
            for m in models:
                if HAS_RICH:
                    print_detail_rich(m, __import__("rich.console", fromlist=["Console"]).Console(), is_embed)
                else:
                    print_detail_plain(m, is_embed)
        else:
            if fmt == "table" and HAS_RICH:
                print_table_rich(models, show_providers=args.show_providers, show_tps=fetch_tps, is_embed=is_embed)
            else:
                print_table_plain(models, show_providers=args.show_providers, show_tps=fetch_tps, is_embed=is_embed)


def _resolve_format(args) -> str:
    if args.json:
        return "json"
    return args.format or ("table" if HAS_RICH else "plain")


def run_compare(args) -> None:
    """Side-by-side model comparison mode."""
    if not args.model_ids:
        print("compare mode requires at least one model id.", file=sys.stderr)
        sys.exit(2)

    with OpenRouterClient(api_key=args.api_key) as client:
        models = prepare_comparison(
            client,
            args.model_ids,
            tps=args.tps or args.min_tps is not None,
            show_providers=args.show_providers,
        )

        if not models:
            print("No models found matching your criteria.")
            return

        rows = compare_attributes(models)
        fmt = _resolve_format(args)

        if fmt == "json":
            print_compare_json(models, rows)
        elif fmt == "jsonl":
            print_compare_jsonl(models, rows)
        elif fmt == "csv":
            print_compare_csv(models, rows)
        elif fmt == "tsv":
            print_compare_tsv(models, rows)
        elif fmt == "plain" or not HAS_RICH:
            print_compare_plain(models, rows)
        else:
            print_compare_rich(models, rows)


def run_benchmarks(args) -> None:
    """Benchmark leaderboards mode."""
    category = args.model_ids[0] if args.model_ids else None

    with OpenRouterClient(api_key=args.api_key) as client:
        snapshot = fetch_benchmarks(client)

        fmt = _resolve_format(args)

        if not category:
            if fmt == "json":
                print_benchmarks_json(snapshot, None, None)
            elif fmt == "jsonl":
                for cid in list_categories(snapshot):
                    for r in get_rows(snapshot, cid):
                        print_benchmarks_jsonl([r], cid)
            else:
                print_categories_plain(snapshot)
            return

        rows = filter_rows(get_rows(snapshot, category), args.search, args.limit or args.top)
        joined = enrich_with_models(rows, client) if args.prices else None

        if fmt == "json":
            print_benchmarks_json(snapshot, category, rows)
        elif fmt == "jsonl":
            print_benchmarks_jsonl(rows, category)
        elif fmt == "csv":
            print_benchmarks_csv(rows, category)
        elif fmt == "tsv":
            print_benchmarks_tsv(rows, category)
        elif fmt == "plain" or not HAS_RICH:
            print_benchmarks_plain(rows, category, joined)
        else:
            print_benchmarks_rich(rows, category, joined)


def run_aa(args) -> None:
    """ArtificialAnalysis.ai multi-analysis mode."""
    with OpenRouterClient(api_key=args.api_key) as client:
        try:
            models = fetch_aa_full(client)
        except Exception:
            models = fetch_aa_dataset(client, page="/models")
        query = (args.search or "").lower()
        if query:
            models = [m for m in models if query in m.name.lower() or query in m.slug.lower() or query in m.host.lower()]
        if args.limit or args.top:
            models = models[: args.limit or args.top]
        fmt = _resolve_format(args)
        from sdk.sources.artificialanalysis_output import print_aa_json, print_aa_jsonl, print_aa_plain, print_aa_rich
        if fmt == "json":
            print_aa_json(models)
        elif fmt == "jsonl":
            print_aa_jsonl(models)
        elif fmt == "csv":
            from sdk.sources.artificialanalysis_output import print_aa_csv
            print_aa_csv(models)
        elif fmt == "tsv":
            from sdk.sources.artificialanalysis_output import print_aa_csv
            print_aa_csv(models, delimiter=chr(9))
        elif fmt in ("plain", "table") and not HAS_RICH:
            print_aa_plain(models)
        elif fmt == "table":
            print_aa_rich(models)
        else:
            print_aa_plain(models)


def run_modelsdev(args) -> None:
    """models.dev catalog mode."""
    with OpenRouterClient(api_key=args.api_key) as client:
        dataset = fetch_models_dev(client)
        models = dataset.all_models
        query = (args.search or "").lower()
        if query:
            models = [m for m in models if query in m.id.lower() or query in m.name.lower()]
        if args.limit or args.top:
            models = models[: args.limit or args.top]
        fmt = _resolve_format(args)
        from sdk.sources.modelsdev_output import print_md_csv, print_md_json, print_md_jsonl, print_md_plain, print_md_rich
        if fmt == "json":
            print_md_json(models)
        elif fmt == "jsonl":
            print_md_jsonl(models)
        elif fmt == "csv":
            print_md_csv(models)
        elif fmt == "tsv":
            print_md_csv(models, delimiter=chr(9))
        elif fmt == "table" and HAS_RICH:
            print_md_rich(models)
        else:
            print_md_plain(models)


def run_analyze(args) -> None:
    """Cross-source analysis for one model id."""
    if not args.model_ids:
        print("analyze mode requires one model id.", file=sys.stderr)
        sys.exit(2)
    model_id = args.model_ids[0]

    with OpenRouterClient(api_key=args.api_key) as client:
        analysis = analyze_model(
            client,
            model_id,
            tps=args.tps or args.min_tps is not None,
            show_providers=args.show_providers,
        )
        fmt = _resolve_format(args)
        if fmt == "jsonl":
            print_analysis_jsonl(analysis)
        elif fmt in ("json", "csv", "tsv", "table", "plain"):
            if fmt == "json":
                print_analysis_json(analysis)
            else:
                print_analysis_plain(analysis)


if __name__ == "__main__":
    main()