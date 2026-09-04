"""Benchmark rankings mode.

Lists or searches OpenRouter's internal benchmark leaderboards:

- **Design-arena (daData)** categories: `models-website`, `models-3d`,
  `models-codecategories`, `models-dataviz`, `models-image`, `agents-*`,
  etc. Rows carry `score`, `win_rate`, `avg_generation_time_ms`.
- **Adaptive-arena (aaData)** groups: `intelligence`, `coding`, `agentic`
  with reasoning-effort-tagged scores, plus per-slug percentiles.

Usage from cli.py:
    python cli.py benchmarks \[category\] \[-s search\] \[-n limit\]
    python cli.py benchmarks                                # list categories
    python cli.py benchmarks models-codecategories          # top of a category
    python cli.py benchmarks intelligence -s deepseek       # search within group
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from sdk.core.client import OpenRouterClient
from sdk.openrouter.api import fetch_benchmarks
from sdk.core.types import BenchmarkEntry, BenchmarkSnapshot


def list_categories(snapshot: BenchmarkSnapshot) -> list[str]:
    """Return da + aa category ids (aa first, grouped)."""
    return snapshot.aa_category_ids + snapshot.da_category_ids


def get_rows(snapshot: BenchmarkSnapshot, category: str) -> list[BenchmarkEntry]:
    """Rows for a category; da categories and aa groups both supported."""
    if category in snapshot.aa_categories:
        return snapshot.aa_categories[category]
    return snapshot.da_categories.get(category, [])


def enrich_with_models(
    rows: list[BenchmarkEntry],
    client: OpenRouterClient,
) -> dict[str, dict]:
    """Join benchmark entries with live catalog data by openrouter_id.

    Returns {openrouter_id: {id, input_price, output_price, context_length,
    supports_reasoning, tps_p50}} so renderers can show cost + capacity
    alongside benchmark scores.
    """
    from sdk.openrouter.api import fetch_models_v1

    try:
        catalog = fetch_models_v1(client)
    except Exception:
        return {}

    by_id: dict[str, dict] = {}
    for m in catalog:
        by_id[m.id] = {
            "id": m.id,
            "input_price": m.input_price,
            "output_price": m.output_price,
            "context_length": m.context_length,
            "supports_reasoning": m.supports_reasoning,
            "tps_p50": m.tps_p50,
        }

    out: dict[str, dict] = {}
    for r in rows:
        key = r.openrouter_id or r.da_model_id or ""
        if key in by_id:
            out[key] = by_id[key]
    return out


def filter_rows(rows: list[BenchmarkEntry], search: Optional[str], top: Optional[int]) -> list[BenchmarkEntry]:
    if search:
        q = search.lower()
        rows = [
            r for r in rows
            if q in (r.permaslug or "").lower()
            or q in (r.display_name or "").lower()
            or q in (r.openrouter_id or "").lower()
        ]
    if top:
        rows = rows[:top]
    return rows


# ── plain renderers ────────────────────────────────────────────────────


def print_categories_plain(snapshot: BenchmarkSnapshot) -> None:
    print("Adaptive-arena groups (aaData):")
    for cid in snapshot.aa_category_ids:
        print(f"  {cid:<20} {len(snapshot.aa_categories[cid]):>4} models")
    print("\nDesign-arena categories (daData):")
    for cid in snapshot.da_category_ids:
        print(f"  {cid:<32} {len(snapshot.da_categories[cid]):>4} models")


def print_benchmarks_plain(rows: list[BenchmarkEntry], category: str, models: Optional[dict] = None) -> None:
    if not rows:
        print("No benchmark entries found.")
        return
    print(f"Benchmark: {category}  ({len(rows)} models)")
    if models:
        print(f"{'#':>3}  {'Model':<36} {'Score':>7} {'Win%':>6} {'In $/M':>8} {'Out $/M':>8} {'Ctx':>9}")
        print("-" * 80)
        for i, r in enumerate(rows, 1):
            name = (r.display_name or r.da_model_id or r.openrouter_id or r.permaslug)[:36]
            score = f"{r.score:.1f}" if r.score else "-"
            win = f"{r.win_rate:.1f}" if r.win_rate else "-"
            key = r.openrouter_id or ""
            m = (models or {}).get(key, {})
            if m:
                inp = f"{m['input_price']:.3f}"
                out = f"{m['output_price']:.3f}"
                ctx = f"{m['context_length']:,}" if m['context_length'] else "-"
            else:
                inp = out = ctx = "-"
            print(f"{i:>3}  {name:<36} {score:>7} {win:>6} {inp:>8} {out:>8} {ctx:>9}")
    else:
        print(f"{'#':>3}  {'Model':<42} {'Score':>8} {'Win%':>7} {'Avg gen ms':>12}")
        print("-" * 78)
        for i, r in enumerate(rows, 1):
            name = r.display_name or r.da_model_id or r.openrouter_id or r.permaslug
            score = f"{r.score:.1f}" if r.score else "-"
            win = f"{r.win_rate:.1f}" if r.win_rate else "-"
            gen = f"{r.avg_generation_time_ms:,.0f}" if r.avg_generation_time_ms else "-"
            print(f"{i:>3}  {name[:42]:<42} {score:>8} {win:>7} {gen:>12}")


def print_benchmarks_json(snapshot: BenchmarkSnapshot, category: Optional[str], rows: Optional[list[BenchmarkEntry]]) -> None:
    if category and rows is not None:
        payload = {
            "category": category,
            "entries": [r.__dict__ for r in rows],
        }
    else:
        payload = {
            "categories": {
                "aa": {
                    cid: [r.__dict__ for r in rows]
                    for cid, rows in snapshot.aa_categories.items()
                },
                "da": {
                    cid: [r.__dict__ for r in rows]
                    for cid, rows in snapshot.da_categories.items()
                },
            },
            "weighted_input_prices": snapshot.weighted_input_prices,
            "cost_per_request": snapshot.cost_per_request,
        }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


# ── rich renderer ──────────────────────────────────────────────────────


def print_benchmarks_rich(rows: list[BenchmarkEntry], category: str, models: Optional[dict] = None) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print_benchmarks_plain(rows, category)
        return

    if not rows:
        Console().print("[yellow]No benchmark entries found.[/yellow]")
        return

    table = Table(title=f"Benchmark: {category}", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan", max_width=46, overflow="crop")
    table.add_column("Score", style="green", justify="right")
    table.add_column("Win %", style="yellow", justify="right")
    if models:
        table.add_column("In $/M", style="green", justify="right")
        table.add_column("Out $/M", style="green", justify="right")
        table.add_column("Ctx", style="yellow", justify="right")
    else:
        table.add_column("Avg gen ms", style="bright_blue", justify="right")

    for i, r in enumerate(rows, 1):
        name = r.display_name or r.da_model_id or r.openrouter_id or r.permaslug
        score = f"{r.score:.1f}" if r.score else "-"
        win = f"{r.win_rate:.1f}" if r.win_rate else "-"
        key = r.openrouter_id or ""
        m = (models or {}).get(key, {})
        if models:
            inp = f"{m['input_price']:.3f}" if m else "-"
            out = f"{m['output_price']:.3f}" if m else "-"
            ctx = f"{m['context_length']:,}" if m and m['context_length'] else "-"
            table.add_row(str(i), name, score, win, inp, out, ctx)
        else:
            gen = f"{r.avg_generation_time_ms:,.0f}" if r.avg_generation_time_ms else "-"
            table.add_row(str(i), name, score, win, gen)

    Console().print(table)

# ── machine formats ────────────────────────────────────────────────────

BENCHMARK_CSV_FIELDS = [
    ("rank", lambda r, i: i),
    ("da_model_id", lambda r, i: r.da_model_id),
    ("permaslug", lambda r, i: r.permaslug),
    ("openrouter_id", lambda r, i: r.openrouter_id),
    ("display_name", lambda r, i: r.display_name),
    ("score", lambda r, i: r.score),
    ("win_rate", lambda r, i: r.win_rate),
    ("avg_generation_time_ms", lambda r, i: r.avg_generation_time_ms),
    ("uid", lambda r, i: r.uid),
    ("aa_name", lambda r, i: r.aa_name),
]


def print_benchmarks_csv(rows: list[BenchmarkEntry], category: str) -> None:
    import csv
    writer = csv.writer(sys.stdout, lineterminator=chr(10))
    writer.writerow([h for h, _ in BENCHMARK_CSV_FIELDS])
    for i, r in enumerate(rows, 1):
        writer.writerow([str(fn(r, i)) for _, fn in BENCHMARK_CSV_FIELDS])


def print_benchmarks_tsv(rows: list[BenchmarkEntry], category: str) -> None:
    import csv
    writer = csv.writer(sys.stdout, delimiter=chr(9), lineterminator=chr(10))
    writer.writerow([h for h, _ in BENCHMARK_CSV_FIELDS])
    for i, r in enumerate(rows, 1):
        writer.writerow([str(fn(r, i)) for _, fn in BENCHMARK_CSV_FIELDS])


def print_benchmarks_jsonl(rows: list[BenchmarkEntry], category: str) -> None:
    import json
    for r in rows:
        json.dump({"category": category, "entry": r.__dict__}, sys.stdout, separators=(",", ":"))
        sys.stdout.write(chr(10))
