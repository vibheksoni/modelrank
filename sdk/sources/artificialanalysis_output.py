"""Rendering helpers for artificialanalysis.ai data (plain + machine formats)."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from typing import Optional

from sdk.sources.artificialanalysis import AAModel

AA_CSV_FIELDS = [
    ("name", lambda m: m.name),
    ("slug", lambda m: m.slug),
    ("host", lambda m: m.host),
    ("context_window_tokens", lambda m: m.context_window_tokens),
    ("price_1m_input_tokens", lambda m: m.price_1m_input_tokens),
    ("price_1m_output_tokens", lambda m: m.price_1m_output_tokens),
    ("cache_hit_price", lambda m: m.cache_hit_price),
    ("cache_hit_discount_percent", lambda m: m.cache_hit_discount_percent),
    ("blended_0_3_1", lambda m: m.price_1m_blended_0_3_1),
    ("blended_7_2_1", lambda m: m.price_1m_blended_7_2_1),
    ("blended_0_1_1", lambda m: m.price_1m_blended_0_1_1),
    ("median_output_speed", lambda m: m.output_speed_variance.median if m.output_speed_variance else ""),
    ("p50_time_to_first_chunk", lambda m: m.time_to_first_chunk_variance.median if m.time_to_first_chunk_variance else ""),
    ("end_to_end_total_ms", lambda m: m.end_to_end_response_time.total if m.end_to_end_response_time else ""),
    ("time_to_first_answer_ms", lambda m: m.time_to_first_answer_token.total if m.time_to_first_answer_token else ""),
    ("briefcase_total_cost", lambda m: m.briefcase_total_cost),
    ("intelligence_index", lambda m: m.benchmarks.intelligence_index if m.benchmarks else ""),
    ("agentic_index", lambda m: m.benchmarks.agentic_index if m.benchmarks else ""),
    ("omniscience", lambda m: m.benchmarks.omniscience if m.benchmarks else ""),
    ("hle", lambda m: m.benchmarks.hle if m.benchmarks else ""),
    ("gpqa", lambda m: m.benchmarks.gpqa if m.benchmarks else ""),
    ("scicode", lambda m: m.benchmarks.scicode if m.benchmarks else ""),
    ("terminalbench_v21", lambda m: m.benchmarks.terminalbench_v21 if m.benchmarks else ""),
    ("lcr", lambda m: m.benchmarks.lcr if m.benchmarks else ""),
    ("reasoning_tokens", lambda m: m.reasoning_tokens),
    ("reasoning", lambda m: m.reasoning),
]


def print_aa_plain(models: list[AAModel]) -> None:
    if not models:
        print("No models found matching your criteria.")
        return
    has_bench = any(m.benchmarks and m.benchmarks.intelligence_index is not None for m in models)
    header = f"{'#':>3}  {'Model':<40} {'Host':<16} {'Ctx':>8} {'In $/M':>8} {'Out $/M':>8} {'TPS':>6}"
    if has_bench:
        header += f" {'Intel':>7} {'Agentic':>8} {'R':>2}"
    print(header)
    print("-" * (97 if has_bench else 88))
    for i, m in enumerate(models, 1):
        name = (m.name or m.slug)[:40]
        ctx = f"{m.context_window_tokens:,}" if m.context_window_tokens else "-"
        inp = f"{m.price_1m_input_tokens:.2f}" if m.price_1m_input_tokens else "-"
        out = f"{m.price_1m_output_tokens:.2f}" if m.price_1m_output_tokens else "-"
        tps = f"{m.output_speed_variance.median:.0f}" if m.output_speed_variance and m.output_speed_variance.median else "-"
        line = f"{i:>3}  {name:<40} {(m.host or '')[:16]:<16} {ctx:>8} {inp:>8} {out:>8} {tps:>6}"
        if has_bench:
            b = m.benchmarks
            ii = f"{b.intelligence_index:.1f}" if b and b.intelligence_index is not None else "-"
            ai = f"{b.agentic_index:.1f}" if b and b.agentic_index is not None else "-"
            r = "R" if m.reasoning else ""
            line += f" {ii:>7} {ai:>8} {r:>2}"
        print(line)


def _write_csv(models: list[AAModel], delimiter: str) -> None:
    writer = csv.writer(sys.stdout, delimiter=delimiter, lineterminator=chr(10))
    writer.writerow([h for h, _ in AA_CSV_FIELDS])
    for m in models:
        writer.writerow([str(fn(m)) for _, fn in AA_CSV_FIELDS])


def print_aa_csv(models: list[AAModel], delimiter: str = ",") -> None:
    _write_csv(models, delimiter)


def print_aa_json(models: list[AAModel]) -> None:
    json.dump([asdict(m) for m in models], sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def print_aa_jsonl(models: list[AAModel]) -> None:
    for m in models:
        json.dump(asdict(m), sys.stdout, separators=(",", ":"), ensure_ascii=False)
        sys.stdout.write(chr(10))


def print_aa_rich(models: list[AAModel]) -> None:
    """Rich table for AA models (falls back to plain if rich is unavailable)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print_aa_plain(models)
        return
    if not models:
        Console().print("[yellow]No models found matching your criteria.[/yellow]")
        return

    has_bench = any(m.benchmarks and m.benchmarks.intelligence_index is not None for m in models)
    table = Table(title="ArtificialAnalysis.ai Models", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan", max_width=42, overflow="fold")
    table.add_column("Host", style="white", max_width=16, overflow="fold")
    table.add_column("Ctx", style="yellow", justify="right", min_width=9)
    table.add_column("In $/M", style="green", justify="right", min_width=7)
    table.add_column("Out $/M", style="green", justify="right", min_width=7)
    table.add_column("TPS", style="bright_blue", justify="right", min_width=5)
    if has_bench:
        table.add_column("Intel", style="magenta", justify="right", min_width=6)
        table.add_column("Agentic", style="magenta", justify="right", min_width=7)
        table.add_column("R", style="magenta", width=1)

    def _safe(v: str) -> str:
        """Strip non-cp1252 characters so the Windows console never mojibakes."""
        return v.encode("ascii", "replace").decode("ascii")

    for i, m in enumerate(models, 1):
        ctx = f"{m.context_window_tokens:,}" if m.context_window_tokens else "-"
        inp = f"{m.price_1m_input_tokens:.2f}" if m.price_1m_input_tokens else "-"
        out = f"{m.price_1m_output_tokens:.2f}" if m.price_1m_output_tokens else "-"
        tps = f"{m.output_speed_variance.median:.0f}" if m.output_speed_variance and m.output_speed_variance.median else "-"
        row = [str(i), _safe((m.name or m.slug)[:42]), _safe((m.host or "")[:16]), ctx, inp, out, tps]
        if has_bench:
            b = m.benchmarks
            ii = f"{b.intelligence_index:.1f}" if b and b.intelligence_index is not None else "-"
            ai = f"{b.agentic_index:.1f}" if b and b.agentic_index is not None else "-"
            row += [ii, ai, "R" if m.reasoning else ""]
        table.add_row(*row)

    Console().print(table)
