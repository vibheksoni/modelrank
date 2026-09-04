"""Rendering helpers for models.dev data (plain + machine formats)."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict

from sdk.sources.modelsdev import MDModel

MD_CSV_FIELDS = [
    ("id", lambda m: m.id),
    ("name", lambda m: m.name),
    ("family", lambda m: m.family),
    ("reasoning", lambda m: m.reasoning),
    ("tool_call", lambda m: m.tool_call),
    ("structured_output", lambda m: m.structured_output),
    ("open_weights", lambda m: m.open_weights),
    ("context", lambda m: m.limit.context if m.limit else ""),
    ("output_limit", lambda m: m.limit.output if m.limit else ""),
    ("input_cost_per_1m", lambda m: m.cost.input if m.cost else ""),
    ("output_cost_per_1m", lambda m: m.cost.output if m.cost else ""),
    ("cache_read_per_1m", lambda m: m.cost.cache_read if m.cost else ""),
    ("cache_write_per_1m", lambda m: m.cost.cache_write if m.cost else ""),
    ("input_modalities", lambda m: ",".join(m.modalities.input) if m.modalities else ""),
    ("output_modalities", lambda m: ",".join(m.modalities.output) if m.modalities else ""),
    ("knowledge", lambda m: m.knowledge),
    ("release_date", lambda m: m.release_date),
]


def print_md_plain(models: list[MDModel]) -> None:
    if not models:
        print("No models found matching your criteria.")
        return
    header = f"{'#':>3}  {'Model':<46} {'In $/M':>8} {'Out $/M':>8} {'Ctx':>9} {'R':>2} {'Tool':>5} {'OW':>3}"
    print(header)
    print("-" * 92)
    for i, m in enumerate(models, 1):
        name = (m.id or m.name)[:46]
        inp = f"{m.cost.input:.3f}" if m.cost and m.cost.input else "-"
        out = f"{m.cost.output:.3f}" if m.cost and m.cost.output else "-"
        ctx = f"{m.limit.context:,}" if m.limit and m.limit.context else "-"
        r = "R" if m.reasoning else ""
        t = "Y" if m.tool_call else ""
        ow = "Y" if m.open_weights else ""
        print(f"{i:>3}  {name:<46} {inp:>8} {out:>8} {ctx:>9} {r:>2} {t:>5} {ow:>3}")


def _write_csv(models: list[MDModel], delimiter: str) -> None:
    writer = csv.writer(sys.stdout, delimiter=delimiter, lineterminator=chr(10))
    writer.writerow([h for h, _ in MD_CSV_FIELDS])
    for m in models:
        writer.writerow([str(fn(m)) for _, fn in MD_CSV_FIELDS])


def print_md_csv(models: list[MDModel], delimiter: str = ",") -> None:
    _write_csv(models, delimiter)


def print_md_json(models: list[MDModel]) -> None:
    json.dump([asdict(m) for m in models], sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


def print_md_jsonl(models: list[MDModel]) -> None:
    for m in models:
        json.dump(asdict(m), sys.stdout, separators=(",", ":"), ensure_ascii=False)
        sys.stdout.write(chr(10))


def print_md_rich(models) -> None:
    """Rich table for models.dev models (falls back to plain)."""
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print_md_plain(models)
        return
    if not models:
        Console().print("[yellow]No models found matching your criteria.[/yellow]")
        return

    def _safe(v: str) -> str:
        return v.encode("ascii", "replace").decode("ascii")

    table = Table(title="models.dev Catalog", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan", max_width=44, overflow="fold")
    table.add_column("In $/M", style="green", justify="right", min_width=7)
    table.add_column("Out $/M", style="green", justify="right", min_width=7)
    table.add_column("Ctx", style="yellow", justify="right", min_width=9)
    table.add_column("CacheR", style="dim", justify="right", min_width=6)
    table.add_column("R", style="magenta", width=1)
    table.add_column("Tool", style="cyan", width=4)
    table.add_column("OW", style="green", width=2)

    for i, m in enumerate(models, 1):
        inp = f"{m.cost.input:.3f}" if m.cost and m.cost.input else "-"
        out = f"{m.cost.output:.3f}" if m.cost and m.cost.output else "-"
        ctx = f"{m.limit.context:,}" if m.limit and m.limit.context else "-"
        cr = f"{m.cost.cache_read:.3f}" if m.cost and m.cost.cache_read else "-"
        table.add_row(
            str(i),
            _safe((m.id or m.name)[:44]),
            inp, out, ctx, cr,
            "R" if m.reasoning else "",
            "Y" if m.tool_call else "",
            "Y" if m.open_weights else "",
        )

    Console().print(table)
