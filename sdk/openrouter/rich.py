"""Rich-powered table rendering (optional dependency)."""

from __future__ import annotations

from sdk.core.types import ModelEntry, ProviderEndpoint

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _tps_str(m: ModelEntry) -> str:
    if not m.tps_p50:
        return "-"
    return f"{m.tps_p50:,}"


def print_embed_table_rich(models: list[ModelEntry], console: Console, show_tps: bool = False) -> None:
    table = Table(title="OpenRouter Embedding Models", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan", max_width=45, overflow="crop")
    table.add_column("Provider", style="white", max_width=15, overflow="crop")
    table.add_column("$/1M tokens", style="green", justify="right", max_width=12, overflow="crop")
    table.add_column("Context", style="yellow", justify="right", max_width=10, overflow="crop")
    if show_tps:
        table.add_column("TPS p50", style="bright_blue", justify="right", max_width=8, overflow="crop")
    table.add_column("Quant", style="dim", max_width=8, overflow="crop")

    for i, m in enumerate(models, 1):
        price = "FREE" if m.is_free else f"${m.input_price:.3f}"
        ctx = f"{m.context_length:,}" if m.context_length else "-"
        row = [str(i), m.id, m.provider, price, ctx]
        if show_tps:
            row.append(_tps_str(m))
        row.append(m.quantization or "-")
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]{len(models)} embedding models found[/dim]")


def print_llm_table_rich(models: list[ModelEntry], console: Console, show_tps: bool = False) -> None:
    table = Table(title="OpenRouter LLM Models", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Model", style="cyan", max_width=45, overflow="crop")
    table.add_column("Provider", style="white", max_width=14, overflow="crop")
    table.add_column("In $/1M", style="green", justify="right", max_width=10, overflow="crop")
    table.add_column("Out $/1M", style="green", justify="right", max_width=10, overflow="crop")
    table.add_column("Context", style="yellow", justify="right", max_width=10, overflow="crop")
    if show_tps:
        table.add_column("TPS p50", style="bright_blue", justify="right", max_width=8, overflow="crop")
    table.add_column("R", style="magenta", width=1)

    for i, m in enumerate(models, 1):
        if m.is_free:
            in_str = "FREE"
            out_str = "FREE"
        else:
            in_str = f"${m.input_price:.2f}"
            out_str = f"${m.output_price:.2f}"
        ctx = f"{m.context_length:,}" if m.context_length else "-"
        r = "R" if m.supports_reasoning else ""
        row = [str(i), m.id, m.provider, in_str, out_str, ctx]
        if show_tps:
            row.append(_tps_str(m))
        row.append(r)
        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]{len(models)} LLM models found  (R = reasoning)[/dim]")


def print_detail_rich(m: ModelEntry, console: Console, is_embed: bool) -> None:
    console.print(f"\n[bold cyan]{m.id}[/bold cyan]")
    console.print(f"  Name:        {m.name}")
    console.print(f"  Provider:    {m.provider}")
    console.print(f"  Context:     {m.context_length:,} tokens")

    if is_embed:
        price = "FREE" if m.is_free else f"${m.input_price:.3f}/1M tokens"
        console.print(f"  Price:       {price}")
    else:
        if m.is_free:
            console.print("  Price:       FREE")
        else:
            console.print(f"  Input:       ${m.input_price:.2f}/1M tokens")
            console.print(f"  Output:      ${m.output_price:.2f}/1M tokens")
        console.print(f"  Reasoning:   {'yes' if m.supports_reasoning else 'no'}")
        mods_in = m.input_modalities or "-"
        mods_out = m.output_modalities or "-"
        console.print(f"  Modalities:  {mods_in} -> {mods_out}")

    console.print(f"  Quant:       {m.quantization or '-'}")

    if m.tps_p50:
        console.print(f"  TPS p50:     {m.tps_p50:,} tok/s")
        if m.tps_p90:
            console.print(f"  TPS p90:     {m.tps_p90:,} tok/s")
        console.print(f"  Latency p50: {m.tps_latency:,.0f} ms")
        console.print(f"  Best on:     {m.tps_provider}")
        console.print(f"  Requests:    {m.tps_requests:,} (30min window)")

    if m.description:
        desc = m.description[:300].encode("ascii", "replace").decode("ascii")
        console.print(f"  Description: {desc}")


def print_table_rich(
    models: list[ModelEntry],
    show_providers: bool = False,
    show_tps: bool = False,
    is_embed: bool = False,
) -> None:
    console = Console()

    if is_embed:
        print_embed_table_rich(models, console, show_tps=show_tps)
    else:
        print_llm_table_rich(models, console, show_tps=show_tps)

    if show_providers:
        for m in models:
            if m.endpoints:
                _print_endpoints_rich(m.endpoints, console)

    console.print(f"\n[dim]Total: {len(models)} model(s)[/dim]")


def _print_endpoints_rich(endpoints: list[ProviderEndpoint], console: Console) -> None:
    eps = sorted(endpoints, key=lambda e: e.total_price)
    table = Table(show_lines=True)
    table.add_column("Provider", style="white", max_width=28, overflow="crop")
    table.add_column("Tag", style="dim", max_width=28, overflow="crop")
    table.add_column("In $/M", style="green", justify="right")
    table.add_column("Out $/M", style="green", justify="right")
    table.add_column("Total", style="green", justify="right")
    table.add_column("Context", style="yellow", justify="right")
    table.add_column("Up 30m", style="bright_blue", justify="right")

    for e in eps:
        ctx_str = f"{e.context_length:,}" if e.context_length else "-"
        up_str = f"{e.uptime_30m:.0f}%" if e.uptime_30m is not None else "-"
        table.add_row(
            e.provider_name, e.tag,
            f"{e.input_price:.4f}", f"{e.output_price:.4f}", f"{e.total_price:.4f}",
            ctx_str, up_str,
        )

    console.print(table)

# ── comparison rendering ───────────────────────────────────────────────


def print_compare_rich(models, rows, console: Console = None) -> None:
    """Render the attribute x model comparison table via rich."""
    if not models:
        if console:
            console.print("[yellow]No models found matching your criteria.[/yellow]")
        else:
            print("No models found matching your criteria.")
        return
    console = console or Console()

    table = Table(title="Model Comparison", show_lines=True)
    table.add_column("Attribute", style="bold cyan", max_width=24, overflow="crop")
    for m in models:
        table.add_column(m.id[:20], max_width=26, overflow="crop", style="white")

    for label, values in rows:
        table.add_row(label, *[str(v)[:24] for v in values])

    console.print(table)
