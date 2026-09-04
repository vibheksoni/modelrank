"""Plain-text rendering (no rich dependency) and JSON output."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from sdk.core.types import ModelEntry, ProviderEndpoint


def _tps_str(m: ModelEntry) -> str:
    if not m.tps_p50:
        return "-"
    return f"{m.tps_p50:,}"


def print_table_plain(
    models: list[ModelEntry],
    show_providers: bool = False,
    show_tps: bool = False,
    is_embed: bool = False,
) -> None:
    if not models:
        print("No models match the given filters.")
        return

    if is_embed:
        header = (
            f"{'#':>3}  {'Model ID':<45} {'Provider':<15} "
            f"{'$/1M':>8} {'Context':>10} "
        )
        if show_tps:
            header += f"{'TPS p50':>10} "
        header += f"{'Quant':<8}"
        print(header)
        print("-" * 110)

        for i, m in enumerate(models, 1):
            ctx_str = f"{m.context_length:,}" if m.context_length else "-"
            price = "FREE" if m.is_free else f"{m.input_price:.4f}"
            row = f"{i:>3}  {m.id:<45} {m.provider:<15} {price:>8} {ctx_str:>10} "
            if show_tps:
                row += f"{_tps_str(m):>10} "
            row += f"{m.quantization or '-':<8}"
            print(row)
    else:
        header = (
            f"{'#':>3}  {'Model ID':<45} {'Provider':<14} "
            f"{'In $/M':>8} {'Out $/M':>8} {'Total':>8} {'Ctx':>10} "
            f"{'In Mod':<18} {'Out Mod':<12} {'Free':>4}"
        )
        if show_tps:
            header += f" {'TPS p50':>10}"
        header += f" {'R':>1}"
        print(header)
        print("-" * 150)

        for i, m in enumerate(models, 1):
            ctx_str = f"{m.context_length:,}" if m.context_length else "-"
            free_str = "YES" if m.is_free else ""
            r_str = "R" if m.supports_reasoning else ""
            row = (
                f"{i:>3}  {m.id:<45} {m.provider:<14} "
                f"{m.input_price:>8.4f} {m.output_price:>8.4f} {m.total_price:>8.4f} "
                f"{ctx_str:>10} {m.input_modalities:<18} {m.output_modalities:<12} {free_str:>4}"
            )
            if show_tps:
                row += f" {_tps_str(m):>10}"
            row += f" {r_str:>1}"
            print(row)

            if show_providers and m.endpoints:
                _print_endpoints_plain(m.endpoints)

    print(f"\nTotal: {len(models)} model(s)" + ("  (R = reasoning)" if not is_embed else ""))


def _print_endpoints_plain(endpoints: list[ProviderEndpoint]) -> None:
    eps = sorted(endpoints, key=lambda e: e.total_price)
    print(
        f"{'':>3}  {'  -> Provider':<30} {'Tag':<28} "
        f"{'In $/M':>9} {'Out $/M':>9} {'Total':>9} {'Ctx':>10} {'Up 30m':>7}"
    )
    print(f"{'':>3}  {'-'*118}")
    for e in eps:
        ctx_str = f"{e.context_length:,}" if e.context_length else "-"
        up_str = f"{e.uptime_30m:.0f}%" if e.uptime_30m is not None else "-"
        print(
            f"{'':>3}    {e.provider_name:<28} {e.tag:<28} "
            f"{e.input_price:>9.4f} {e.output_price:>9.4f} {e.total_price:>9.4f} "
            f"{ctx_str:>10} {up_str:>7}"
        )


def print_detail_plain(m: ModelEntry, is_embed: bool) -> None:
    print(f"\n  {m.id}")
    print(f"    Name:        {m.name}")
    print(f"    Provider:    {m.provider}")
    print(f"    Context:     {m.context_length:,} tokens")

    if is_embed:
        price = "FREE" if m.is_free else f"${m.input_price:.3f}/1M tokens"
        print(f"    Price:       {price}")
    else:
        if m.is_free:
            print("    Price:       FREE")
        else:
            print(f"    Input:       ${m.input_price:.2f}/1M tokens")
            print(f"    Output:      ${m.output_price:.2f}/1M tokens")
        print(f"    Reasoning:   {'yes' if m.supports_reasoning else 'no'}")
        mods_in = m.input_modalities or "-"
        mods_out = m.output_modalities or "-"
        print(f"    Modalities:  {mods_in} -> {mods_out}")

    print(f"    Quant:       {m.quantization or '-'}")

    if m.tps_p50:
        print(f"    TPS p50:     {m.tps_p50:,} tok/s")
        if m.tps_p90:
            print(f"    TPS p90:     {m.tps_p90:,} tok/s")
        print(f"    Latency p50: {m.tps_latency:,.0f} ms")
        print(f"    Best on:     {m.tps_provider}")
        print(f"    Requests:    {m.tps_requests:,} (30min window)")

    if m.description:
        desc = m.description[:300].encode("ascii", "replace").decode("ascii")
        print(f"    Description: {desc}")


def print_json(models: list[ModelEntry]) -> None:
    payload = [asdict(m) for m in models]
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")

# ── comparison rendering ───────────────────────────────────────────────


def print_compare_plain(models, rows) -> None:
    """Render the attribute x model comparison table (plain text)."""
    if not models:
        print("No models found matching your criteria.")
        return

    header = f"{'Attribute':<22}"
    for m in models:
        header += f" {m.id[:28]:>28}"
    print(header)
    print("-" * (22 + 30 * len(models)))

    for label, values in rows:
        line = f"{label:<22}"
        for v in values:
            line += f" {v[:28]:>28}"
        print(line)


def print_compare_json(models, rows) -> None:
    """Emit a machine-readable comparison payload."""
    payload = {
        "models": [asdict(m) for m in models],
        "comparison": {label: values for label, values in rows},
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")






# ── machine formats (csv / tsv / jsonl) ────────────────────────────────

MODEL_CSV_FIELDS = [
    ("id", lambda m: m.id),
    ("permaslug", lambda m: m.permaslug),
    ("name", lambda m: m.name),
    ("provider", lambda m: m.provider),
    ("input_price_per_1m", lambda m: m.input_price),
    ("output_price_per_1m", lambda m: m.output_price),
    ("total_price_per_1m", lambda m: m.total_price),
    ("is_free", lambda m: m.is_free),
    ("context_length", lambda m: m.context_length),
    ("input_modalities", lambda m: m.input_modalities),
    ("output_modalities", lambda m: m.output_modalities),
    ("supports_reasoning", lambda m: m.supports_reasoning),
    ("quantization", lambda m: m.quantization),
    ("tps_p50", lambda m: m.tps_p50),
    ("tps_p90", lambda m: m.tps_p90),
    ("tps_latency_ms", lambda m: m.tps_latency),
    ("tps_provider", lambda m: m.tps_provider),
    ("tps_requests", lambda m: m.tps_requests),
    ("model_version_group_id", lambda m: m.model_version_group_id),
    ("is_alias", lambda m: m.is_alias),
    ("canonical_id", lambda m: m.canonical_id),
    ("author", lambda m: m.author),
    ("knowledge_cutoff", lambda m: m.knowledge_cutoff),
    ("limit_rpm", lambda m: m.limit_rpm),
    ("limit_rpd", lambda m: m.limit_rpd),
    ("endpoint_count", lambda m: len(m.endpoints)),
    ("provider_regions", lambda m: ";".join(sorted({e.provider_region for e in m.endpoints if e.provider_region}))),
    ("byok", lambda m: any(e.is_byok for e in m.endpoints)),
    ("deranked", lambda m: any(e.is_deranked for e in m.endpoints)),
    ("disabled", lambda m: any(e.is_disabled for e in m.endpoints)),
]


def _write_csv(models, delimiter: str) -> None:
    import csv
    writer = csv.writer(sys.stdout, delimiter=delimiter, lineterminator=chr(10))
    writer.writerow([h for h, _ in MODEL_CSV_FIELDS])
    for m in models:
        writer.writerow([str(fn(m)) for _, fn in MODEL_CSV_FIELDS])


def print_csv(models: list[ModelEntry]) -> None:
    """One row per model, comma-delimited, header line included."""
    _write_csv(models, ",")


def print_tsv(models: list[ModelEntry]) -> None:
    """One row per model, tab-delimited, header line included."""
    _write_csv(models, chr(9))


def print_jsonl(models: list[ModelEntry]) -> None:
    """NDJSON: one compact JSON object per model per line."""
    for m in models:
        json.dump(asdict(m), sys.stdout, separators=(",", ":"))
        sys.stdout.write(chr(10))


OUTPUT_FORMATS = ("table", "plain", "json", "jsonl", "csv", "tsv")
