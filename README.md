# modelrank

A CLI tool to list, filter, sort, compare, and analyze LLM models from multiple data sources. All from the terminal. No API key required.

Made for agents and humans who need to pick the right model for a task without browsing a website or guessing at endpoints.

## Table of Contents

- [Why did I make this?](#why-did-i-make-this)
- [What it does](#what-it-does)
- [Install](#install)
- [Usage](#usage)
  - [Filtering and sorting](#filtering-and-sorting)
  - [Embedding and LLM modes](#embedding-and-llm-modes)
  - [Compare mode](#compare-mode)
  - [Cross-source analysis](#cross-source-analysis)
  - [ArtificialAnalysis.ai leaderboard](#artificialanalysisai-leaderboard)
  - [models.dev catalog](#modelsdev-catalog)
  - [Benchmark leaderboards](#benchmark-leaderboards)
  - [Output formats](#output-formats)
- [Modality filtering](#modality-filtering)
- [Flags](#flags)
- [Architecture](#architecture)
- [API quirks](#api-quirks)
- [Data sources](#data-sources)

## Why did I make this?

Too many times I needed a model for a task. Cheapest, smartest, fastest, longest context, vision capable, whatever the constraint was. The process was always the same friction. Open a browser, tab through OpenRouter catalog, check ArtificialAnalysis for benchmarks, cross reference pricing on models.dev, try to remember which provider had the lowest latency, and eventually just pick something by gut feeling.

That is inefficient. The data is all available through APIs. The problem is that those APIs are spread across three different sites. One of them encrypts its data payload with AES-GCM and embeds the key in a React Server Component flight manifest. Another returns one row per provider endpoint instead of one row per model. The third ignores its own query parameters server side. None of them are documented for programmatic use.

This tool exists so that instead of browsing or guessing or brute forcing endpoints, you just run a command. An agent can pipe `--json` output directly into its next decision. A human can scan a sorted table in two seconds. The cheapest model with 128k context that supports image input and reasoning? One command. The benchmark scores and real world throughput for a specific model across all sources? One command. Side by side comparison of three candidates with per provider pricing? One command.

Let the model run the commands. That is the whole point.

## What it does

- **List** models with pricing, context length, modalities, reasoning, provider, and quantization.
- **Filter** by price, context, provider, name, modality, free or paid, reasoning, and throughput.
- **Sort** by input price, output price, total price, context, name, provider, or throughput.
- **Compare** models side by side with 26 attributes.
- **Analyze** a single model across all four data sources in one report.
- **Benchmark** leaderboards with optional live pricing join.
- **Enrich** with live throughput stats and per provider endpoints.
- **Output** in six formats. Rich table, plain ASCII, JSON, JSONL, CSV, and TSV.

## Install

```bash
pip install curl_cffi rich
```

`rich` is optional. The tool falls back to plain text tables automatically.

## Usage

### Filtering and sorting

```bash
# Cheapest 20 models by input price
python cli.py --sort input --top 20

# Free models only
python cli.py --free-only

# Anthropic models under $1/M input, sorted by output price
python cli.py --provider anthropic --max-input 1.0 --sort output

# Search and show all upstream providers per model
python cli.py --search claude --show-providers

# 128k+ context under $0.50/M total
python cli.py --min-context 128000 --max-total 0.50 --sort total

# Models that accept text+image+video input (must support ALL three)
python cli.py --input-modality text,image,video --sort input

# Most expensive models (descending)
python cli.py --sort total --desc --top 10
```

### Embedding and LLM modes

```bash
# Embedding models under $0.03/1M, cheapest first
python cli.py embed --max-price 0.03 --sort input --limit 12

# LLM models sorted by live throughput
python cli.py llm --tps --sort tps -n 10

# Reasoning models with TPS, no free tier
python cli.py llm -r --no-free --tps

# Models with at least 150 tok/s p50 throughput
python cli.py llm --min-tps 150
```

### Compare mode

```bash
# Side by side attributes (ids, ~aliases, short slugs all resolve)
python cli.py compare gpt-5 claude-sonnet-4 glm-5.3-flash

# With live TPS or per provider flags
python cli.py compare gpt-5 glm-5.3-flash --tps
python cli.py compare gpt-5 glm-5.3-flash --show-providers

# Any output format
python cli.py compare gpt-5 glm-5.3-flash --json
```

### Cross-source analysis

```bash
# One model id across all four data sources
python cli.py analyze glm-5.3-flash
python cli.py analyze gpt-5 --format json
```

### ArtificialAnalysis.ai leaderboard

```bash
# 367 merged models with benchmark indices, performance, and pricing
python cli.py aa -s glm -n 10
python cli.py aa -s claude --format csv
```

### models.dev catalog

```bash
# 203 providers and 7343 models with cost, limits, and capabilities
python cli.py modelsdev -s deepseek-v4 -n 5
python cli.py modelsdev -s qwen --format jsonl
```

### Benchmark leaderboards

```bash
# List all benchmark categories
python cli.py benchmarks

# Top coding models with price join
python cli.py benchmarks models-codecategories -n 10 --prices

# Search a category, output CSV
python cli.py benchmarks intelligence -s deepseek --format csv
```

### Output formats

All six formats work across every mode.

```bash
python cli.py llm -s claude --format csv
python cli.py compare gpt-5 glm-5.3-flash --format tsv
python cli.py aa -s glm --format json
```

## Modality filtering

**Available input modalities.** `text`, `image`, `file`, `video`, `audio`

**Available output modalities.** `text`, `image`, `audio`

When you pass comma separated values like `--input-modality text,image,audio`, the model must support all listed modalities. Use a single value to filter by one modality.

<details>
<summary>Flags</summary>

| Flag | Type | Description |
|---|---|---|
| `embed` / `llm` | positional | Use frontend find API for embedding or text models |
| `compare` | positional | Compare mode, accepts full ids, `~` aliases, short slugs |
| `benchmarks` | positional | Benchmark leaderboards |
| `aa` | positional | ArtificialAnalysis.ai merged leaderboard |
| `modelsdev` | positional | models.dev catalog |
| `analyze` | positional | Cross source analysis |
| `--api-key` | string | OpenRouter API key, optional, raises rate limits |
| `--min-input` | float | Min input price in USD per 1M tokens |
| `--max-input` | float | Max input price in USD per 1M tokens |
| `--min-output` | float | Min output price in USD per 1M tokens |
| `--max-output` | float | Max output price in USD per 1M tokens |
| `--max-output-price` | float | Alias for `--max-output` |
| `--min-total` | float | Min total price in USD per 1M tokens |
| `--max-total` | float | Max total price in USD per 1M tokens |
| `--max-price`, `-p` | float | Max input price, passed server side and applied client side |
| `--min-context`, `-c` | int | Min context length in tokens |
| `--max-context` | int | Max context length in tokens |
| `--provider` | string | Filter by provider prefix like anthropic, openai, google |
| `--search`, `-s` | string | Search query, server side for frontend API, client side for public API |
| `--order`, `-o` | enum | Frontend API server side order, most-popular, newest, cheapest, name |
| `--input-modality` | string | Filter by input modality, model must support all listed |
| `--output-modality` | string | Filter by output modality, model must support all listed |
| `--free-only` | flag | Only show free models |
| `--paid-only` | flag | Only show paid models |
| `--no-free` | flag | Exclude free tier models |
| `--reasoning`, `-r` | flag | Only show models with reasoning support |
| `--tps` | flag | Fetch live throughput stats |
| `--min-tps` | int | Minimum p50 throughput in tok/s, implies `--tps` |
| `--sort` | enum | Sort by input, output, total, context, name, provider, tps |
| `--desc` | flag | Sort descending |
| `--top` | int | Limit to top N results |
| `--limit`, `-n` | int | Alias for `--top` |
| `--detail`, `-d` | flag | Show detailed info per model |
| `--format` | enum | table, plain, json, jsonl, csv, tsv, works across all modes |
| `--json` | flag | Alias for `--format json` |
| `--show-providers` | flag | Expand each model into all upstream provider endpoints |
| `--prices` | flag | Benchmarks mode, join leaderboard entries with live catalog pricing |

</details>

<details>
<summary>Architecture</summary>

The project is a SDK and a CLI. You can use the CLI directly or import the SDK into your own codebase.

```python
from sdk import OpenRouterClient, fetch_models_v1, filter_models, sort_models

client = OpenRouterClient()
models = fetch_models_v1(client)
filtered = filter_models(models, min_context=128000, max_output=0.50)
sorted_models = sort_models(filtered, "input")
```

```
cli.py                              # entry point, 7 modes
sdk/
  __init__.py                       # re-exports everything for easy import
  analysis.py                       # cross source analyze, merges all 4 datasets
  core/
    __init__.py
    constants.py                    # endpoint URLs, headers, modality and order maps
    types.py                        # 13 dataclasses, 204+ typed fields
    client.py                       # curl_cffi client with Chrome TLS and multi host DNS pinning
  openrouter/
    __init__.py
    api.py                          # all OpenRouter API calls
    filters.py                      # filter and dedupe
    sorting.py                      # sort fields and sort logic
    relations.py                    # alias resolution and version family grouping
    compare.py                      # compare mode with by id fetch and attribute matrix
    output.py                       # plain tables, JSON, CSV, TSV, JSONL serializers
    rich.py                         # rich tables, optional, falls back to plain
    benchmarks.py                   # benchmark leaderboards and pricing enrichment
  sources/
    __init__.py
    artificialanalysis.py           # AA decryption, RSC parsing, typed models with benchmarks
    artificialanalysis_output.py    # AA renderers
    modelsdev.py                    # models.dev catalog with typed models and providers
    modelsdev_output.py             # models.dev renderers
```

The HTTP client uses `curl_cffi` with Chrome TLS impersonation and multi host DNS pinning so requests keep working even when libcurl DNS resolver fails.

</details>

<details>
<summary>API quirks</summary>

1. **`output_modalities` is ignored server side** on the find endpoint. It returns models of every modality. The tool filters client side.
2. **`max_price` is raw USD per 1M tokens** on the find endpoint. Also applied client side so results are correct either way.
3. **`active=true` is required** for clean find results.
4. **TPS is an aggregate best provider stat**, not your endpoint real speed. Treat TPS as a relative ranking signal and verify with a real call before committing to a model.
5. **TPS matching uses `permaslug`**. The find endpoint returns short aliases while the performance endpoint uses dated slugs. The `permaslug` field bridges them.
6. **Negative sentinel prices** from auto routing models are normalized to 0.0.
7. **Frontend endpoints can change**. They are internal and unauthenticated. If `embed` or `llm` mode fails, the tool falls back to the public API with client side modality filtering.
8. **Rows are deduplicated**. The find endpoint returns one row per provider endpoint. The tool collapses duplicates to the cheapest variant per model id after filtering. Use `--show-providers` to see the per provider breakdown.
9. **AA data is AES-GCM encrypted**. The manifest file is encrypted with a key embedded in the page RSC flight payload, then gzip compressed. The tool replicates the decryption logic from the site client side JS.
10. **Windows console is cp1252**. No unicode arrows or em dashes in output. Rich tables use ASCII safe cell values and `overflow="fold"` to avoid mojibake.
11. **Progress messages go to stderr**. `--json` stdout is clean and pipeable.

</details>

## Data sources

- [OpenRouter](https://openrouter.ai)
- [ArtificialAnalysis.ai](https://artificialanalysis.ai)
- [models.dev](https://models.dev)
