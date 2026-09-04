# modelrank

A CLI tool and SDK to list, filter, sort, compare, and analyze LLM models across OpenRouter, ArtificialAnalysis.ai, and models.dev.

All from the terminal. No API key required.

## Why

Too many times you need a model for a task. Cheapest, smartest, fastest, longest context, vision capable, whatever the constraint is. The process is always the same friction. Open a browser, tab through catalogs, check benchmarks, cross reference pricing, and eventually just pick something by gut feeling.

That is inefficient. The data is all available through APIs. This tool exists so that instead of browsing or guessing, you just run a command. An agent can pipe `--json` output directly into its next decision. A human can scan a sorted table in two seconds.

Let the model run the commands. That is the whole point.

## Install

```bash
pip install -r requirements.txt
```

`rich` is optional. The tool falls back to plain text tables automatically.

## Quick Start

```bash
# Cheapest 20 models by input price
python cli.py --sort input --top 20

# Free models only
python cli.py --free-only

# Compare models side by side
python cli.py compare gpt-5 claude-sonnet-4 glm-5.3-flash

# Analyze a model across all data sources
python cli.py analyze glm-5.3-flash

# ArtificialAnalysis.ai leaderboard with benchmarks
python cli.py aa -s glm -n 10
```

## SDK Usage

```python
from sdk import OpenRouterClient, fetch_models_v1, filter_models, sort_models

client = OpenRouterClient()
models = fetch_models_v1(client)
filtered = filter_models(models, min_context=128000, max_output=0.50)
sorted_models = sort_models(filtered, "input")
```

## Data Sources

- [OpenRouter](https://openrouter.ai)
- [ArtificialAnalysis.ai](https://artificialanalysis.ai)
- [models.dev](https://models.dev)
