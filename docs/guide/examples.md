# Usage Examples

## Finding the cheapest capable model

```bash
# Cheapest models with 128k context that support reasoning
python cli.py --min-context 128000 --reasoning --sort input --top 10

# Cheapest vision models (image input, text output)
python cli.py --input-modality image --output-modality text --sort input --top 10
```

## Comparing candidates

```bash
# Compare three models with live TPS
python cli.py compare gpt-5 claude-sonnet-4 glm-5.3-flash --tps

# Compare with per provider breakdown
python cli.py compare gpt-5 glm-5.3-flash --show-providers

# Export comparison as CSV
python cli.py compare gpt-5 glm-5.3-flash --format csv > comparison.csv
```

## Cross-source analysis

```bash
# Get everything about one model from all four sources
python cli.py analyze glm-5.3-flash

# JSON output for programmatic use
python cli.py analyze gpt-5 --format json | python -m json.tool
```

## Benchmark leaderboards

```bash
# List all benchmark categories
python cli.py benchmarks

# Top coding models with live pricing
python cli.py benchmarks models-codecategories -n 10 --prices

# Search benchmarks for a specific model
python cli.py benchmarks intelligence -s deepseek --format csv
```

## ArtificialAnalysis.ai

```bash
# Leaderboard with benchmark indices
python cli.py aa -s glm -n 10

# Rich table with intelligence and agentic columns
python cli.py aa -s glm --format table

# CSV export
python cli.py aa -s claude --format csv > aa.csv
```

## Agent-friendly output

```bash
# JSON to stdout, progress to stderr
python cli.py llm -s deepseek --tps --sort tps --json | python -c "
import sys, json
models = json.load(sys.stdin)
for m in models:
    print(f\"{m['id']}: {m['tps_p50']} tok/s, \${m['input_price']}/M\")
"
```

## SDK integration

```python
from sdk import OpenRouterClient, fetch_models_v1, filter_models, sort_models
from sdk.sources.artificialanalysis import fetch_aa_full

client = OpenRouterClient()

# Get OpenRouter models
models = fetch_models_v1(client)
filtered = filter_models(models, min_context=128000, max_output=0.50)
sorted_models = sort_models(filtered, "input")

# Get ArtificialAnalysis benchmarks
aa_models = fetch_aa_full(client)
glm = next(m for m in aa_models if "glm-5-3" in (m.slug or ""))
print(f"Intelligence: {glm.benchmarks.intelligence_index}")
print(f"Agentic: {glm.benchmarks.agentic_index}")
```
