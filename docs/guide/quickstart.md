# Quick Start

## Install

```bash
pip install -r requirements.txt
```

Dependencies are `curl_cffi` and `rich`. The `rich` package is optional and the tool falls back to plain text tables automatically.

## CLI

The entry point is `cli.py` in the project root. There are 7 modes.

### List and filter

```bash
# Cheapest 20 models by input price
python cli.py --sort input --top 20

# Free models only
python cli.py --free-only

# Anthropic models under $1/M input
python cli.py --provider anthropic --max-input 1.0 --sort output

# 128k+ context under $0.50/M total
python cli.py --min-context 128000 --max-total 0.50 --sort total

# Models that accept text, image, and video input
python cli.py --input-modality text,image,video --sort input
```

### Embedding and LLM modes

```bash
# Embedding models under $0.03/1M
python cli.py embed --max-price 0.03 --sort input --limit 12

# LLM models sorted by live throughput
python cli.py llm --tps --sort tps -n 10

# Reasoning models with TPS, no free tier
python cli.py llm -r --no-free --tps
```

### Compare

```bash
python cli.py compare gpt-5 claude-sonnet-4 glm-5.3-flash
python cli.py compare gpt-5 glm-5.3-flash --tps
python cli.py compare gpt-5 glm-5.3-flash --show-providers
```

### Analyze

```bash
python cli.py analyze glm-5.3-flash
python cli.py analyze gpt-5 --format json
```

### ArtificialAnalysis.ai

```bash
python cli.py aa -s glm -n 10
python cli.py aa -s claude --format csv
```

### models.dev

```bash
python cli.py modelsdev -s deepseek-v4 -n 5
python cli.py modelsdev -s qwen --format jsonl
```

### Benchmarks

```bash
python cli.py benchmarks
python cli.py benchmarks models-codecategories -n 10 --prices
python cli.py benchmarks intelligence -s deepseek --format csv
```

### Output formats

All six formats work across every mode.

```bash
python cli.py llm -s claude --format csv
python cli.py compare gpt-5 glm-5.3-flash --format tsv
python cli.py aa -s glm --format json
```

## SDK

```python
from sdk import OpenRouterClient, fetch_models_v1, filter_models, sort_models

client = OpenRouterClient()
models = fetch_models_v1(client)
filtered = filter_models(models, min_context=128000, max_output=0.50)
sorted_models = sort_models(filtered, "input")
```
