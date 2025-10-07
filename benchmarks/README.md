# DevAgent Benchmarks

Benchmark suite for evaluating LLM provider performance on DevAgent tasks.

## Quick Start

```bash
# Run all tests on all providers
python3 benchmarks/run_benchmarks.py

# Run on specific provider (fuzzy match)
python3 benchmarks/run_benchmarks.py --provider grok
python3 benchmarks/run_benchmarks.py --provider "claude sonnet"

# Run limited tests
python3 benchmarks/run_benchmarks.py --limit 3 --tests 5
```

## Files

- `run_benchmarks.py` - Main benchmark runner
- `benchmark_runner.py` - Core execution engine
- `test_cases.py` - 18 test case definitions
- `providers_config.yaml` - 18 provider configurations
- `results/` - Output files (JSON, CSV, Markdown)

## Results

Results saved in three formats:
- `results/*_results.json` - Machine-readable
- `results/*_results.csv` - Spreadsheet
- `results/*_summary.md` - Human-readable
