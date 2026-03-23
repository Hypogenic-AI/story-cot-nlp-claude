# Story Chain-of-Thought (Story CoT)

Does prompting LLMs to reason in narrative/story form improve performance compared to standard step-by-step chain-of-thought?

## Key Findings

- **Story CoT matches standard CoT performance** across all four tested benchmarks (GSM8K, CommonsenseQA, StrategyQA, ARC-Challenge) using GPT-4.1
- **No statistically significant differences** between Story CoT and Few-shot CoT on any benchmark (all McNemar p > 0.47)
- **Both CoT formats dramatically outperform direct prompting** on math (+36pp) and science (+19pp), confirming the value of structured reasoning
- **CoT format doesn't matter at the frontier**: The null result supports the computational scaffolding hypothesis — CoT works by providing intermediate tokens for computation, not through specific reasoning formats
- **Story CoT produces qualitatively different but equally effective reasoning traces**, suggesting practical applications in explainable AI and educational tools

## Results Summary

| Dataset | Direct | Zero-shot CoT | Few-shot CoT | Story CoT |
|---------|--------|---------------|--------------|-----------|
| GSM8K (Math) | 59.3% | **96.0%** | 95.3% | 95.3% |
| CommonsenseQA | 88.7% | 74.7% | 88.0% | **89.3%** |
| StrategyQA | 84.7% | 86.0% | **89.3%** | 88.0% |
| ARC-Challenge | 75.3% | 72.0% | 93.3% | **94.7%** |

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install openai numpy pandas matplotlib seaborn scipy datasets pyarrow tqdm

# Run experiments (requires OPENAI_API_KEY)
python src/run_experiments.py
python src/run_strategyqa.py

# Analyze
python src/analyze_results.py
```

## File Structure

```
├── REPORT.md              # Full research report with results
├── README.md              # This file
├── planning.md            # Research plan and methodology
├── literature_review.md   # Comprehensive literature review
├── resources.md           # Catalog of gathered resources
├── src/
│   ├── prompts.py         # All prompt templates (Direct, CoT, Story CoT)
│   ├── run_experiments.py # Main experiment runner
│   ├── run_strategyqa.py  # StrategyQA-specific runner
│   └── analyze_results.py # Statistical analysis and plotting
├── results/
│   ├── all_results.json   # Raw results (2,400 predictions)
│   ├── config.json        # Experiment configuration
│   ├── analysis.json      # Statistical analysis output
│   ├── trace_examples.json# Qualitative reasoning examples
│   └── plots/             # Visualizations
├── datasets/              # Pre-downloaded evaluation datasets
├── papers/                # Reference papers (PDFs)
└── code/                  # Baseline code repositories
```

See [REPORT.md](REPORT.md) for full methodology, analysis, and discussion.
