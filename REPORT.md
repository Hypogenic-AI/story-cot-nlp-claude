# Story Chain-of-Thought: Does Narrative Reasoning Improve LLM Performance?

## 1. Executive Summary

We conducted the first systematic comparison of story-form chain-of-thought (Story CoT) reasoning against standard CoT approaches across four diverse reasoning benchmarks using GPT-4.1. Our key finding is that **Story CoT performs comparably to standard few-shot CoT across all tested domains** — it neither significantly improves nor degrades performance. Across GSM8K (math), CommonsenseQA, StrategyQA (multi-hop), and ARC-Challenge (science), the accuracy differences between Story CoT and standard CoT were 0-1.3 percentage points and none reached statistical significance (all p > 0.47). This suggests that for state-of-the-art LLMs, the **structure and presence of intermediate reasoning matters more than the specific format** (step-by-step vs. narrative), supporting the hypothesis that CoT benefits arise from computational scaffolding rather than logical form.

## 2. Goal

**Hypothesis**: Prompting LLMs to reason in narrative/story form (Story CoT) — similar to how humans recount memories or predict outcomes through mental simulation — will improve performance on certain tasks compared to standard step-by-step CoT, particularly on tasks with inherent causal or temporal structure.

**Why this matters**: Current CoT methods use formulaic step-by-step decomposition, but cognitive science suggests humans often reason through narrative mental simulation. If narrative reasoning taps into different (and potentially stronger) model capabilities than standard CoT, this could improve performance and reveal fundamental insights about how LLMs process reasoning.

**Gap in existing work**: The Narrative-of-Thought (NOT) paper (Zhang et al., 2024) showed dramatic improvements with narrative reasoning for temporal graph generation, but no study has systematically compared story-form CoT against standard CoT across diverse reasoning domains (math, commonsense, multi-hop, science).

## 3. Data Construction

### Dataset Description

| Dataset | Source | Task Type | N (sampled) | Split |
|---------|--------|-----------|-------------|-------|
| GSM8K | OpenAI | Math word problems | 150 | Test |
| CommonsenseQA | Allen AI | 5-way commonsense QA | 150 | Validation |
| StrategyQA | BIU NLP | Yes/No multi-hop QA | 150 | Train |
| ARC-Challenge | Allen AI | 4-way science QA | 150 | Test |

### Example Samples

**GSM8K**: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" → **72**

**CommonsenseQA**: "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? (A) ignore (B) enforce (C) authoritarian (D) yell at (E) avoid" → **A**

**StrategyQA**: "Is Mixed martial arts totally original from Roman Colosseum games?" → **No**

**ARC-Challenge**: "Which property of a mineral can be determined just by looking at it? (A) luster (B) mass (C) weight (D) hardness" → **A**

### Preprocessing Steps
1. Loaded datasets from pre-downloaded HuggingFace arrow files
2. Randomly sampled 150 questions per dataset (seed=42)
3. Extracted gold answers: numeric for GSM8K, letter for CommonsenseQA/ARC, boolean→Yes/No for StrategyQA

### Data Quality
- All 600 samples had valid questions and gold answers
- No missing values or duplicates in sampled sets
- Balanced label distributions checked for multiple-choice datasets

## 4. Experiment Description

### Methodology

#### High-Level Approach
We compare four prompting strategies on the same 150 questions per dataset, all using GPT-4.1 at temperature=0 for deterministic outputs:

1. **Direct**: Answer immediately with no reasoning
2. **Zero-shot CoT**: "Let's think step by step" (no exemplars)
3. **Few-shot CoT**: 3 exemplars with standard step-by-step reasoning
4. **Story CoT**: 3 exemplars with narrative-form reasoning

#### Why This Method?
- **Prompt-based comparison** is the standard in CoT literature (Wei et al., 2022; Zhang et al., 2024)
- **Same model, same questions** enables paired statistical testing (McNemar's test)
- **Multiple benchmarks** test whether effects are domain-specific
- **GPT-4.1** is a state-of-the-art model, ensuring results are relevant to current practice

### Implementation Details

#### Tools and Libraries
- OpenAI API: GPT-4.1 (2025)
- Python 3.12.8
- NumPy 2.3.0, Pandas, SciPy, Matplotlib, Seaborn
- HuggingFace Datasets (for data loading)

#### Prompting Design

**Few-shot CoT exemplar** (GSM8K):
> Step 1: We start with 15 trees.
> Step 2: After planting, there are 21 trees.
> Step 3: The number planted is 21 - 15 = 6.

**Story CoT exemplar** (same question):
> Imagine a grove at dawn with 15 trees standing tall. A team of workers arrives with saplings on a truck. They spend the morning digging holes and carefully planting new trees among the existing ones. By sunset, someone walks through and counts every tree — there are now 21 in total. The original 15 haven't changed, so the workers must have added 21 minus 15, which is 6 new trees during their day's work.

Story CoT exemplars embed the same mathematical/logical steps within a vivid narrative, using characters, settings, and temporal progression.

#### Hyperparameters

| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| Model | GPT-4.1 | Latest available |
| Temperature | 0.0 | Deterministic |
| Max tokens | 1024 | Sufficient for all tasks |
| Few-shot examples | 3 | Standard in CoT literature |
| Samples per dataset | 150 | Power analysis (~80% power for 10% diff) |
| Random seed | 42 | Convention |

### Experimental Protocol

#### Reproducibility Information
- Single run per condition (temperature=0 is deterministic)
- Random seed: 42 (for data sampling)
- Hardware: 4× NVIDIA RTX A6000 (48GB) — GPUs not used (API-based experiment)
- Execution time: ~6 minutes total for all 2,400 API calls
- 10 parallel workers for API calls

#### Evaluation Metrics
- **Accuracy**: Primary metric. Exact match of extracted answer vs. gold answer
- **Response length**: Secondary. Measures verbosity of reasoning traces (characters)
- **McNemar's test**: For pairwise strategy comparison (paired binary outcomes on same questions)
- **Bootstrap 95% CIs**: Non-parametric confidence intervals (10,000 resamples)

### Raw Results

#### Main Results Table

| Dataset | Direct | Zero-shot CoT | Few-shot CoT | Story CoT |
|---------|--------|---------------|--------------|-----------|
| **GSM8K** (Math) | 59.3% [51.3, 67.3] | **96.0%** [92.7, 98.7] | 95.3% [92.0, 98.7] | 95.3% [92.0, 98.7] |
| **CommonsenseQA** | **88.7%** [83.3, 93.3] | 74.7% [67.3, 81.3] | 88.0% [82.7, 93.3] | **89.3%** [84.0, 94.0] |
| **StrategyQA** (Multi-hop) | 84.7% [78.7, 90.0] | 86.0% [80.0, 91.3] | **89.3%** [84.0, 94.0] | 88.0% [82.7, 92.7] |
| **ARC-Challenge** (Science) | 75.3% [68.7, 82.0] | 72.0% [64.7, 79.3] | 93.3% [89.3, 97.3] | **94.7%** [90.7, 98.0] |

Values are accuracy (%) with 95% bootstrap confidence intervals.

#### Statistical Tests: Story CoT vs. Few-shot CoT (McNemar's test)

| Dataset | Story wins | CoT wins | χ² | p-value | Significance |
|---------|-----------|----------|-----|---------|-------------|
| GSM8K | 3 | 3 | 0.17 | 0.683 | n.s. |
| CommonsenseQA | 6 | 4 | 0.10 | 0.752 | n.s. |
| StrategyQA | 5 | 7 | 0.08 | 0.773 | n.s. |
| ARC-Challenge | 2 | 0 | 0.50 | 0.480 | n.s. |

**None of the comparisons reach statistical significance.** Story CoT and Few-shot CoT produce virtually identical results.

#### Average Response Length (characters)

| Dataset | Direct | Zero-shot CoT | Few-shot CoT | Story CoT |
|---------|--------|---------------|--------------|-----------|
| GSM8K | 8 | 633 | 407 | 679 |
| CommonsenseQA | 1 | 837 | 568 | 488 |
| StrategyQA | 2 | 602 | 457 | 581 |
| ARC-Challenge | 35 | 1,065 | 553 | 572 |

Story CoT responses are similar in length to few-shot CoT, except on GSM8K where narrative framing produces longer responses.

#### Visualizations

Results are saved as:
- `results/plots/accuracy_comparison.png` — Grouped bar chart with 95% CIs
- `results/plots/accuracy_heatmap.png` — Accuracy heatmap across all conditions
- `results/plots/response_lengths.png` — Box plots of response lengths
- `results/plots/story_vs_cot_scatter.png` — Scatter plot comparing Story CoT vs Few-shot CoT

#### Token Usage and Cost
- Total tokens: 443,836 input + 218,968 output = 662,804
- Estimated cost: ~$2.64

## 5. Result Analysis

### Key Findings

1. **Story CoT matches Few-shot CoT performance across all domains.** The largest observed difference was 1.3 percentage points (CommonsenseQA: Story 89.3% vs. CoT 88.0%; ARC: Story 94.7% vs. CoT 93.3%), and none reached statistical significance.

2. **Both Few-shot CoT and Story CoT dramatically outperform Direct prompting** on tasks requiring multi-step reasoning (GSM8K: +36pp; ARC: +18-19pp), confirming the value of structured reasoning.

3. **Zero-shot CoT underperforms Few-shot methods on 3 of 4 datasets.** Notably, Zero-shot CoT *degrades* performance on CommonsenseQA (74.7% vs. 88.7% direct), consistent with literature showing CoT can hurt on certain task types.

4. **Story CoT produces qualitatively different reasoning** that embeds identical logical steps within narrative context (characters, settings, temporal progression), yet achieves the same accuracy.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| **H1**: Story CoT improves on causal/commonsense tasks | **Partially supported** | Story CoT achieves highest accuracy on CommonsenseQA (89.3%) and ARC (94.7%), but differences are not statistically significant |
| **H2**: Story CoT degrades on math tasks | **Refuted** | Story CoT matches Few-shot CoT exactly on GSM8K (95.3% each) |
| **H3**: Story CoT produces more diverse reasoning | **Supported qualitatively** | Story traces use different vocabulary and framing while reaching same conclusions |
| **H4**: Effectiveness correlates with task narrative naturalness | **Weakly supported** | Largest (non-significant) gains on commonsense/science; neutral on math |

### Qualitative Analysis of Reasoning Traces

**Where Story CoT Wins (CommonsenseQA example):**
- Q: "How might releasing energy that has built up feel?"
- Story CoT correctly identified "cathartic" by narrating a stressed person exercising and feeling relief
- Standard CoT chose a less apt answer through abstract step-by-step analysis

**Where Story CoT Wins (ARC example):**
- Q: "If HCl is added to Zn, what would be an expected product?"
- Story CoT narrated a chemistry lab scene, vividly describing bubbles forming (hydrogen gas)
- Standard CoT reasoned abstractly but selected the wrong product

**Key pattern**: Story CoT wins tend to occur on questions where situational/experiential knowledge helps — the narrative frames the problem in a way that activates relevant world knowledge.

### Surprises and Insights

1. **Zero-shot CoT can be harmful.** On CommonsenseQA, "Let's think step by step" reduced accuracy by 14 percentage points compared to direct answering. This aligns with literature (Schaeffer et al., 2023) showing that CoT structure matters more than content.

2. **Story CoT doesn't hurt math.** We hypothesized narrative framing would impair precise calculation, but GPT-4.1 successfully embeds calculations within stories (e.g., "the cashier rings up 5 times $3, which comes to $15").

3. **Format equivalence at the frontier.** The null result is itself informative: for GPT-4.1, the benefit of CoT comes from having *any* structured intermediate reasoning, not from the specific format. This supports the computational scaffolding hypothesis.

### Error Analysis

- **GSM8K errors** (both methods): Complex multi-step problems with large numbers, where arithmetic errors propagate. Both CoT formats fail similarly.
- **CommonsenseQA errors** (both methods): Ambiguous questions where multiple answers seem plausible. Neither format provides an advantage.
- **ARC errors**: Rare for both methods. The 2 cases where Story CoT won involved chemistry/physics scenarios where narrative context helped.

### Limitations

1. **Single model tested.** Results may differ for smaller models where CoT is an emergent capability, or for models with different training data distributions.
2. **English-only, single run.** Temperature=0 means no variance across runs, but also no measure of model uncertainty.
3. **Sample size.** 150 samples per dataset provides ~80% power to detect 10% accuracy differences, but not smaller effects.
4. **Prompt engineering scope.** Only one Story CoT prompt design was tested. Alternative narrative styles (first-person, mystery format, dialogue) might yield different results.
5. **Task scope.** The NOT paper showed Story CoT excels at temporal graph generation — a structured output task we did not test. Story CoT's benefits may be task-type specific rather than domain-specific.
6. **No training-time intervention.** We only tested prompting. Training models to produce story-form CoT (as the original hypothesis suggests) could yield different results.

## 6. Conclusions

### Summary
Story CoT performs **equivalently** to standard few-shot CoT across math, commonsense, multi-hop, and science reasoning benchmarks using GPT-4.1. The narrative framing neither helps nor hurts, suggesting that for state-of-the-art LLMs, the benefit of chain-of-thought reasoning comes from structured intermediate computation rather than the specific reasoning format.

### Implications

**Practical**: Practitioners can use narrative-form reasoning prompts without performance penalty. This may be valuable for applications where natural-sounding reasoning is preferred (explainable AI, educational tools, conversational agents).

**Theoretical**: The format-independence of CoT benefits supports the *computational scaffolding hypothesis* — CoT works by providing intermediate token positions for computation, not by simulating human-like logical reasoning. This aligns with Schaeffer et al. (2023)'s finding that even invalid logic in CoT maintains performance.

### Confidence in Findings
- **High confidence** that Story CoT does not significantly differ from Few-shot CoT on these benchmarks with GPT-4.1
- **Moderate confidence** in generalization to other state-of-the-art models
- **Lower confidence** about smaller models, where format effects may be larger

## 7. Next Steps

### Immediate Follow-ups
1. **Test on temporal/causal tasks**: Replicate the NOT paper's temporal graph generation to confirm Story CoT's domain-specific advantages
2. **Test on smaller models**: GPT-3.5-class models, Llama-8B, etc. where CoT format effects may emerge
3. **Narrative style variants**: Test first-person narratives, dialogue, mystery-solving, etc.

### Alternative Approaches
- **Fine-tuning with story-form CoT**: Train models to produce narrative reasoning (the original hypothesis)
- **Self-consistency with Story CoT**: Test if narrative diversity improves majority voting
- **Hybrid approaches**: Standard CoT for math, Story CoT for commonsense, auto-selected by task type

### Open Questions
1. Why does narrative reasoning dramatically help temporal graph generation (Zhang et al., 2024) but not standard QA? Is it about structured output vs. answer extraction?
2. At what model size threshold does CoT format start to matter?
3. Can narrative structure improve CoT *faithfulness* (the reasoning-answer alignment issue identified by Arcuschin et al., 2025)?

## References

1. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. NeurIPS 2022.
2. Wang, X., et al. (2022). Self-consistency improves chain of thought reasoning in language models. ICLR 2023.
3. Zhang, X.F., Beauchamp, N., Wang, L. (2024). Narrative-of-Thought: Improving temporal reasoning via recounted narratives. Findings of EMNLP 2024.
4. Yao, S., et al. (2023). Tree of thoughts: Deliberate problem solving with large language models. NeurIPS 2023.
5. Schaeffer, R., Pistunova, K., Khanna, S. (2023). Invalid logic, equivalent gains: The bizarreness of reasoning in language model prompting.
6. Arcuschin, I., et al. (2025). Chain-of-thought reasoning in the wild is not always faithful.
7. Chen, Q., et al. (2025). Towards reasoning era: A survey of long chain-of-thought for reasoning LLMs.
