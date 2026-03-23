# Resources Catalog

## Summary
This document catalogs all resources gathered for the Story CoT research project. The project investigates whether training/prompting language models to use chain-of-thought reasoning in story/narrative form improves performance compared to standard CoT approaches.

## Papers
Total papers downloaded: 18

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Narrative-of-Thought (NOT) | Zhang, Beauchamp, Wang | 2024 | papers/narrative_of_thought_temporal_reasoning.pdf | **Core paper.** Story-form CoT for temporal reasoning |
| Chain-of-Thought Prompting | Wei et al. | 2022 | papers/wei2022_chain_of_thought_prompting.pdf | Foundational CoT paper |
| Self-Consistency | Wang et al. | 2022 | papers/wang2022_self_consistency_cot.pdf | Majority voting over CoT paths |
| Auto-CoT | Zhang et al. | 2022 | papers/zhang2022_automatic_cot.pdf | Automated CoT demonstration generation |
| Tree of Thoughts | Yao et al. | 2023 | papers/yao2023_tree_of_thoughts.pdf | Tree-structured reasoning |
| Graph-of-Thought | Yao et al. | 2023 | papers/yao2023_graph_of_thought.pdf | Graph-structured reasoning |
| Buffer of Thoughts | Yang et al. | 2024 | papers/yang2024_buffer_of_thoughts.pdf | Template-based reasoning |
| CoT Without Prompting | Wang & Zhou | 2024 | papers/wang2024_cot_without_prompting.pdf | Emergent CoT via decoding |
| Contrastive CoT | Chia et al. | 2023 | papers/chia2023_contrastive_cot.pdf | Positive+negative exemplars |
| Chain of Code | Li et al. | 2023 | papers/li2023_chain_of_code.pdf | Code as reasoning medium |
| Implicit CoT Distillation | Deng et al. | 2023 | papers/deng2023_implicit_cot_distillation.pdf | Internalized CoT |
| Diffusion of Thoughts | Ye et al. | 2024 | papers/ye2024_diffusion_of_thoughts.pdf | CoT in diffusion LMs |
| CoT Not Faithful | Arcuschin et al. | 2025 | papers/arcuschin2025_cot_not_faithful.pdf | CoT faithfulness analysis |
| Invalid Logic Gains | Schaeffer et al. | 2023 | papers/schaeffer2023_invalid_logic_reasoning.pdf | Invalid CoT still helps |
| Long CoT Survey | Chen et al. | 2025 | papers/chen2025_long_cot_survey.pdf | Comprehensive CoT survey |
| CoT Strategies Survey | Yu et al. | 2023 | papers/yu2023_cot_strategies_survey.pdf | CoT taxonomy |
| Plan-Guided Narrative Summarization | Grenander et al. | 2025 | papers/grenander2025_plan_guided_summarization_narrative.pdf | Narrative processing |
| SEED-Story | Yang et al. | 2024 | papers/seed_story_multimodal_long_story.pdf | Story generation with LLMs |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 7

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| ProScript | AI2 | 6,414 scenarios | Temporal graph generation | datasets/proscript/ | Primary evaluation dataset from NOT |
| Schema-11 | NoT repo | 11 scenarios | Temporal graph generation | datasets/schema11/ | News domain temporal graphs |
| GSM8K | HuggingFace | 8,792 examples | Math word problems | datasets/gsm8k/ | Standard CoT benchmark |
| StrategyQA | HuggingFace | 2,290 examples | Multi-hop yes/no QA | datasets/strategyqa/ | Implicit reasoning required |
| CommonsenseQA | HuggingFace | 12,102 examples | 5-way commonsense QA | datasets/commonsenseqa/ | Commonsense reasoning |
| ARC-Challenge | HuggingFace | 2,590 examples | Science QA | datasets/arc_challenge/ | Challenging science questions |
| BIG-bench Causal | HuggingFace | 190 examples | Causal judgment | datasets/bigbench_causal_judgment/ | Causal reasoning scenarios |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| NoT | github.com/launchnlp/NoT | Narrative-of-Thought implementation | code/NoT/ | Core implementation, includes data processing |
| zero_shot_cot | github.com/kojima-takeshi188/zero_shot_cot | Zero-shot CoT baseline | code/zero_shot_cot/ | Standard CoT baseline implementation |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with queries: "chain of thought reasoning story narrative language models", "chain of thought prompting reasoning", "story narrative reasoning natural language processing"
2. Searched arXiv API for targeted papers on CoT variants and narrative reasoning
3. Identified the key "Narrative-of-Thought" paper (Zhang et al., 2024) as the closest existing work to Story CoT
4. Downloaded datasets used in NOT paper (ProScript, Schema-11) plus standard CoT benchmarks

### Selection Criteria
- **Papers**: Prioritized (1) papers directly combining narrative/story with CoT, (2) foundational CoT papers, (3) CoT variants/alternatives, (4) CoT analysis/faithfulness studies
- **Datasets**: Selected for coverage across reasoning types: temporal (ProScript, Schema-11), mathematical (GSM8K), commonsense (CommonsenseQA, StrategyQA, ARC), and causal (BIG-bench)
- **Code**: Focused on implementations directly relevant to reproducing or extending Story CoT experiments

### Key Finding
The **Narrative-of-Thought (NOT)** paper (Zhang et al., 2024) is essentially a proof-of-concept for Story CoT applied to temporal reasoning. It demonstrates that:
- Standard CoT fails on temporal reasoning tasks
- Narrative-form reasoning significantly outperforms standard CoT (16-71% F1 improvement)
- Small models with NOT can match GPT-3.5 performance
- Quality narratives require conciseness, simplicity, and factuality

### Gaps
1. NOT only evaluates temporal reasoning - Story CoT should be tested on broader tasks
2. No existing work compares narrative CoT vs. standard CoT across multiple reasoning domains
3. No training-based Story CoT (only prompting-based NOT exists)
4. Limited analysis of when narrative reasoning helps vs. hurts

## Recommendations for Experiment Design

Based on gathered resources:

1. **Primary dataset(s)**:
   - **ProScript** for temporal reasoning (direct comparison with NOT)
   - **GSM8K** for math reasoning (test if story CoT helps/hurts vs. standard CoT)
   - **StrategyQA** for multi-hop reasoning (stories may help connect facts)
   - **CommonsenseQA** for commonsense (narratives are natural for commonsense)

2. **Baseline methods**:
   - Standard prompting (no CoT)
   - Zero-shot CoT ("Let's think step by step")
   - Few-shot CoT (standard step-by-step exemplars)
   - NOT (for temporal tasks, using code from NoT repo)
   - Story CoT variants: "Let's think about this as a story", few-shot with narrative exemplars

3. **Evaluation metrics**:
   - Accuracy (QA tasks)
   - F1 / GED (temporal graph tasks)
   - Self-faithfulness (narrative-answer alignment)
   - Qualitative analysis of generated narratives

4. **Code to adapt/reuse**:
   - **NoT repo**: Adapt prompting templates for Story CoT across different tasks
   - **zero_shot_cot**: Baseline implementation
   - Both repos provide evaluation pipelines that can be extended

5. **Experimental design suggestions**:
   - Compare Story CoT prompts: "Tell a story about...", "Imagine a scenario where...", "Once upon a time...", "Here's what happened..."
   - Test whether story-form CoT works better for tasks with causal/temporal structure vs. pure logic
   - Analyze whether models capable of better story generation also produce better Story CoT
   - Test hypothesis: "Does CoT work because of logical reasoning or structural/narrative coherence?" by using coherent but logically irrelevant narratives
