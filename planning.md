# Research Plan: Story Chain-of-Thought (Story CoT)

## Motivation & Novelty Assessment

### Why This Research Matters
Current chain-of-thought (CoT) methods use step-by-step logical decomposition, but humans often reason through narratives—recounting scenarios, simulating situations, and predicting outcomes through story-like mental simulations. If narrative-form reasoning taps into different (and potentially stronger) model capabilities than standard CoT, this could unlock improved performance and reveal fundamental insights about how LLMs reason.

### Gap in Existing Work
The Narrative-of-Thought (NOT) paper (Zhang et al., 2024) demonstrated that story-form CoT dramatically outperforms standard CoT for temporal reasoning (16-71% F1 improvement). However: (1) NOT was only tested on temporal graph generation—not on standard reasoning benchmarks like math, commonsense, or science; (2) No study systematically compares story-form CoT against standard CoT across multiple reasoning domains; (3) Schaeffer et al. (2023) showed invalid logic in CoT still helps, suggesting structure matters more than logical validity—but no one has tested whether narrative structure specifically contributes to this effect.

### Our Novel Contribution
We conduct the first systematic comparison of story-form CoT vs. standard CoT across four diverse reasoning benchmarks (math, commonsense, multi-hop, science) using state-of-the-art LLMs. We test whether narrative framing improves, maintains, or degrades performance compared to standard CoT, and analyze when and why narrative reasoning helps.

### Experiment Justification
- **Experiment 1 (Multi-benchmark comparison)**: Tests whether story CoT generalizes beyond temporal reasoning to math, commonsense, multi-hop, and science tasks. This is the core gap in the literature.
- **Experiment 2 (Reasoning trace analysis)**: Analyzes qualitative differences between standard and story CoT traces to understand *why* any performance differences occur.
- **Experiment 3 (Story prompt variants)**: Tests different narrative elicitation strategies to find the most effective way to induce story-form reasoning.

## Research Question
Does prompting LLMs to reason in narrative/story form (Story CoT) improve, match, or degrade task performance compared to standard step-by-step CoT, and does the effect vary by reasoning domain?

## Hypothesis Decomposition
1. **H1**: Story CoT improves performance on tasks with inherent causal/temporal structure (commonsense, multi-hop) compared to standard CoT.
2. **H2**: Story CoT degrades or matches performance on tasks requiring precise logical/mathematical reasoning (GSM8K, ARC).
3. **H3**: Story CoT produces more diverse reasoning paths than standard CoT, potentially benefiting self-consistency voting.
4. **H4**: The effectiveness of Story CoT correlates with the narrative naturalness of the task domain.

## Proposed Methodology

### Approach
We use the OpenAI API (GPT-4.1) to compare four prompting strategies across four benchmarks, evaluating accuracy as the primary metric. We sample 150 questions per dataset for cost-efficiency while maintaining statistical power.

### Experimental Steps
1. Load and sample datasets (GSM8K, CommonsenseQA, StrategyQA, ARC-Challenge)
2. Design prompts: Direct, Zero-shot CoT, Few-shot CoT (3-shot), Story CoT (3-shot with narrative exemplars)
3. Run all conditions through GPT-4.1 API
4. Parse answers and compute accuracy
5. Statistical comparison (McNemar's test for paired comparisons)
6. Qualitative analysis of reasoning traces

### Baselines
1. **Direct prompting**: No reasoning, answer directly
2. **Zero-shot CoT**: "Let's think step by step"
3. **Few-shot CoT**: 3 exemplars with standard step-by-step reasoning
4. **Story CoT**: 3 exemplars with narrative-form reasoning ("Let me think through this as a story...")

### Evaluation Metrics
- **Primary**: Accuracy (exact match for math, letter match for multiple choice, boolean for yes/no)
- **Secondary**: Response length, reasoning diversity (for self-consistency analysis)

### Statistical Analysis Plan
- McNemar's test for pairwise condition comparisons (paired binary outcomes)
- 95% confidence intervals via bootstrap
- Effect sizes reported as accuracy differences with CIs
- Bonferroni correction for multiple comparisons

## Expected Outcomes
- Story CoT ≥ standard CoT on commonsense and multi-hop tasks (StrategyQA, CommonsenseQA)
- Story CoT ≤ standard CoT on math (GSM8K) due to imprecise narrative reasoning
- Mixed results on science (ARC) depending on question type

## Timeline and Milestones
1. Environment & data setup: 10 min
2. Prompt design & implementation: 30 min
3. API experiments (4 datasets × 4 conditions × 150 samples): 60-90 min
4. Analysis & visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
- API rate limits → use exponential backoff with retries
- Answer parsing ambiguity → use robust regex + fallback parsing
- Cost management → 150 samples × 4 conditions × 4 datasets = 2,400 API calls, est. $20-40

## Success Criteria
- Complete results for all 4 conditions on all 4 datasets
- Statistical tests with p-values and confidence intervals
- Clear answer to whether Story CoT helps, hurts, or is neutral vs. standard CoT
- Qualitative insights into when narrative reasoning is beneficial
