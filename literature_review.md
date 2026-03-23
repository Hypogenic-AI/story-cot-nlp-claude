# Literature Review: Story Chain-of-Thought (Story CoT)

## Research Area Overview

This literature review examines the intersection of chain-of-thought (CoT) reasoning and narrative/story-based reasoning in large language models (LLMs). The core hypothesis is that training or prompting LLMs to use story-form CoT reasoning—similar to how humans recount memories or narrate experiences—may improve performance on certain reasoning tasks compared to standard CoT approaches.

The review covers: (1) foundational CoT work, (2) CoT variants and structural alternatives, (3) narrative and story-based reasoning, and (4) analysis of when and why CoT works or fails.

---

## Key Papers

### Paper 1: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Authors**: Jason Wei, Xuezhi Wang, Dale Schuurmans, et al.
- **Year**: 2022
- **Source**: NeurIPS 2022 (arXiv: 2201.11903)
- **Key Contribution**: Introduced chain-of-thought (CoT) prompting—providing intermediate reasoning steps as exemplars—that significantly improves LLM performance on complex reasoning tasks.
- **Methodology**: Few-shot prompting with manually written CoT exemplars. Tested on arithmetic (GSM8K), commonsense (CommonsenseQA, StrategyQA), and symbolic reasoning tasks.
- **Datasets Used**: GSM8K, SVAMP, ASDiv, AQuA, CommonsenseQA, StrategyQA, letter concatenation, coin flip
- **Results**: 540B PaLM with CoT achieves SOTA on GSM8K (56.9%), surpassing finetuned GPT-3. CoT is an emergent ability—effective only in models ≥100B parameters.
- **Code Available**: No official repo
- **Relevance to Story CoT**: This is the foundational work. Standard CoT uses step-by-step logical decomposition. Story CoT proposes an alternative: narrative-form reasoning that may tap into different model capabilities. The question is whether narrative structure adds something beyond linear step decomposition.

### Paper 2: Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **Authors**: Xuezhi Wang, Jason Wei, Dale Schuurmans, et al.
- **Year**: 2022
- **Source**: ICLR 2023 (arXiv: 2203.11171)
- **Key Contribution**: Proposed self-consistency—sampling multiple reasoning paths and selecting the most consistent answer via majority voting.
- **Methodology**: Sample diverse CoT paths (temperature > 0), then marginalize over paths to find the most consistent answer.
- **Datasets Used**: GSM8K, SVAMP, AQuA, StrategyQA, ARC
- **Results**: Significant improvements over greedy CoT (e.g., GSM8K: 56.5% → 74.4% with PaLM-540B). Works across different models and tasks.
- **Relevance to Story CoT**: Self-consistency could be applied to Story CoT—generating multiple narrative reasoning paths and selecting the most consistent. Different narrative styles might produce more diverse reasoning paths than standard CoT.

### Paper 3: NARRATIVE-OF-THOUGHT: Improving Temporal Reasoning via Recounted Narratives
- **Authors**: Xinliang Frederick Zhang, Nick Beauchamp, Lu Wang
- **Year**: 2024
- **Source**: Findings of EMNLP 2024 (arXiv: 2410.05558)
- **Key Contribution**: **Most directly relevant paper.** Proposed NARRATIVE-OF-THOUGHT (NOT)—a prompting technique where LLMs first generate a temporally grounded narrative, then use it to produce a temporal graph. This is essentially "Story CoT" applied to temporal reasoning.
- **Methodology**:
  1. Convert events to Python class structure
  2. Prompt LLM to generate a narrative linking events in temporal order
  3. Use narrative to guide temporal graph generation
  4. Use narrative-aware few-shot demonstrations with reference narratives
- **Datasets Used**: ProScript (2,077 scenarios, daily activities), Schema-11 (11 news scenarios), WikiHow Script (2,991 how-to scenarios)
- **Results**:
  - NOT boosts F1 by 16-71% over standard prompting across all base models
  - NOT-augmented LLaMA3-8B achieves F1 comparable to GPT-3.5 (42.2 vs. 45.7)
  - Outperforms GPT-3.5/4 on structural similarity (GED)
  - Standard CoT actually *degrades* performance on temporal reasoning
  - Key narrative qualities: **conciseness, simplicity, factuality**
  - 72.8% self-faithfulness between narratives and temporal graphs
- **Code Available**: Yes, https://github.com/launchnlp/NoT
- **Relevance to Story CoT**: This IS Story CoT for temporal reasoning. Shows that narrative-form reasoning outperforms standard CoT specifically for temporal tasks. Critical finding: CoT fails on temporal reasoning, but narrative-form reasoning succeeds. The hypothesis that story-form CoT may reveal different model capabilities is directly supported.

### Paper 4: Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- **Authors**: Shunyu Yao, Dian Yu, Jeffrey Zhao, et al.
- **Year**: 2023
- **Source**: NeurIPS 2023 (arXiv: 2305.10601)
- **Key Contribution**: Extended CoT from linear chains to tree structures, enabling exploration and backtracking in reasoning.
- **Methodology**: LLM generates and evaluates intermediate "thoughts," explores multiple paths via BFS/DFS with self-evaluation.
- **Datasets Used**: Game of 24, Creative Writing, Mini Crosswords
- **Results**: Significantly outperforms CoT on tasks requiring exploration (e.g., Game of 24: 4% → 74%).
- **Relevance to Story CoT**: ToT represents structural exploration of reasoning space. Story CoT could be seen as an alternative to tree exploration—using narrative structure to guide reasoning in a more human-like way rather than systematic search.

### Paper 5: Chain-of-Thought Reasoning in the Wild Is Not Always Faithful
- **Authors**: Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, et al.
- **Year**: 2025
- **Source**: arXiv: 2503.08679
- **Key Contribution**: Demonstrated that CoT reasoning in LLMs is not always faithful—the stated reasoning may not reflect the actual computation driving the answer.
- **Relevance to Story CoT**: This raises a critical question for Story CoT: are narrative-form CoTs more or less faithful than standard CoTs? The NOT paper found 72.8% self-faithfulness. If standard CoT can be unfaithful, perhaps story-form reasoning, being more constrained by narrative coherence, could be more faithful.

### Paper 6: Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning LLMs
- **Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, et al.
- **Year**: 2025
- **Source**: arXiv: 2503.09567
- **Key Contribution**: Comprehensive survey of long CoT reasoning, covering training methods, prompting strategies, and evaluation for reasoning LLMs.
- **Relevance to Story CoT**: Provides context on the state-of-the-art in CoT research. Story CoT naturally produces longer reasoning traces (narratives), so findings about long CoT performance and challenges are directly relevant.

### Paper 7: Contrastive Chain-of-Thought Prompting
- **Authors**: Yew Ken Chia, Guizhen Chen, Luu Anh Tuan, et al.
- **Year**: 2023
- **Source**: arXiv: 2311.09277
- **Key Contribution**: Added both positive and negative (incorrect) exemplars to CoT demonstrations, helping models learn from mistakes.
- **Relevance to Story CoT**: Contrastive examples could be applied to Story CoT—showing both good and bad narrative reasoning paths.

### Paper 8: Chain of Code: Reasoning with a Language Model-Augmented Code Emulator
- **Authors**: Chengshu Li, Jacky Liang, Andy Zeng, et al.
- **Year**: 2023
- **Source**: arXiv: 2312.04474
- **Key Contribution**: Uses code as the reasoning medium, with LLM emulating code execution for non-deterministic steps.
- **Relevance to Story CoT**: Represents the "code as reasoning" paradigm. The NOT paper already combines narrative with code representation. Story CoT generalizes the question: what is the optimal reasoning *medium* (natural language steps, code, narrative, etc.)?

### Paper 9: Implicit Chain of Thought Reasoning via Knowledge Distillation
- **Authors**: Yuntian Deng, Kiran Prasad, Roland Fernandez, et al.
- **Year**: 2023
- **Source**: arXiv: 2311.01460
- **Key Contribution**: Distills CoT reasoning into a model's internal representations, enabling reasoning without explicit intermediate tokens.
- **Relevance to Story CoT**: Raises the question of whether story-form CoT could be distilled into models, potentially allowing "narrative thinking" without explicit narrative generation.

### Paper 10: Invalid Logic, Equivalent Gains: The Bizarreness of Reasoning in Language Model Prompting
- **Authors**: Rylan Schaeffer, Kateryna Pistunova, Samar Khanna
- **Year**: 2023
- **Source**: arXiv: 2307.10573
- **Key Contribution**: Shows that even logically invalid CoT can improve performance, suggesting CoT may work through mechanisms other than logical reasoning.
- **Relevance to Story CoT**: **Highly relevant.** If invalid logic still helps, this supports the hypothesis that CoT may exploit hidden patterns rather than performing genuine reasoning. Story CoT could test this—if narrative-form reasoning works even when the narrative logic is different from step-by-step logic, it would suggest the benefit comes from structure/coherence rather than logical validity.

### Paper 11: Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models
- **Authors**: Ling Yang, Zhaochen Yu, Tianjun Zhang, et al.
- **Year**: 2024
- **Source**: arXiv: 2406.04271
- **Key Contribution**: Stores and retrieves high-level "thought templates" for reasoning, combining meta-reasoning with task-specific knowledge.
- **Relevance to Story CoT**: Story templates (narrative schemas) could serve as thought templates for Story CoT, paralleling how humans use familiar narrative structures to reason about new situations.

### Paper 12: Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models
- **Authors**: Jiacheng Ye, Shansan Gong, Liheng Chen, et al.
- **Year**: 2024
- **Source**: arXiv: 2402.07754
- **Key Contribution**: Adapts CoT to diffusion language models, generating all reasoning tokens simultaneously rather than sequentially.
- **Relevance to Story CoT**: Explores alternative generation paradigms for CoT. Story CoT's sequential narrative structure may be more suited to autoregressive generation, but the parallel generation of reasoning could inform how narratives are structured.

### Paper 13: Towards Better Chain-of-Thought Prompting Strategies: A Survey
- **Authors**: Zihan Yu, Liang He, Zhen Wu, et al.
- **Year**: 2023
- **Source**: arXiv: 2310.04959
- **Key Contribution**: Comprehensive survey categorizing CoT strategies by their prompting approaches, including zero-shot, few-shot, and various structural modifications.
- **Relevance to Story CoT**: Provides taxonomic context for where Story CoT fits among CoT variants.

---

## Common Methodologies

- **Few-shot prompting with exemplars**: Used in Wei et al. (2022), NOT, and most CoT variants. Story CoT would similarly use narrative-form exemplars.
- **Structural representation (code-based)**: Used in NOT, Chain of Code, PAL. Converting problems to code helps LLMs reason more accurately.
- **Self-consistency / multiple sampling**: Used in Wang et al. (2022). Applicable to Story CoT by generating multiple narrative reasoning paths.
- **Meta-prompting / template-based**: Used in Buffer of Thoughts. Story schemas could serve as reasoning templates.

## Standard Baselines

- **Standard prompting** (no CoT): Direct question → answer
- **Zero-shot CoT**: "Let's think step by step"
- **Few-shot CoT**: Manually crafted step-by-step exemplars
- **Self-consistency CoT**: Multiple sampled CoT paths with majority voting
- **GPT-3.5/4 baselines**: Common reference points for performance comparison

## Evaluation Metrics

- **Accuracy**: For classification/QA tasks (GSM8K, CommonsenseQA, StrategyQA)
- **F1 Score**: For temporal graph generation (ProScript, Schema-11)
- **Graph Edit Distance (GED)**: Structural similarity for graph outputs
- **Self-faithfulness**: Alignment between intermediate reasoning and final answer
- **Consistency**: Agreement across different input orderings

## Datasets in the Literature

| Dataset | Used In | Task | Domain |
|---------|---------|------|--------|
| ProScript | NOT (Zhang et al., 2024) | Temporal graph generation | Daily activities |
| Schema-11 | NOT (Zhang et al., 2024) | Temporal graph generation | News |
| WikiHow Script | NOT (Zhang et al., 2024) | Temporal graph generation | Daily activities |
| GSM8K | Wei et al. (2022), Wang et al. (2022), most CoT papers | Math reasoning | Arithmetic |
| CommonsenseQA | Wei et al. (2022), most CoT papers | Commonsense QA | General |
| StrategyQA | Wei et al. (2022), Wang et al. (2022) | Multi-hop QA | General |
| ARC-Challenge | Multiple CoT papers | Science QA | Science |
| BIG-bench | Multiple reasoning papers | Various reasoning tasks | Various |

## Evolution of CoT and Where Story CoT Fits

The literature reveals a clear progression: Wei (2022) established CoT as natural-language reasoning steps; Wang (2022) showed multiple reasoning narratives improve accuracy via self-consistency; Yao (2023) introduced branching thought structures (ToT); Chen (2025) characterized the emerging Long CoT paradigm with deep reasoning, exploration, and reflection. Story CoT sits at the intersection of these threads—it uses explicit narrative structure (characters, settings, causal progressions, plot arcs) as the organizing principle for intermediate reasoning. Long CoT's characteristics map onto narrative structures: deep reasoning parallels story development; extensive exploration parallels subplot branching; reflection parallels a narrator revisiting earlier assumptions. Story CoT could provide a natural framework for organizing Long CoT, using narrative conventions to guide when to conclude and potentially mitigate "overthinking."

A critical cautionary note: Arcuschin et al. (2025) show CoT reasoning is unfaithful even on realistic prompts—models construct post-hoc rationalizations. Story CoT faces the same risk: models might generate compelling stories that don't reflect actual reasoning. Design must account for faithfulness via consistency checks.

## Cross-Cutting Themes Supporting Story CoT

1. **Structure over logic**: Schaeffer et al. (2023) show that logically *invalid* CoT prompts achieve similar gains to valid ones on BIG-Bench Hard. Chia et al. (2023) show that contrastive examples (positive+negative) reduce errors. Together, these suggest that the **format/structure** of reasoning matters as much or more than logical validity—directly supporting narrative as a reasoning scaffold.

2. **Alternative reasoning representations**: Chain of Code (Li et al., 2023) uses code; Implicit CoT (Deng et al., 2023) uses hidden states; Diffusion of Thoughts (Ye et al., 2024) uses parallel diffusion. Narrative is another alternative representation for reasoning, and each format may excel at different task types.

3. **Intrinsic reasoning capabilities**: Wang & Zhou (2024) show CoT paths exist latently in LLMs and can be elicited through decoding strategies. Story-form reasoning patterns may similarly exist latently, given that LLMs are trained extensively on narrative text.

4. **Reusable reasoning patterns**: Buffer of Thoughts (Yang et al., 2024) stores abstract "thought-templates" for retrieval. Narrative structures (hero's journey, problem-solution, cause-effect) are natural templates that could serve the same function.

## Gaps and Opportunities

1. **Limited task coverage for narrative CoT**: NOT only tests temporal reasoning. Story CoT should be evaluated on commonsense, mathematical, and other reasoning tasks.
2. **No comparison across narrative styles**: NOT evaluates narrative "quality" but doesn't compare narrative CoT vs. standard CoT across diverse tasks.
3. **Faithfulness of story-form reasoning**: Only 72.8% self-faithfulness in NOT. Can narrative coherence constraints improve faithfulness?
4. **Training with story-form CoT**: All existing work is prompting-based. No work explores fine-tuning models to produce story-form reasoning traces.
5. **Does narrative structure reveal hidden patterns?**: The "invalid logic" finding (Schaeffer et al., 2023) suggests CoT may work through mechanisms other than reasoning. Story CoT could help disentangle structure from logic.
6. **Cross-task generalization**: Does Story CoT help on tasks where standard CoT fails (temporal reasoning) but hurt on tasks where standard CoT excels (math)?

## Recommendations for Our Experiment

Based on the literature review:

### Recommended Datasets
1. **ProScript** (primary): Direct comparison with NOT; temporal reasoning where story CoT has proven effective
2. **GSM8K**: Standard math reasoning benchmark; test if story CoT helps/hurts vs. standard CoT
3. **CommonsenseQA**: Commonsense reasoning; narratives are natural for commonsense
4. **StrategyQA**: Multi-hop reasoning requiring world knowledge; stories may help connect reasoning steps
5. **ARC-Challenge**: Science reasoning; test generalization
6. **BIG-bench Causal Judgment**: Causal reasoning with narrative-like scenarios

### Recommended Baselines
1. Standard prompting (no CoT)
2. Zero-shot CoT ("Let's think step by step")
3. Few-shot CoT (standard step-by-step)
4. NOT (narrative-of-thought, for temporal tasks)
5. Self-consistency variants of above

### Recommended Metrics
1. Task-specific accuracy (for QA tasks)
2. F1 / GED (for temporal graph tasks)
3. Self-faithfulness (narrative-answer alignment)
4. Narrative quality metrics (coherence, factuality)

### Methodological Considerations
- **Narrative prompts**: Use "Let's think about this as a story" or similar prompts to elicit story-form reasoning
- **Narrative exemplars**: Provide few-shot examples with narrative-form reasoning chains
- **Comparison of reasoning formats**: Same problem solved with (a) standard CoT, (b) story-form CoT, (c) code-form CoT
- **Model size effects**: CoT is an emergent ability in large models; test whether Story CoT emerges at different scales
- **Narrative quality analysis**: Evaluate generated narratives for coherence, temporal grounding, and factual accuracy
