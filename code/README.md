# Cloned Repositories

## Repo 1: Narrative-of-Thought (NoT)
- **URL**: https://github.com/launchnlp/NoT
- **Purpose**: Official implementation of the Narrative-of-Thought prompting technique for temporal reasoning. Most directly relevant codebase for Story CoT research.
- **Location**: code/NoT/
- **Key files**:
  - `Data/` - Dataset processing scripts and Schema-11 annotations
  - `Data/README.md` - Instructions for downloading ProScript, Schema-11, WikiHow Script
  - `Data/schema11/test_raw.json` - Re-annotated Schema-11 evaluation data
  - `model/` - Model inference code
  - `requirements.sh` - Dependency setup
- **How to use**: Provides the prompting templates, data processing, and evaluation pipeline for NOT. Can be adapted for Story CoT experiments by modifying the narrative generation prompts.
- **Notes**: Built on LlamaFactory but can be reproduced with HuggingFace alone. Requires GPU for inference with local models.

## Repo 2: Zero-shot Chain-of-Thought
- **URL**: https://github.com/kojima-takeshi188/zero_shot_cot
- **Purpose**: Implementation of "Let's think step by step" zero-shot CoT prompting (Kojima et al., 2022).
- **Location**: code/zero_shot_cot/
- **Key files**: Contains prompting templates and evaluation scripts for standard zero-shot CoT.
- **How to use**: Baseline implementation for comparing zero-shot Story CoT against zero-shot standard CoT.
- **Notes**: Provides a clean reference for how zero-shot CoT prompting is implemented.
