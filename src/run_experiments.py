"""Main experiment runner for Story CoT research.

Compares four prompting strategies (Direct, Zero-shot CoT, Few-shot CoT, Story CoT)
across four benchmarks (GSM8K, CommonsenseQA, StrategyQA, ARC-Challenge) using GPT-4.1.
"""

import json
import os
import re
import sys
import time
import random
import hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from prompts import (
    format_gsm8k_prompt, format_csqa_prompt,
    format_strategyqa_prompt, format_arc_prompt
)

# Configuration
SEED = 42
MODEL = "gpt-4.1"
TEMPERATURE = 0.0  # deterministic for reproducibility
MAX_TOKENS = 1024
SAMPLES_PER_DATASET = 150
MAX_WORKERS = 10  # parallel API calls

random.seed(SEED)
np.random.seed(SEED)

client = OpenAI()

RESULTS_DIR = Path("/workspaces/story-cot-nlp-claude/results")
RESULTS_DIR.mkdir(exist_ok=True)

STRATEGIES = ["direct", "zero_shot_cot", "few_shot_cot", "story_cot"]


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_arrow(path):
    """Load a HuggingFace dataset saved in arrow format."""
    from datasets import load_from_disk
    return load_from_disk(path)


def load_gsm8k(n=SAMPLES_PER_DATASET):
    """Load GSM8K test set samples."""
    ds = load_dataset_arrow("/workspaces/story-cot-nlp-claude/datasets/gsm8k/test")
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    samples = []
    for i in indices:
        item = ds[i]
        # Extract numeric answer from "#### <number>"
        answer_text = item["answer"]
        match = re.search(r"####\s*([\d,\.\-]+)", answer_text)
        gold = match.group(1).replace(",", "") if match else ""
        samples.append({
            "id": f"gsm8k_{i}",
            "question": item["question"],
            "gold_answer": gold,
        })
    return samples


def load_commonsenseqa(n=SAMPLES_PER_DATASET):
    """Load CommonsenseQA validation set samples."""
    ds = load_dataset_arrow("/workspaces/story-cot-nlp-claude/datasets/commonsenseqa/validation")
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    samples = []
    for i in indices:
        item = ds[i]
        samples.append({
            "id": f"csqa_{i}",
            "question": item["question"],
            "choices": item["choices"],
            "gold_answer": item["answerKey"],
        })
    return samples


def load_strategyqa(n=SAMPLES_PER_DATASET):
    """Load StrategyQA samples from the JSON file."""
    with open("/workspaces/story-cot-nlp-claude/datasets/strategyqa/samples/train_samples.json") as f:
        data = json.load(f)
    random.shuffle(data)
    samples = []
    for i, item in enumerate(data[:n]):
        samples.append({
            "id": f"sqa_{i}",
            "question": item["question"],
            "gold_answer": "Yes" if item["answer"] else "No",
        })
    return samples


def load_arc_challenge(n=SAMPLES_PER_DATASET):
    """Load ARC-Challenge test set samples."""
    ds = load_dataset_arrow("/workspaces/story-cot-nlp-claude/datasets/arc_challenge/test")
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    samples = []
    for i in indices:
        item = ds[i]
        samples.append({
            "id": f"arc_{i}",
            "question": item["question"],
            "choices": item["choices"],
            "gold_answer": item["answerKey"],
        })
    return samples


# =============================================================================
# API Calling
# =============================================================================

def call_api(prompt, max_retries=5):
    """Call OpenAI API with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return {
                "text": response.choices[0].message.content,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt + random.random()
                print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                return {"text": f"ERROR: {e}", "usage": {"input_tokens": 0, "output_tokens": 0}}


# =============================================================================
# Answer Extraction
# =============================================================================

def extract_gsm8k_answer(response_text):
    """Extract numeric answer from GSM8K response."""
    # Look for #### pattern first
    match = re.search(r"####\s*([\d,\.\-]+)", response_text)
    if match:
        return match.group(1).replace(",", "")
    # Look for "the answer is X" pattern
    match = re.search(r"(?:the answer is|answer:?)\s*\$?([\d,\.\-]+)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    # Last number in response
    numbers = re.findall(r"(?<!\w)([\d,]+\.?\d*)(?!\w)", response_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def extract_letter_answer(response_text):
    """Extract letter answer (A-E) from multiple choice response."""
    # Look for "Answer: X" pattern
    match = re.search(r"(?:answer|Answer)\s*[:=]?\s*\(?([A-Ea-e1-5])\)?", response_text)
    if match:
        ans = match.group(1).upper()
        # Convert numbers to letters if needed
        if ans.isdigit():
            ans = chr(ord('A') + int(ans) - 1)
        return ans
    # Look for standalone letter at end
    match = re.search(r"\(?([A-Ea-e])\)?\s*[.!]?\s*$", response_text.strip())
    if match:
        return match.group(1).upper()
    # Look for "is (X)" pattern
    match = re.search(r"is\s+\(?([A-Ea-e])\)?", response_text)
    if match:
        return match.group(1).upper()
    # First letter mentioned
    match = re.search(r"\(([A-Ea-e])\)", response_text)
    if match:
        return match.group(1).upper()
    return ""


def extract_yesno_answer(response_text):
    """Extract Yes/No answer from response."""
    # Look for "Answer: Yes/No"
    match = re.search(r"(?:answer|Answer)\s*[:=]?\s*(yes|no)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    # Check last line
    last_line = response_text.strip().split("\n")[-1].strip()
    if re.search(r"\byes\b", last_line, re.IGNORECASE):
        return "Yes"
    if re.search(r"\bno\b", last_line, re.IGNORECASE):
        return "No"
    # Check whole text
    text_lower = response_text.lower()
    yes_count = len(re.findall(r"\byes\b", text_lower))
    no_count = len(re.findall(r"\bno\b", text_lower))
    if yes_count > no_count:
        return "Yes"
    elif no_count > yes_count:
        return "No"
    return ""


def check_gsm8k_correct(predicted, gold):
    """Check if GSM8K answer is correct (numeric comparison)."""
    try:
        return abs(float(predicted) - float(gold)) < 0.01
    except (ValueError, TypeError):
        return False


# =============================================================================
# Experiment Runner
# =============================================================================

def run_single_experiment(dataset_name, sample, strategy):
    """Run a single prompt and return results."""
    if dataset_name == "gsm8k":
        prompt = format_gsm8k_prompt(sample["question"], strategy)
    elif dataset_name == "commonsenseqa":
        prompt = format_csqa_prompt(sample["question"], sample["choices"], strategy)
    elif dataset_name == "strategyqa":
        prompt = format_strategyqa_prompt(sample["question"], strategy)
    elif dataset_name == "arc_challenge":
        prompt = format_arc_prompt(sample["question"], sample["choices"], strategy)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    response = call_api(prompt)

    # Extract answer
    if dataset_name == "gsm8k":
        predicted = extract_gsm8k_answer(response["text"])
        correct = check_gsm8k_correct(predicted, sample["gold_answer"])
    elif dataset_name in ["commonsenseqa", "arc_challenge"]:
        predicted = extract_letter_answer(response["text"])
        correct = predicted == sample["gold_answer"]
    elif dataset_name == "strategyqa":
        predicted = extract_yesno_answer(response["text"])
        correct = predicted == sample["gold_answer"]

    return {
        "id": sample["id"],
        "dataset": dataset_name,
        "strategy": strategy,
        "question": sample["question"],
        "gold_answer": sample["gold_answer"],
        "predicted": predicted,
        "correct": correct,
        "response_text": response["text"],
        "response_length": len(response["text"]),
        "input_tokens": response["usage"]["input_tokens"],
        "output_tokens": response["usage"]["output_tokens"],
    }


def run_dataset_experiment(dataset_name, samples, strategy):
    """Run experiment for one dataset and one strategy with parallel API calls."""
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for sample in samples:
            future = executor.submit(run_single_experiment, dataset_name, sample, strategy)
            futures[future] = sample["id"]

        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"  {dataset_name}/{strategy}",
            ncols=80
        )
        for future in pbar:
            result = future.result()
            results.append(result)

    return results


def main():
    """Main experiment loop."""
    print("=" * 70)
    print("Story CoT Experiment")
    print(f"Model: {MODEL} | Samples/dataset: {SAMPLES_PER_DATASET} | Seed: {SEED}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    datasets["gsm8k"] = load_gsm8k()
    print(f"  GSM8K: {len(datasets['gsm8k'])} samples")
    datasets["commonsenseqa"] = load_commonsenseqa()
    print(f"  CommonsenseQA: {len(datasets['commonsenseqa'])} samples")
    datasets["strategyqa"] = load_strategyqa()
    print(f"  StrategyQA: {len(datasets['strategyqa'])} samples")
    datasets["arc_challenge"] = load_arc_challenge()
    print(f"  ARC-Challenge: {len(datasets['arc_challenge'])} samples")

    # Run experiments
    all_results = []
    total_experiments = len(datasets) * len(STRATEGIES)
    exp_num = 0

    for dataset_name, samples in datasets.items():
        for strategy in STRATEGIES:
            exp_num += 1
            print(f"\n[{exp_num}/{total_experiments}] Running {dataset_name} / {strategy}...")
            results = run_dataset_experiment(dataset_name, samples, strategy)
            all_results.extend(results)

            # Print interim accuracy
            correct = sum(1 for r in results if r["correct"])
            total = len(results)
            print(f"  Accuracy: {correct}/{total} = {correct/total:.1%}")

            # Save intermediate results
            with open(RESULTS_DIR / "all_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Save final results
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save config
    config = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "seed": SEED,
        "strategies": STRATEGIES,
        "datasets": list(datasets.keys()),
        "timestamp": datetime.now().isoformat(),
        "total_results": len(all_results),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for dataset_name in datasets:
        print(f"\n{dataset_name}:")
        for strategy in STRATEGIES:
            subset = [r for r in all_results if r["dataset"] == dataset_name and r["strategy"] == strategy]
            correct = sum(1 for r in subset if r["correct"])
            total = len(subset)
            avg_len = np.mean([r["response_length"] for r in subset])
            print(f"  {strategy:20s}: {correct}/{total} = {correct/total:.1%}  (avg response len: {avg_len:.0f})")

    # Total token usage
    total_input = sum(r["input_tokens"] for r in all_results)
    total_output = sum(r["output_tokens"] for r in all_results)
    print(f"\nTotal tokens: {total_input:,} input + {total_output:,} output = {total_input+total_output:,}")
    print(f"Estimated cost: ~${total_input/1e6 * 2 + total_output/1e6 * 8:.2f}")
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
