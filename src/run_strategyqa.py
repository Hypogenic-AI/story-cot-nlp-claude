"""Re-run StrategyQA experiments with proper data loading from arrow dataset."""

import json
import os
import re
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prompts import format_strategyqa_prompt

SEED = 42
MODEL = "gpt-4.1"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
SAMPLES = 150
MAX_WORKERS = 10

random.seed(SEED)
np.random.seed(SEED)

client = OpenAI()
RESULTS_DIR = Path("/workspaces/story-cot-nlp-claude/results")
STRATEGIES = ["direct", "zero_shot_cot", "few_shot_cot", "story_cot"]


def load_strategyqa():
    from datasets import load_from_disk
    ds = load_from_disk("/workspaces/story-cot-nlp-claude/datasets/strategyqa/train/")
    indices = random.sample(range(len(ds)), min(SAMPLES, len(ds)))
    samples = []
    for i in indices:
        item = ds[i]
        samples.append({
            "id": f"sqa_{i}",
            "question": item["question"],
            "gold_answer": "Yes" if item["answer"] else "No",
        })
    return samples


def call_api(prompt, max_retries=5):
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
                time.sleep(2 ** attempt + random.random())
            else:
                return {"text": f"ERROR: {e}", "usage": {"input_tokens": 0, "output_tokens": 0}}


def extract_yesno_answer(text):
    match = re.search(r"(?:answer|Answer)\s*[:=]?\s*(yes|no)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    last_line = text.strip().split("\n")[-1].strip()
    if re.search(r"\byes\b", last_line, re.IGNORECASE):
        return "Yes"
    if re.search(r"\bno\b", last_line, re.IGNORECASE):
        return "No"
    text_lower = text.lower()
    if text_lower.count("yes") > text_lower.count("no"):
        return "Yes"
    elif text_lower.count("no") > text_lower.count("yes"):
        return "No"
    return ""


def main():
    print("Running StrategyQA experiments...")
    samples = load_strategyqa()
    print(f"Loaded {len(samples)} samples")

    all_results = []
    for strategy in STRATEGIES:
        print(f"\n  Strategy: {strategy}")
        results = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for sample in samples:
                prompt = format_strategyqa_prompt(sample["question"], strategy)
                future = executor.submit(call_api, prompt)
                futures[future] = sample

            for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
                sample = futures[future]
                response = future.result()
                predicted = extract_yesno_answer(response["text"])
                correct = predicted == sample["gold_answer"]
                results.append({
                    "id": sample["id"],
                    "dataset": "strategyqa",
                    "strategy": strategy,
                    "question": sample["question"],
                    "gold_answer": sample["gold_answer"],
                    "predicted": predicted,
                    "correct": correct,
                    "response_text": response["text"],
                    "response_length": len(response["text"]),
                    "input_tokens": response["usage"]["input_tokens"],
                    "output_tokens": response["usage"]["output_tokens"],
                })

        correct = sum(1 for r in results if r["correct"])
        print(f"  Accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")
        all_results.extend(results)

    # Save StrategyQA results
    with open(RESULTS_DIR / "strategyqa_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Merge with main results
    with open(RESULTS_DIR / "all_results.json") as f:
        main_results = json.load(f)

    # Remove old strategyqa results
    main_results = [r for r in main_results if r["dataset"] != "strategyqa"]
    main_results.extend(all_results)

    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(main_results, f, indent=2)

    print(f"\nDone! Updated all_results.json with {len(all_results)} StrategyQA results")


if __name__ == "__main__":
    main()
