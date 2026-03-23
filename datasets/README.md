# Downloaded Datasets

This directory contains datasets for the Story CoT research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: ProScript

### Overview
- **Source**: https://storage.googleapis.com/ai2-mosaic-public/projects/proscript/proscript_v1a.zip
- **Size**: 6,414 scenarios (train: 3,252, dev: 1,085, test: 2,077)
- **Format**: JSONL files
- **Task**: Temporal graph generation (ordering events into temporal graphs)
- **License**: Apache 2.0 (AI2)

### Download Instructions

```python
import requests, zipfile, io, os
url = 'https://storage.googleapis.com/ai2-mosaic-public/projects/proscript/proscript_v1a.zip'
r = requests.get(url, timeout=60)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('datasets/')
os.rename('datasets/proscript_v1a', 'datasets/proscript')
```

### Loading the Dataset

```python
import json
with open("datasets/proscript/train.jsonl") as f:
    data = [json.loads(line) for line in f]
```

### Notes
- Primary dataset from the Narrative-of-Thought paper
- Each scenario has events, temporal edges, and optional context
- Events are daily activities (e.g., "create a video game", "walk into a store")

---

## Dataset 2: Schema-11

### Overview
- **Source**: Re-annotated from https://github.com/CogComp/Zero_Shot_Schema_Induction
- **Size**: 11 scenarios (news topics)
- **Format**: JSON
- **Task**: Temporal graph generation for newsworthy event schemas
- **Domain**: News journalism (armed robbery, business change, etc.)

### Download Instructions

Already included in the NoT repository:
```bash
git clone https://github.com/launchnlp/NoT.git
cp -r NoT/Data/schema11 datasets/schema11
```

### Notes
- Re-annotated by Zhang et al. (2024) for the NOT paper
- Small but challenging evaluation set
- Contains non-linear temporal graphs (27% have branches)

---

## Dataset 3: GSM8K

### Overview
- **Source**: HuggingFace `openai/gsm8k`
- **Size**: train: 7,473, test: 1,319
- **Format**: HuggingFace Dataset
- **Task**: Grade school math word problems
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k")
```

### Notes
- Standard CoT reasoning benchmark
- Each example has a question and step-by-step solution with final numerical answer
- Good for comparing story CoT vs standard CoT on math reasoning

---

## Dataset 4: StrategyQA

### Overview
- **Source**: HuggingFace `ChilleD/StrategyQA`
- **Size**: train: 1,603, test: 687
- **Format**: HuggingFace Dataset
- **Task**: Multi-hop yes/no questions requiring implicit reasoning
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("ChilleD/StrategyQA")
dataset.save_to_disk("datasets/strategyqa")
```

### Notes
- Questions require combining multiple facts (e.g., "Did Aristotle use a laptop?")
- Good for testing whether narrative reasoning helps connect disparate facts

---

## Dataset 5: CommonsenseQA

### Overview
- **Source**: HuggingFace `tau/commonsense_qa`
- **Size**: train: 9,741, validation: 1,221, test: 1,140
- **Format**: HuggingFace Dataset
- **Task**: 5-way multiple choice commonsense questions
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("tau/commonsense_qa")
dataset.save_to_disk("datasets/commonsenseqa")
```

### Notes
- Standard commonsense reasoning benchmark
- Narratives may help by situating choices in real-world scenarios

---

## Dataset 6: ARC-Challenge

### Overview
- **Source**: HuggingFace `allenai/ai2_arc` (ARC-Challenge split)
- **Size**: train: 1,119, validation: 299, test: 1,172
- **Format**: HuggingFace Dataset
- **Task**: Multiple choice science questions (challenging subset)
- **License**: CC BY-SA 4.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
dataset.save_to_disk("datasets/arc_challenge")
```

---

## Dataset 7: BIG-bench Causal Judgment

### Overview
- **Source**: HuggingFace `tasksource/bigbench` (causal_judgment)
- **Size**: train: 152, validation: 38
- **Format**: HuggingFace Dataset
- **Task**: Causal reasoning about scenarios

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("tasksource/bigbench", "causal_judgment")
dataset.save_to_disk("datasets/bigbench_causal_judgment")
```

### Notes
- Small but directly relevant - causal judgment from scenarios
- Tests whether narrative reasoning helps with causal understanding
