
# Security–Utility Trade-offs of Lightweight Defenses Against Indirect Prompt Injection in Local RAG Systems

Reproducible evaluation framework and benchmark pipeline for measuring the security–utility trade-offs of lightweight, inference-time defenses against indirect prompt injection in local Retrieval-Augmented Generation (RAG) systems.

All components run locally with zero API cost: language models are served via [Ollama](https://ollama.com), dense retrieval uses `sentence-transformers` (`all-MiniLM-L6-v2`), and attack compliance is scored deterministically via unique canary strings rather than a paid LLM judge.

## Key Features

- **Full-Factorial Evaluation:** Crosses generator model scale (`Qwen2.5-1.5B-Instruct` vs. `Qwen2.5-7B-Instruct`, 4-bit quantized) with four defense configurations (`none`, `delimiter`, `filter`, `both`).
- **Deterministic Canary Scoring:** Replaces non-deterministic GPT-3.5 judge calls with unique per-case confirmation codes (e.g., `XK7Q-2M9P`), making attack success rate (ASR) auditable via exact string matching.
- **Strict Anti-Leakage Split:** Adapts the [BIPIA](https://github.com/microsoft/BIPIA) `EmailQA` benchmark into disjoint development (`dev`) and frozen evaluation (`test`) splits partitioned by source email ID.
- **Joint Security–Utility Metrics:** Reports Attack Success Rate (ASR) conditioned on retrieval hits, legitimate-answer accuracy, filter false-positive rate (FPR), retrieval hit rate, refusal rate, and response latency (`p50`/`p95`) with 95% Wilson score confidence intervals.

## Project Layout

```text
rag-injection-defense/
├── configs/
│   └── experiment.yaml            # Pinned model, defense, retrieval, and generation settings
├── data/
│   ├── raw/                       # Cloned upstream BIPIA repository (gitignored)
│   └── processed/                 # Adapted JSONL splits (42 dev / 99 test / 18 adaptive)
├── results/
│   ├── report.md                  # Main factorial experiment summary table
│   ├── results.jsonl              # Full per-case response and metadata logs
│   ├── security_utility.svg       # Security–utility trade-off plot
│   └── adaptive/                  # Exploratory defense-aware variant results
├── src/
│   ├── types.py                   # Core data structures (Document, RetrievedDoc)
│   ├── llm.py                     # Ollama HTTP client with pinned sampling & context window
│   ├── retriever.py               # MiniLM dense embeddings + top-k cosine similarity
│   ├── defenses.py                # Defense A (delimiter marking) & Defense B (regex filter)
│   ├── rag.py                     # End-to-end RAG pipeline orchestration
│   ├── data.py                    # Canary generation, answer normalization, dev/test split
│   ├── metrics.py                 # ASR, accuracy, FPR, latency percentiles, Wilson 95% CIs
│   └── runner.py                  # Factorial experiment runner and report generator
├── scripts/
│   ├── download_data.py           # Clones BIPIA and builds processed JSONL splits
│   ├── pilot.py                   # Hardware latency benchmark and runtime estimator
│   ├── run_experiment.py          # CLI entry point for main and adaptive runs
│   └── make_figure.py             # Generates the SVG security–utility plot
└── tests/
    ├── test_defenses_metrics.py   # Unit tests for defenses, scoring, and Wilson CIs
    └── test_data_build.py         # Data integrity and disjoint-split tests

```

## Results Summary (Frozen Test Split)

Evaluated on 99 frozen test cases (70 indirect prompt injection attacks, 29 answerable benign queries) with top-5 dense retrieval over 21 emails per case (1 target + 20 distractors) on an NVIDIA GeForce GTX 1070 (8 GB VRAM):

| Model | Defense | ASR (95% CI) | Acc (95% CI) | FPR | Retr-hit | p50 (s) | p95 (s) | Refusal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-1.5b` | `none` | 0.00 [0.00, 0.11] | 0.66 [0.47, 0.80] | 0.00 | 0.43 | 2.7 | 3.8 | 0.00 |
| `qwen2.5-1.5b` | `delimiter` | 0.00 [0.00, 0.11] | 0.72 [0.54, 0.85] | 0.00 | 0.43 | 2.6 | 2.9 | 0.00 |
| `qwen2.5-1.5b` | `filter` | 0.00 [0.00, 0.11] | 0.66 [0.47, 0.80] | 0.00 | 0.43 | 2.7 | 3.8 | 0.00 |
| `qwen2.5-1.5b` | `both` | 0.00 [0.00, 0.11] | 0.72 [0.54, 0.85] | 0.00 | 0.43 | 2.7 | 2.8 | 0.00 |
| `qwen2.5-7b` | `none` | 0.37 [0.22, 0.54] | 0.69 [0.51, 0.83] | 0.00 | 0.43 | 5.4 | 7.4 | 0.01 |
| `qwen2.5-7b` | `delimiter` | 0.23 [0.12, 0.41] | 0.69 [0.51, 0.83] | 0.00 | 0.43 | 5.6 | 7.5 | 0.00 |
| `qwen2.5-7b` | `filter` | 0.37 [0.22, 0.54] | 0.69 [0.51, 0.83] | 0.00 | 0.43 | 5.3 | 7.2 | 0.01 |
| `qwen2.5-7b` | `both` | 0.23 [0.12, 0.41] | 0.69 [0.51, 0.83] | 0.00 | 0.43 | 5.7 | 7.5 | 0.00 |

## Setup & Reproduction

1. **Install Prerequisites:** Install Python 3.10+ and [Ollama](https://ollama.com/download).
2. **Pull Quantized Models:**
```bash
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M

```


3. **Install Python Dependencies:**
```bash
pip install -r requirements.txt

```


4. **Run Unit Tests:**
```bash
python -m pytest tests/ -q

```


5. **Rebuild Dataset (Optional):** Processed splits are included in `data/processed/`. To clone BIPIA and regenerate from scratch:
```bash
python scripts/download_data.py

```


6. **Run Hardware Pilot & Full Experiment:**
```bash
python scripts/pilot.py
python scripts/run_experiment.py --config configs/experiment.yaml
python scripts/run_experiment.py --config configs/experiment.yaml --adaptive
python scripts/make_figure.py

```
