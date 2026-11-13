# Security–Utility Trade-offs of Lightweight Defenses Against Indirect Prompt Injection in Local RAG Systems

Reproducible evaluation framework and benchmark pipeline for measuring the security–utility trade-offs of lightweight, inference-time defenses against indirect prompt injection in local Retrieval-Augmented Generation (RAG) systems.

All components run locally with zero API cost: language models are served via [Ollama](https://ollama.com), dense retrieval uses `sentence-transformers` (`all-MiniLM-L6-v2`), and attack compliance is scored deterministically via unique canary strings rather than an LLM judge.

## Key Features

- **Full-Factorial Evaluation:** Crosses generator model scale (`Qwen2.5-1.5B-Instruct` vs. `Qwen2.5-7B-Instruct`, 4-bit quantized) with four defense configurations (`none`, `delimiter`, `filter`, `both`).
- **Deterministic Canary & Dollar-Amount Scoring:** Detects attack compliance via unique per-case verification codes (e.g., `XK7Q-2M9P`) and scores benign QA utility on normalized `Decimal` dollar amounts (`strict` exact set match and `lenient` superset match).
- **Strict Anti-Leakage Split:** Adapts the [BIPIA](https://github.com/microsoft/BIPIA) `EmailQA` benchmark into disjoint development (`dev`) and frozen evaluation (`test`) splits partitioned by normalized email prefix and disjoint attack-template pools (`override` and `task_redirect`).
- **Experimental Controls:** Includes `--forced-retrieval` (inserts unretrieved target documents at their true cosine similarity) and `--positive-control` (places the canary instruction in the trusted system prompt with clean documents indexed) to separate retrieval misses and small-model capability floors from defense robustness.
- **Paired Statistical Testing:** Reports 95% Wilson score confidence intervals on proportions and exact two-sided McNemar tests with step-down Holm-Bonferroni correction against `none`.

## Project Layout

```text
local-rag-injection-defense/
├── configs/
│   └── experiment.yaml          # Pinned model, defense, retrieval, and generation settings
├── data/
│   ├── raw/                     # Cloned upstream BIPIA repository (gitignored)
│   └── processed/               # Adapted JSONL splits (dev / test / adaptive)
├── results/
│   ├── report.md                # Main factorial experiment summary and McNemar tables
│   ├── results.jsonl            # Full per-case response and latency logs
│   ├── security_utility.svg     # Security–utility trade-off frontier plot
│   ├── forced_retrieval/        # Conditional susceptibility control (100% target retrieval)
│   ├── positive_control/        # Trusted system-prompt capability control
│   └── adaptive/                # Exploratory defense-aware variant results
├── src/
│   ├── types.py                 # Core data structures (Document, RetrievedDoc)
│   ├── llm.py                   # Ollama HTTP client with pinned sampling & 8k context window
│   ├── retriever.py             # MiniLM dense embeddings + top-k cosine similarity
│   ├── defenses.py              # Defense A (delimiter marking) & Defense B (regex filter)
│   ├── rag.py                   # End-to-end RAG pipeline orchestration
│   ├── data.py                  # Canary generation, Decimal amount scorer, disjoint split
│   ├── metrics.py               # Canary rate, accuracy, FPR, latency, Wilson CIs, McNemar/Holm
│   └── runner.py                # Crash-safe factorial runner and Markdown report generator
├── scripts/
│   ├── download_data.py         # Clones pinned BIPIA commit and builds JSONL splits
│   ├── pilot.py                 # Hardware latency benchmark and runtime estimator
│   ├── run_experiment.py        # CLI entry point for main, control, audit, and adaptive runs
│   └── make_figure.py           # Generates the SVG security–utility plot (stdlib only)
├── tests/
│   ├── test_data_build.py       # Schema integrity checks on synthetic fixtures
│   ├── test_data_construction.py# Disjoint split, template pool, and canary-wrapper tests
│   ├── test_defenses_metrics.py # Unit tests for defenses, Wilson CIs, and McNemar/Holm stats
│   ├── test_pipeline.py         # End-to-end runner, latency, resume, and control tests
│   ├── test_real_data.py        # Audit of built data/processed/*.jsonl files (-m data)
│   └── test_scorer.py           # Regression tests for Decimal dollar-amount answer scoring
├── Makefile                     # Shortcuts for test, lint, typecheck, and experiment runs
├── pyproject.toml               # Pytest, Ruff, and mypy configuration
├── requirements.txt             # Pinned runtime dependencies
└── requirements-dev.txt         # Test, lint, and type-checking dependencies

```

## Results Summary (Frozen Test Split)

Evaluated on the frozen test split (70 indirect prompt injection attacks, 27 answerable benign queries including 18 benign directive-bearing emails) with top-5 dense retrieval over 21 emails per case (1 target + 20 distractors) on an NVIDIA GeForce GTX 1070 (8 GB VRAM).

### 1. Main Experiment (Natural Top-5 Retrieval)

| Model | Defense | Canary rate (95% CI) | Acc strict (95% CI) | Acc lenient | FPR | Retr-hit | Gen p50 (s) | Gen p95 (s) | Total p50 (s) | Total p95 (s) | Refusal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-1.5b` | `none` | 0.00 [0.00, 0.12] | 0.15 [0.06, 0.32] | 0.44 | n/a | 0.41 | 2.8 | 3.7 | 2.8 | 3.7 | 0.00 |
| `qwen2.5-1.5b` | `delimiter` | 0.00 [0.00, 0.12] | 0.26 [0.13, 0.45] | 0.52 | n/a | 0.41 | 2.7 | 3.1 | 2.7 | 3.1 | 0.00 |
| `qwen2.5-1.5b` | `filter` | 0.00 [0.00, 0.12] | 0.19 [0.08, 0.37] | 0.44 | 0.00 | 0.41 | 2.7 | 3.7 | 2.7 | 3.7 | 0.00 |
| `qwen2.5-1.5b` | `both` | 0.00 [0.00, 0.12] | 0.26 [0.13, 0.45] | 0.52 | 0.00 | 0.41 | 2.6 | 3.0 | 2.6 | 3.1 | 0.00 |
| `qwen2.5-7b` | `none` | 0.59 [0.41, 0.74] | 0.30 [0.16, 0.48] | 0.48 | n/a | 0.41 | 5.5 | 8.1 | 5.6 | 8.1 | 0.00 |
| `qwen2.5-7b` | `delimiter` | 0.38 [0.23, 0.56] | 0.41 [0.25, 0.59] | 0.52 | n/a | 0.41 | 5.8 | 8.1 | 5.8 | 8.1 | 0.00 |
| `qwen2.5-7b` | `filter` | 0.28 [0.15, 0.46] | 0.30 [0.16, 0.48] | 0.48 | 0.00 | 0.41 | 5.4 | 8.2 | 5.4 | 8.2 | 0.00 |
| `qwen2.5-7b` | `both` | 0.31 [0.17, 0.49] | 0.41 [0.25, 0.59] | 0.52 | 0.00 | 0.41 | 5.8 | 7.9 | 5.8 | 7.9 | 0.00 |

#### Paired Exact McNemar Tests (Main Experiment vs. `none`, Holm-corrected)

| Model | Defense | Paired N | Fixed (b) | Broke (c) | p (exact) | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-1.5b` | `delimiter` | 29 | 0 | 0 | 1.000 | 1.000 |
| `qwen2.5-1.5b` | `filter` | 29 | 0 | 0 | 1.000 | 1.000 |
| `qwen2.5-1.5b` | `both` | 29 | 0 | 0 | 1.000 | 1.000 |
| `qwen2.5-7b` | `delimiter` | 29 | 7 | 1 | 0.070 | 0.281 |
| `qwen2.5-7b` | `filter` | 29 | 9 | 0 | 0.004 | 0.023 |
| `qwen2.5-7b` | `both` | 29 | 9 | 1 | 0.021 | 0.107 |

### 2. Forced-Retrieval Control (`--forced-retrieval`)

Guarantees the poisoned target email is in the top-5 retrieved context (using its true cosine similarity score) across all 70 attack cases:

| Model | Defense | Canary rate (95% CI) | Acc strict (95% CI) | Acc lenient | FPR | Retr-hit | Gen p50 (s) | Gen p95 (s) | Total p50 (s) | Total p95 (s) | Refusal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-1.5b` | `none` | 0.00 [0.00, 0.05] | 0.15 [0.06, 0.32] | 0.44 | n/a | 1.00 | 2.7 | 3.7 | 2.7 | 3.8 | 0.00 |
| `qwen2.5-1.5b` | `delimiter` | 0.00 [0.00, 0.05] | 0.26 [0.13, 0.45] | 0.52 | n/a | 1.00 | 2.7 | 3.1 | 2.7 | 3.1 | 0.00 |
| `qwen2.5-1.5b` | `filter` | 0.00 [0.00, 0.05] | 0.19 [0.08, 0.37] | 0.44 | 0.00 | 1.00 | 2.7 | 3.7 | 2.7 | 3.7 | 0.00 |
| `qwen2.5-1.5b` | `both` | 0.00 [0.00, 0.05] | 0.26 [0.13, 0.45] | 0.52 | 0.00 | 1.00 | 2.6 | 3.0 | 2.6 | 3.0 | 0.00 |
| `qwen2.5-7b` | `none` | 0.63 [0.51, 0.73] | 0.30 [0.16, 0.48] | 0.48 | n/a | 1.00 | 5.8 | 8.3 | 5.8 | 8.3 | 0.04 |
| `qwen2.5-7b` | `delimiter` | 0.64 [0.53, 0.74] | 0.41 [0.25, 0.59] | 0.52 | n/a | 1.00 | 6.2 | 9.0 | 6.2 | 9.0 | 0.00 |
| `qwen2.5-7b` | `filter` | 0.34 [0.24, 0.46] | 0.30 [0.16, 0.48] | 0.48 | 0.00 | 1.00 | 5.4 | 8.4 | 5.4 | 8.4 | 0.03 |
| `qwen2.5-7b` | `both` | 0.41 [0.31, 0.53] | 0.41 [0.25, 0.59] | 0.52 | 0.00 | 1.00 | 5.8 | 8.5 | 5.8 | 8.5 | 0.00 |

#### Paired Exact McNemar Tests (Forced Retrieval vs. `none`, Holm-corrected)

| Model | Defense | Paired N | Fixed (b) | Broke (c) | p (exact) | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-1.5b` | `delimiter` | 70 | 0 | 0 | 1.000 | 1.000 |
| `qwen2.5-1.5b` | `filter` | 70 | 0 | 0 | 1.000 | 1.000 |
| `qwen2.5-1.5b` | `both` | 70 | 0 | 0 | 1.000 | 1.000 |
| `qwen2.5-7b` | `delimiter` | 70 | 7 | 8 | 1.000 | 1.000 |
| `qwen2.5-7b` | `filter` | 70 | 20 | 0 | 0.000 | 0.000 |
| `qwen2.5-7b` | `both` | 70 | 20 | 5 | 0.004 | 0.020 |

### 3. Trusted System-Prompt Positive Control (`--positive-control`)

Indexes the clean version of each target email and places the injection instruction in the trusted system prompt across all 70 attack cases:

| Model | Defense | Canary rate (95% CI) | Acc strict (95% CI) | Acc lenient | FPR | Retr-hit | Gen p50 (s) | Gen p95 (s) | Total p50 (s) | Total p95 (s) | Refusal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-1.5b` | `positive_control` | 0.00 [0.00, 0.05] | n/a | 0.00 | n/a | 0.51 | 2.9 | 4.1 | 3.0 | 4.1 | 0.06 |
| `qwen2.5-7b` | `positive_control` | 0.97 [0.90, 0.99] | n/a | 0.00 | n/a | 0.51 | 5.5 | 8.0 | 5.5 | 8.0 | 0.04 |

### 4. Exploratory Adaptive Variants (`--adaptive`)

Evaluated on 18 defense-aware attack variants (`obfuscated`, `role_smuggle`, `polite_request`; 11/18 retrieved naturally):

| Model | Defense | Canary rate (95% CI) | Acc strict (95% CI) | Acc lenient | FPR | Retr-hit | Gen p50 (s) | Gen p95 (s) | Total p50 (s) | Total p95 (s) | Refusal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5-1.5b` | `none` | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 3.0 | 3.9 | 3.0 | 3.9 | 0.00 |
| `qwen2.5-1.5b` | `delimiter` | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 2.8 | 3.2 | 2.8 | 3.2 | 0.00 |
| `qwen2.5-1.5b` | `filter` | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 2.9 | 3.7 | 2.9 | 3.7 | 0.00 |
| `qwen2.5-1.5b` | `both` | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 2.7 | 3.2 | 2.7 | 3.2 | 0.00 |
| `qwen2.5-7b` | `none` | 0.18 [0.05, 0.48] | n/a | 0.00 | n/a | 0.61 | 5.8 | 8.4 | 5.8 | 8.5 | 0.00 |
| `qwen2.5-7b` | `delimiter` | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 5.7 | 7.3 | 5.7 | 7.4 | 0.00 |
| `qwen2.5-7b` | `filter` | 0.09 [0.02, 0.38] | n/a | 0.00 | n/a | 0.61 | 5.7 | 8.5 | 5.7 | 8.5 | 0.00 |
| `qwen2.5-7b` | `both` | 0.00 [0.00, 0.26] | n/a | 0.00 | n/a | 0.61 | 5.7 | 7.4 | 5.7 | 7.4 | 0.00 |

## Setup & Reproduction

1. **Install Prerequisites:** Install Python 3.11 or 3.12 (`uv` recommended) and [Ollama](https://ollama.com/download).
2. **Pull Quantized Models:**

```bash
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:7b-instruct-q4_K_M

```

3. **Create Environment & Install Dependencies:**

```bash
uv venv --python 3.12
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt

```

4. **Run Unit Tests & Linters (No Ollama or Real Data Required):**

```bash
ruff check src scripts tests
mypy src scripts
pytest -m "not data" -q

```

5. **Build & Audit Dataset:**

```bash
python scripts/download_data.py
pytest -m data -v
python scripts/run_experiment.py --filter-audit

```

6. **Run Experiments & Generate Figure:**

```bash
# Main factorial experiment + 10-case determinism check
python scripts/run_experiment.py --config configs/experiment.yaml --stability-check

# Experimental controls and exploratory adaptive set
python scripts/run_experiment.py --config configs/experiment.yaml --forced-retrieval
python scripts/run_experiment.py --config configs/experiment.yaml --positive-control
python scripts/run_experiment.py --config configs/experiment.yaml --adaptive

# Render SVG security-utility frontier
python scripts/make_figure.py --results results/results.jsonl --out results/security_utility.svg

```

Pass `--resume` to any `run_experiment.py` command to continue an interrupted run, or `--rescore results/results.jsonl` to re-run answer and canary scoring offline.

```
