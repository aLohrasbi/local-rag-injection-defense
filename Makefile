PY ?= python
CONFIG ?= configs/experiment.yaml
# Extra CLI flags for run/forced/control, e.g.  make run ARGS="--resume --max-cases 5"
ARGS ?=

.PHONY: data run forced control report test check

data:    ## rebuild data/processed/ from BIPIA (needs network)
	$(PY) scripts/download_data.py

run:     ## main factorial experiment (needs Ollama)
	$(PY) scripts/run_experiment.py --config $(CONFIG) $(ARGS)

forced:  ## forced-retrieval variant (needs Ollama)
	$(PY) scripts/run_experiment.py --config $(CONFIG) --forced-retrieval $(ARGS)

control: ## trusted-prompt positive control (needs Ollama)
	$(PY) scripts/run_experiment.py --config $(CONFIG) --positive-control $(ARGS)

report:  ## re-score results/results.jsonl and redraw the figure (no LLM calls)
	$(PY) scripts/run_experiment.py --config $(CONFIG) --rescore results/results.jsonl
	$(PY) scripts/make_figure.py

test:    ## unit tests (fixtures only; the real-data audit is `pytest -m data`)
	$(PY) -m pytest -m "not data"

check:   ## what CI runs
	$(PY) -m ruff check .
	$(PY) -m mypy src scripts tests
	$(PY) -m pytest -m "not data"
