.PHONY: help setup data demo full eval test check report clean gate

PY := python
VENV := .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN := $(VENV)/Scripts
endif

help:
	@echo "setup  - create venv + install deps (~3 min)"
	@echo "data   - fetch twcs.csv and build data/processed/*.jsonl"
	@echo "demo   - REPLAY cached LLM calls, NO API KEY NEEDED (~4 min) -> headline table + figures"
	@echo "full   - live API run: rebuilds the cache from scratch (needs OPENAI_API_KEY, costs money)"
	@echo "eval   - evaluate all three systems -> reports/results.json"
	@echo "test   - pytest"
	@echo "check  - test + lint-ish sanity + submission gate checks"
	@echo "report - regenerate every figure from reports/results.json"

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(BIN)/python -m pip install -r requirements.txt
	@echo "setup done"

data:
	bash scripts/get_data.sh
	$(BIN)/python -m src.support_agent.ingest --build
	$(BIN)/python -m src.support_agent.ingest --verify

# The reproduction path a grader runs. LLM_OFFLINE=1 makes any cache miss a hard error,
# so this target cannot silently fall back to a live call it does not have a key for.
demo:
	LLM_OFFLINE=1 $(BIN)/python -m eval.run_eval --system all --split golden --offline
	LLM_OFFLINE=1 $(BIN)/python -m eval.figures
	$(BIN)/python -m eval.run_eval --headline

full:
	bash scripts/get_data.sh
	$(BIN)/python -m src.support_agent.ingest --build
	$(BIN)/python -m src.support_agent.taxonomy --induce
	$(BIN)/python -m eval.run_eval --system all --split golden
	$(BIN)/python -m eval.judge_validation --report --probes
	$(BIN)/python -m eval.figures

eval:
	$(BIN)/python -m eval.run_eval --system all --split golden

test:
	$(BIN)/python -m pytest tests -q

check: test
	$(BIN)/python scripts/submission_gate.py

report:
	$(BIN)/python -m eval.figures

gate:
	$(BIN)/python scripts/submission_gate.py

clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
