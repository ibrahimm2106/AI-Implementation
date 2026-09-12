.PHONY: install test lint train

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

train:
	python scripts/run_experiment.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir artifacts
