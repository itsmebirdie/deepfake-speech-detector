.PHONY: install lint test data weights baseline
install:
	pip install -r requirements.txt && pip install -e .
lint:
	ruff check src tests scripts
test:
	pytest -q
data:
	bash scripts/download_data.sh
weights:
	bash scripts/download_weights.sh
baseline:
	bash scripts/setup_baseline.sh
