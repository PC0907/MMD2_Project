.PHONY: help install test eda direct reverse evasion clean

help:
	@echo "install   - editable install + deps"
	@echo "test      - run unit tests (no GPU/network)"
	@echo "eda       - run exploratory data analysis"
	@echo "direct    - train+eval direct clarity baseline"
	@echo "reverse   - train+eval reverse hierarchical pipeline"
	@echo "evasion   - train evasion baseline + joint"
	@echo "clean     - remove caches and runs"

install:
	pip install -e ".[dev]"

test:
	pytest -q tests/

eda:
	python -m clarity_evasion.cli.eda --out_dir runs/eda

direct:
	python -m clarity_evasion.cli.train --config configs/direct.yaml
	python -m clarity_evasion.cli.evaluate --config configs/direct.yaml --pipeline direct --model_dir runs/direct-deberta/best --out_dir runs/direct-deberta/eval

reverse:
	python -m clarity_evasion.cli.train --config configs/reverse.yaml
	python -m clarity_evasion.cli.evaluate --config configs/reverse.yaml --pipeline reverse --model_dir runs/reverse-deberta/best --out_dir runs/reverse-deberta/eval

evasion:
	python -m clarity_evasion.cli.train --config configs/evasion_baseline.yaml
	python -m clarity_evasion.cli.train --config configs/evasion_joint.yaml

clean:
	rm -rf runs logs **/__pycache__ .pytest_cache *.egg-info
