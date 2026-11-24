PY ?= python

.PHONY: help stamp data dataset baselines model interp reproduce test clean

help:
	@echo "make stamp       freeze + hash the pre-registration, then commit"
	@echo "make data        ~1.5 GB of bookTicker archives (ends 2024-03-30)"
	@echo "make dataset     load -> labels -> features"
	@echo "make baselines   THE BAR. Must exist before the model stage."
	@echo "make model       causal transformer vs LightGBM (needs torch CPU)"
	@echo "make interp      probes, SAE, activation patching"
	@echo "make test        unit tests"

stamp:
	PYTHONPATH=src $(PY) scripts/stamp_prereg.py

data:
	PYTHONPATH=src $(PY) scripts/download_bookticker.py --days 7 --end 2024-03-30

dataset:
	PYTHONPATH=src $(PY) scripts/build_dataset.py

baselines:
	PYTHONPATH=src $(PY) scripts/run_baselines.py

model:
	PYTHONPATH=src $(PY) scripts/run_model.py

interp:
	PYTHONPATH=src $(PY) scripts/run_interp.py

reproduce: dataset baselines model interp
	@echo "reproduced: baselines bar exists and the transformer was compared against it."

test:
	PYTHONPATH=src $(PY) -m pytest -q tests

clean:
	rm -rf outputs .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
