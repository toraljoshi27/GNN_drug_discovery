help:
	@echo 'Targets: setup, download, train, eval'

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

download:
	python src/data/download_moleculenet.py

train:
	python src/train.py --config $(CFG)

eval:
	python src/eval.py --config $(CFG)
