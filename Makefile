SYSTEM_PYTHON ?= python3.12
PYTHON ?= .venv/bin/python

.PHONY: setup replay test figures paper verify

setup:
	$(SYSTEM_PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

replay:
	$(PYTHON) repro/e29_14/replay_scores.py --repository-root "$(CURDIR)"
	$(PYTHON) repro/e33_1/replay_score.py --repository "$(CURDIR)"

test:
	$(PYTHON) -m pytest -q

figures:
	$(PYTHON) paper/make_figures.py

paper: figures
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error Supplementary_Information.tex

verify: replay test figures

