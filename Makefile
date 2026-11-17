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
	cp paper/figures/fig_micro.pdf paper/Fig1.pdf
	cp paper/figures/fig_scale.pdf paper/Fig2.pdf
	cp paper/figures/fig_backup.pdf paper/Fig3.pdf
	cp paper/figures/fig_replication.pdf paper/Fig4.pdf

paper: figures
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error Supplementary_Information.tex

verify: replay test figures

