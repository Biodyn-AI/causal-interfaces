# Reproducibility guide

## Supported environment

The submission release was validated with Python 3.12.15, PyTorch 2.13.0,
NumPy 2.5.1, Transformers 5.14.1, Safetensors 0.8.0, Einops 0.8.2,
Matplotlib 3.11.1, and Pytest 9.1.1.

## Strict saved-output replay

From the repository root:

```bash
make setup
make replay
make test
```

The replay validates all registered artifact hashes before calculating scores.
It never overwrites sealed material.
The make targets also remove macOS AppleDouble (`._*`) sidecars from the
disposable `.venv` before imports; no repository source or evidence file is
removed.

## GPT-2 Medium fresh inference

Download the exact checkpoint revision:

```bash
hf download openai-community/gpt2-medium \
  --revision 6dcaa7a952f72f9298047fd5137cd6e4f05f41da \
  --local-dir /absolute/path/to/gpt2-medium
```

Then run all four confirmation stages into a new or empty directory:

```bash
.venv/bin/python -m repro.e29_14.reproduce_inference \
  --repository-root "$PWD" \
  --model-snapshot /absolute/path/to/gpt2-medium \
  --output-directory /absolute/path/to/new-e29-output
```

Use `--experiments a` for the shortest headline check. The full run takes
substantially longer. The output directory must be absent or empty.

## GPT-2 Small fresh inference

Download `gpt2` at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, then run:

```bash
.venv/bin/python -m experiments.e33_ioi_scope_replication.workflow acquire \
  --registration notes/E33.1-IOI-SCOPE-REGISTRATION-V2.json \
  --registration-sha256 39a5a589678c76d51da022e1847a99463a6e289e0ba39ebe94bd0a91756d583e \
  --protocol-note notes/E33.1-IOI-SCOPE-PROTOCOL.md \
  --model-snapshot /absolute/path/to/gpt2-small \
  --output /absolute/path/to/new-e33-output \
  --confirm-run
```

If the new labels are byte-identical to the sealed labels, the registered
scoring-repair workflow can reproduce the final score. See `repro/e33_1/README.md`
for the exact command and repair history.

## Manuscript and figures

```bash
make figures
make paper
```

The manuscript uses the official Springer Nature LaTeX class. Its figures are
generated only from sealed score artifacts.

