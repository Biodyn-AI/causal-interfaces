# Causal interfaces across transformer scales

This repository contains the code, prospective registrations, saved model outputs,
tests, figures, and manuscript sources for:

> Ihor Kendiukhov. *Patching only changed tokens can match patching all
> tokens across dozens of transformer components.* Prepared for submission to
> *Scientific Reports*.

The paper tests a simple question: if an activation patch is restricted to the
token positions where two prompts differ, does it retain the effect of patching
every position when the intervention expands across much of a transformer?
The answer is yes in two controlled settings: GPT-2 Medium sequence completion
and GPT-2 Small indirect-object identification. The result is deliberately
narrow; it is not evidence of universality, a minimal circuit, or a natural
computation path.

## Reproduce the published scores without model downloads

The lightweight path verifies every registered input hash and recomputes the
paper's scientific summaries from the saved per-example outputs.

```bash
git clone https://github.com/Biodyn-AI/causal-interfaces.git
cd causal-interfaces

make setup
make replay
```

Successful output includes:

```text
"all_replays_match_published_scientific_fields": true
"all_artifact_hashes_match": true
"saved_label_score_match_within_declared_tolerance": true
```

Run the portable publication suite (145 selected tests) with:

```bash
make test
```

The test suite and score replay make no network requests and no model calls.
Both saved-label replays require exact structure and exact non-floating values;
floating-point values must agree within absolute and relative tolerances of
`1e-12`. In a clean clone without a local GPT-2 Medium tokenizer snapshot,
Pytest reports `142 passed, 3 skipped, 11 deselected`. The three skipped tests
check prompt-token alignment against the pinned tokenizer; all 145 selected
tests pass when that optional snapshot is locally available. Eleven additional
historical custody tests are marked `original_checkout`
because their sealed expected values contain immutable absolute paths from the
acquisition machine. They passed in that original checkout and are excluded in
a clone; see [docs/PROVENANCE.md](docs/PROVENANCE.md).

The make targets remove macOS AppleDouble (`._*`) sidecar files from the
disposable `.venv` before imports. These metadata files can otherwise appear on
external drives and be mistaken for Python source files. Repository source and
evidence files are never removed.

## Regenerate the figures

```bash
make figures
```

The script reads sealed score JSON files rather than reimplementing the
scientific analysis. It writes vector PDF and high-resolution PNG files to
`paper/figures/`.

## Repeat inference from the pinned checkpoints

Fresh inference is optional and substantially heavier. The exact model
revisions are:

- GPT-2 Medium: `openai-community/gpt2-medium` at
  `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`
- GPT-2 Small: `gpt2` at
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for checkpoint download,
fresh-inference commands, expected runtimes, and the boundary between exact
score replay and a new model run.

## Repository map

```text
paper/          Scientific Reports manuscript, supplement, and figures
experiments/    Registered acquisition, protocol, and scoring code
repro/          Portable score replay and optional fresh inference
notes/          Prospective registrations, protocols, results, and claim freeze
runs/           Sealed labels, scores, masks, custody, and attempt records
tests/          Focused publication test suite
docs/           Reproduction, artifact, and provenance documentation
```

The machine-readable map of publication artifacts is documented in
[docs/ARTIFACTS.md](docs/ARTIFACTS.md). Historical registrations are immutable;
some contain inert absolute paths from the acquisition machine. The portable
manifests use repository-relative paths. See
[docs/PROVENANCE.md](docs/PROVENANCE.md).

The manuscript bibliography contains 45 contextually cited references. Every
entry was checked against a primary online record; the dated audit and URLs are
in [paper/REFERENCE_VALIDATION.md](paper/REFERENCE_VALIDATION.md). The focused
reference test also prevents uncited entries, missing URLs, or fewer than 40
references.

## Requirements

- Python 3.12
- CPU is sufficient for saved-output replay and tests
- Fresh inference requires downloading the two GPT-2 checkpoints
- `pdflatex` and `latexmk` are required only to rebuild the manuscript

The release was validated with Python 3.12.15, PyTorch 2.13.0,
Transformers 5.14.1, NumPy 2.5.1, and Matplotlib 3.11.1.

## Licensing

Original code is licensed under the MIT License. Original saved outputs,
figures, manuscript text, and research documentation are licensed under
CC BY 4.0. Third-party packages and model checkpoints retain their own terms;
the checkpoints are not redistributed.

## Citation

Use `CITATION.cff` for machine-readable citation metadata. Until a journal DOI
is assigned, cite the versioned GitHub release and the manuscript title above.
