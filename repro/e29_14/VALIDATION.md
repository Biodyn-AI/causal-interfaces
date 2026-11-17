# Release validation — 2026-11-16

The release wrapper was tested from the current checkout with Python 3.12.15,
PyTorch 2.13.0, Transformers 5.14.1, Safetensors 0.8.0, and macOS 27.0.1.

## Saved-label replay

`replay_scores.py` hash-verified and rescored E29.14a, E29.14b, E29.14c, and
E29.14d. Every replayed scientific field exactly matched its published JSON
summary.

## Fresh model inference

Two independent inference-to-score smoke runs used the hash-pinned GPT-2
Medium checkpoint and wrote results outside the repository:

| Panel | Runtime | Scientific-score comparison |
|---|---:|---|
| E29.14a | 19.94 s | exact match to the published scientific fields |
| E29.14b | 111.65 s | exact match to the published scientific fields |

These runs used the portable wrapper rather than the historical custody CLI.
They therefore test model inference, intervention execution, and scoring across
an OS/Python patch-version change without claiming a new prospective seal.

## Tests

The focused E29.14 release and protocol suite completed with `146 passed`.
The portable CLI was additionally invoked from `/tmp`, outside the repository,
and resolved all repository inputs through `--repository-root`.

