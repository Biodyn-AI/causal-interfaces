# E29.14 publication reproduction

This directory provides two release paths while leaving the original sealed
registration, acquisition, custody, label, and score files byte-for-byte
unchanged.

1. **Score replay** is portable and makes no model calls. It verifies every
   registration, label, protocol, and published-summary digest, invokes the
   original pure scientific scorers, and requires exact agreement with every
   published scientific field.
2. **Independent inference reproduction** verifies the pinned GPT-2 Medium
   checkpoint, reacquires the frozen panels on CPU in float64, and writes new
   labels and scores. It does not claim to extend the historical prospective
   custody chain.

## Environment

From the repository root:

```bash
python3.12 -m venv .venv-paper
.venv-paper/bin/python -m pip install --upgrade pip
.venv-paper/bin/python -m pip install -r repro/e29_14/requirements.txt
```

The tested environment is Python 3.12.15. The historical acquisition used
Python 3.12.9 and macOS 26.3; those exact platform values remain recorded in
the sealed registrations but are intentionally not required for score replay.

## Portable score replay

```bash
.venv-paper/bin/python repro/e29_14/replay_scores.py \
  --repository-root "$PWD" \
  --output-directory /tmp/e29_14_replayed_scores
```

Success prints
`"all_replays_match_published_scientific_fields": true`. The optional output
directory contains newly serialized scientific scores and never overwrites a
sealed artifact.

## Independent inference-to-score reproduction

Download the exact checkpoint revision into any local directory:

```bash
hf download openai-community/gpt2-medium \
  --revision 6dcaa7a952f72f9298047fd5137cd6e4f05f41da \
  --local-dir /tmp/gpt2-medium-e29
```

Then reproduce all four panels:

```bash
.venv-paper/bin/python -m repro.e29_14.reproduce_inference \
  --repository-root "$PWD" \
  --model-snapshot /tmp/gpt2-medium-e29 \
  --output-directory /tmp/e29_14_independent_reproduction
```

Use `--experiments a` for the shortest headline check. The full run acquires
the same logical panels as E29.14a--d and can take substantially longer. The
output directory must be absent or empty; reruns therefore cannot silently
overwrite evidence.

## Integrity boundary

- `manifest.json` pins every publication input and the complete checkpoint.
- The portable scorer validates scientific labels and scoring, not the
  historical absolute filesystem paths.
- The independent run validates the scientific protocol and checkpoint, but
  records its own runtime environment and explicitly sets
  `historical_custody_extended` to `false`.
- The original registrations remain the source of truth for which analyses
  were prospective.

