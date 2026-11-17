# E33.1 cross-model and cross-task replication: reproducibility

This bundle verifies and replays the paper's GPT-2 Small IOI
replication. The scientific claim is deliberately narrow: on the fixed,
unfiltered 16-context panel, patching only the three changed name-token
positions reproduced the causal effect of all-position patching. It does not
claim model universality, a minimal circuit, or an identified abstract variable.

## Exact environment

The sealed run used Python 3.12.15, PyTorch 2.13.0, Transformers 5.14.1,
CPU float64, eager attention, and GPT-2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The active registration contains SHA-256 digests for every required checkpoint
file. A checkpoint snapshot supplied to the acquisition command is rejected if
any byte differs.

## Verify custody and replay the saved-label score

```bash
.venv/bin/python repro/e33_1/replay_score.py
```

This verifies the registration, scoring addendum, sealed labels, final score,
protocol note, and results note against `manifest.json`. It then runs the
registered calculations over the saved per-context labels and requires exact
equality with every field of the published score. It makes no model calls and
writes no files.

Expected terminal fields are:

```json
{
  "all_artifact_hashes_match": true,
  "all_registered_gates_passed": true,
  "exact_saved_label_score_match": true
}
```

## Repeat fresh inference

Set explicit paths to the pinned Hugging Face snapshot and a new, empty output
directory. Do not point `E33_OUTPUT` at the published run: acquisition is
write-once and intentionally refuses to overwrite it.

```bash
export E33_MODEL_SNAPSHOT=/absolute/path/to/gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
export E33_OUTPUT=/absolute/path/to/new-empty-e33-output

.venv/bin/python -m experiments.e33_ioi_scope_replication.workflow acquire \
  --registration notes/E33.1-IOI-SCOPE-REGISTRATION-V2.json \
  --registration-sha256 39a5a589678c76d51da022e1847a99463a6e289e0ba39ebe94bd0a91756d583e \
  --protocol-note notes/E33.1-IOI-SCOPE-PROTOCOL.md \
  --model-snapshot "$E33_MODEL_SNAPSHOT" \
  --output "$E33_OUTPUT" \
  --confirm-run
```

The registered acquisition performs 384 logical model examples: 32 clean and
352 patched. On the sealed CPU configuration it took several minutes. If the
fresh `labels.json` is byte-identical to the sealed labels, replay the registered
post-acquisition scoring repair with:

```bash
.venv/bin/python -m experiments.e33_ioi_scope_replication.score_repair_workflow score \
  --addendum notes/E33.1-SCORING-REPAIR-REGISTRATION.json \
  --addendum-sha256 ed2a268cddadf61d48905f3d665b22b5267375c36047a941c6a9255519ae9e9c \
  --output "$E33_OUTPUT" \
  --confirm-run
```

The addendum deliberately requires the sealed labels hash. This makes it a
strict exact-reproduction check rather than a tolerant cross-platform scoring
entry point.

## Transparent serialization repairs

Registration V1 was superseded before output creation, attempt consumption,
model load, or response because a tuple/list JSON round-trip check failed. V2
changed no prompt, condition, or gate. After labels were sealed, the original
scorer stopped on alphabetically sorted JSON dictionary keys before reading any
scientific value. The separately registered, outcome-blind adapter restores the
already registered condition order and delegates all calculations and gates to
the unchanged scorer. It changed no value, condition, threshold, or scientific
calculation.

See `notes/E33.1-IOI-SCOPE-PROTOCOL.md` for the design and
`notes/E33.1-IOI-SCOPE-RESULTS.md` for the complete result table.
