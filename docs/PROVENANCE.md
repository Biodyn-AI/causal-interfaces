# Provenance and integrity boundaries

The prospective registrations and saved outputs are preserved exactly as they
were written. Their SHA-256 digests are checked by the replay tools.

Several registration files contain absolute paths from the original acquisition
machine. These strings are inert metadata, not active release paths. They are
left unchanged because sanitizing them would alter registered hashes. Portable
reproduction uses the relative paths in `repro/e29_14/manifest.json` and
`repro/e33_1/manifest.json`.

The default test command excludes 11 tests marked `original_checkout`. Those
tests reconstruct registrations or custody objects whose expected values
intentionally include the original absolute paths; they therefore pass only in
the acquisition checkout. The release runs 143 portable tests instead. This is
a portability boundary, not a waiver of scientific checks: the replay tools
verify every sealed publication artifact and recompute all reported scores.

The repository distinguishes three operations:

1. **Saved-output replay** verifies hashes and recomputes the scientific
   summaries without model calls. Both gates are exact for structure and
   non-floating values and use `1e-12` absolute and relative tolerances for
   floats.
2. **Fresh inference** downloads the pinned checkpoint and creates new output
   in an empty directory. It tests scientific portability but does not extend
   the historical custody chain.
3. **New analysis** is outside the sealed study and must write new files with a
   new registration or an explicit post-hoc label.

Transparent repairs are retained rather than hidden. E29.14c V1 was superseded
before the model ran because of a metadata link. E33.1 V1 was superseded before
the model ran because of a JSON round-trip issue; a later registered adapter
restored registered key order after labels were sealed. Neither repair changed
a scientific value, condition, threshold, or calculation.

