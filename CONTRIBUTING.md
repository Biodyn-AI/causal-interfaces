# Contributing

This repository is the versioned research record for a submitted manuscript.
The sealed registrations and saved evidence under `notes/` and `runs/` must not
be edited because their exact hashes are part of the scientific provenance.

Bug reports and reproduction questions are welcome through GitHub Issues.
Proposed code changes should include a focused test and must not overwrite or
silently reinterpret sealed artifacts. New analyses should write to new paths
and state clearly whether they are prospective, exploratory, or post hoc.

Before opening a pull request, run:

```bash
make replay
make test
make figures
```

