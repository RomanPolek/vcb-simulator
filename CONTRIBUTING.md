# Contributing

Install the dependencies in `requirements.txt` and put GCC on PATH. Run from
the repository root:

```sh
python -m unittest discover -s tests
```

Keep changes focused on the supported native engine and viewer. For logic or
runtime changes, add a small circuit regression using the ASCII helpers in
`tests/logic_test_harness.py`, and compare native results with the Python
reference where applicable. For renderer changes, run the renderer tests and
check the viewer interactively.

Describe the behavior being changed and the checks you ran. Include a minimal
reproducer with bug reports. Keep generated binaries, caches, private projects,
and benchmark output out of contributions. Only contribute material you have
permission to distribute under the repository's MIT license.

## Pull requests and merging

Fork the repository and open a pull request against `main`. CI runs on Linux
and Windows with Python 3.11 and 3.13. Changes must pass the required checks,
be up to date with `main`, and have review conversations resolved before merging.
The maintainer, @RomanPolek, handles merges. Pull requests use squash merges;
direct pushes, force pushes, and deletion of `main` are blocked.

Approval counts are not required so the solo maintainer can merge their own
pull requests after CI passes. CODEOWNERS requests the maintainer's review on
contributions. Please keep discussions constructive and focused on the work.
