# Contributing

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,plot,notebooks]"
```

## Checks

Run these before opening a pull request. CI runs the same commands.

```bash
ruff check src tests tools benchmarks examples
pytest -q
```

## Pull requests

- Work on a branch and open a pull request into `main`. Direct pushes to `main` are refused.
- Fill in the pull request template: the change, the verification, and whether it needs a changelog line.
- Commits must be authored from an address on the repository's allow-list, and must not carry a `Co-authored-by` trailer. The `commit-policy` check enforces both on every pull request.
- A change that alters user-visible behaviour adds a line under `[Unreleased]` in `CHANGELOG.md`.

## Releases

See [RELEASING.md](RELEASING.md). A merge to `main` that changes the version in `pyproject.toml` is tagged and published by the release workflow.

## Documentation

Documentation lives in `docs/`, in the README, and in docstrings. Each claim in `docs/proofs/` either names the test that checks it or says how it is known.
