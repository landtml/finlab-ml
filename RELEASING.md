# Releasing

The Release workflow (`.github/workflows/release.yml`) tags and publishes each release. It runs after the
CI workflow passes on a push to `main`.

## Release pull request

A release pull request to `main` contains:

1. The new version in `pyproject.toml`.
2. A `## [X.Y.Z] - YYYY-MM-DD` section in `CHANGELOG.md` with the former `[Unreleased]` entries, and the
   updated link references at the end of the file.
3. The new version in the README install command (`@vX.Y.Z`).
4. Optional release notes in `.github/releases/vX.Y.Z.md`, in the format of `v0.1.1.md`.

## Workflow

When the version in `pyproject.toml` has no tag, the workflow:

- tags the commit `vX.Y.Z`;
- builds the sdist and the wheel;
- publishes the release "finlab X.Y.Z" with both files attached.

The release notes come from `.github/releases/vX.Y.Z.md`. Without that file, they are an install line
followed by the `CHANGELOG.md` section.

A version that already has a tag is skipped, unless its GitHub release exists but lacks the sdist or the
wheel. In that case the run builds both from the tag and uploads them to that release, replacing any file
with the same name; no tag is created. A version with no `CHANGELOG.md` section fails the run and
publishes nothing. A README that names another version raises a warning. A failed run is retried by the
next passing CI run on `main`, or from the Actions tab.
