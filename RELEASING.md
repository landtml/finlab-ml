# Releasing

Releases are published by `.github/workflows/release.yml`. Nobody pushes tags by hand.

## Steps

In one pull request to `main`:

1. Set the new version in `pyproject.toml`.
2. In `CHANGELOG.md`, move the `[Unreleased]` entries under `## [X.Y.Z] - YYYY-MM-DD` and update
   the link references at the bottom.
3. Update the version in the README's install command (`@vX.Y.Z`).
4. Optional: write the release notes in `.github/releases/vX.Y.Z.md`. Without that file, the
   release uses an install line and the changelog section. `v0.1.1.md` is an example.

Merge the pull request. When CI passes on `main`, the workflow sees a version with no tag. It
then:

- tags the merge commit `vX.Y.Z`;
- builds the sdist and the wheel;
- publishes the release "finlab X.Y.Z" with both files attached.

## What the workflow checks

- It runs only after the CI workflow passes on a push to `main`.
- A version that is already tagged is skipped, so other merges to `main` release nothing.
- A version with no `CHANGELOG.md` section fails the run and publishes nothing.
- A README that still names the old version gives a warning but does not stop the release.

If a run fails, fix the cause on `main`. The next CI pass retries the release. You can also
start the workflow by hand from the Actions tab.
