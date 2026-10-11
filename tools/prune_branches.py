"""Delete merged or abandoned feature branches in this repository.

Two modes:

* ``event``: a pull request was closed. Delete its head branch when the branch
  lives in this repository, is not protected, and no other open PR uses it.
* ``prune``: sweep every branch whose PRs are all closed. A branch with no PR
  at all is kept, because nothing shows that it was abandoned.

Protected branches are ``main`` and any ``release-*`` branch.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from typing import Any

PROTECTED = ("main", "release-*")


def is_protected(branch: str) -> bool:
    return any(fnmatch.fnmatchcase(branch, pattern) for pattern in PROTECTED)


def plan_prune(branches: list[str], prs: list[dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    """Return (branches to delete, {branch: reason it is kept}) for the prune sweep."""
    by_head: dict[str, list[dict[str, Any]]] = {}
    for pr in prs:
        by_head.setdefault(pr["head"]["ref"], []).append(pr)

    delete: list[str] = []
    keep: dict[str, str] = {}
    for branch in branches:
        if is_protected(branch):
            keep[branch] = "protected"
            continue
        head_prs = by_head.get(branch, [])
        if not head_prs:
            keep[branch] = "no pull request"
        elif any(pr["state"] == "open" for pr in head_prs):
            keep[branch] = "open pull request"
        else:
            delete.append(branch)
    return delete, keep


def plan_event(pr: dict[str, Any], repo: str, prs: list[dict[str, Any]]) -> str | None:
    """Return the head branch to delete after ``pr`` closed, or None to keep it."""
    head = pr["head"]
    branch = head["ref"]
    if head["repo"]["full_name"] != repo or is_protected(branch):
        return None
    still_open = [
        other for other in prs
        if other["head"]["ref"] == branch and other["state"] == "open" and other["number"] != pr["number"]
    ]
    return None if still_open else branch


def _gh_json(path: str) -> list[Any]:
    out = subprocess.run(
        ["gh", "api", "--paginate", path],
        check=True, capture_output=True, text=True,
    ).stdout
    # --paginate concatenates one JSON array per page.
    decoder = json.JSONDecoder()
    items: list[Any] = []
    pos = 0
    while pos < len(out):
        if out[pos].isspace():
            pos += 1
            continue
        value, pos = decoder.raw_decode(out, pos)
        items.extend(value if isinstance(value, list) else [value])
    return items


def _delete(repo: str, branch: str, dry_run: bool) -> None:
    if dry_run:
        print(f"dry run: would delete {branch}")
        return
    subprocess.run(
        ["gh", "api", "-X", "DELETE", f"repos/{repo}/git/refs/heads/{branch}"],
        check=True,
    )
    print(f"deleted {branch}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["event", "prune"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo = os.environ["REPO"]
    prs = _gh_json(f"repos/{repo}/pulls?state=all&per_page=100")

    if args.mode == "event":
        with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as fh:
            pr = json.load(fh)["pull_request"]
        branch = plan_event(pr, repo, prs)
        if branch is None:
            print("keep: branch is protected, external, or used by another open PR")
        else:
            _delete(repo, branch, args.dry_run)
        return 0

    branches = [b["name"] for b in _gh_json(f"repos/{repo}/branches?per_page=100")]
    delete, keep = plan_prune(branches, prs)
    for branch, reason in sorted(keep.items()):
        print(f"keep {branch}: {reason}")
    for branch in delete:
        _delete(repo, branch, args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
