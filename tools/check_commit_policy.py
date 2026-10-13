"""Reject commits that add a co-author trailer or use an author outside the allow-list.

The check reads each commit in a revision range. A commit fails when its message has a
``Co-authored-by`` trailer, or when its author email is not on the allow-list. The committer
is not checked, because GitHub sets it to its own address when it merges a pull request.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

ALLOWED_AUTHOR_EMAILS = frozenset(
    {
        "92649783+landtml@users.noreply.github.com",
        "landtml@users.noreply.github.com",
    }
)

_RECORD_END = "\x1e"
_FIELD_END = "\x1f"
_CO_AUTHOR_TRAILER = re.compile(r"^\s*co-authored-by\s*:", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Commit:
    sha: str
    author_email: str
    message: str


def parse_log(text: str) -> list[Commit]:
    """Parse the output of ``git log`` written with the separators defined above."""
    commits = []
    for record in text.split(_RECORD_END):
        record = record.strip("\n")
        if not record:
            continue
        sha, author_email, message = record.split(_FIELD_END, 2)
        commits.append(Commit(sha.strip(), author_email.strip(), message))
    return commits


def find_violations(commits: list[Commit]) -> list[str]:
    """Return one line per policy violation, in commit order."""
    violations = []
    for commit in commits:
        if _CO_AUTHOR_TRAILER.search(commit.message):
            violations.append(f"{commit.sha[:12]}: co-authored-by trailer")
        if commit.author_email.lower() not in ALLOWED_AUTHOR_EMAILS:
            violations.append(f"{commit.sha[:12]}: author email is not on the allow-list")
    return violations


def read_range(revision_range: str) -> list[Commit]:
    fmt = f"%H{_FIELD_END}%ae{_FIELD_END}%B{_RECORD_END}"
    result = subprocess.run(
        ["git", "log", f"--format={fmt}", revision_range],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_log(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--range", required=True, help="revision range, for example base..head")
    args = parser.parse_args(argv)

    commits = read_range(args.range)
    violations = find_violations(commits)
    for line in violations:
        print(line)
    if violations:
        print(f"commit policy: {len(violations)} violation(s) in {len(commits)} commit(s)")
        return 1
    print(f"commit policy: no violations in {len(commits)} commit(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
