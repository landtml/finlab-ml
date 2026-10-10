"""Fail when commit messages or a pull request body carry AI attribution.

Usage:
    python tools/check_attribution.py --commits <range>   # git log --format=%B <range>
    python tools/check_attribution.py --body-file <path>

Exit status: 0 when clean, 1 when a violation is found, 2 on a usage or git error.

The project's own attribution block (the ``claude-projects-attribution`` marker and
the ``_Requested by`` line that links a project thread) is allowed. Session links
(``claude.ai/code/session_``) are never allowed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

_FLAGS = re.IGNORECASE

# A Co-Authored-By trailer that names an AI vendor or model.
_CO_AUTHOR = re.compile(
    r"^\s*co-authored-by:.*\b(claude|anthropic|gpt|openai|gemini|llama|mistral|copilot)\b",
    _FLAGS,
)
_CLAUDE_SESSION = re.compile(r"claude-session", _FLAGS)
_GENERATED_WITH = re.compile(r"generated (with|by)\b.*\bclaude\b", _FLAGS)
_GENERATED_FOOTER_CLAUDE_CODE = re.compile(r"^\s*_?generated\b.*\bclaude code\b", _FLAGS)
_SESSION_LINK = re.compile(r"claude\.ai/code/session_", _FLAGS)

_ALLOWED_MARKER = "claude-projects-attribution"
_REQUESTED_BY = re.compile(r"^\s*_?requested by\b", _FLAGS)
_PROJECT_LINK = "claude.ai/code/project/"


def _is_allowed_block(line: str) -> bool:
    if _ALLOWED_MARKER in line.lower():
        return True
    return bool(_REQUESTED_BY.match(line)) and _PROJECT_LINK in line.lower()


def find_violations(text: str) -> list[str]:
    """Return the forbidden lines in ``text``, in order, each reported once."""
    violations: list[str] = []
    for line in text.splitlines():
        if _SESSION_LINK.search(line):
            flagged = True
        elif _is_allowed_block(line):
            flagged = False
        else:
            flagged = any(
                pattern.search(line)
                for pattern in (
                    _CO_AUTHOR,
                    _CLAUDE_SESSION,
                    _GENERATED_WITH,
                    _GENERATED_FOOTER_CLAUDE_CODE,
                )
            )
        if flagged:
            violations.append(line.strip())
    return violations


def _commit_messages(rev_range: str) -> str:
    result = subprocess.run(
        ["git", "log", "--format=%B", rev_range],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git log failed for {rev_range}")
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--commits", metavar="RANGE", help="git revision range to check")
    source.add_argument("--body-file", metavar="PATH", help="file holding a PR body")
    args = parser.parse_args(argv)

    try:
        if args.commits is not None:
            text = _commit_messages(args.commits)
        else:
            with open(args.body_file, encoding="utf-8") as fh:
                text = fh.read()
    except (OSError, RuntimeError) as exc:
        print(f"check_attribution: {exc}", file=sys.stderr)
        return 2

    violations = find_violations(text)
    if not violations:
        print("check_attribution: no AI attribution found")
        return 0
    print("check_attribution: AI attribution found in:")
    for line in violations:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
