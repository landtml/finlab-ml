"""Print the release notes for a version, for the release workflow.

Uses .github/releases/v<version>.md when it exists. Otherwise uses the
CHANGELOG.md section for the version. Fails if CHANGELOG.md has no section for
the version, so a version bump cannot be released without a changelog entry.

Usage: python .github/scripts/release_notes.py <version>
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/landtml/finlab-ml"


def changelog_section(text: str, version: str) -> str | None:
    heading = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n", re.MULTILINE)
    match = heading.search(text)
    if match is None:
        return None
    # The section ends at the next version heading or at the link references.
    end = re.compile(r"^(## \[|\[[^\]]+\]: )", re.MULTILINE).search(text, match.end())
    return text[match.end() : end.start() if end else len(text)].strip()


def main() -> int:
    version = sys.argv[1]
    section = changelog_section((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version)
    if not section:
        print(f"CHANGELOG.md has no '## [{version}]' section", file=sys.stderr)
        return 1

    notes = ROOT / ".github" / "releases" / f"v{version}.md"
    if notes.exists():
        print(notes.read_text(encoding="utf-8").strip())
        return 0

    print(
        f"## Install\n\n```bash\n"
        f'pip install "git+{REPO}@v{version}"\n'
        f"```\n\n{section}\n\n"
        f"The full changelog is in [CHANGELOG.md]({REPO}/blob/v{version}/CHANGELOG.md)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
