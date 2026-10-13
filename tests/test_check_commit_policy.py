import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_commit_policy", ROOT / "tools" / "check_commit_policy.py"
)
policy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = policy
_spec.loader.exec_module(policy)

ALLOWED = "92649783+landtml@users.noreply.github.com"


def _commit(message: str, email: str = ALLOWED, sha: str = "a" * 40) -> policy.Commit:
    return policy.Commit(sha=sha, author_email=email, message=message)


def test_commit_from_the_account_passes():
    assert policy.find_violations([_commit("Add a parser\n\nWhy it changed.")]) == []


@pytest.mark.parametrize(
    "trailer",
    [
        "Co-Authored-By: Someone <someone@example.com>",
        "co-authored-by: Someone <someone@example.com>",
        "CO-AUTHORED-BY : Someone <someone@example.com>",
    ],
)
def test_co_author_trailer_fails_whatever_the_name(trailer):
    violations = policy.find_violations([_commit(f"Fix parser\n\n{trailer}\n")])
    assert violations == [f"{'a' * 12}: co-authored-by trailer"]


def test_text_that_only_mentions_co_authoring_passes():
    message = "Document how co-authored-by lines are checked"
    assert policy.find_violations([_commit(message)]) == []


def test_author_outside_the_allow_list_fails():
    violations = policy.find_violations([_commit("Fix parser", email="someone@example.com")])
    assert violations == [f"{'a' * 12}: author email is not on the allow-list"]


def test_the_older_noreply_address_is_allowed():
    email = "landtml@users.noreply.github.com"
    assert policy.find_violations([_commit("Fix parser", email=email)]) == []


def test_author_email_comparison_ignores_case():
    email = ALLOWED.upper()
    assert policy.find_violations([_commit("Fix parser", email=email)]) == []


def test_each_violation_is_reported_in_commit_order():
    commits = [
        _commit("First\n\nCo-authored-by: X <x@example.com>", sha="1" * 40),
        _commit("Second", email="y@example.com", sha="2" * 40),
    ]
    assert policy.find_violations(commits) == [
        f"{'1' * 12}: co-authored-by trailer",
        f"{'2' * 12}: author email is not on the allow-list",
    ]


def test_parse_log_reads_every_record():
    fs, rs = policy._FIELD_END, policy._RECORD_END
    text = f"{'a' * 40}{fs}{ALLOWED}{fs}First line\n\nBody\n{rs}\n{'b' * 40}{fs}x@y.z{fs}Second{rs}"
    commits = policy.parse_log(text)
    assert [c.sha for c in commits] == ["a" * 40, "b" * 40]
    assert commits[0].message == "First line\n\nBody"
    assert commits[1].author_email == "x@y.z"


def _git(repo: Path, *args: str, email: str = ALLOWED) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    result = subprocess.run(
        ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
    )
    return result.stdout


def _repo_with_commits(tmp_path: Path, commits: list[tuple[str, str]]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "commit.gpgsign", "false")
    for index, (message, email) in enumerate(commits):
        (repo / f"f{index}.txt").write_text(str(index), encoding="utf-8")
        _git(repo, "add", ".")
        _git(repo, "commit", "-q", "-m", message, email=email)
    return repo


def test_main_passes_a_range_of_allowed_commits(tmp_path, capsys):
    repo = _repo_with_commits(tmp_path, [("Start", ALLOWED), ("Add file", ALLOWED)])
    cwd = Path.cwd()
    os.chdir(repo)
    try:
        assert policy.main(["--range", "HEAD~1..HEAD"]) == 0
    finally:
        os.chdir(cwd)
    assert "no violations in 1 commit(s)" in capsys.readouterr().out


def test_main_fails_a_range_with_a_co_author_trailer(tmp_path, capsys):
    repo = _repo_with_commits(
        tmp_path,
        [
            ("Start", ALLOWED),
            ("Add file\n\nCo-Authored-By: Someone <someone@example.com>", ALLOWED),
        ],
    )
    cwd = Path.cwd()
    os.chdir(repo)
    try:
        assert policy.main(["--range", "HEAD~1..HEAD"]) == 1
    finally:
        os.chdir(cwd)
    out = capsys.readouterr().out
    assert "co-authored-by trailer" in out
    assert "1 violation(s) in 1 commit(s)" in out
