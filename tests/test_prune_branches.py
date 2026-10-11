"""Tests for tools/prune_branches.py branch selection (no GitHub calls)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "prune_branches.py"
_spec = importlib.util.spec_from_file_location("prune_branches", TOOL)
prune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prune)


def _pr(number, head, state="closed", repo="landtml/finlab-ml"):
    return {"number": number, "state": state, "head": {"ref": head, "repo": {"full_name": repo}}}


def test_protected_branches_are_never_deleted():
    delete, keep = prune.plan_prune(["main", "release-0.1.2"], [_pr(1, "main"), _pr(2, "release-0.1.2")])
    assert delete == []
    assert keep == {"main": "protected", "release-0.1.2": "protected"}


def test_branch_with_only_closed_prs_is_deleted():
    delete, _ = prune.plan_prune(["feat-a"], [_pr(3, "feat-a"), _pr(4, "feat-a")])
    assert delete == ["feat-a"]


def test_branch_with_an_open_pr_is_kept():
    delete, keep = prune.plan_prune(["feat-b"], [_pr(5, "feat-b", state="closed"), _pr(6, "feat-b", state="open")])
    assert delete == []
    assert keep == {"feat-b": "open pull request"}


def test_branch_without_any_pr_is_kept():
    delete, keep = prune.plan_prune(["scratch"], [])
    assert delete == []
    assert keep == {"scratch": "no pull request"}


def test_closed_pr_deletes_its_same_repo_head():
    pr = _pr(7, "feat-c")
    assert prune.plan_event(pr, "landtml/finlab-ml", [pr]) == "feat-c"


def test_closed_pr_keeps_fork_head_and_protected_head():
    fork = _pr(8, "feat-d", repo="someone/fork")
    assert prune.plan_event(fork, "landtml/finlab-ml", [fork]) is None
    main_pr = _pr(9, "main")
    assert prune.plan_event(main_pr, "landtml/finlab-ml", [main_pr]) is None


def test_closed_pr_keeps_branch_still_used_by_another_open_pr():
    closing = _pr(10, "feat-e")
    other = _pr(11, "feat-e", state="open")
    assert prune.plan_event(closing, "landtml/finlab-ml", [closing, other]) is None
