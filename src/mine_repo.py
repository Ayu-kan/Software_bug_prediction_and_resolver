"""
mine_repo.py
------------
Step 1 of the pipeline.

Mines the full commit history of ANY git repository (local path or GitHub URL)
using PyDriller, and saves one row per (commit, modified file) into a CSV.

This script is REPO-AGNOSTIC on purpose: the hackathon evaluator may hand you
a completely different repo, so nothing here is hardcoded to a specific project.

Usage:
    python mine_repo.py <repo_path_or_url> <output_csv>

Example:
    python mine_repo.py https://github.com/psf/requests raw_commit_data.csv
    python mine_repo.py C:\\Users\\me\\Desktop\\my_repo raw_commit_data.csv
"""

import sys
import re
import csv
from pydriller import Repository

# Heuristic keyword list used to guess whether a commit is a bug-fix.
# This is documented as a heuristic on purpose (see analysis doc, Problem 7.1) -
# commit messages don't always say "bug" even when they fix one, so this is an
# approximation, not ground truth.
BUG_KEYWORDS = re.compile(
    r"\b(fix|fixes|fixed|bug|bugs|error|errors|issue|issues|defect|defects|"
    r"crash|crashes|patch|resolve|resolves|resolved|fault|faulty|broken|"
    r"regression|hotfix)\b",
    re.IGNORECASE,
)


def is_bug_fixing_commit(message: str) -> bool:
    return bool(BUG_KEYWORDS.search(message or ""))


def mine(repo_path: str, output_csv: str):
    rows = []
    print(f"Mining commit history from: {repo_path}")
    print("This can take a while for large repos...")

    commit_count = 0
    for commit in Repository(repo_path).traverse_commits():
        commit_count += 1
        is_bug = is_bug_fixing_commit(commit.msg)

        for mod in commit.modified_files:
            filepath = mod.new_path or mod.old_path
            if filepath is None:
                continue
            rows.append(
                {
                    "commit_hash": commit.hash,
                    "commit_date": commit.committer_date.isoformat(),
                    "author": commit.author.name,
                    "file": filepath,
                    "lines_added": mod.added_lines,
                    "lines_deleted": mod.deleted_lines,
                    "is_bug_fix": int(is_bug),
                    "message": (commit.msg or "").replace("\n", " ")[:200],
                }
            )

        if commit_count % 200 == 0:
            print(f"  ...processed {commit_count} commits")

    if not rows:
        print("No commits found. Check the repo path/URL.")
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {commit_count} commits, {len(rows)} file-change rows -> {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mine_repo.py <repo_path_or_url> <output_csv>")
        sys.exit(1)
    mine(sys.argv[1], sys.argv[2])
