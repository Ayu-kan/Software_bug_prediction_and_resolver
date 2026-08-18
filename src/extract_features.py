"""
src/extract_features.py
--------------------
Upgraded extract_features.py incorporating:
- Multi-language static code parsing (Phase 4)
- File filtering & prioritization (Phase 3)
- Dependency graph calculation (Phase 5)
- Architecture layer classification (Phase 6)
- Hybrid Risk formulation (Phase 7)
"""

import sys
import os
import re
import csv
from datetime import timedelta
from collections import defaultdict

from pydriller import Repository

# Add local path imports
sys.path.insert(0, os.path.dirname(__file__))
from filtering.file_filter import is_ignored_path, calculate_file_importance
from analyzers.base_analyzer import MultiLanguageAnalyzer
from graph.dependency_graph import DependencyGraphBuilder, detect_architecture_role

BUG_KEYWORDS = re.compile(
    r"\b(fix|fixes|fixed|bug|bugs|error|errors|issue|issues|defect|defects|"
    r"crash|crashes|patch|resolve|resolves|resolved|fault|faulty|broken|"
    r"regression|hotfix)\b",
    re.IGNORECASE,
)

RECENT_WINDOW_DAYS = 90


def is_bug_fixing_commit(message: str) -> bool:
    return bool(BUG_KEYWORDS.search(message or ""))


def extract(repo_path: str, output_csv: str, cutoff_ratio: float = 0.8):
    print(f"Reading full commit history from: {repo_path}")
    commits = list(Repository(repo_path).traverse_commits())
    if len(commits) < 10:
        print("Warning: very few commits found - dataset will be small/unreliable.")

    cutoff_index = max(1, int(len(commits) * cutoff_ratio))
    cutoff_date = commits[cutoff_index - 1].committer_date
    recent_start = cutoff_date - timedelta(days=RECENT_WINDOW_DAYS)

    # Per-file running stats
    stats = defaultdict(lambda: {
        "commit_count": 0,
        "developers": set(),
        "lines_added": 0,
        "lines_deleted": 0,
        "recent_commit_count": 0,
        "last_modified": None,
        "previous_bug_count": 0,
        "last_source_code": "",
    })

    files_touched_after_cutoff_by_bug = set()

    for idx, commit in enumerate(commits):
        before_cutoff = idx < cutoff_index
        is_bug = is_bug_fixing_commit(commit.msg)

        try:
            mods = commit.modified_files
        except Exception as err:
            print(f"Warning: Skipping modified files extraction for commit {commit.hash[:7]}: {err}")
            continue

        for mod in mods:
            filepath = mod.new_path or mod.old_path
            if filepath is None or is_ignored_path(filepath):
                continue

            if before_cutoff:
                s = stats[filepath]
                s["commit_count"] += 1
                s["developers"].add(commit.author.email or commit.author.name)
                try:
                    s["lines_added"] += mod.added_lines or 0
                    s["lines_deleted"] += mod.deleted_lines or 0
                except Exception:
                    pass
                s["last_modified"] = commit.committer_date
                if commit.committer_date >= recent_start:
                    s["recent_commit_count"] += 1
                if is_bug:
                    s["previous_bug_count"] += 1
                try:
                    if mod.source_code:
                        s["last_source_code"] = mod.source_code
                except Exception:
                    pass
            else:
                if is_bug:
                    files_touched_after_cutoff_by_bug.add(filepath)

    analyzer = MultiLanguageAnalyzer()
    file_sources = {f: s["last_source_code"] for f, s in stats.items()}
    
    # Graph metrics
    graph_builder = DependencyGraphBuilder()
    graph_builder.build_graph(file_sources)
    graph_metrics = graph_builder.calculate_graph_metrics()

    rows = []
    for filepath, s in stats.items():
        if not s["last_modified"]:
            continue

        # Static Code Metrics
        metrics = analyzer.analyze(s["last_source_code"], filepath)
        days_since_last_change = (cutoff_date - s["last_modified"]).days
        code_churn = s["lines_added"] + s["lines_deleted"]
        future_bug = 1 if filepath in files_touched_after_cutoff_by_bug else 0

        # Architecture & Importance
        arch_info = detect_architecture_role(filepath, s["last_source_code"])
        is_supported, importance = calculate_file_importance(filepath)
        g_meta = graph_metrics.get(filepath, {"fan_in": 0, "fan_out": 0, "betweenness": 0, "dependency_risk": 0})

        row = {
            "file": filepath,
            "loc": metrics["loc"],
            "complexity": metrics["complexity"],
            "function_count": metrics["function_count"],
            "avg_function_size": metrics["avg_function_size"],
            "max_function_size": metrics["max_function_size"],
            "dependency_count": metrics["dependency_count"],
            "commit_count": s["commit_count"],
            "developer_count": len(s["developers"]),
            "lines_added": s["lines_added"],
            "lines_deleted": s["lines_deleted"],
            "code_churn": code_churn,
            "recent_commit_count": s["recent_commit_count"],
            "days_since_last_change": days_since_last_change,
            "previous_bug_count": s["previous_bug_count"],
            "fan_in": g_meta["fan_in"],
            "fan_out": g_meta["fan_out"],
            "dependency_risk": g_meta["dependency_risk"],
            "architecture_role": arch_info["architecture_role"],
            "architecture_risk": arch_info["architecture_risk"],
            "file_importance": importance,
            "future_bug": future_bug,
        }
        rows.append(row)

    if not rows:
        print("No valid code feature rows produced.")
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Features extracted successfully to: {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_features.py <repo_path_or_url> <output_csv> [cutoff_ratio]")
        sys.exit(1)
    ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    extract(sys.argv[1], sys.argv[2], ratio)
