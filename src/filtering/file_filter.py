"""
src/filtering/file_filter.py
----------------------------
Phase 3 Task 7 & Task 8: File Filtering and Prioritization.
"""

import os
from typing import Tuple

IGNORED_EXTENSIONS = {
    ".md", ".txt", ".log", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".mp4", ".mp3", ".wav", ".exe",
    ".dll", ".so", ".dylib", ".pyc", ".pyo", ".class", ".o", ".obj",
    ".csv", ".tsv", ".parquet", ".db", ".sqlite", ".bin",
    ".json", ".yaml", ".yml", ".toml", ".rst", ".xml", ".properties"
}

IGNORED_PATTERNS = [
    "env", ".env", "env.example", "env_example", ".env.example", ".env.local",
    "dockerfile", "makefile", "license", "changelog", ".gitignore", ".dockerignore",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"
]

IGNORED_DIRS = {
    "node_modules", ".venv", "venv", "env", "__pycache__", ".git", ".idea",
    ".vscode", "build", "dist", "target", "bin", "obj", ".pytest_cache", ".mypy_cache",
    ".github", "docs", "documentation"
}

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
}

HIGH_PRIORITY_KEYWORDS = [
    "api", "route", "router", "endpoint", "controller", "service",
    "db", "database", "model", "schema", "auth", "jwt", "login",
    "token", "security", "permission", "middleware", "config"
]


def is_ignored_path(filepath: str) -> bool:
    """Check if filepath should be filtered out."""
    norm_path = filepath.replace("\\", "/").lower()
    parts = norm_path.split("/")
    filename = os.path.basename(norm_path)
    
    # Check directory ignores
    if any(part in IGNORED_DIRS for part in parts):
        return True

    # Check extension ignores
    _, ext = os.path.splitext(norm_path)
    if ext in IGNORED_EXTENSIONS:
        return True

    # Check file pattern ignores (e.g. env_example, .gitignore)
    if filename in IGNORED_PATTERNS or any(p in filename for p in ["env_example", "env.example", "example.env"]):
        return True

    # Only include supported source code extensions
    if ext not in SUPPORTED_EXTENSIONS:
        return True

    return False


def get_file_language(filepath: str) -> str:
    """Return normalized language name or 'unknown'."""
    _, ext = os.path.splitext(filepath.lower())
    return SUPPORTED_EXTENSIONS.get(ext, "other")


def calculate_file_importance(filepath: str) -> Tuple[bool, int]:
    """
    Returns (is_supported_code_file, importance_score 1-5).
    """
    if is_ignored_path(filepath):
        return False, 0

    lang = get_file_language(filepath)
    if lang == "other":
        return False, 1

    score = 2
    norm = filepath.lower()
    for kw in HIGH_PRIORITY_KEYWORDS:
        if kw in norm:
            score += 1

    return True, min(score, 5)
