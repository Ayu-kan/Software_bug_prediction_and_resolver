"""
src/repo/validator.py
---------------------
Phase 2 Task 3 & Task 6: Repository Validation and Limits handling.
"""

import os
import re
import shutil
import tempfile
import subprocess
from typing import Dict, Any, Tuple

GITHUB_URL_REGEX = re.compile(r"^(https?://|git@)(github\.com|gitlab\.com|bitbucket\.org)/[\w.-]+/[\w.-]+(?:\.git)?$")

# Guard limits
MAX_REPO_SIZE_MB = 500
MAX_FILES_COUNT = 50000
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB


def validate_repo_input(repo_input: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates repository input (URL or local path).
    Returns (is_valid, error_message, metadata).
    """
    repo_input = repo_input.strip()
    if not repo_input:
        return False, "Repository path or URL cannot be empty.", {}

    is_url = bool(repo_input.startswith(("http://", "https://", "git@")) or repo_input.endswith(".git"))
    metadata = {"is_url": is_url, "path_or_url": repo_input}

    if is_url:
        if not GITHUB_URL_REGEX.match(repo_input) and not repo_input.endswith(".git"):
            return False, "Invalid repository URL format.", metadata
        return True, "", metadata
    else:
        if not os.path.exists(repo_input):
            return False, f"Local directory does not exist: '{repo_input}'", metadata
        if not os.path.isdir(repo_input):
            return False, f"Path is not a directory: '{repo_input}'", metadata
        
        # Check size & file count for local repo
        total_size = 0
        file_count = 0
        for root, _, files in os.walk(repo_input):
            if ".git" in root or "node_modules" in root or ".venv" in root:
                continue
            for f in files:
                file_count += 1
                fp = os.path.join(root, f)
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
        
        size_mb = total_size / (1024 * 1024)
        metadata["size_mb"] = round(size_mb, 2)
        metadata["file_count"] = file_count

        if size_mb > MAX_REPO_SIZE_MB:
            return False, f"Repository size ({size_mb:.1f} MB) exceeds limit of {MAX_REPO_SIZE_MB} MB.", metadata
        if file_count > MAX_FILES_COUNT:
            return False, f"Repository file count ({file_count}) exceeds limit of {MAX_FILES_COUNT}.", metadata

        return True, "", metadata


class TemporaryClone:
    """Context manager for temporary git cloning."""
    def __init__(self, repo_url: str, depth: int = 50):
        self.repo_url = repo_url
        self.depth = depth
        self.temp_dir = None

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="shallow_repo_")
        cmd = ["git", "clone"]
        if self.depth:
            cmd.extend(["--depth", str(self.depth)])
        cmd.extend([self.repo_url, self.temp_dir])
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to clone repository '{self.repo_url}': {e}")
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
