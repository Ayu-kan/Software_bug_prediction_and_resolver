"""
src/history/persistence.py
--------------------------
Phase 10 Tasks 33-35: Analysis History & Solution Feedback Storage.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "history.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_path TEXT,
            timestamp TEXT,
            total_files INTEGER,
            high_risk_count INTEGER,
            summary_json TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS solution_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            timestamp TEXT,
            status TEXT,
            comment TEXT
        )
    """)
    conn.commit()
    conn.close()


def record_analysis_run(repo_path: str, total_files: int, high_risk_count: int, summary_data: List[Dict[str, Any]]):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO analysis_history (repo_path, timestamp, total_files, high_risk_count, summary_json)
        VALUES (?, ?, ?, ?, ?)
    """, (repo_path, now, total_files, high_risk_count, json.dumps(summary_data[:50])))
    conn.commit()
    conn.close()


def get_analysis_history() -> List[Dict[str, Any]]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, repo_path, timestamp, total_files, high_risk_count FROM analysis_history ORDER BY id DESC LIMIT 20")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def record_solution_feedback(file_path: str, status: str, comment: str = ""):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute("""
        INSERT INTO solution_feedback (file_path, timestamp, status, comment)
        VALUES (?, ?, ?, ?)
    """, (file_path, now, status, comment))
    conn.commit()
    conn.close()
