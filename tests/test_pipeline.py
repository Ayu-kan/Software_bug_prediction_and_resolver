"""
tests/test_pipeline.py
-----------------------
Phase 12 Task 46: Tests for file filtering, validation, static code analysis, and graph building.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from filtering.file_filter import is_ignored_path, calculate_file_importance
from repo.validator import validate_repo_input
from analyzers.base_analyzer import MultiLanguageAnalyzer
from graph.dependency_graph import DependencyGraphBuilder, detect_architecture_role


class TestBugPredictionPipeline(unittest.TestCase):
    def test_file_filtering(self):
        self.assertTrue(is_ignored_path("node_modules/express/index.js"))
        self.assertTrue(is_ignored_path("venv/lib/python3.9/site-packages/pkg.py"))
        self.assertTrue(is_ignored_path("image.png"))
        self.assertTrue(is_ignored_path("README.md"))
        self.assertTrue(is_ignored_path("notes.txt"))
        self.assertTrue(is_ignored_path("package.json"))
        self.assertTrue(is_ignored_path("env.example"))
        self.assertTrue(is_ignored_path(".env"))
        self.assertFalse(is_ignored_path("src/controllers/auth_controller.py"))

    def test_repo_validation(self):
        is_valid, msg, meta = validate_repo_input("https://github.com/psf/requests")
        self.assertTrue(is_valid)
        self.assertTrue(meta["is_url"])

        is_valid_empty, _, _ = validate_repo_input("")
        self.assertFalse(is_valid_empty)

    def test_analyzers(self):
        analyzer = MultiLanguageAnalyzer()
        py_code = "def hello():\n    if True:\n        print('hi')\n"
        py_res = analyzer.analyze(py_code, "test.py")
        self.assertEqual(py_res["loc"], 3)
        self.assertGreaterEqual(py_res["complexity"], 1)

        js_code = "function test() { if (a && b) { return 1; } }"
        js_res = analyzer.analyze(js_code, "test.js")
        self.assertEqual(js_res["function_count"], 1)

    def test_architecture_detection(self):
        info = detect_architecture_role("src/auth/jwt_handler.py")
        self.assertEqual(info["architecture_role"], "Authentication & Security")
        self.assertEqual(info["architecture_risk"], 90.0)


if __name__ == "__main__":
    unittest.main()
