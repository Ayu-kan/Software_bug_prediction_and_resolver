"""
src/analyzers/base_analyzer.py
------------------------------
Phase 4 Task 10 - Task 14: Modular Static Code Analyzers for Python, JS, TS, Java.
"""

import ast
import re
from typing import Dict, Any

from radon.raw import analyze as radon_raw_analyze
from radon.complexity import cc_visit


class BaseAnalyzer:
    def analyze(self, source_code: str, filepath: str) -> Dict[str, Any]:
        raise NotImplementedError


class MultiLanguageAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.import_re = re.compile(r"^\s*(import|from|require|export)\s+\S+", re.MULTILINE)
        self.js_fn_re = re.compile(r"\bfunction\b|\=\>|\bclass\b", re.MULTILINE)

    def analyze(self, source_code: str, filepath: str) -> Dict[str, Any]:
        if not source_code:
            return self._empty_metrics()

        ext = filepath.lower().split(".")[-1]
        if ext == "py":
            return self._analyze_python(source_code)
        elif ext in ["js", "jsx", "ts", "tsx"]:
            return self._analyze_js_ts(source_code)
        elif ext == "java":
            return self._analyze_java(source_code)
        else:
            return self._analyze_generic(source_code)

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "loc": 0,
            "complexity": 0,
            "function_count": 0,
            "avg_function_size": 0,
            "max_function_size": 0,
            "dependency_count": 0,
        }

    def _analyze_python(self, source_code: str) -> Dict[str, Any]:
        try:
            raw = radon_raw_analyze(source_code)
            loc = raw.loc
        except Exception:
            loc = len(source_code.splitlines())

        try:
            blocks = cc_visit(source_code)
            complexity = sum(b.complexity for b in blocks)
            function_count = len(blocks)
            sizes = [(b.endline - b.lineno + 1) for b in blocks if b.endline and b.lineno]
            avg_fn_size = sum(sizes) / len(sizes) if sizes else 0
            max_fn_size = max(sizes) if sizes else 0
        except Exception:
            complexity, function_count, avg_fn_size, max_fn_size = 0, 0, 0, 0

        dep_count = len(re.findall(r"^\s*(import|from)\s+\S+", source_code, re.MULTILINE))
        return {
            "loc": loc,
            "complexity": complexity,
            "function_count": function_count,
            "avg_function_size": round(avg_fn_size, 2),
            "max_function_size": max_fn_size,
            "dependency_count": dep_count,
        }

    def _analyze_js_ts(self, source_code: str) -> Dict[str, Any]:
        lines = source_code.splitlines()
        loc = len([l for l in lines if l.strip() and not l.strip().startswith(("//", "/*", "*"))])
        
        fns = len(self.js_fn_re.findall(source_code))
        dep_count = len(self.import_re.findall(source_code))

        # Basic Cyclomatic Estimate for JS/TS
        decision_points = len(re.findall(r"\b(if|else|for|while|switch|case|catch|\?\?|\?)\b", source_code))
        complexity = decision_points + 1

        avg_fn_size = round(loc / fns, 2) if fns > 0 else 0
        return {
            "loc": loc,
            "complexity": complexity,
            "function_count": fns,
            "avg_function_size": avg_fn_size,
            "max_function_size": avg_fn_size,
            "dependency_count": dep_count,
        }

    def _analyze_java(self, source_code: str) -> Dict[str, Any]:
        lines = source_code.splitlines()
        loc = len([l for l in lines if l.strip() and not l.strip().startswith(("//", "/*", "*"))])
        
        methods = len(re.findall(r"\b(public|private|protected|static)\s+[\w\<\>\[\]]+\s+\w+\s*\(", source_code))
        dep_count = len(re.findall(r"^\s*import\s+[\w\.]+;", source_code, re.MULTILINE))

        decision_points = len(re.findall(r"\b(if|else|for|while|switch|case|catch)\b", source_code))
        complexity = decision_points + 1
        avg_fn_size = round(loc / methods, 2) if methods > 0 else 0

        return {
            "loc": loc,
            "complexity": complexity,
            "function_count": methods,
            "avg_function_size": avg_fn_size,
            "max_function_size": avg_fn_size,
            "dependency_count": dep_count,
        }

    def _analyze_generic(self, source_code: str) -> Dict[str, Any]:
        lines = source_code.splitlines()
        loc = len(lines)
        return {
            "loc": loc,
            "complexity": 1,
            "function_count": 0,
            "avg_function_size": 0,
            "max_function_size": 0,
            "dependency_count": 0,
        }
