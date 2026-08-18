"""
src/graph/dependency_graph.py
-----------------------------
Phase 5 & Phase 6: Dependency Graph Building & Architectural Layer Detection.
"""

import os
import re
from typing import Dict, Any, List, Set
import networkx as nx


class DependencyGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, file_sources: Dict[str, str]) -> nx.DiGraph:
        """
        file_sources: dict mapping relative filepath -> source_code string
        """
        self.graph.clear()
        file_list = list(file_sources.keys())
        for f in file_list:
            self.graph.add_node(f)

        # Detect internal imports
        for src_file, code in file_sources.items():
            if not code:
                continue
            imported_files = self._find_imported_files(src_file, code, file_list)
            for imp in imported_files:
                if imp != src_file:
                    self.graph.add_edge(src_file, imp)

        return self.graph

    def calculate_graph_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics = {}
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())

        try:
            betweenness = nx.betweenness_centrality(self.graph)
        except Exception:
            betweenness = {node: 0.0 for node in self.graph.nodes()}

        for node in self.graph.nodes():
            fan_in = in_degree.get(node, 0)
            fan_out = out_degree.get(node, 0)
            centrality = betweenness.get(node, 0.0)
            
            # Risk = combination of high fan-in (dependents) + centrality
            dep_risk = round(min((fan_in * 1.5 + fan_out * 0.5 + centrality * 10), 100), 1)

            metrics[node] = {
                "fan_in": fan_in,
                "fan_out": fan_out,
                "betweenness": round(centrality, 3),
                "dependency_risk": dep_risk
            }
        return metrics

    def _find_imported_files(self, current_file: str, code: str, all_files: List[str]) -> Set[str]:
        imported = set()
        # Find module name references
        lines = code.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith(("import ", "from ", "require(", "import {")):
                for target_file in all_files:
                    target_name = os.path.splitext(os.path.basename(target_file))[0]
                    if target_name and len(target_name) > 2 and target_name in line:
                        imported.add(target_file)
        return imported


def detect_architecture_role(filepath: str, source_code: str = "") -> Dict[str, Any]:
    """
    Phase 6: Classifies architecture role and security/auth risk level.
    """
    path_lower = filepath.replace("\\", "/").lower()

    role = "Utility / Business Logic"
    risk_score = 10.0

    if any(k in path_lower for k in ["auth", "jwt", "login", "password", "security", "token", "permission"]):
        role = "Authentication & Security"
        risk_score = 90.0
    elif any(k in path_lower for k in ["db", "database", "model", "schema", "orm", "migration", "query"]):
        role = "Database Layer"
        risk_score = 80.0
    elif any(k in path_lower for k in ["api", "route", "router", "controller", "endpoint", "views"]):
        role = "Backend API Layer"
        risk_score = 70.0
    elif any(k in path_lower for k in ["service", "handler", "manager"]):
        role = "Core Service Layer"
        risk_score = 60.0
    elif any(k in path_lower for k in ["component", "frontend", "ui", "pages", "src/app"]):
        role = "Frontend Layer"
        risk_score = 40.0

    return {
        "architecture_role": role,
        "architecture_risk": risk_score
    }
