"""
Context Layer: Graph-based Eval Knowledge Store
================================================

Stores eval test cases, runs, verdicts, and failure patterns as a graph.
Provides two backends:
  - Local: NetworkX + SQLite (runs anywhere, used for this demo)
  - Neo4j Aura: production backend (requires network access)

Graph Schema:
  (TestCase)-[:HAS_OUTPUT]->(Output)
  (Output)-[:JUDGED_BY]->(Verdict {method, result, correct})
  (TestCase)-[:BELONGS_TO]->(Category)
  (EvalRun)-[:CONTAINS]->(Verdict)
  (Verdict)-[:EXHIBITS]->(FailureMode)
  (TestCase)-[:HARDER_THAN]->(TestCase)  -- derived from judge miss rate
"""

import networkx as nx
import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "eval_context.db"


class ContextLayer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.db = sqlite3.connect(str(DB_PATH))
        self._init_db()
        self.current_run_id = None

    def _init_db(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                model TEXT,
                config TEXT,
                summary TEXT
            );
            CREATE TABLE IF NOT EXISTS verdicts (
                verdict_id TEXT PRIMARY KEY,
                run_id TEXT,
                test_id TEXT,
                method TEXT,
                result TEXT,
                correct INTEGER,
                explanation TEXT,
                latency_ms REAL,
                FOREIGN KEY(run_id) REFERENCES eval_runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS test_cases (
                test_id TEXT PRIMARY KEY,
                category TEXT,
                prompt TEXT,
                output TEXT,
                ground_truth TEXT,
                is_actually_correct INTEGER
            );
            CREATE TABLE IF NOT EXISTS failure_patterns (
                pattern_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                test_ids TEXT
            );
        """)
        self.db.commit()

    def start_run(self, model, config=None):
        self.current_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_node = f"EvalRun:{self.current_run_id}"
        self.graph.add_node(run_node, type="EvalRun", model=model,
                           timestamp=datetime.now().isoformat(),
                           config=json.dumps(config or {}))
        self.db.execute(
            "INSERT INTO eval_runs VALUES (?, ?, ?, ?, ?)",
            (self.current_run_id, datetime.now().isoformat(), model,
             json.dumps(config or {}), None)
        )
        self.db.commit()
        return self.current_run_id

    def add_test_case(self, test_id, category, prompt, output, ground_truth,
                      is_actually_correct=False):
        tc_node = f"TestCase:{test_id}"
        cat_node = f"Category:{category}"

        self.graph.add_node(tc_node, type="TestCase", prompt=prompt,
                           ground_truth=ground_truth)
        self.graph.add_node(cat_node, type="Category", name=category)
        self.graph.add_edge(tc_node, cat_node, relation="BELONGS_TO")

        out_node = f"Output:{test_id}"
        self.graph.add_node(out_node, type="Output", text=output[:200],
                           is_correct=is_actually_correct)
        self.graph.add_edge(tc_node, out_node, relation="HAS_OUTPUT")

        self.db.execute(
            "INSERT OR REPLACE INTO test_cases VALUES (?, ?, ?, ?, ?, ?)",
            (str(test_id), category, prompt, output, ground_truth,
             int(is_actually_correct))
        )
        self.db.commit()
        return tc_node

    def record_verdict(self, test_id, method, result, correct, explanation="",
                       latency_ms=0):
        vid = hashlib.md5(
            f"{self.current_run_id}:{test_id}:{method}".encode()
        ).hexdigest()[:12]
        verdict_node = f"Verdict:{vid}"

        self.graph.add_node(verdict_node, type="Verdict", method=method,
                           result=result, correct=correct,
                           explanation=explanation[:100])

        out_node = f"Output:{test_id}"
        if out_node in self.graph:
            self.graph.add_edge(out_node, verdict_node, relation="JUDGED_BY")

        run_node = f"EvalRun:{self.current_run_id}"
        if run_node in self.graph:
            self.graph.add_edge(run_node, verdict_node, relation="CONTAINS")

        if not correct:
            failure_type = f"FalsePass:{method}" if result == "PASS" else f"FalseNeg:{method}"
            fm_node = f"FailureMode:{failure_type}"
            self.graph.add_node(fm_node, type="FailureMode", name=failure_type)
            self.graph.add_edge(verdict_node, fm_node, relation="EXHIBITS")

        self.db.execute(
            "INSERT OR REPLACE INTO verdicts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (vid, self.current_run_id, str(test_id), method, result,
             int(correct), explanation, latency_ms)
        )
        self.db.commit()
        return verdict_node

    def derive_difficulty_edges(self):
        """Add HARDER_THAN edges based on judge miss rate across runs."""
        cursor = self.db.execute("""
            SELECT test_id, method,
                   COUNT(*) as total,
                   SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as misses
            FROM verdicts
            WHERE method = 'llm_judge'
            GROUP BY test_id
        """)
        miss_rates = {}
        for row in cursor:
            test_id, method, total, misses = row
            miss_rates[test_id] = misses / total if total > 0 else 0

        sorted_tests = sorted(miss_rates.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(sorted_tests) - 1):
            harder_id, harder_rate = sorted_tests[i]
            easier_id, easier_rate = sorted_tests[i + 1]
            if harder_rate > easier_rate:
                self.graph.add_edge(
                    f"TestCase:{harder_id}", f"TestCase:{easier_id}",
                    relation="HARDER_THAN",
                    miss_rate_delta=harder_rate - easier_rate
                )

    def get_graph_stats(self):
        nodes_by_type = {}
        for node, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1

        edges_by_type = {}
        for u, v, data in self.graph.edges(data=True):
            r = data.get("relation", "unknown")
            edges_by_type[r] = edges_by_type.get(r, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
        }

    def get_false_pass_analysis(self):
        """Which categories does the LLM judge fail on most?"""
        cursor = self.db.execute("""
            SELECT t.category, v.method,
                   COUNT(*) as total,
                   SUM(CASE WHEN v.correct = 0 AND v.result = 'PASS' THEN 1 ELSE 0 END) as false_passes
            FROM verdicts v
            JOIN test_cases t ON v.test_id = t.test_id
            GROUP BY t.category, v.method
            ORDER BY false_passes DESC
        """)
        return cursor.fetchall()

    def get_method_comparison(self):
        """Head-to-head: LLM judge vs deterministic on the same test set."""
        cursor = self.db.execute("""
            SELECT v.method,
                   COUNT(*) as total,
                   SUM(v.correct) as correct,
                   ROUND(AVG(v.correct) * 100, 1) as accuracy_pct,
                   SUM(CASE WHEN v.correct = 0 AND v.result = 'PASS' THEN 1 ELSE 0 END) as false_passes,
                   ROUND(AVG(v.latency_ms), 0) as avg_latency_ms
            FROM verdicts v
            GROUP BY v.method
        """)
        return cursor.fetchall()

    def export_neo4j_cypher(self):
        """Generate Cypher statements to load this graph into Neo4j Aura."""
        statements = [
            "// Auto-generated from eval context layer",
            "// Run against your Neo4j Aura instance",
            "",
            "// Clear previous data",
            "MATCH (n) DETACH DELETE n;",
            "",
        ]

        for node, data in self.graph.nodes(data=True):
            ntype = data.get("type", "Node")
            props = {k: v for k, v in data.items() if k != "type"}
            props["_id"] = node
            props_str = ", ".join(
                f"{k}: {json.dumps(v)}" for k, v in props.items()
            )
            statements.append(f"CREATE (:{ntype} {{{props_str}}});")

        statements.append("")

        for u, v, data in self.graph.edges(data=True):
            rel = data.get("relation", "RELATED_TO")
            props = {k: v for k, v in data.items() if k != "relation"}
            props_str = ""
            if props:
                props_str = " {" + ", ".join(
                    f"{k}: {json.dumps(v)}" for k, v in props.items()
                ) + "}"
            statements.append(
                f"MATCH (a {{_id: {json.dumps(u)}}}), (b {{_id: {json.dumps(v)}}}) "
                f"CREATE (a)-[:{rel}{props_str}]->(b);"
            )

        return "\n".join(statements)

    def close(self):
        self.db.commit()
        self.db.close()
