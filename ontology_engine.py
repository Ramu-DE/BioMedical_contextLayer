"""
Ontology Engine — Palantir Foundry-inspired Hydration Layer
============================================================

Implements the Palantir "Hydrate the Ontology" pattern:

  DATA + MODELS  →  ONTOLOGY  →  ANALYTICS / WORKFLOWS / INTEGRATIONS

Each Object Type has four facets (matching the Ontology Engine diagram):
  - Properties   : typed attributes stored on the object
  - Functions    : computed/derived values (fx)
  - Actions      : mutations that write back through the ontology
  - Automations  : triggered rules when conditions are met

Objects are connected by typed Links.
WRITES flow in from data pipelines (hydration).
READS flow out to applications (queries).

Domain: BioMedical AI Evaluation Framework
"""

import json
import sqlite3
import hashlib
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from enum import Enum

import boto3
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DB_PATH = Path(__file__).parent / "ontology.db"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ONTOLOGY SCHEMA — Object Types, Link Types, Property Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ObjectType(Enum):
    TEST_CASE = "TestCase"
    JUDGE = "Judge"
    EVAL_RUN = "EvalRun"
    VERDICT = "Verdict"
    BIAS_PATTERN = "BiasPattern"
    COST_RECORD = "CostRecord"
    CATEGORY = "Category"
    FAILURE_MODE = "FailureMode"


class LinkType(Enum):
    JUDGES = "JUDGES"                # Judge -> TestCase
    PRODUCES = "PRODUCES"            # Judge -> Verdict
    EVALUATED_IN = "EVALUATED_IN"    # Verdict -> EvalRun
    TESTS = "TESTS"                  # Verdict -> TestCase
    EXHIBITS = "EXHIBITS"            # Verdict -> BiasPattern
    COSTS = "COSTS"                  # Verdict -> CostRecord
    BELONGS_TO = "BELONGS_TO"        # TestCase -> Category
    AGREES_WITH = "AGREES_WITH"      # Judge -> Judge (inter-rater)
    TRIGGERS = "TRIGGERS"            # FailureMode -> Automation
    HARDER_THAN = "HARDER_THAN"      # TestCase -> TestCase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ONTOLOGY OBJECTS — Each with Properties, Functions, Actions, Automations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OntologyObject:
    """Base for all ontology objects."""
    object_type: str
    object_id: str
    properties: dict = field(default_factory=dict)
    _engine: Any = field(default=None, repr=False)

    def prop(self, key: str) -> Any:
        return self.properties.get(key)

    def set_prop(self, key: str, value: Any):
        self.properties[key] = value
        if self._engine:
            self._engine._persist_object(self)

    def links(self, link_type: str = None) -> list:
        if self._engine:
            return self._engine.get_links(self.object_id, link_type)
        return []

    def linked_objects(self, link_type: str) -> list:
        if self._engine:
            return self._engine.get_linked_objects(self.object_id, link_type)
        return []


class TestCase(OntologyObject):
    """A single evaluation test case."""

    def __init__(self, test_id, **props):
        super().__init__(ObjectType.TEST_CASE.value, test_id, props)

    # -- Functions (fx) --
    @property
    def difficulty_score(self) -> float:
        verdicts = self.linked_objects("TESTS")
        if not verdicts:
            return 0.0
        incorrect = sum(1 for v in verdicts if not v.prop("correct"))
        return incorrect / len(verdicts)

    @property
    def judge_disagreement_rate(self) -> float:
        verdicts = self.linked_objects("TESTS")
        if len(verdicts) < 2:
            return 0.0
        results = [v.prop("result") for v in verdicts]
        unique = len(set(results))
        return (unique - 1) / len(results)

    @property
    def avg_latency_ms(self) -> float:
        verdicts = self.linked_objects("TESTS")
        latencies = [v.prop("latency_ms") for v in verdicts if v.prop("latency_ms")]
        return sum(latencies) / len(latencies) if latencies else 0.0


class Judge(OntologyObject):
    """An LLM judge configuration."""

    def __init__(self, judge_id, **props):
        super().__init__(ObjectType.JUDGE.value, judge_id, props)

    # -- Functions (fx) --
    @property
    def accuracy(self) -> float:
        verdicts = self.linked_objects("PRODUCES")
        if not verdicts:
            return 0.0
        correct = sum(1 for v in verdicts if v.prop("correct"))
        return correct / len(verdicts)

    @property
    def false_pass_rate(self) -> float:
        verdicts = self.linked_objects("PRODUCES")
        testable = [v for v in verdicts if v.prop("expected") == "FAIL"]
        if not testable:
            return 0.0
        fps = sum(1 for v in testable if v.prop("result") == "PASS")
        return fps / len(testable)

    @property
    def bias_profile(self) -> dict:
        verdicts = self.linked_objects("PRODUCES")
        biases = defaultdict(int)
        for v in verdicts:
            for bp in v.linked_objects("EXHIBITS"):
                biases[bp.prop("bias_type")] += 1
        return dict(biases)

    @property
    def total_cost(self) -> float:
        verdicts = self.linked_objects("PRODUCES")
        costs = [v.prop("cost_usd") for v in verdicts if v.prop("cost_usd")]
        return sum(costs)

    @property
    def avg_latency_ms(self) -> float:
        verdicts = self.linked_objects("PRODUCES")
        lats = [v.prop("latency_ms") for v in verdicts if v.prop("latency_ms")]
        return sum(lats) / len(lats) if lats else 0.0


class EvalRun(OntologyObject):
    """A batch evaluation run."""

    def __init__(self, run_id, **props):
        super().__init__(ObjectType.EVAL_RUN.value, run_id, props)

    # -- Functions (fx) --
    @property
    def accuracy_by_judge(self) -> dict:
        verdicts = self.linked_objects("EVALUATED_IN")
        by_judge = defaultdict(lambda: {"correct": 0, "total": 0})
        for v in verdicts:
            j = v.prop("judge_method")
            by_judge[j]["total"] += 1
            if v.prop("correct"):
                by_judge[j]["correct"] += 1
        return {j: d["correct"] / d["total"] if d["total"] else 0
                for j, d in by_judge.items()}

    @property
    def accuracy_by_category(self) -> dict:
        verdicts = self.linked_objects("EVALUATED_IN")
        by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
        for v in verdicts:
            cat = v.prop("category")
            by_cat[cat]["total"] += 1
            if v.prop("correct"):
                by_cat[cat]["correct"] += 1
        return {c: d["correct"] / d["total"] if d["total"] else 0
                for c, d in by_cat.items()}

    @property
    def total_cost(self) -> float:
        verdicts = self.linked_objects("EVALUATED_IN")
        return sum(v.prop("cost_usd") or 0 for v in verdicts)

    @property
    def bias_summary(self) -> dict:
        verdicts = self.linked_objects("EVALUATED_IN")
        biases = defaultdict(int)
        for v in verdicts:
            for bp in v.linked_objects("EXHIBITS"):
                biases[bp.prop("bias_type")] += 1
        total = len(verdicts)
        return {k: {"count": c, "rate": c / total if total else 0}
                for k, c in biases.items()}


class Verdict(OntologyObject):
    """A single judge's decision on a test case."""

    def __init__(self, verdict_id, **props):
        super().__init__(ObjectType.VERDICT.value, verdict_id, props)

    @property
    def is_false_positive(self) -> bool:
        return self.prop("expected") == "FAIL" and self.prop("result") == "PASS"

    @property
    def is_false_negative(self) -> bool:
        return self.prop("expected") == "PASS" and self.prop("result") == "FAIL"

    @property
    def cost_usd(self) -> float:
        in_tok = self.prop("input_tokens") or 0
        out_tok = self.prop("output_tokens") or 0
        return (in_tok * 0.25 + out_tok * 1.25) / 1_000_000


class BiasPattern(OntologyObject):
    """A detected bias in a verdict."""

    def __init__(self, bias_id, **props):
        super().__init__(ObjectType.BIAS_PATTERN.value, bias_id, props)


class Category(OntologyObject):
    """A test category (domain grouping)."""

    def __init__(self, cat_id, **props):
        super().__init__(ObjectType.CATEGORY.value, cat_id, props)

    @property
    def test_count(self) -> int:
        return len(self.linked_objects("BELONGS_TO"))

    @property
    def avg_accuracy(self) -> float:
        tests = self.linked_objects("BELONGS_TO")
        if not tests:
            return 0.0
        scores = [1 - t.difficulty_score for t in tests]
        return sum(scores) / len(scores)


class FailureMode(OntologyObject):
    """A class of failure (false pass, bias, hallucination, etc.)."""

    def __init__(self, fm_id, **props):
        super().__init__(ObjectType.FAILURE_MODE.value, fm_id, props)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ONTOLOGY ENGINE — The core runtime (WRITES / READS / Links)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OntologyEngine:
    """
    Central ontology runtime. Manages objects, links, persistence.

    Mirrors the Palantir Ontology Engine pattern:
      WRITES (left)  →  [Objects + Links]  →  READS (right)
    """

    OBJECT_CLASSES = {
        ObjectType.TEST_CASE.value: TestCase,
        ObjectType.JUDGE.value: Judge,
        ObjectType.EVAL_RUN.value: EvalRun,
        ObjectType.VERDICT.value: Verdict,
        ObjectType.BIAS_PATTERN.value: BiasPattern,
        ObjectType.CATEGORY.value: Category,
        ObjectType.FAILURE_MODE.value: FailureMode,
    }

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.objects: dict[str, OntologyObject] = {}
        self.links: list[dict] = []
        self.automations: list[dict] = []
        self._action_log: list[dict] = []
        self.db = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS objects (
                object_id TEXT PRIMARY KEY,
                object_type TEXT NOT NULL,
                properties TEXT NOT NULL,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS links (
                link_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS action_log (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                object_id TEXT,
                payload TEXT,
                timestamp TEXT,
                status TEXT
            );
            CREATE TABLE IF NOT EXISTS automations (
                automation_id TEXT PRIMARY KEY,
                name TEXT,
                trigger_condition TEXT,
                action TEXT,
                enabled INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_id);
            CREATE INDEX IF NOT EXISTS idx_links_type ON links(link_type);
            CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(object_type);
        """)
        self.db.commit()

    # ── WRITES: Create / Update Objects ────────────────────────────────

    def create_object(self, obj: OntologyObject) -> OntologyObject:
        obj._engine = self
        self.objects[obj.object_id] = obj
        self._persist_object(obj)
        self._check_automations("object_created", obj)
        return obj

    def create_link(self, source_id: str, target_id: str, link_type: str,
                    properties: dict = None) -> dict:
        link_id = hashlib.md5(f"{source_id}:{link_type}:{target_id}".encode()).hexdigest()[:12]
        link = {
            "link_id": link_id,
            "source_id": source_id,
            "target_id": target_id,
            "link_type": link_type,
            "properties": properties or {},
        }
        self.links.append(link)
        now = datetime.utcnow().isoformat()
        self.db.execute(
            "INSERT OR REPLACE INTO links VALUES (?,?,?,?,?,?)",
            (link_id, source_id, target_id, link_type,
             json.dumps(properties or {}), now)
        )
        self.db.commit()
        return link

    def _persist_object(self, obj: OntologyObject):
        now = datetime.utcnow().isoformat()
        self.db.execute(
            "INSERT OR REPLACE INTO objects VALUES (?,?,?,?,?)",
            (obj.object_id, obj.object_type,
             json.dumps(obj.properties), now, now)
        )
        self.db.commit()

    # ── READS: Query Objects and Links ─────────────────────────────────

    def get_object(self, object_id: str) -> Optional[OntologyObject]:
        if object_id in self.objects:
            return self.objects[object_id]
        row = self.db.execute(
            "SELECT object_type, properties FROM objects WHERE object_id=?",
            (object_id,)
        ).fetchone()
        if row:
            obj_type, props_json = row
            cls = self.OBJECT_CLASSES.get(obj_type, OntologyObject)
            if cls == OntologyObject:
                obj = OntologyObject(obj_type, object_id, json.loads(props_json))
            else:
                obj = cls.__new__(cls)
                OntologyObject.__init__(obj, obj_type, object_id, json.loads(props_json))
            obj._engine = self
            self.objects[object_id] = obj
            return obj
        return None

    def get_objects_by_type(self, object_type: str) -> list[OntologyObject]:
        rows = self.db.execute(
            "SELECT object_id FROM objects WHERE object_type=?",
            (object_type,)
        ).fetchall()
        return [self.get_object(r[0]) for r in rows]

    def get_links(self, object_id: str, link_type: str = None) -> list[dict]:
        if link_type:
            rows = self.db.execute(
                "SELECT * FROM links WHERE (source_id=? OR target_id=?) AND link_type=?",
                (object_id, object_id, link_type)
            ).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM links WHERE source_id=? OR target_id=?",
                (object_id, object_id)
            ).fetchall()
        return [{"link_id": r[0], "source_id": r[1], "target_id": r[2],
                 "link_type": r[3], "properties": json.loads(r[4])} for r in rows]

    def get_linked_objects(self, object_id: str, link_type: str) -> list[OntologyObject]:
        links = self.get_links(object_id, link_type)
        result = []
        for l in links:
            other_id = l["target_id"] if l["source_id"] == object_id else l["source_id"]
            obj = self.get_object(other_id)
            if obj:
                result.append(obj)
        return result

    def query(self, object_type: str, **filters) -> list[OntologyObject]:
        objs = self.get_objects_by_type(object_type)
        for key, value in filters.items():
            objs = [o for o in objs if o.prop(key) == value]
        return objs

    # ── ACTIONS: Mutations through the ontology ────────────────────────

    def execute_action(self, action_type: str, object_id: str = None,
                       payload: dict = None) -> dict:
        action_id = hashlib.md5(
            f"{action_type}:{object_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]

        result = {"action_id": action_id, "status": "pending"}

        if action_type == "run_eval":
            result = self._action_run_eval(payload)
        elif action_type == "flag_bias":
            result = self._action_flag_bias(object_id, payload)
        elif action_type == "promote_judge":
            result = self._action_promote_judge(object_id)
        elif action_type == "rerun_failed":
            result = self._action_rerun_failed(object_id)

        now = datetime.utcnow().isoformat()
        self.db.execute(
            "INSERT INTO action_log VALUES (?,?,?,?,?,?)",
            (action_id, action_type, object_id,
             json.dumps(payload or {}), now, result.get("status", "completed"))
        )
        self.db.commit()
        self._action_log.append(result)
        return result

    def _action_flag_bias(self, verdict_id: str, payload: dict) -> dict:
        verdict = self.get_object(verdict_id)
        if not verdict:
            return {"status": "error", "message": f"Verdict {verdict_id} not found"}
        bias_type = payload.get("bias_type", "unknown")
        bp = BiasPattern(
            f"bias-{verdict_id}-{bias_type}",
            bias_type=bias_type,
            severity=payload.get("severity", "medium"),
            verdict_id=verdict_id,
        )
        self.create_object(bp)
        self.create_link(verdict_id, bp.object_id, LinkType.EXHIBITS.value)
        return {"status": "completed", "bias_pattern_id": bp.object_id}

    def _action_promote_judge(self, judge_id: str) -> dict:
        judge = self.get_object(judge_id)
        if not judge:
            return {"status": "error", "message": f"Judge {judge_id} not found"}
        judge.set_prop("promoted", True)
        judge.set_prop("promoted_at", datetime.utcnow().isoformat())
        return {"status": "completed", "judge_id": judge_id}

    def _action_run_eval(self, payload: dict) -> dict:
        return {"status": "queued", "message": "Eval run queued for execution"}

    def _action_rerun_failed(self, run_id: str) -> dict:
        run = self.get_object(run_id)
        if not run:
            return {"status": "error"}
        verdicts = self.get_linked_objects(run_id, "EVALUATED_IN")
        failed = [v for v in verdicts if not v.prop("correct")]
        return {"status": "queued", "failed_count": len(failed),
                "message": f"Re-running {len(failed)} failed test cases"}

    # ── AUTOMATIONS: Triggered rules ───────────────────────────────────

    def register_automation(self, name: str, trigger: str, action: callable):
        auto_id = hashlib.md5(name.encode()).hexdigest()[:12]
        self.automations.append({
            "automation_id": auto_id,
            "name": name,
            "trigger": trigger,
            "action": action,
            "enabled": True,
        })
        self.db.execute(
            "INSERT OR REPLACE INTO automations VALUES (?,?,?,?,?)",
            (auto_id, name, trigger, action.__name__ if callable(action) else str(action), 1)
        )
        self.db.commit()

    def _check_automations(self, event_type: str, obj: OntologyObject):
        for auto in self.automations:
            if not auto["enabled"]:
                continue
            if auto["trigger"] == event_type:
                try:
                    auto["action"](self, obj)
                except Exception as e:
                    print(f"Automation '{auto['name']}' failed: {e}")

    # ── STATS: Ontology-wide metrics ───────────────────────────────────

    def stats(self) -> dict:
        counts = {}
        for ot in ObjectType:
            row = self.db.execute(
                "SELECT COUNT(*) FROM objects WHERE object_type=?", (ot.value,)
            ).fetchone()
            counts[ot.value] = row[0]
        link_count = self.db.execute("SELECT COUNT(*) FROM links").fetchone()[0]
        action_count = self.db.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]
        return {
            "objects": counts,
            "total_objects": sum(counts.values()),
            "total_links": link_count,
            "total_actions": action_count,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HYDRATION PIPELINE — WRITES: Raw data → Ontology Objects
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HydrationPipeline:
    """
    Palantir-style hydration: transforms raw eval data into typed
    Ontology Objects with Links, Functions, and computed properties.
    """

    def __init__(self, engine: OntologyEngine):
        self.engine = engine
        self.hydrated = {"objects": 0, "links": 0}

    def _detect_schema(self, results: list[dict]) -> str:
        if not results:
            return "empty"
        keys = set(results[0].keys())
        if "module" in keys and "det_verdict" in keys:
            return "comprehensive"
        if "no_rubric_verdict" in keys and "domain_expert_verdict" in keys:
            return "poc_live"
        if "judge_verdict" in keys:
            return "poc_basic"
        if "no_rubric_verdict" in keys and "with_rubric_verdict" in keys:
            return "poc_subtle"
        return "unknown"

    def _ensure_judge(self, judge_id: str, method: str, label: str, model: str):
        if not self.engine.get_object(judge_id):
            j = Judge(judge_id, method=method, label=label, model=model)
            self.engine.create_object(j)
            self.hydrated["objects"] += 1

    def _create_verdict(self, verdict_id, run_id, test_id, judge_id, **props):
        v = Verdict(verdict_id, **props)
        v.properties["cost_usd"] = v.cost_usd
        self.engine.create_object(v)
        self.engine.create_link(verdict_id, run_id, LinkType.EVALUATED_IN.value)
        self.engine.create_link(judge_id, verdict_id, LinkType.PRODUCES.value)
        self.engine.create_link(verdict_id, test_id, LinkType.TESTS.value)
        self.hydrated["objects"] += 1
        self.hydrated["links"] += 3

        fp = props.get("is_false_pass", False)
        if fp:
            bp = BiasPattern(
                f"bias-fp-{verdict_id}",
                bias_type="false_positive", severity="high",
                description=f"Judge passed a known-bad test case",
            )
            self.engine.create_object(bp)
            self.engine.create_link(verdict_id, bp.object_id, LinkType.EXHIBITS.value)
            self.hydrated["objects"] += 1
            self.hydrated["links"] += 1
        return v

    def hydrate_from_results(self, results_path: str, run_label: str = None):
        with open(results_path) as f:
            data = json.load(f)

        run_id = data.get("run_id", run_label or Path(results_path).stem)
        results = data.get("results", [])
        summary = data.get("summary", {})

        if not results:
            print(f"Skipping {Path(results_path).name}: no results")
            return self.hydrated

        schema = self._detect_schema(results)
        print(f"Hydrating from {Path(results_path).name}: {len(results)} results (schema={schema})")

        eval_run = EvalRun(
            run_id,
            source_file=Path(results_path).name,
            schema=schema,
            timestamp=datetime.utcnow().isoformat(),
            result_count=len(results),
            summary=summary,
        )
        self.engine.create_object(eval_run)
        self.hydrated["objects"] += 1

        categories_seen = set()

        for r in results:
            test_id = r.get("test_id", f"test-{hash(str(r)) % 10000}")
            category = r.get("category", "unknown")
            module = r.get("module", "P")

            cat_id = f"cat-{category.lower().replace(' ', '-')}"
            if cat_id not in categories_seen:
                cat = Category(cat_id, name=category, module=module)
                self.engine.create_object(cat)
                categories_seen.add(cat_id)
                self.hydrated["objects"] += 1

            tc = TestCase(
                test_id,
                category=category,
                module=module,
                is_correct=r.get("is_correct"),
                error_type=r.get("error_type"),
            )
            self.engine.create_object(tc)
            self.engine.create_link(test_id, cat_id, LinkType.BELONGS_TO.value)
            self.hydrated["objects"] += 1
            self.hydrated["links"] += 1

            if schema == "comprehensive":
                self._hydrate_comprehensive(r, run_id, test_id, category)
            elif schema == "poc_live":
                self._hydrate_poc_live(r, run_id, test_id, category)
            elif schema == "poc_basic":
                self._hydrate_poc_basic(r, run_id, test_id, category)
            elif schema == "poc_subtle":
                self._hydrate_poc_subtle(r, run_id, test_id, category)

        print(f"  Hydrated: {self.hydrated['objects']} objects, {self.hydrated['links']} links")
        return self.hydrated

    def _hydrate_comprehensive(self, r, run_id, test_id, category):
        module = r.get("module", "E")

        if module == "E":
            for method, label, model in [
                ("det", "Deterministic", "rule-based"),
                ("nr", "Narrow LLM", "claude-haiku-4.5"),
                ("wr", "Wide-Retrieval LLM", "claude-haiku-4.5"),
            ]:
                correct_key = f"{method}_correct"
                verdict_key = f"{method}_verdict"
                if correct_key not in r and verdict_key not in r:
                    continue
                judge_id = f"judge-{method}"
                self._ensure_judge(judge_id, method, label, model)
                self._create_verdict(
                    f"v-{run_id}-{test_id}-{method}",
                    run_id, test_id, judge_id,
                    judge_method=method,
                    result=r.get(verdict_key),
                    correct=r.get(correct_key, False),
                    expected="PASS" if r.get("is_correct") else "FAIL",
                    category=category,
                    latency_ms=r.get(f"{method}_ms"),
                    input_tokens=r.get(f"{method}_in_tok", 0),
                    output_tokens=r.get(f"{method}_out_tok", 0),
                    is_false_pass=r.get(f"{method}_fp", False),
                    is_false_fail=r.get(f"{method}_ff", False),
                )

        elif module == "A":
            if category in ("position_bias", "verbosity_bias", "adversarial"):
                judge_id = "judge-bias_test"
                self._ensure_judge(judge_id, "bias_test", "Bias Probe", "rule-based")
                verdict_id = f"v-{run_id}-{test_id}-bias"
                v = Verdict(
                    verdict_id,
                    judge_method="bias_test",
                    result="biased" if not r.get("is_correct") else "clean",
                    correct=r.get("is_correct", False),
                    category=category,
                    bias_type=category,
                )
                self.engine.create_object(v)
                self.engine.create_link(verdict_id, run_id, LinkType.EVALUATED_IN.value)
                self.engine.create_link(verdict_id, test_id, LinkType.TESTS.value)
                self.hydrated["objects"] += 1
                self.hydrated["links"] += 2

                if not r.get("is_correct"):
                    bp = BiasPattern(
                        f"bias-{category}-{test_id}",
                        bias_type=category, severity="medium",
                    )
                    self.engine.create_object(bp)
                    self.engine.create_link(verdict_id, bp.object_id, LinkType.EXHIBITS.value)
                    self.hydrated["objects"] += 1
                    self.hydrated["links"] += 1

    def _hydrate_poc_live(self, r, run_id, test_id, category):
        for method, label in [
            ("no_rubric", "No-Rubric LLM"),
            ("with_rubric", "With-Rubric LLM"),
            ("domain_expert", "Domain Expert LLM"),
        ]:
            correct_key = f"{method}_correct"
            verdict_key = f"{method}_verdict"
            if correct_key not in r:
                continue
            judge_id = f"judge-{method}"
            self._ensure_judge(judge_id, method, label, "claude-haiku-4.5")
            self._create_verdict(
                f"v-{run_id}-{test_id}-{method}",
                run_id, test_id, judge_id,
                judge_method=method,
                result=r.get(verdict_key),
                correct=r.get(correct_key, False),
                expected="PASS" if r.get("is_correct") else "FAIL",
                category=category,
                latency_ms=r.get(f"{method}_ms"),
                is_false_pass=r.get(f"{method}_fp", False),
                is_false_fail=r.get(f"{method}_ff", False),
            )
        if "det_correct" in r:
            self._ensure_judge("judge-det", "det", "Deterministic", "rule-based")
            self._create_verdict(
                f"v-{run_id}-{test_id}-det",
                run_id, test_id, "judge-det",
                judge_method="det",
                result=r.get("det_result"),
                correct=r.get("det_correct", False),
                expected="PASS" if r.get("is_correct") else "FAIL",
                category=category,
            )

    def _hydrate_poc_basic(self, r, run_id, test_id, category):
        self._ensure_judge("judge-llm_basic", "llm_basic", "Basic LLM Judge", "claude-haiku-4.5")
        self._create_verdict(
            f"v-{run_id}-{test_id}-llm_basic",
            run_id, test_id, "judge-llm_basic",
            judge_method="llm_basic",
            result=r.get("judge_verdict"),
            correct=r.get("judge_correct", False),
            expected="PASS" if r.get("is_correct") else "FAIL",
            category=category,
            latency_ms=r.get("judge_ms"),
            is_false_pass=r.get("is_false_pass", False),
            is_false_fail=r.get("is_false_fail", False),
        )
        if "det_correct" in r:
            self._ensure_judge("judge-det", "det", "Deterministic", "rule-based")
            self._create_verdict(
                f"v-{run_id}-{test_id}-det",
                run_id, test_id, "judge-det",
                judge_method="det",
                result=r.get("det_result"),
                correct=r.get("det_correct", False),
                expected="PASS" if r.get("is_correct") else "FAIL",
                category=category,
            )

    def _hydrate_poc_subtle(self, r, run_id, test_id, category):
        for method, label in [
            ("no_rubric", "No-Rubric LLM"),
            ("with_rubric", "With-Rubric LLM"),
        ]:
            correct_key = f"{method}_correct"
            verdict_key = f"{method}_verdict"
            if correct_key not in r:
                continue
            judge_id = f"judge-{method}"
            self._ensure_judge(judge_id, method, label, "claude-haiku-4.5")
            self._create_verdict(
                f"v-{run_id}-{test_id}-{method}",
                run_id, test_id, judge_id,
                judge_method=method,
                result=r.get(verdict_key),
                correct=r.get(correct_key, False),
                expected="PASS" if r.get("is_correct") else "FAIL",
                category=category,
                latency_ms=r.get(f"{method}_ms"),
                is_false_pass=r.get(f"{method}_false_pass", False),
            )

    def hydrate_all(self, evals_dir: str = None):
        evals_dir = evals_dir or str(Path(__file__).parent)
        result_files = sorted(Path(evals_dir).glob("*_results.json"))
        total = {"objects": 0, "links": 0}
        for rf in result_files:
            h = self.hydrate_from_results(str(rf))
            total["objects"] += h["objects"]
            total["links"] += h["links"]
            self.hydrated = {"objects": 0, "links": 0}
        print(f"\nTotal hydrated: {total['objects']} objects, {total['links']} links")
        return total


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  READ LAYER — Semantic queries over the ontology
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OntologyReader:
    """
    READS side of the ontology — semantic queries that power
    analytics, workflows, and integrations.
    """

    def __init__(self, engine: OntologyEngine):
        self.engine = engine

    def accuracy_heatmap(self) -> dict:
        runs = self.engine.get_objects_by_type(ObjectType.EVAL_RUN.value)
        heatmap = {}
        for run in runs:
            heatmap[run.object_id] = run.accuracy_by_judge
        return heatmap

    def bias_report(self) -> dict:
        biases = self.engine.get_objects_by_type(ObjectType.BIAS_PATTERN.value)
        by_type = defaultdict(list)
        for b in biases:
            by_type[b.prop("bias_type")].append(b.object_id)
        return {k: {"count": len(v), "ids": v} for k, v in by_type.items()}

    def hardest_tests(self, top_n: int = 5) -> list[dict]:
        tests = self.engine.get_objects_by_type(ObjectType.TEST_CASE.value)
        scored = []
        for tc in tests:
            score = tc.difficulty_score
            if score > 0:
                scored.append({
                    "test_id": tc.object_id,
                    "category": tc.prop("category"),
                    "difficulty": round(score, 2),
                })
        scored.sort(key=lambda x: x["difficulty"], reverse=True)
        return scored[:top_n]

    def judge_leaderboard(self) -> list[dict]:
        judges = self.engine.get_objects_by_type(ObjectType.JUDGE.value)
        board = []
        for j in judges:
            board.append({
                "judge_id": j.object_id,
                "label": j.prop("label"),
                "method": j.prop("method"),
                "accuracy": round(j.accuracy, 3),
                "false_pass_rate": round(j.false_pass_rate, 3),
                "total_cost": round(j.total_cost, 6),
                "bias_profile": j.bias_profile,
            })
        board.sort(key=lambda x: x["accuracy"], reverse=True)
        return board

    def category_breakdown(self) -> list[dict]:
        categories = self.engine.get_objects_by_type(ObjectType.CATEGORY.value)
        breakdown = []
        for cat in categories:
            breakdown.append({
                "category": cat.prop("name"),
                "module": cat.prop("module"),
                "test_count": cat.test_count,
            })
        return breakdown

    def cost_analysis(self) -> dict:
        verdicts = self.engine.get_objects_by_type(ObjectType.VERDICT.value)
        total_cost = sum(v.prop("cost_usd") or 0 for v in verdicts)
        by_judge = defaultdict(float)
        for v in verdicts:
            by_judge[v.prop("judge_method")] += v.prop("cost_usd") or 0
        return {
            "total_cost_usd": round(total_cost, 6),
            "by_judge": {k: round(v, 6) for k, v in by_judge.items()},
            "total_verdicts": len(verdicts),
            "cost_per_verdict": round(total_cost / len(verdicts), 6) if verdicts else 0,
        }

    def full_dashboard(self) -> dict:
        return {
            "ontology_stats": self.engine.stats(),
            "judge_leaderboard": self.judge_leaderboard(),
            "bias_report": self.bias_report(),
            "hardest_tests": self.hardest_tests(),
            "cost_analysis": self.cost_analysis(),
            "category_breakdown": self.category_breakdown(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN — Hydrate + Query Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("ONTOLOGY ENGINE — Palantir-style Hydration")
    print("=" * 70)

    # Clean slate
    if DB_PATH.exists():
        DB_PATH.unlink()

    # 1. Initialize engine
    engine = OntologyEngine()
    print(f"\nEngine initialized: {engine.db_path}")

    # 2. Register automations
    def on_bias_detected(eng, obj):
        if obj.object_type == ObjectType.BIAS_PATTERN.value:
            print(f"  [AUTOMATION] Bias detected: {obj.prop('bias_type')} — {obj.object_id}")

    engine.register_automation(
        "bias_alert", "object_created", on_bias_detected
    )

    # 3. HYDRATE — Write raw eval data into ontology
    print("\n" + "─" * 70)
    print("HYDRATION PIPELINE (WRITES)")
    print("─" * 70)
    pipeline = HydrationPipeline(engine)
    pipeline.hydrate_all()

    # 4. READ — Query the ontology
    print("\n" + "─" * 70)
    print("ONTOLOGY READS")
    print("─" * 70)
    reader = OntologyReader(engine)

    stats = engine.stats()
    print(f"\nOntology Stats:")
    for obj_type, count in stats["objects"].items():
        if count > 0:
            print(f"  {obj_type:<20}: {count}")
    print(f"  {'Total Objects':<20}: {stats['total_objects']}")
    print(f"  {'Total Links':<20}: {stats['total_links']}")

    print(f"\nJudge Leaderboard:")
    for j in reader.judge_leaderboard():
        print(f"  {j['label']:<20} acc={j['accuracy']:.1%}  FP={j['false_pass_rate']:.1%}  "
              f"cost=${j['total_cost']:.4f}  biases={j['bias_profile']}")

    print(f"\nBias Report:")
    for bias_type, info in reader.bias_report().items():
        print(f"  {bias_type:<20}: {info['count']} instances")

    print(f"\nHardest Tests:")
    for t in reader.hardest_tests():
        print(f"  {t['test_id']:<30} [{t['category']}] difficulty={t['difficulty']}")

    print(f"\nCost Analysis:")
    cost = reader.cost_analysis()
    print(f"  Total: ${cost['total_cost_usd']:.4f} across {cost['total_verdicts']} verdicts")
    print(f"  Per verdict: ${cost['cost_per_verdict']:.6f}")
    for method, c in cost["by_judge"].items():
        print(f"    {method}: ${c:.6f}")

    # 5. ACTIONS — Execute through the ontology
    print(f"\n" + "─" * 70)
    print("ACTIONS")
    print("─" * 70)
    result = engine.execute_action("promote_judge", "judge-wr")
    print(f"  Promote judge-wr: {result['status']}")

    result = engine.execute_action("rerun_failed", "comprehensive_20260905_035430")
    print(f"  Rerun failed: {result}")

    # Save dashboard
    dashboard = reader.full_dashboard()
    dash_path = Path(__file__).parent / "ontology_dashboard.json"
    with open(dash_path, "w") as f:
        json.dump(dashboard, f, indent=2, default=str)
    print(f"\nDashboard saved: {dash_path.name}")

    print("\n" + "=" * 70)
    print("ONTOLOGY HYDRATED — Ready for Analytics, Workflows, Integrations")
    print("=" * 70)
