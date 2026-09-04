#!/usr/bin/env python3
"""Verify Neo4j connectivity and survey the curated knowledge graph.

Runs independently of OpenSearch so Phase 1 can proceed before the vector
store is configured. Never prints credential values.

Usage:
    .venv/bin/python scripts/verify_neo4j.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config as config_module  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402
from neo4j.exceptions import AuthError, ClientError, ConfigurationError  # noqa: E402


def try_auth(uri: str, user: str, password: str) -> tuple[bool, str]:
    """Attempt connectivity with one username. Returns (ok, detail)."""
    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return True, "authenticated"
    except AuthError as e:
        return False, f"auth failed: {e.code if hasattr(e, 'code') else e}"
    except (ConfigurationError, ClientError) as e:
        return False, f"client error: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        if driver is not None:
            driver.close()


def survey(driver, database: str) -> dict:
    """Describe what is actually in the graph."""
    out: dict = {}
    with driver.session(database=database) as s:
        out["labels"] = sorted(
            r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")
        )
        out["rel_types"] = sorted(
            r["relationshipType"]
            for r in s.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType"
            )
        )
        out["node_count"] = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        out["rel_count"] = s.run(
            "MATCH ()-[r]->() RETURN count(r) AS c"
        ).single()["c"]
        out["per_label"] = {
            r["label"]: r["c"]
            for r in s.run(
                "MATCH (n) UNWIND labels(n) AS label "
                "RETURN label, count(*) AS c ORDER BY c DESC LIMIT 40"
            )
        }
        try:
            out["edition"] = s.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN edition LIMIT 1"
            ).single()["edition"]
        except Exception:  # noqa: BLE001
            out["edition"] = "unknown"
    return out


def check_write(driver, database: str, prefix: str) -> tuple[bool, str]:
    """Confirm we can write, then remove the probe. Uses the Ctx_ namespace."""
    label = f"{prefix}WriteProbe"
    try:
        with driver.session(database=database) as s:
            s.run(
                f"CREATE (n:`{label}` {{probe_id: $id, at: datetime()}})",
                id="verify-probe",
            )
            found = s.run(
                f"MATCH (n:`{label}` {{probe_id: $id}}) RETURN count(n) AS c",
                id="verify-probe",
            ).single()["c"]
            s.run(
                f"MATCH (n:`{label}` {{probe_id: $id}}) DELETE n", id="verify-probe"
            )
        return (found == 1), f"wrote and removed a :{label} node"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def list_databases(driver) -> list[str]:
    """User-visible databases, excluding system. Aura may name the single
    database after the instance ID rather than 'neo4j'."""
    try:
        with driver.session(database="system") as s:
            names = [
                r["name"] for r in s.run("SHOW DATABASES YIELD name RETURN name")
            ]
        seen: dict[str, None] = {}
        for n in names:
            if n != "system":
                seen.setdefault(n, None)
        return list(seen)
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    cfg = config_module.load()
    if not cfg.neo4j_uri or not cfg.neo4j_password:
        print("NEO4J_URI or NEO4J_PASSWORD is not set — fill in .env first")
        return 1

    print(f"uri       {cfg.neo4j_uri}")
    print(f"password  set [{cfg.neo4j_password.fingerprint()}]")
    print()

    # Aura's default user is "neo4j", but some instances use the instance ID.
    instance_id = cfg.neo4j_uri.split("//")[-1].split(".")[0]
    candidates = [cfg.neo4j_user]
    for extra in ("neo4j", instance_id):
        if extra not in candidates:
            candidates.append(extra)

    working: str | None = None
    print("username candidates:")
    for user in candidates:
        ok, detail = try_auth(cfg.neo4j_uri, user, str(cfg.neo4j_password))
        print(f"  {user:<12} {'OK  ' if ok else 'FAIL'} {detail}")
        if ok and working is None:
            working = user
    print()

    if working is None:
        print("no username authenticated — check the password in Aura Console")
        return 1
    if working != cfg.neo4j_user:
        print(f"ACTION: set NEO4J_USER={working} in .env (currently {cfg.neo4j_user!r})")
        print()

    driver = GraphDatabase.driver(
        cfg.neo4j_uri, auth=(working, str(cfg.neo4j_password))
    )
    try:
        dbs = list_databases(driver)
        kg_db = cfg.neo4j_kg_database
        if dbs:
            print(f"user databases: {dbs}")
            if kg_db not in dbs:
                fallback = dbs[0]
                print(
                    f"ACTION: NEO4J_KG_DATABASE={kg_db!r} does not exist. "
                    f"Set NEO4J_KG_DATABASE and NEO4J_CTX_DATABASE to {fallback!r}."
                )
                kg_db = fallback
            if len(dbs) == 1:
                print(
                    f"single database — runtime context isolated by label prefix "
                    f"{cfg.ctx_label_prefix!r} (multi-db unavailable)"
                )
            print()

        try:
            info = survey(driver, kg_db)
        except ClientError as e:
            print(f"cannot survey database {kg_db!r}: {e}")
            return 1

        print(f"database       {kg_db}")
        print(f"edition        {info['edition']}")
        print(f"nodes          {info['node_count']:,}")
        print(f"relationships  {info['rel_count']:,}")
        print(f"labels ({len(info['labels'])})      {info['labels'][:25]}")
        print(f"rel types ({len(info['rel_types'])})   {info['rel_types'][:25]}")
        if info["per_label"]:
            print("\ntop labels by count:")
            for label, count in list(info["per_label"].items())[:20]:
                print(f"  {count:>8,}  {label}")

        ok, detail = check_write(driver, kg_db, cfg.ctx_label_prefix)
        print(f"\nwrite access   {'OK' if ok else 'FAIL'} — {detail}")

        if info["node_count"] == 0:
            print(
                "\nWARNING: the graph is empty. Task 1.6 (load_kg.py) will need to "
                "load the 69 node/relationship CSVs before linking can work."
            )
        return 0 if ok else 2
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
