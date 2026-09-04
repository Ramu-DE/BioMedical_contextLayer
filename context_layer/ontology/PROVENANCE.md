# Vendored assets

Source: https://github.com/ramu-de/BioMedical_KnowledgeGraph_ontology_MCP
Commit: f8fb947e06b17a255c7343b69b5262fbc315cc04
Vendored: 2026-08-27T17:42:10Z
Path:   neo4j-neptune-mcp-platform/

| Asset | Destination | Count |
|---|---|---|
| OWL/TTL ontology modules | `ontology/*.ttl` | 8 |
| SHACL shapes | `src/context_layer/knowledge/shapes/*.ttl` | 10 |
| Node CSVs | `data/nodes/*.csv` | 32 |
| Relationship CSVs | `data/relationships/*.csv` | 37 |

Vendored because /tmp is volatile; the loader must not depend on a clone
surviving between sessions.
