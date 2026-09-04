#!/usr/bin/env bash
# Run the context layer locally.
#
# PYTHONPATH must be set BEFORE python starts: `python -m` resolves the module
# before server.py can extend sys.path. The Dockerfile sets the same value.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/src"
export APP_PORT="${APP_PORT:-18080}"
export HOOKS_PORT="${HOOKS_PORT:-19000}"
# 8080 is often occupied by other local services; default to 18080.
echo "context layer -> http://localhost:${APP_PORT}"
echo "  health: curl localhost:${APP_PORT}/health"
echo "  ask:    curl -X POST localhost:${APP_PORT}/invoke -H 'Content-Type: application/json' \\"
echo "            -d '{\"question\":\"...\"}'"
exec .venv/bin/python -m context_layer.agent.server
