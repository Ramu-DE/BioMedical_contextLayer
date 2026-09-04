#!/usr/bin/env bash
# Context Layer dashboard, served through this workshop's existing nginx proxy.
#
# nginx already has:  location /app -> http://localhost:8081/app  (with websocket
# upgrade headers), so the dashboard is reachable at the same CloudFront URL as
# the IDE. No nginx edit, no security group change, no extra port exposed.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/src"
export UI_HOST="${UI_HOST:-127.0.0.1}"
export UI_PORT="${UI_PORT:-8081}"
export UI_ROOT_PATH="${UI_ROOT_PATH:-/ctx}"

if ! python3 -c "
import socket,sys
s=socket.socket()
try: s.bind(('127.0.0.1', ${UI_PORT}))
except OSError: sys.exit(1)
finally: s.close()" 2>/dev/null; then
  echo "ERROR: port ${UI_PORT} is in use." >&2
  exit 1
fi

cat <<BANNER

  ┌──────────────────────────────────────────────────────────────┐
  │  Context Layer dashboard                                     │
  │                                                              │
  │  Open the SAME URL as your IDE, with /ctx/ on the end:        │
  │                                                              │
  │      https://<your-workshop-domain>/ctx/                      │
  │                                                              │
  │  (Not localhost — this runs on an EC2 box behind CloudFront.) │
  └──────────────────────────────────────────────────────────────┘

BANNER

exec .venv/bin/python -m context_layer.ui.app
