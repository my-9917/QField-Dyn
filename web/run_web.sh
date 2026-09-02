#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QFIELD_ROOT="$ROOT_DIR"
export QFIELD_WEB_HOST="${QFIELD_WEB_HOST:-127.0.0.1}"
export QFIELD_WEB_PORT="${QFIELD_WEB_PORT:-8765}"
exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/web/server.py"
