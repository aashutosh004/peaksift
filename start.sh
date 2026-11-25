#!/usr/bin/env bash
# Start the PeakSift web app (opens http://localhost:8000). Ctrl+C to stop.
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Run ./setup.sh first."; exit 1; }
exec .venv/bin/python -m peaksift serve "$@"
