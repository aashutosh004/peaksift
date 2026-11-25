#!/usr/bin/env bash
# PeakSift command line from this folder:  ./peaksift.sh clip "<link>"   ./peaksift.sh --help
DIR="$(cd "$(dirname "$0")" && pwd)"
[ -x "$DIR/.venv/bin/python" ] || { echo "Run ./setup.sh first."; exit 1; }
exec "$DIR/.venv/bin/python" -m peaksift "$@"
