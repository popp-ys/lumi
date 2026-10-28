#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Backend venv missing. Run ./install.sh first.' >&2
  exit 1
fi
exec .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8765
