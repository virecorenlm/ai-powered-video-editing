#!/usr/bin/env bash
set -euo pipefail

python -m compileall main.py editor.py storyteller.py
pytest -q || {
  status=$?
  if [ "$status" -eq 5 ]; then
    echo "No tests discovered (pytest exit code 5)."
    exit 0
  fi
  exit "$status"
}
