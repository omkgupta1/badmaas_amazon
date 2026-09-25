#!/usr/bin/env bash
# End-to-end: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.
# Usage:  ./run_all.sh [configs/default.yaml|configs/dev.yaml]
# Every stage caches its result under work/, so an interrupted run resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")"
CFG="${1:-configs/default.yaml}"
# use the active virtualenv, else ./.venv if it exists, else whatever `python` is on PATH
if [ -z "${VIRTUAL_ENV:-}" ] && [ -x .venv/bin/python ]; then
  PYBIN=.venv/bin/python
else
  PYBIN="${PYTHON:-python}"
fi
PY=("$PYBIN" -m src.run_pipeline --config "$CFG")

# keep the Mac awake for the whole run (no-op elsewhere)
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ]; then
  export CAFFEINATED=1
  exec caffeinate -dimsu "$0" "$@"
fi

"${PY[@]}" prep --split train
"${PY[@]}" prep --split test
"${PY[@]}" block --split train
"${PY[@]}" block --split test
"${PY[@]}" features --split train
"${PY[@]}" features --split test
"${PY[@]}" train-a
"${PY[@]}" predict-a --split test
"${PY[@]}" collective --split train
"${PY[@]}" collective --split test
"${PY[@]}" all          # neural (if enabled) + stage B + decision + outputs + summary
