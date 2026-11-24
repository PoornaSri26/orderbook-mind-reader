#!/usr/bin/env bash
# Stand-in for `make reproduce` where make is unavailable (Windows).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then stages=(dataset baselines model interp); fi
for s in "${stages[@]}"; do
  echo "== stage: $s"
  case "$s" in
    data)     PYTHONPATH=src "$PY" scripts/download_bookticker.py --days "${DAYS:-7}" --end 2024-03-30 ;;
    dataset)  PYTHONPATH=src "$PY" scripts/build_dataset.py ;;
    baselines) PYTHONPATH=src "$PY" scripts/run_baselines.py ;;
    model)    PYTHONPATH=src "$PY" scripts/run_model.py ;;
    interp)   PYTHONPATH=src "$PY" scripts/run_interp.py ;;
    *) echo "unknown stage: $s" >&2; exit 2 ;;
  esac
done
