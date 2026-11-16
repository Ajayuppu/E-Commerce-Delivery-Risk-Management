#!/bin/bash
# ml-project2-v2: build the project environment inside the SageMaker Studio JupyterLab space,
# register the Jupyter kernel "Python (ml-project2-v2)", execute all notebooks in order and
# copy the executed notebooks + outputs back to S3.
#   Usage (Studio terminal):  bash ~/ml-project2-v2/aws/studio_bootstrap.sh [--no-run]
set -uo pipefail
PROJ="$HOME/ml-project2-v2"
BUCKET="${PROJECT_BUCKET:-ml-project2-olist-044450716277}"
PREFIX="${PROJECT_PREFIX:-ml-project2-v2}"
LOG="$PROJ/logs/studio_run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$PROJ/logs"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] bootstrap start on $(hostname) as $(whoami)"
cat /opt/ml/metadata/resource-metadata.json 2>/dev/null; echo
PY=$(command -v python3)
echo "base python: $PY $($PY --version)"

# 1) isolated virtual environment with the pinned versions (does not touch the Studio base env)
if [ ! -x "$PROJ/.venv/bin/python" ]; then
  $PY -m venv "$PROJ/.venv"
fi
"$PROJ/.venv/bin/pip" install -q --upgrade pip
"$PROJ/.venv/bin/pip" install -q -r "$PROJ/requirements.txt"
"$PROJ/.venv/bin/python" -m ipykernel install --user --name mlproj2 --display-name "Python (ml-project2-v2)"
"$PROJ/.venv/bin/pip" freeze > "$PROJ/logs/pip_freeze_studio.txt"
echo "[$(date)] environment ready"

# 2) raw data (never re-downloaded if present)
mkdir -p "$PROJ/data/raw"
aws s3 sync "s3://$BUCKET/$PREFIX/data/raw/" "$PROJ/data/raw/" --only-show-errors
ls -l "$PROJ/data/raw"

if [ "${1:-}" == "--no-run" ]; then echo "environment only"; exit 0; fi

# 3) execute all notebooks in order (outputs are written into the notebooks themselves)
cd "$PROJ"
START=$(date +%s)
"$PROJ/.venv/bin/python" run_all.py
RC=$?
echo "[$(date)] run_all exit code $RC, $(( $(date +%s) - START ))s"

# 4) copy executed notebooks + outputs back to S3 as evidence of the Studio run
RUN="studio-run"
aws s3 sync "$PROJ/notebooks" "s3://$BUCKET/$PREFIX/$RUN/notebooks/" --only-show-errors
aws s3 sync "$PROJ/outputs" "s3://$BUCKET/$PREFIX/$RUN/outputs/" --only-show-errors --exclude "models/*"
aws s3 sync "$PROJ/outputs/models" "s3://$BUCKET/$PREFIX/$RUN/models/" --only-show-errors
aws s3 cp "$LOG" "s3://$BUCKET/$PREFIX/$RUN/logs/" --only-show-errors
aws s3 cp "$PROJ/logs/pip_freeze_studio.txt" "s3://$BUCKET/$PREFIX/$RUN/logs/" --only-show-errors
echo "$RC $(date -Iseconds)" > "$PROJ/logs/RUN_DONE"
aws s3 cp "$PROJ/logs/RUN_DONE" "s3://$BUCKET/$PREFIX/$RUN/RUN_DONE" --only-show-errors
echo "[$(date)] bootstrap finished"
