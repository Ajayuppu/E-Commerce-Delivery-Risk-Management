#!/bin/bash
# Lifecycle configuration (JupyterLab app, domain machinelearningproj2ajay-ecommerce).
# 1) First start: copy the project from S3 into the Studio home folder (never overwrites later edits).
# 2) Until a complete run exists (logs/RUN_DONE), start the bootstrap in the background:
#    venv + kernel "Python (ml-project2-v2)" + execute notebooks 00-06 + copy results to S3.
set -eux
PROJ=/home/sagemaker-user/ml-project2-v2
BUCKET=ml-project2-olist-044450716277
PREFIX=ml-project2-v2
if [ ! -f "$PROJ/.synced_from_s3" ]; then
  mkdir -p "$PROJ"
  aws s3 sync "s3://$BUCKET/$PREFIX/studio/" "$PROJ/" --only-show-errors
  date > "$PROJ/.synced_from_s3"
fi
if [ ! -f "$PROJ/logs/RUN_DONE" ]; then
  chmod +x "$PROJ/aws/studio_bootstrap.sh"
  if [ "$(id -u)" = "0" ]; then
    chown -R 1000:100 "$PROJ"
    nohup sudo -u sagemaker-user -H bash "$PROJ/aws/studio_bootstrap.sh" > /home/sagemaker-user/ml-project2-v2-bootstrap.out 2>&1 &
  else
    nohup bash "$PROJ/aws/studio_bootstrap.sh" > /home/sagemaker-user/ml-project2-v2-bootstrap.out 2>&1 &
  fi
fi
