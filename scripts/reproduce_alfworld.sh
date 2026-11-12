#!/usr/bin/env bash
set -euo pipefail
collection_config=${1:?Provide a collection YAML}
training_config=${2:?Provide an online-training YAML}
evaluation_config=${3:?Provide a held-out evaluation YAML}
work_dir=${4:?Provide a new output directory for verified data and tables}
python scripts/run_pipeline.py --collection "$collection_config" --training "$training_config" --evaluation "$evaluation_config" --out "$work_dir"
