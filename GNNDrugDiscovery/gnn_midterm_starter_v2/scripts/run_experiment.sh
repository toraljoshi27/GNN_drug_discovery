#!/usr/bin/env bash
set -e
CFG=${1:-configs/exp_tox21_gcn.yaml}
python src/train.py --config $CFG
python src/eval.py --config $CFG
