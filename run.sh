#!/bin/bash
# run.sh — Run QBase strategies via AlphaForge
# Usage: ./run.sh strategies/ag/trend_v1.py --symbols AG --freq daily --start 2022

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AF_PATH=$(python3 -c "import yaml; c=yaml.safe_load(open('$SCRIPT_DIR/config.yaml')); from pathlib import Path; print(Path(c['alphaforge']['path']).expanduser())")

export PYTHONPATH="$SCRIPT_DIR:$AF_PATH:$PYTHONPATH"
af run "$@"
