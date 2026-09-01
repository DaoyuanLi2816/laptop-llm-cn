#!/usr/bin/env bash
set -euo pipefail

python -m laptop_llm pipeline --config configs/smoke.yaml
python -m pytest -q

echo "Smoke pipeline 与测试均已通过。"
