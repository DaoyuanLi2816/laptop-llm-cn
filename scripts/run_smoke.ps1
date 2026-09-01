$ErrorActionPreference = "Stop"

python -m laptop_llm pipeline --config configs/smoke.yaml
python -m pytest -q

Write-Host "Smoke pipeline 与测试均已通过。"
