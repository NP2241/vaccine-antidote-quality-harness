#!/usr/bin/env bash
# Minimal behavioral eval + guardrail-style judge (local two-step demo).
# Run from repo root after: pip install -e ".[dev]" && export HF_TOKEN=...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
beq eval-generate \
  --instruction-path examples/eval/sample_prompts.json \
  --output-path examples/outputs/sample_generations.json \
  --model-folder "${MODEL_FOLDER:-meta-llama/Llama-2-7b-hf}" \
  --num-test-data 2 \
  --max-new-tokens 64
beq eval-judge \
  --input-path examples/outputs/sample_generations.json \
  --output-path examples/outputs/sample_generations_judged.json
