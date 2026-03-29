# Quality workflow

This document describes the **config → run → artifact → eval → judge → report** flow implemented in this demo and what a contributor would **inspect** at each step.

## Flow

1. **Config (optional alignment)** — JSON under `configs/` describes model and alignment method. Running `beq align` (or `beq pipeline`) executes training and writes:
   - `artifact.json` — stable metadata and paths to saved weights/tokenizer
   - `train_metrics.json` — lightweight training diagnostics
   - `merged_model/`, `tokenizer/` — on disk under `alignment.output_dir`

2. **Behavioral eval generation** — `beq eval-generate` reads instructions (JSON file or BeaverTails subset) and writes a **JSON array** of `{ "instruction": "...", "output": "..." }`.

3. **Guardrail-style judging** — `beq eval-judge` consumes that JSON and writes a file with:
   - `summary` — e.g. sample count, flagged count, aggregate rate, moderation model id
   - `results` — original rows plus `violated_categories` per row

4. **Report (optional)** — `write_pipeline_report` (used from `align` / `pipeline` with `--write-report` / `--report-out`) writes `pipeline_report.json` with UTC timestamp, artifact path, and steps. **Today** eval output paths are **not** auto-attached; run eval separately and merge reporting in a future iteration if needed.

## What to inspect

| Output | What to look for |
|--------|-------------------|
| Generations JSON | Row count, sensible `output` text for each `instruction` |
| Judged JSON | `summary` aggregates; per-row `violated_categories` for debugging |
| `artifact.json` | Keys match expectations in tests (`tests/test_artifact_schema.py`) |
| `pipeline_report.json` | When alignment ran, path to `artifact.json`, timestamp |

## How this could evolve

- **Single orchestrated command** that runs align → generate → judge and fills `eval_paths` in the report.
- **CI** that runs `pytest` on every PR and, with policy, optional GPU jobs or scheduled evals.
- **Threshold gate** that exits non-zero when aggregate metrics regress beyond a budget.
- **Richer diagnostics** (structured logs, trace ids) for failures in generate/judge.

These extensions align with **OSS quality ecosystems** and **Gemini CLI–style** iteration without changing the core idea: **structured eval artifacts** and **automated judging** as first-class outputs.
