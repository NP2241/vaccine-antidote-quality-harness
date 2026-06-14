# Quality workflow

End-to-end flow: **config → train (optional) → generate → judge → inspect JSON**.

## Steps

### 1. Alignment (optional)

`beq align --config configs/<method>_example.json`

Writes under `alignment.output_dir` (e.g. `examples/outputs/sft_demo/`):

| File | Contents |
|------|----------|
| `artifact.json` | `method_name`, model paths, `stage2_ready`, base model id |
| `train_metrics.json` | Step count, last loss |
| `merged_model/`, `tokenizer/` | Saved weights |

Swap method via config: `sft`, `vaccine`, or `repnoise`.

### 2. Behavioral eval generation

`beq eval-generate --instruction-path ... --output-path ...`

Input: JSON array of `{ "instruction": "..." }` or the `BeaverTails` dataset keyword.

Output: JSON array of `{ "instruction", "output" }`.

### 3. Moderation judge

`beq eval-judge --input-path <generations.json> --output-path <judged.json>`

Output shape (see `examples/eval/sample_generations_judged.placeholder.json`):

- **`summary`** — `samples`, `flagged_count`, `harmful_rate_percent`, `moderation_model`
- **`results`** — each row plus `violated_categories`

### 4. Pipeline report (optional)

`beq pipeline --config ... --report-out examples/outputs/pipeline_report.json`

Runs alignment only and writes a timestamped report. Eval paths are **not** auto-filled — run generate/judge separately and merge reporting manually if needed.

## What to inspect

| Output | Check |
|--------|-------|
| Generations JSON | Row count; outputs match prompt intent |
| Judged JSON | `summary` aggregates; per-row categories for failures |
| `artifact.json` | Keys match `tests/test_artifact_schema.py` |
| `pipeline_report.json` | Artifact path and timestamp when alignment ran |

## Example judged output (placeholder)

```json
{
  "summary": {
    "samples": 2,
    "flagged_count": 0,
    "harmful_rate_percent": 0.0,
    "moderation_model": "PKU-Alignment/beaver-dam-7b"
  },
  "results": [
    {
      "instruction": "...",
      "output": "...",
      "violated_categories": []
    }
  ]
}
```
