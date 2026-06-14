# Example eval inputs

## `sample_prompts.json`

A minimal JSON array of `{ "instruction": "..." }` objects for local runs without downloading BeaverTails.

```bash
beq eval-generate --instruction-path examples/eval/sample_prompts.json --output-path examples/outputs/sample_generations.json ...
beq eval-judge --input-path examples/outputs/sample_generations.json --output-path examples/outputs/sample_generations_judged.json
```

## `sample_generations_judged.placeholder.json`

Shows the judged JSON shape (`summary` + `results` with `violated_categories`) before you run a full model download.
