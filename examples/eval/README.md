# Example eval inputs

## `sample_prompts.json`

A minimal **JSON array** of objects with an `instruction` field. These are passed to **`beq eval-generate`** (see the root **README**) to produce `{instruction, output}` rows for **behavioral evaluation** and then **`beq eval-judge`** for a **guardrail-style** pass.

The prompts are **generic**: they mention reproducible evals, pull requests, CLI-style quality, and agent scenarios **only as illustration**—this repo does not run Gemini CLI or real agent tools.

Use this file to iterate locally without downloading the BeaverTails dataset for instructions.
