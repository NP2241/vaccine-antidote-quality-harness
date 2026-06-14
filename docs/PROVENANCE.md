# Provenance

This repository is a **cleaned public snapshot** of code from the [**CS639 class project**](https://github.com/zidage/cs639_project/tree/yurun_sft) (`yurun_sft` branch). The original tree mixed alignment experiments, PKU Antidote eval utilities, notebooks, and large JSON artifacts. This repo keeps the **runnable core** and documents what was left behind.

## Original repository

| | |
|---|---|
| **Repo** | [github.com/zidage/cs639_project (`yurun_sft`)](https://github.com/zidage/cs639_project/tree/yurun_sft) |
| **Also consolidated from** | `Anda_Vaccine_Repnoise_baseline_(in_progress)` branch (alignment methods) |
| **This repo** | [github.com/NP2241/vaccine-antidote-quality-harness](https://github.com/NP2241/vaccine-antidote-quality-harness) |

## File mapping (class project → this repo)

### Alignment stage (`Anda_Vaccine_Repnoise_baseline` branch)

| Original (`cs639_project`) | Here | Notes |
|----------------------------|------|-------|
| `main.py` | `src/beq/cli.py` | Flat script → `beq` subcommands |
| `run_alignment.py` | `src/beq/core/run_alignment.py` | Package imports |
| `artifact_utils.py` | `src/beq/artifacts/io.py` | Renamed |
| `config_utils.py` | `src/beq/core/config.py` | Unchanged behavior |
| `model_utils.py` | `src/beq/core/models.py` | Unchanged behavior |
| `data_utils.py` | `src/beq/data/datasets.py` | Minor style edits |
| `vaccine_aligner.py` | `src/beq/methods/vaccine.py` | |
| `repnoise_aligner.py` | `src/beq/methods/repnoise.py` | |
| `sft_aligner.py` | `src/beq/methods/sft.py` | |
| `configs/*.json` | `configs/*.json` | Paths → `examples/data/`, `examples/outputs/` |

### Eval stage (`yurun_env_setup_sft_baseline` branch)

| Original (`Antidote/poison/evaluation/`) | Here | Notes |
|--------------------------------------------|------|-------|
| `constants.py` | `src/beq/evals/constants.py` | |
| `utils.py` | `src/beq/evals/pku_utils.py` | Renamed; Apache-2.0 header kept |
| `moderation.py` | `src/beq/evals/moderation.py` | Removed hardcoded cache path |
| `pred.py` | `src/beq/evals/generate.py` | Rewritten as `run_generate`; env token |
| `eval_sentiment.py` | `src/beq/evals/judge.py` | Rewritten as `run_judge`; fixed JSON shape |

### New in this repo (not copied)

- `src/beq/cli.py` — unified entrypoint
- `src/beq/evals/report.py` — `pipeline_report.json` helper
- `tests/` — import and schema smoke tests
- `examples/` — tiny JSON/JSONL fixtures
- `scripts/run_local_demo.sh`
- `docs/` — findings, workflow, this file

## Intentionally omitted

- `__pycache__`, committed `.pyc` files
- `huggingface_token.txt` and similar secret-on-disk patterns
- Large committed eval JSON (multi-thousand-line runs)
- Notebooks, paper PDFs, posters
- Slurm / large finetune script grids
- Full Antidote `train.py` / `trainer.py` / GSM8K stack
- Duplicate `fabien_eval_baseline` tree under `hw3_advancements/`

## Third-party attribution

- **PKU-Alignment** — `src/beq/evals/constants.py`, `pku_utils.py`, `moderation.py` (Apache-2.0)
- **git-disl** — Vaccine / Antidote research context; methods here are simplified teaching implementations

## Operational notes

- Set `HF_TOKEN` for `meta-llama/*` and `PKU-Alignment/beaver-dam-7b`
- Alignment and full eval runs expect a capable GPU
- CI runs `pytest` only (CPU, no model downloads)
