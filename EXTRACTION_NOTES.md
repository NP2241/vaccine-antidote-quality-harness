# Provenance and lineage

This file records **where the code came from**, what was **omitted**, and **licensing** for adapted third-party evaluation code. It supports **auditability**; the **purpose** of the repository is documented in the **README** and **`docs/GSOC_CONTEXT.md`** (behavioral evals, guardrail-style judging, artifacts, contributor CLI).

The codebase was consolidated from earlier **course/research** work into a **compact, proposal-facing** sample. It is **not** a full reproduction of any published method.

## Source branches and files

### Branch `Anda_Vaccine_Repnoise_baseline_(in_progress)` (cs639_project)

| Original file | New location | Notes |
|---------------|--------------|--------|
| `main.py` | `src/beq/cli.py` | Replaced flat script with `beq` subcommands (`align`, `eval-generate`, `eval-judge`, `pipeline`). |
| `run_alignment.py` | `src/beq/core/run_alignment.py` | Imports updated to `beq.methods.*`. |
| `artifact_utils.py` | `src/beq/artifacts/io.py` | Renamed for clarity; logic preserved. |
| `config_utils.py` | `src/beq/core/config.py` | Same behavior (`load_json`, `ensure_dir`). |
| `model_utils.py` | `src/beq/core/models.py` | Same behavior. |
| `data_utils.py` | `src/beq/data/datasets.py` | Same behavior; minor loop style edits. |
| `vaccine_aligner.py` | `src/beq/methods/vaccine.py` | Imports repointed. |
| `repnoise_aligner.py` | `src/beq/methods/repnoise.py` | Imports repointed; harmless forward result assigned to `_`. |
| `sft_aligner.py` | `src/beq/methods/sft.py` | Imports repointed. |
| `configs/*.json` | `configs/*.json` | Paths retargeted to `examples/data/` and `examples/outputs/`. |
| `README.md` | `README.md` | Rewritten for this repo (structure, quickstart, attribution). |

### Branch `yurun_env_setup_sft_baseline` (cs639_project)

| Original file | New location | Notes |
|---------------|--------------|--------|
| `Antidote/poison/evaluation/constants.py` | `src/beq/evals/constants.py` | Imports fixed to package. |
| `Antidote/poison/evaluation/utils.py` | `src/beq/evals/pku_utils.py` | Renamed to avoid confusion with generic “utils”; PKU license header kept. |
| `Antidote/poison/evaluation/moderation.py` | `src/beq/evals/moderation.py` | **Hardcoded `../../cache` removed**; `cache_dir` and `token` parameters added to `from_pretrained`. |
| `Antidote/poison/evaluation/pred.py` | `src/beq/evals/generate.py` | **Rewritten** as `run_generate`: no `huggingface_token.txt`; uses `HF_TOKEN` / `--token`; `.cuda()` replaced with device from model; argparse replaced by CLI wrapper. |
| `Antidote/poison/evaluation/eval_sentiment.py` | `src/beq/evals/judge.py` | **Rewritten** as `run_judge`: fixed JSON shape (`summary` + `results`), renamed `violated_categories`, removed `sys.path` hack. |

### Intentionally omitted

- `__pycache__`, committed `.pyc` files
- `huggingface_token.txt` and any token file pattern
- Large JSON eval artifacts (multi-thousand-line runs)
- Notebooks, paper PDFs, posters
- Slurm / large finetune script grids
- Full Antidote `train.py` / `trainer.py` / GSM8K stack (would balloon scope; can be added later as optional modules)
- `fabien_eval_baseline` duplicate tree under `hw3_advancements/` (content merged conceptually via yurun paths)

### New glue code (not copied from source)

- `src/beq/cli.py` — unified CLI
- `src/beq/evals/report.py` — lightweight `pipeline_report.json` helper
- `tests/*` — minimal import and schema checks
- `examples/*` — tiny JSONL/JSON fixtures
- `scripts/run_local_demo.sh` — optional two-step eval demo
- `docs/*`, `CONTRIBUTING.md` — contributor and reviewer-facing docs

## Follow-ups

- **GPU / memory**: Alignment and full eval runs assume a capable GPU and Hugging Face access for Llama and moderation models.
- **Gated models**: Set `HF_TOKEN` (or pass `--token`) for `meta-llama/*` and `PKU-Alignment/beaver-dam-7b`.
- **CI**: Repository may include a **GitHub Actions** workflow that runs `pytest` on push/PR (CPU-only; no heavy eval jobs). Local full runs remain separate.
- **Transformers versions**: Pin was relaxed vs. original Antidote; if you hit edge cases in `QAModeration`, pin `transformers` explicitly in your deployment.
- **Further consolidation**: `eval-generate` and any future GSM8K path could share a single `GenerationSpec` config object.

## Attribution

- Alignment-stage code and artifact contract derive from internal experimentation on **Vaccine / RepNoise / SFT**-style loops (simplified, not official reproductions).
- `constants.py`, `pku_utils.py`, and `moderation.py` contain **PKU-Alignment** Apache-2.0 headers and were adapted from the Antidote evaluation utilities.
