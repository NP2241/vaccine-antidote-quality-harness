# Behavioral evals & quality gates — OSS quality harness (demo)

This repository is a **small, proposal-facing demo** of engineering patterns that are **relevant to [Google Summer of Code](https://summerofcode.withgoogle.com/) work on [Gemini CLI](https://github.com/google-gemini/gemini-cli) **quality**: behavioral evaluation, guardrail-style checks, stable artifacts, and contributor-oriented entrypoints. It is **not** part of Gemini CLI and does **not** ship Gemini or agent-runtime code.

The implementation uses an **open Hugging Face stack** (causal LM generation + a moderation-style judge) as a **stand-in**: the transferable ideas are **structured eval I/O, automated scoring, artifact contracts, and a single CLI**—the same disciplines that apply when iterating on prompts, tools, and release quality in a CLI or agent product.

## Why this repo exists / GSoC relevance

Quality-focused GSoC projects depend on **repeatable behavioral evals**, **signals that behave like guardrails or regression checks**, and **OSS workflows** that let contributors run and extend validation without guesswork. This repo demonstrates:

- **Behavioral eval generation** — instructions → model outputs as **JSON** for downstream checks.
- **Guardrail-style judging** — an automated pass that scores each `(instruction, output)` and emits **aggregate and per-row** results.
- **Artifact contracts** — predictable files (`artifact.json`, metrics, reports) that downstream tooling can consume.
- **Iterative validation** — diff-friendly outputs and lightweight tests that document expected schemas.

See **[`docs/GSOC_CONTEXT.md`](docs/GSOC_CONTEXT.md)** for a concise mapping to the Gemini CLI quality scope and **[`docs/quality_workflow.md`](docs/quality_workflow.md)** for the end-to-end flow.

## What this repo does

1. **`beq eval-generate`** — **Behavioral eval generation**: load a causal LM (optional LoRA merge), run instructions from a **JSON file** of `{ "instruction": "..." }` or from **BeaverTails**, write a list of `{instruction, output}`.
2. **`beq eval-judge`** — **Guardrail-style judging**: run **`PKU-Alignment/beaver-dam-7b`** (QAModeration) on each pair and write JSON with a **`summary`** block (e.g. aggregate rates) plus **`results`** (including `violated_categories`).
3. **`beq align`** *(optional)* — Run one of three **alignment methods** (SFT, simplified Vaccine, simplified RepNoise) from a JSON config; writes `merged_model/`, `tokenizer/`, `artifact.json`, and `train_metrics.json` for workflows that need a trained artifact before eval.
4. **`beq pipeline`** — Runs **alignment only** and writes a small **`pipeline_report.json`** pointing at the alignment artifact. It does **not** chain eval steps; run `eval-generate` / `eval-judge` separately (see [Limitations](#limitations-and-future-work)).

Together, the core quality story is: **config → run → artifact → behavioral generations → guardrail-style report** (alignment is optional).

### Mapping: components → GSoC scope

| Repo component | What it demonstrates | Why it matters for Gemini CLI quality (GSoC) |
|----------------|----------------------|-----------------------------------------------|
| `beq eval-generate` | Structured **behavioral eval** outputs (JSON) | Same discipline as capturing and evaluating **turn-level behavior** before ship. |
| `beq eval-judge` | **Guardrail-style** automated scoring + summaries | Analogous to **regression-aware** checks and **pre-release** quality signals (when wired to policy/automation). |
| `artifact.json` / `train_metrics.json` | **Stable artifact contract** after training | Downstream steps need **predictable paths and keys**—like build artifacts feeding validation. |
| `write_pipeline_report` / `pipeline_report.json` | **Traceability** (timestamp, paths, steps) | Foundation for **inspectability** and richer CI/reporting later. |
| `beq` CLI | **Contributor-facing** commands | Single entrypoints mirror how OSS projects expose **quality workflows**. |
| Tests under `tests/` | Schema + import **smoke** checks | **Contributor confidence** and lightweight automation (see [Tests](#tests)). |

## Quality workflow

```text
                    ┌─────────────────────┐
  configs/*.json ──►│ align (optional)    │──► artifact.json, train_metrics.json
                    └──────────┬──────────┘
                               │
  prompts JSON / dataset ──────┼──► eval-generate ──► generations.json
                               │                           │
                               │                           ▼
                               │                    eval-judge ──► judged.json
                               │                    (summary + results)
                               ▼
                    pipeline_report.json (optional; alignment-centric today)
```

**Inspect:** JSON under `examples/outputs/` (or your chosen paths); see [Artifacts produced](#artifacts-produced).

## Quickstart

### 1. Environment

```bash
cd beq_oss_proposal_1
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export HF_TOKEN=...   # required for gated models (Llama, beaver-dam-7b)
```

Paths in `configs/` are relative to the current working directory.

### 2. Behavioral eval (emphasized): generate then judge

Using the **tiny** local prompt file (no BeaverTails download for prompts):

```bash
beq eval-generate \
  --instruction-path examples/eval/sample_prompts.json \
  --output-path examples/outputs/sample_generations.json \
  --model-folder meta-llama/Llama-2-7b-hf \
  --num-test-data 2 \
  --max-new-tokens 128

beq eval-judge \
  --input-path examples/outputs/sample_generations.json \
  --output-path examples/outputs/sample_generations_judged.json
```

Optional script:

```bash
bash scripts/run_local_demo.sh
```

### 3. Optional alignment stage (GPU; downloads base model)

```bash
beq align --config configs/sft_example.json --write-report examples/outputs/pipeline_report.json
```

**Artifacts** (under `alignment.output_dir`, e.g. `examples/outputs/sft_demo/`):

- `artifact.json` — `method_name`, `model_path`, `tokenizer_path`, `stage2_ready`, `base_model_name`
- `train_metrics.json` — step count and last loss (lightweight)
- `merged_model/`, `tokenizer/` — saved weights

### 4. Pipeline helper (alignment only)

Runs alignment and writes a short report path (eval steps are **not** included):

```bash
beq pipeline --config configs/sft_example.json --report-out examples/outputs/pipeline_report.json
```

## Artifacts produced

| Stage | Output | Purpose |
|--------|--------|---------|
| Eval generate | `*.json` list of `{instruction, output}` | Input to judges or human review; **diff-friendly** behavioral eval rows |
| Eval judge | JSON with `summary` + `results` | Aggregate metrics, per-row `violated_categories`; **machine-readable** quality signal |
| Align | `artifact.json` | Stable contract for downstream training or eval consumers |
| Align | `train_metrics.json` | Lightweight training diagnostics |
| Optional report | `pipeline_report.json` | Timestamp, artifact path, steps (eval paths reserved; not auto-filled today) |

## Extending the repo

### How to add a new eval

1. Add a module under `src/beq/evals/` (e.g. `my_eval.py`) that exposes a function taking an `argparse.Namespace` or a small dataclass.
2. Register a subcommand in `src/beq/cli.py` mirroring `eval-generate` / `eval-judge`.
3. Prefer **JSON or JSONL** inputs/outputs so results can be **diffed** in automation once CI is extended.
4. Document required env vars (e.g. `HF_TOKEN`) in the command help text.

Details: **[`CONTRIBUTING.md`](CONTRIBUTING.md)**.

### How to add a new alignment method

1. Implement `prepare()` and `train()` like `src/beq/methods/sft.py`, returning the dict from `save_artifact`.
2. Register the method name in `src/beq/core/run_alignment.py`.
3. Add `configs/your_method_example.json` with any new `method_params` block.

Keep methods **small and documented**; this sample favors clarity over covering every research variant.

## Tests

```bash
pytest -q
```

These tests are **import and schema smoke checks** (no GPU, no full eval run). They help **stabilize contracts** for contributors. **GitHub Actions** runs `pytest` on push and pull request (see `.github/workflows/ci.yml`).

## Limitations and future work

This repo **does not** implement Gemini CLI integration, agent **tools**, **subagents**, **skills**, or **chat-log → eval** generation. It does **not** yet provide threshold-based failing gates, rich structured logging, or a single command that runs align + eval + judge end-to-end. Those are natural extensions aligned with the GSoC scope; see **`docs/GSOC_CONTEXT.md`**.

## Provenance and attribution

Lineage from prior coursework/research codebases and third-party eval utilities is documented in **`EXTRACTION_NOTES.md`** (file mapping, omissions, licensing). That document supports **provenance**; the **product story** of this repo is **behavioral evals, guardrails, and artifacts**, not reproduction of a specific paper.

## License

Files under `src/beq/evals/constants.py`, `pku_utils.py`, and `moderation.py` retain **Apache-2.0** headers from the PKU-Alignment project. Other code is provided for demonstration; confirm licensing before redistributing as part of a larger OSS effort.

## Repository structure

```text
beq_oss_proposal_1/
├── README.md
├── CONTRIBUTING.md
├── EXTRACTION_NOTES.md       # Provenance (sources, omissions)
├── docs/
│   ├── GSOC_CONTEXT.md       # GSoC / Gemini CLI quality mapping
│   └── quality_workflow.md   # End-to-end flow
├── pyproject.toml
├── configs/
├── src/beq/
│   ├── cli.py
│   ├── core/
│   ├── artifacts/
│   ├── data/
│   ├── methods/
│   └── evals/
├── examples/
│   ├── data/
│   ├── eval/
│   └── outputs/              # Default output dir (gitignored when used)
├── scripts/
│   └── run_local_demo.sh
└── tests/
```
