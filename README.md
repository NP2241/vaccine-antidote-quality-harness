# Vaccine, Antidote & alignment evals

Companion code for a LinkedIn write-up on what I learned experimenting with **harmful fine-tuning defenses** — simplified [Vaccine](https://github.com/git-disl/Vaccine), [RepNoise](https://arxiv.org/abs/2402.01109)-style, and SFT baselines — plus a small **behavioral eval + moderation judge** loop.

**Original repo:** [zidage/cs639_project (`yurun_sft` branch)](https://github.com/zidage/cs639_project/tree/yurun_sft) — our CS639 class project with notebooks, scattered scripts, result dumps, and PKU Antidote eval code.

**This repo** is a cleaner refactoring of that codebase: one CLI (`beq`), stable JSON artifact contracts, tiny fixtures, and a layout that is easier to parse and run. If you want the raw group-project history, start there; if you want the distilled harness, you are in the right place.

> **Not a paper reproduction.** These are teaching-scale implementations meant to compare methods and run evals — not official releases from the git-disl or PKU-Alignment projects.

## What I took away

The longer version lives in [`docs/FINDINGS.md`](docs/FINDINGS.md). In short:

1. **Shared artifact contracts beat one-off scripts.** When SFT, Vaccine, and RepNoise all write the same `artifact.json` + `train_metrics.json` shape, swapping methods is a config change, not a rewrite.
2. **Generate → judge as JSON is the useful quality loop.** Instructions in, completions out, then an automated moderation pass with aggregate and per-row `violated_categories`. Diff-friendly and CI-friendly.
3. **Class-project glue is where time goes.** Token files, `sys.path` hacks, and hardcoded `../../cache` paths were the real cleanup work — the alignment math was already there.
4. **A moderation judge is a guardrail-shaped signal, not a gate.** Beaver-dam-7b gives useful regression-style summaries; wiring thresholds and CI failure is deliberately left as an exercise.

## What this repo does

| Command | Purpose |
|---------|---------|
| `beq align` | Train with **SFT**, **Vaccine**, or **RepNoise** from a JSON config → `artifact.json`, weights, metrics |
| `beq eval-generate` | Run instructions through a causal LM → JSON list of `{instruction, output}` |
| `beq eval-judge` | Score each pair with **beaver-dam-7b** → `summary` + `results` JSON |
| `beq pipeline` | Alignment only + optional `pipeline_report.json` (eval steps are separate today) |

```text
configs/*.json ──► align (optional) ──► artifact.json, train_metrics.json
                                              │
prompts / dataset ────────────────────────────┼──► eval-generate ──► generations.json
                                              │                              │
                                              │                              ▼
                                              │                       eval-judge ──► judged.json
```

See [`docs/quality_workflow.md`](docs/quality_workflow.md) for step-by-step inspection notes.

## Quickstart

**Requirements:** Python 3.10+, a GPU for alignment and full eval runs, and a Hugging Face token for gated models (`meta-llama/*`, `PKU-Alignment/beaver-dam-7b`).

```bash
git clone https://github.com/NP2241/vaccine-antidote-quality-harness.git
cd vaccine-antidote-quality-harness

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

export HF_TOKEN=...   # or pass --token on each command
```

**Behavioral eval (two steps, tiny local prompts):**

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

Or run both via `bash scripts/run_local_demo.sh`.

**Optional alignment (GPU, downloads base model):**

```bash
beq align --config configs/sft_example.json
# configs/vaccine_example.json and configs/repnoise_example.json swap the method
```

**Tests (CPU, no model downloads):**

```bash
pytest -q
```

## Repository layout

```text
vaccine-antidote-quality-harness/
├── README.md
├── docs/
│   ├── FINDINGS.md          # Blog-style takeaways and limitations
│   ├── PROVENANCE.md        # Lineage from cs639_project + third-party code
│   └── quality_workflow.md  # What to inspect at each stage
├── configs/                 # Example alignment configs (SFT / Vaccine / RepNoise)
├── src/beq/                 # CLI, alignment methods, eval pipeline
├── examples/
│   ├── data/                # Tiny JSONL training fixtures
│   └── eval/                # Sample prompts + judged JSON shape
├── scripts/run_local_demo.sh
└── tests/
```

## Provenance

- **Original codebase:** [github.com/zidage/cs639_project (`yurun_sft`)](https://github.com/zidage/cs639_project/tree/yurun_sft) — CS639 group project; this repo refactors the runnable alignment + eval pieces from that tree.
- **Eval utilities:** Adapted from PKU-Alignment Antidote evaluation code (`constants.py`, `pku_utils.py`, `moderation.py` — Apache-2.0 headers retained).
- **Research context:** [Vaccine](https://github.com/git-disl/Vaccine), [Antidote](https://github.com/git-disl/Antidote) harmful fine-tuning defense line.

Full file mapping: [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## License

PKU-adapted files under `src/beq/evals/` retain their **Apache-2.0** headers. Other demonstration code is provided as-is for learning and reference.
