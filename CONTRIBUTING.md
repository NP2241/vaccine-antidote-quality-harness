# Contributing

This repository is intentionally **small and demo-oriented**. Changes should stay easy to review and aligned with the **behavioral eval + guardrail + artifact** story.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -q
```

Tests are **smoke-level** (imports, config parsing, documented `artifact.json` keys). They do **not** download gated models or require a GPU.

## Hugging Face token

Full **`beq eval-generate`** / **`beq eval-judge`** / **`beq align`** runs need **`HF_TOKEN`** (or `--token`) for gated models such as `meta-llama/*` and `PKU-Alignment/beaver-dam-7b`.

## Adding a new eval

1. Add logic under `src/beq/evals/` (e.g. new module with a `run_*` function).
2. Register a subcommand in `src/beq/cli.py`.
3. Prefer **JSON/JSONL** inputs and outputs; document env vars in `--help`.

## Adding a new alignment method

1. Implement `prepare()` / `train()` like `src/beq/methods/sft.py`.
2. Register the method name in `src/beq/core/run_alignment.py`.
3. Add an example config under `configs/`.

## Docs

- **[`docs/GSOC_CONTEXT.md`](docs/GSOC_CONTEXT.md)** — mapping to GSoC / Gemini CLI quality themes.
- **[`docs/quality_workflow.md`](docs/quality_workflow.md)** — end-to-end flow.

## Provenance

Third-party and lineage notes live in **`EXTRACTION_NOTES.md`**.
