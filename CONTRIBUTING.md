# Contributing

Small companion repo — keep changes easy to review.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

Smoke-level only: imports, config parsing, `artifact.json` schema. No GPU or gated model downloads.

## Hugging Face token

Full runs need `HF_TOKEN` (or `--token`) for `meta-llama/*` and `PKU-Alignment/beaver-dam-7b`.

## Adding an eval command

1. Add `run_*` under `src/beq/evals/`
2. Register in `src/beq/cli.py`
3. Use JSON/JSONL I/O; document env vars in `--help`

## Adding an alignment method

1. Implement `prepare()` / `train()` like `src/beq/methods/sft.py`
2. Register in `src/beq/core/run_alignment.py`
3. Add `configs/your_method_example.json`

## Docs

- [`docs/FINDINGS.md`](docs/FINDINGS.md) — project takeaways
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — lineage from [zidage/cs639_project (`yurun_sft`)](https://github.com/zidage/cs639_project/tree/yurun_sft)
- [`docs/quality_workflow.md`](docs/quality_workflow.md) — inspect-at-each-step guide
