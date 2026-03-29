# Gemini CLI quality (GSoC) — how this repo relates

This document is for reviewers and contributors. It states what this repository **is**, what it is **not**, and which engineering patterns **transfer** to Google Summer of Code–style work on **Gemini CLI quality** (behavioral evals, guardrails, contributor workflows).

## What this repo is

- A **compact, open-source demo harness** built on **Hugging Face** (causal LM + a moderation-style judge).
- A **behavioral eval** path: `eval-generate` → JSON generations → `eval-judge` → structured summary + per-row results.
- **Artifact discipline**: `artifact.json`, `train_metrics.json`, optional `pipeline_report.json`.
- A **single CLI** (`beq`) with contributor-oriented subcommands and small **pytest** checks for config/schema stability.

## What this repo is not

- **Not** the Gemini CLI codebase, **not** a fork of Google’s agent runtime, and **not** an implementation of Gemini prompts, tools, MCP, skills, or subagents.
- **Not** a production release gate: there is **no** built-in failing threshold on judge metrics (that would be additional tooling).
- **Not** a chat-log-to-eval pipeline; generations come from **instructions** (JSON or a dataset), not from replaying agent transcripts.

## Patterns that map to the GSoC scope

| Idea in this repo | Gemini CLI quality angle |
|-------------------|---------------------------|
| JSON **behavioral eval** rows `{instruction, output}` | Structured captures of **model behavior** for review and automation. |
| **Guardrail-style judge** with aggregate + per-row signals | Same *shape* as **pre-ship checks** and **regression awareness** (policy and wiring are product-specific). |
| Stable **artifact** keys and paths | Lets **tooling and CI** depend on a contract instead of scraping logs. |
| **Diff-friendly** outputs | Supports **iterative validation** and future **eval gap** tooling. |
| **Tests** documenting schema expectations | **Contributor confidence** and safer refactors. |
| **Extension docs** (new eval subcommand, new method) | **OSS contribution** paths around quality code. |

## Future work (honest backlog)

- **CI**: GitHub Actions running `pytest` (and, later, optional heavier jobs if resources allow).
- **Gates**: optional script or exit codes when metrics exceed thresholds.
- **Reporting**: wire `eval_paths` into `pipeline_report.json` after eval steps.
- **Diagnostics**: structured logging, verbosity flags.
- **Chat-log → eval**: importer that normalizes transcripts into the same JSON eval format.

None of the above is required for the current demo to illustrate **transferable patterns**; they are natural next steps aligned with a GSoC project scope.
