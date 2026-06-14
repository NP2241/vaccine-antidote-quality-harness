# Findings

Research notes from our CS639 harmful fine-tuning study, consolidated from the project audit (`639 audits/`). These are **audited observations** from available runs — not universal benchmark claims. Full raw artifacts live in the [original class repo](https://github.com/zidage/cs639_project/tree/yurun_sft); this harness is a cleaned refactor for easier parsing.

## The problem

Fine-tuning-as-a-service creates a real safety risk: a base model can be aligned, then a user (or attacker) fine-tunes it on harmful data and undo that alignment. The git-disl line frames defenses at different stages — alignment-time (**[Vaccine](https://github.com/git-disl/Vaccine)**), fine-tuning-time (Lisa), and post-fine-tuning repair (**[Antidote](https://github.com/git-disl/Antidote)**). We also compared **SFT** as a baseline and **RepNoise** as an alignment-stage alternative.

Our setup: **Llama-2-7B** with LoRA (`r=256`, `alpha=4`), harmful training on `PKU-Alignment/BeaverTails_dangerous` mixed with benign **SST-2**, evaluated on **BeaverTails harmfulness** plus utility tasks (SST-2, GSM8K, AGNews where configured).

## What we ran

Four defenses made it into final quantitative comparison: **SFT** (27 runs), **RepNoise** (16 runs), **Vaccine** (27 runs), **Antidote** (26 runs, 1 failed).

Attack grid (shared framing): harmful ratios **1% / 5% / 10%**, learning rates **1e-5 / 5e-5 / 1e-4**, epochs **5 / 10 / 20**, `sample_num=5000`.

Pipeline per method:

1. Alignment or defended checkpoint (method-specific)
2. Harmful mixed LoRA fine-tuning
3. Optional post-defense (Antidote pruning: checkpoint tags `_dr02_sn2000`)
4. Safety + utility evaluation (coverage differs by method)

## Main results

### 1. Harmful fine-tuning works in strong settings

In the strongest audited attack cells (`r=10%`, `lr=1e-4`, `ep=5/10/20`), BeaverTails harmfulness reached **78.3 / 78.7 / 78.9**. Raw generation files contain explicit unsafe assistance; the moderation evaluator assigns harmful categories (e.g. violence, aiding/abetting) to sampled rows.

**Takeaway:** The threat model is real in our grid — not a edge-case artifact.

### 2. Antidote showed the largest safety gains in selected paired cells

In paired strong cells (`r=10%`, `lr=1e-4`), Antidote reduced harmfulness:

| Epoch | Attack | Antidote | Δ |
|------:|-------:|---------:|--:|
| 5 | 78.3 | 54.5 | 23.8 |
| 10 | 78.7 | 55.0 | 23.7 |
| 20 | 78.9 | 54.8 | 24.1 |

Mean reduction across these three cells: **~23.9 points** (~30% relative).

At ratio=10% over **available cells only**: Antidote mean **54.6** (5/9 cells) vs SFT **77.3** (9/9), RepNoise **78.4** (6/9), Vaccine **72.0** (9/9).

**Takeaway:** Post-fine-tuning repair can materially lower harmfulness in high-strength settings — but evidence is **selected-cell and partial-grid**, not hyperparameter-agnostic.

### 3. RepNoise did not prevent high harmfulness

In available ratio=10% cells, RepNoise stayed in the **high 70s** (76.8–79.5) — similar to or slightly worse than corresponding SFT cells on shared comparisons.

**Takeaway:** Alignment-stage RepNoise, as implemented in our sweep, did not block strong harmful fine-tuning.

### 4. Vaccine helped in weaker cells, degraded under stronger attack

Best ratio=10% cell: **48.6** (`lr=1e-5`, `ep=5`). Under stronger settings (`lr=1e-4`): **77.2 / 79.0 / 79.6**. Low-LR mean ~60.6 vs high-LR mean ~78.6.

**Takeaway:** Vaccine's benefit is **setting-dependent** — effective in weaker attack regimes, not robust across the full grid.

### 5. Safety–utility tradeoffs are messy and protocol-sensitive

**Strict SST-2** (exact `positive`/`negative` label match):

- Attack checkpoint: **831/872**
- Antidote checkpoint: **0/872**

Antidote outputs are verbose, not exact label tokens — so strict exact-match likely **undercounts** capability. A separate moderated interpretation shows **95.18**, explicitly **not** strict accuracy.

One Antidote run failed at **GSM8K utility** with CUDA OOM; safety metrics for that run still completed. Missing utility values are **unavailable**, not zero.

**Takeaway:** "Preserves utility" is not supported under strict SST-2 for the audited Antidote file. Always separate strict vs moderated eval protocols.

### 6. Cross-method comparison is only partially controlled

| Caveat | Detail |
|--------|--------|
| Run counts | SFT 27, RepNoise 16, Vaccine 27, Antidote 26 (1 failed) |
| Metric coverage | Vaccine lacks AdvBench/GSM8K in merged artifacts; utility tasks differ by method |
| Checkpoint lineage | Base LoRA folders differ per method (`_sft`, `_repnoise2`, `_vaccine_2`, `_sft_paper`) |
| Incomplete grids | Antidote 5/9 ratio=10 cells; RepNoise 6/9 |

**Takeaway:** Emphasize **within-method trends** and shared BeaverTails results. Avoid definitive global rankings.

## Engineering lessons (why we refactored)

The class repo reproduced a full pipeline but was hard to share:

| Problem in source tree | Fix in this harness |
|------------------------|---------------------|
| Scattered scripts + notebooks | Single `beq` CLI |
| `huggingface_token.txt` on disk | `HF_TOKEN` env / `--token` |
| Hardcoded cache paths | Parameterized `cache_dir` |
| Multi-thousand-line committed JSON | Gitignored outputs + tiny fixtures |
| Different entrypoints per method | Shared `artifact.json` contract |

**Takeaway:** Packaging and stable artifact schemas took as much effort as the alignment math — and made the results auditable months later.

## Claims to avoid

| Unsafe | Safer |
|--------|-------|
| "Antidote solves harmful fine-tuning" | "Antidote lowers harmfulness in selected high-strength cells" |
| "Antidote preserves utility" | "Strict utility preservation not demonstrated in audited SST-2 file" |
| "RepNoise prevents harmful fine-tuning" | "RepNoise remains high-harmfulness in available strong cells" |
| "Vaccine is best overall" | "Vaccine helps in weaker cells, degrades under stronger settings" |
| "All methods compared identically" | "Comparisons are informative but partially controlled" |
| Missing cells = zero | Missing = unavailable or failed |

## Honest limitations & future work

- GPU + Hugging Face access required; one Antidote GSM8K OOM failure
- Antidote grid incomplete (4/9 ratio=10 cells missing); no broad dense-ratio/sample-count sweep
- LISA and Panacea were planned but not in final quantitative comparison
- No multi-seed variance reporting
- This harness uses tiny fixtures — not the full 27-cell production grid

Next steps that would strengthen conclusions: complete missing Antidote cells, rerun failed GSM8K with memory-aware config, standardize metric coverage across methods, rerun SST-2 with constrained label-only decoding, add multiple seeds.

## References

- Huang et al., [Vaccine: Perturbation-aware Alignment against Harmful Fine-tuning](https://arxiv.org/abs/2402.01109)
- Huang et al., [Antidote: Post-fine-tuning Safety Alignment](https://arxiv.org/abs/2408.09600)
- PKU-Alignment [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) + beaver-dam-7b moderation judge
- Original experiments: [zidage/cs639_project (`yurun_sft`)](https://github.com/zidage/cs639_project/tree/yurun_sft)
