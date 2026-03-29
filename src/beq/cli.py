"""Contributor-facing CLI for behavioral evals, guardrail-style judging, and optional alignment.

This package is an HF-based demo harness. Patterns here (structured JSON I/O, judge pass,
artifact paths) are analogous to quality workflows in a CLI or agent stack but do not
implement Gemini CLI or agent tools.
"""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace

from beq.core.config import load_json
from beq.core.run_alignment import run_alignment
from beq.evals.generate import run_generate
from beq.evals.judge import run_judge
from beq.evals.report import write_pipeline_report


def _cmd_align(args: Namespace) -> None:
    cfg = load_json(args.config)
    artifact = run_alignment(cfg)
    print(json.dumps(artifact, indent=2))
    if getattr(args, "write_report", ""):
        write_pipeline_report(
            args.write_report,
            steps=[{"name": "align", "config": os.path.abspath(args.config)}],
            artifact_path=os.path.join(cfg["alignment"]["output_dir"], "artifact.json"),
            eval_paths={},
        )


def _cmd_eval_generate(args: Namespace) -> None:
    path = run_generate(args)
    print(path)


def _cmd_eval_judge(args: Namespace) -> None:
    path = run_judge(args)
    print(path)


def _cmd_pipeline(args: Namespace) -> None:
    """Run alignment only; write pipeline_report.json. Does not run eval-generate/eval-judge."""
    cfg = load_json(args.config)
    artifact = run_alignment(cfg)
    write_pipeline_report(
        args.report_out,
        steps=[
            {"name": "align", "method": cfg.get("alignment", {}).get("method")},
        ],
        artifact_path=os.path.join(cfg["alignment"]["output_dir"], "artifact.json"),
        eval_paths={},
    )
    print(json.dumps({"artifact": artifact, "report": os.path.abspath(args.report_out)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="beq",
        description=(
            "OSS quality harness (demo): behavioral eval generation, guardrail-style judging, "
            "optional alignment. HF-based stand-in; not Gemini CLI code."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_align = sub.add_parser(
        "align",
        help=(
            "Optional: run alignment/training (SFT / Vaccine / RepNoise) from JSON config; "
            "writes artifact.json and weights for downstream eval."
        ),
    )
    p_align.add_argument("--config", required=True, help="Path to alignment JSON config.")
    p_align.add_argument(
        "--write-report",
        default="",
        help="Optional path to write pipeline_report.json after alignment.",
    )
    p_align.set_defaults(func=_cmd_align)

    p_gen = sub.add_parser(
        "eval-generate",
        help=(
            "Behavioral eval: generate model completions for each instruction; "
            "write JSON list of {instruction, output}."
        ),
    )
    p_gen.add_argument("--model-folder", default="meta-llama/Llama-2-7b-hf")
    p_gen.add_argument("--lora-folder", default="")
    p_gen.add_argument("--lora-folder2", default="")
    p_gen.add_argument(
        "--instruction-path",
        default="BeaverTails",
        help='Use "BeaverTails" for PKU dataset or path to JSON list of {instruction}.',
    )
    p_gen.add_argument("--output-path", required=True)
    p_gen.add_argument("--cache-dir", default="")
    p_gen.add_argument("--num-test-data", type=int, default=8)
    p_gen.add_argument("--max-new-tokens", type=int, default=128)
    p_gen.add_argument("--token", default="", help="Hugging Face token (defaults to HF_TOKEN env).")
    p_gen.set_defaults(func=_cmd_eval_generate)

    p_judge = sub.add_parser(
        "eval-judge",
        help=(
            "Guardrail-style pass: score each (instruction, output) with a moderation model; "
            "write JSON with summary + results (e.g. violated_categories)."
        ),
    )
    p_judge.add_argument("--input-path", required=True, help="JSON list of {instruction, output}.")
    p_judge.add_argument("--output-path", default="", help="Defaults to input_path with _judged.json suffix.")
    p_judge.add_argument("--moderation-model", default="PKU-Alignment/beaver-dam-7b")
    p_judge.add_argument("--cache-dir", default="")
    p_judge.add_argument("--batch-size", type=int, default=8)
    p_judge.add_argument("--threshold", type=float, default=0.5)
    p_judge.add_argument("--token", default="", help="Hugging Face token (defaults to HF_TOKEN env).")
    p_judge.set_defaults(func=_cmd_eval_judge)

    p_pipe = sub.add_parser(
        "pipeline",
        help=(
            "Run alignment only and write pipeline_report.json. "
            "Does not run eval-generate or eval-judge; eval_paths in the report stay empty."
        ),
    )
    p_pipe.add_argument("--config", required=True)
    p_pipe.add_argument("--report-out", default="examples/outputs/pipeline_report.json")
    p_pipe.set_defaults(func=_cmd_pipeline)

    args = parser.parse_args()
    if args.command == "eval-generate":
        args.token = args.token or None
        args.cache_dir = args.cache_dir or None
    if args.command == "eval-judge":
        args.token = args.token or None
        args.cache_dir = args.cache_dir or None

    args.func(args)


if __name__ == "__main__":
    main()
