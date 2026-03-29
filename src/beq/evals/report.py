"""Lightweight pipeline reporting JSON for traceability.

``write_pipeline_report`` records when a run was produced, which artifact path was
written, optional eval output paths, and step metadata. Today the CLI only populates
alignment-centric fields; ``eval_paths`` is reserved for a fuller end-to-end report.
This supports inspectability and could feed richer CI or release documentation later.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def write_pipeline_report(
    path: str,
    *,
    steps: list[dict],
    artifact_path: str | None = None,
    eval_paths: dict[str, str | None] | None = None,
) -> None:
    """Write a single JSON file summarizing pipeline steps (align and/or eval paths when set)."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact_path,
        "eval": eval_paths or {},
        "steps": steps,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
