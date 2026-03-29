"""Alignment-stage artifact I/O: stable ``artifact.json`` and ``train_metrics.json``.

``artifact.json`` holds predictable keys (paths, method name, base model) so downstream
eval or packaging steps can consume outputs without ad hoc parsing. This is the same
*contract* idea as build artifacts feeding validation in a larger quality pipeline.
"""

import json
import os

from beq.core.config import ensure_dir
from beq.core.models import merge_if_needed


def save_artifact(model, tokenizer, cfg: dict, method_name: str) -> dict:
    out_dir = cfg["alignment"]["output_dir"]
    ensure_dir(out_dir)
    model_dir = os.path.join(out_dir, "merged_model")
    tok_dir = os.path.join(out_dir, "tokenizer")
    ensure_dir(model_dir)
    ensure_dir(tok_dir)
    merged_model = merge_if_needed(model)
    merged_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tok_dir)
    artifact = {
        "method_name": method_name,
        "model_path": model_dir,
        "tokenizer_path": tok_dir,
        "stage2_ready": True,
        "base_model_name": cfg["model"]["base_model_name_or_path"],
    }
    path = os.path.join(out_dir, "artifact.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    return artifact


def save_metrics(out_dir: str, metrics: dict) -> None:
    path = os.path.join(out_dir, "train_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
