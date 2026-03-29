"""Dispatch optional alignment/training; produces artifacts for downstream behavioral eval.

This stage is not required to demonstrate eval-generate/eval-judge. When run, it writes
``artifact.json`` and weights under ``alignment.output_dir`` for workflows that need a
fine-tuned model before generation.
"""

from beq.methods.repnoise import RepNoiseAligner
from beq.methods.sft import SFTAligner
from beq.methods.vaccine import VaccineAligner


def run_alignment(cfg: dict):
    name = cfg["alignment"]["method"].lower()
    if name == "sft":
        aligner = SFTAligner(cfg)
    elif name == "vaccine":
        aligner = VaccineAligner(cfg)
    elif name == "repnoise":
        aligner = RepNoiseAligner(cfg)
    else:
        raise ValueError("unknown alignment method: " + name)
    aligner.prepare()
    return aligner.train()
