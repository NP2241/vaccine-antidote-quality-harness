"""Smoke tests: package imports resolve (no full model download).

Imports pull in heavy dependencies (torch, transformers) but avoid running alignment or
eval jobs—suitable for contributor machines and CI that installs the package with dev extras.
"""

def test_import_package():
    import beq  # noqa: F401

    assert beq.__version__


def test_submodules_import():
    from beq.core import load_json, run_alignment
    from beq.evals.moderation import QAModeration

    assert load_json and run_alignment and QAModeration
