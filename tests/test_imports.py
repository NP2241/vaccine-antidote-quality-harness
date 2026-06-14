"""Smoke tests: package imports resolve (no model download or GPU)."""

def test_import_package():
    import beq  # noqa: F401

    assert beq.__version__


def test_submodules_import():
    from beq.core import load_json, run_alignment
    from beq.evals.moderation import QAModeration

    assert load_json and run_alignment and QAModeration
