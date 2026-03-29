"""Contract tests for configs and alignment artifact shape.

These tests run without GPU and give contributors lightweight confidence that example
configs parse and that documented artifact.json keys stay stable—useful for iterative
refactoring and any future CI that runs pytest.
"""

import json
import os


def test_example_configs_parse():
    root = os.path.join(os.path.dirname(__file__), "..")
    for name in ("sft_example.json", "vaccine_example.json", "repnoise_example.json"):
        path = os.path.abspath(os.path.join(root, "configs", name))
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert "alignment" in cfg and "model" in cfg
        assert cfg["alignment"]["method"] in ("sft", "vaccine", "repnoise")


def test_artifact_json_shape():
    """Document expected artifact.json keys produced by alignment stage."""
    expected_keys = {"method_name", "model_path", "tokenizer_path", "stage2_ready", "base_model_name"}
    sample = {
        "method_name": "sft",
        "model_path": "merged_model",
        "tokenizer_path": "tokenizer",
        "stage2_ready": True,
        "base_model_name": "meta-llama/Llama-2-7b-hf",
    }
    assert set(sample.keys()) == expected_keys
