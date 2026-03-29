import json
import os


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
