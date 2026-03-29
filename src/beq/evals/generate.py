"""Behavioral eval generation: instructions to model outputs as structured JSON.

Loads a causal LM (optional LoRA merge), runs each instruction with a fixed chat-style
template string (Alpaca-like). That template is a **stand-in** for prompt iteration in a
product CLI: the transferable pattern is **reproducible, JSON-serialized generations** for
downstream guardrail or regression-style checks—not Gemini CLI prompts or tool calls.

Adapted from prior evaluation code (see EXTRACTION_NOTES.md).
"""

from __future__ import annotations

import json
import os
from argparse import Namespace

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _hf_token(explicit: str | bool | None) -> str | bool | None:
    if explicit is not None and explicit != "":
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return env if env else None


def _load_causal_lm(
    model_folder: str,
    *,
    cache_dir: str | None,
    token: str | bool | None,
    lora_folder: str = "",
    lora_folder2: str = "",
):
    tok = AutoTokenizer.from_pretrained(
        model_folder,
        cache_dir=cache_dir,
        use_fast=True,
        token=token,
    )
    tok.pad_token_id = tok.pad_token_id or 0
    model = AutoModelForCausalLM.from_pretrained(
        model_folder,
        cache_dir=cache_dir,
        device_map="auto",
        token=token,
    )
    if lora_folder:
        model = PeftModel.from_pretrained(model, lora_folder)
        model = model.merge_and_unload()
    if lora_folder2:
        model = PeftModel.from_pretrained(model, lora_folder2)
        model = model.merge_and_unload()
    model.eval()
    return model, tok


def _instructions_from_source(instruction_path: str, num_test_data: int):
    instruction_lst = []
    input_data_lst = []
    if "BeaverTails" in instruction_path:
        from datasets import load_dataset

        dataset = load_dataset("PKU-Alignment/BeaverTails")
        index = 0
        for example in dataset["30k_test"]:
            if index < num_test_data and not example["is_safe"]:
                instance = {"instruction": example["prompt"]}
                instruction_lst.append(example["prompt"])
                input_data_lst.append(instance)
                index += 1
    else:
        with open(instruction_path, "r", encoding="utf-8") as f:
            input_data_lst = json.load(f)
        for data in input_data_lst:
            instruction_lst.append(data["instruction"])
    return instruction_lst, input_data_lst


def run_generate(args: Namespace) -> str:
    """Generate completions for unsafe (or file-provided) instructions; write JSON list."""
    token = _hf_token(getattr(args, "token", None))
    if token is None:
        raise RuntimeError(
            "Set HF_TOKEN (or pass --token) for gated models and BeaverTails-dependent runs."
        )

    model, tokenizer = _load_causal_lm(
        args.model_folder,
        cache_dir=getattr(args, "cache_dir", None) or None,
        token=token,
        lora_folder=getattr(args, "lora_folder", "") or "",
        lora_folder2=getattr(args, "lora_folder2", "") or "",
    )
    device = next(model.parameters()).device

    instruction_lst, input_data_lst = _instructions_from_source(
        args.instruction_path,
        args.num_test_data,
    )

    def query(instruction: str) -> str:
        prompt = (
            "Below is an instruction that describes a task. Write a response that appropriately "
            f"completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        )
        input_dict = tokenizer(prompt, return_tensors="pt")
        input_ids = input_dict["input_ids"].to(device)
        with torch.no_grad():
            generation_output = model.generate(
                inputs=input_ids,
                top_p=1,
                temperature=1.0,
                do_sample=False,
                num_beams=1,
                max_new_tokens=min(getattr(args, "max_new_tokens", 512), 512),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        s = generation_output[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        res = output.split("### Response:")[1].strip()
        return res

    pred_lst = []
    for instruction in tqdm(instruction_lst, desc="generate"):
        pred_lst.append(query(instruction))

    output_lst = []
    for input_data, pred in zip(input_data_lst, pred_lst):
        row = dict(input_data)
        row["output"] = pred
        output_lst.append(row)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_lst, f, indent=2)
    return args.output_path
