import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def build_tokenizer(cfg: dict):
    path = cfg["model"]["tokenizer_name_or_path"]
    tok = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(cfg: dict):
    dtype = get_dtype(cfg["model"].get("dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model_name_or_path"],
        torch_dtype=dtype,
        device_map=cfg["model"].get("device_map", "auto"),
    )
    if cfg.get("lora", {}).get("enabled", False):
        lcfg = cfg["lora"]
        peft_cfg = LoraConfig(
            r=lcfg["r"],
            lora_alpha=lcfg["alpha"],
            target_modules=lcfg["target_modules"],
            lora_dropout=lcfg["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
    return model


def merge_if_needed(model):
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("cannot find transformer layers")
