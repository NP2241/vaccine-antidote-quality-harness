import json

import torch
from torch.utils.data import DataLoader, Dataset


class PromptResponseDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        obj = self.items[idx]
        prompt = obj["prompt"]
        response = obj["response"]
        text = prompt + self.tokenizer.eos_token + response + self.tokenizer.eos_token
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
        ids = enc["input_ids"]
        labels = ids[:]
        return {"input_ids": ids, "labels": labels}


class RepNoiseDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.items.append(obj)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        obj = self.items[idx]
        prompt = obj["prompt"]
        safe_response = obj["safe_response"]
        harmful_response = obj["harmful_response"]
        safe_text = prompt + self.tokenizer.eos_token + safe_response + self.tokenizer.eos_token
        harmful_text = prompt + self.tokenizer.eos_token + harmful_response + self.tokenizer.eos_token
        safe_enc = self.tokenizer(safe_text, truncation=True, max_length=self.max_length)
        harmful_enc = self.tokenizer(harmful_text, truncation=True, max_length=self.max_length)
        return {
            "safe_input_ids": safe_enc["input_ids"],
            "harmful_input_ids": harmful_enc["input_ids"],
        }


def pad_list(x, length, pad_id):
    if len(x) >= length:
        return x[:length]
    return x + [pad_id] * (length - len(x))


def collate_prompt_response(batch, pad_id):
    max_len = 0
    for item in batch:
        max_len = max(max_len, len(item["input_ids"]))
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        ids = item["input_ids"]
        lab = item["labels"]
        input_ids.append(pad_list(ids, max_len, pad_id))
        labels.append(pad_list(lab, max_len, -100))
        attention_mask.append([1] * len(ids) + [0] * (max_len - len(ids)))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def compute_shared_mask(a_ids, b_ids):
    n = min(len(a_ids), len(b_ids))
    out = []
    for i in range(n):
        if a_ids[i] == b_ids[i]:
            out.append(0)
        else:
            out.append(1)
    return out


def collate_repnoise(batch, pad_id):
    max_safe = 0
    max_harm = 0
    for item in batch:
        max_safe = max(max_safe, len(item["safe_input_ids"]))
        max_harm = max(max_harm, len(item["harmful_input_ids"]))
    safe_input_ids = []
    safe_labels = []
    safe_attention = []
    harmful_input_ids = []
    harmful_labels = []
    harmful_attention = []
    shared_masks = []
    for item in batch:
        s = item["safe_input_ids"]
        h = item["harmful_input_ids"]
        safe_input_ids.append(pad_list(s, max_safe, pad_id))
        safe_labels.append(pad_list(s[:], max_safe, -100))
        safe_attention.append([1] * len(s) + [0] * (max_safe - len(s)))
        harmful_input_ids.append(pad_list(h, max_harm, pad_id))
        harmful_labels.append(pad_list(h[:], max_harm, -100))
        harmful_attention.append([1] * len(h) + [0] * (max_harm - len(h)))
        mask = compute_shared_mask(s, h)
        shared_masks.append(pad_list(mask, max_harm, 0))
    return {
        "safe_input_ids": torch.tensor(safe_input_ids, dtype=torch.long),
        "safe_labels": torch.tensor(safe_labels, dtype=torch.long),
        "safe_attention_mask": torch.tensor(safe_attention, dtype=torch.long),
        "harmful_input_ids": torch.tensor(harmful_input_ids, dtype=torch.long),
        "harmful_labels": torch.tensor(harmful_labels, dtype=torch.long),
        "harmful_attention_mask": torch.tensor(harmful_attention, dtype=torch.long),
        "shared_mask": torch.tensor(shared_masks, dtype=torch.float32),
    }


def build_loader(dataset, batch_size, shuffle, collate_fn):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
