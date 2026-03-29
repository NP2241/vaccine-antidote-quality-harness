import torch
from torch.optim import AdamW

from beq.artifacts.io import save_artifact, save_metrics
from beq.core.models import build_model, build_tokenizer
from beq.data.datasets import PromptResponseDataset, build_loader, collate_prompt_response


class SFTAligner:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare(self):
        self.tokenizer = build_tokenizer(self.cfg)
        self.model = build_model(self.cfg)
        ds = PromptResponseDataset(
            self.cfg["alignment"]["train_data"],
            self.tokenizer,
            self.cfg["alignment"]["max_length"],
        )
        self.loader = build_loader(
            ds,
            self.cfg["alignment"]["batch_size"],
            True,
            lambda x: collate_prompt_response(x, self.tokenizer.pad_token_id),
        )

    def train(self):
        self.model.train()
        opt = AdamW(
            self.model.parameters(),
            lr=self.cfg["alignment"]["lr"],
            weight_decay=self.cfg["alignment"]["weight_decay"],
        )
        device = next(self.model.parameters()).device
        grad_accum = self.cfg["alignment"].get("grad_accum", 1)
        metrics = {"method": "sft", "steps": 0, "last_loss": None}
        epoch = 0
        while epoch < self.cfg["alignment"]["epochs"]:
            step_in_epoch = 0
            for batch in self.loader:
                step_in_epoch += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                out = self.model(**batch)
                loss = out.loss / grad_accum
                loss.backward()
                if step_in_epoch % grad_accum == 0:
                    opt.step()
                    opt.zero_grad()
                metrics["steps"] += 1
                metrics["last_loss"] = float(loss.item() * grad_accum)
            epoch += 1
        artifact = save_artifact(self.model, self.tokenizer, self.cfg, "sft")
        save_metrics(self.cfg["alignment"]["output_dir"], metrics)
        return artifact
