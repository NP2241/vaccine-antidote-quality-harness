import torch
import torch.nn.functional as F
from torch.optim import AdamW

from beq.artifacts.io import save_artifact, save_metrics
from beq.core.models import build_model, build_tokenizer, get_transformer_layers
from beq.data.datasets import RepNoiseDataset, build_loader, collate_repnoise


def mean_pool_hidden(x, mask):
    w = mask.unsqueeze(-1)
    s = (x * w).sum(dim=1)
    d = w.sum(dim=1).clamp(min=1.0)
    return s / d


def rbf_kernel(x, y, gamma):
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist = (diff * diff).sum(-1)
    return torch.exp(-gamma * dist)


def mmd_loss(x, y):
    gammas = [0.1, 1.0, 10.0]
    total = 0.0
    for g in gammas:
        kxx = rbf_kernel(x, x, g).mean()
        kyy = rbf_kernel(y, y, g).mean()
        kxy = rbf_kernel(x, y, g).mean()
        total = total + kxx + kyy - 2.0 * kxy
    return total / len(gammas)


class RepNoiseAligner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.saved_outputs = []
        self.hooks = []

    def prepare(self):
        self.tokenizer = build_tokenizer(self.cfg)
        self.model = build_model(self.cfg)
        ds = RepNoiseDataset(
            self.cfg["alignment"]["train_data"],
            self.tokenizer,
            self.cfg["alignment"]["max_length"],
        )
        self.loader = build_loader(
            ds,
            self.cfg["alignment"]["batch_size"],
            True,
            lambda x: collate_repnoise(x, self.tokenizer.pad_token_id),
        )
        self.layers = get_transformer_layers(self.model)

    def _clear_hooks(self):
        self.saved_outputs = []
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _capture_layer_outputs(self):
        self._clear_hooks()

        def make_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    x = out[0]
                else:
                    x = out
                self.saved_outputs.append((idx, x))
                return out

            return hook

        for i, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_hook(make_hook(i)))

    def _layerwise_ascent(self, hidden_states, labels, lm_head, shared_mask):
        total = 0.0
        count = 0
        for _, h in hidden_states:
            logits = lm_head(h)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = shared_mask[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            per = F.cross_entropy(flat_logits, flat_labels, reduction="none").view(shift_labels.size())
            per = per * shift_mask
            denom = shift_mask.sum().clamp(min=1.0)
            total = total + per.sum() / denom
            count += 1
        return total / max(count, 1)

    def _layerwise_noise(self, hidden_states, shared_mask):
        total = 0.0
        count = 0
        for _, h in hidden_states:
            pooled = mean_pool_hidden(h, shared_mask)
            noise = torch.randn_like(pooled)
            total = total + mmd_loss(pooled.float(), noise.float())
            count += 1
        return total / max(count, 1)

    def train(self):
        self.model.train()
        opt = AdamW(
            self.model.parameters(),
            lr=self.cfg["alignment"]["lr"],
            weight_decay=self.cfg["alignment"]["weight_decay"],
        )
        device = next(self.model.parameters()).device
        alpha = self.cfg["method_params"]["repnoise"]["alpha"]
        beta = self.cfg["method_params"]["repnoise"]["beta"]
        grad_accum = self.cfg["alignment"].get("grad_accum", 1)
        metrics = {"method": "repnoise", "steps": 0, "last_loss": None}
        epoch = 0
        while epoch < self.cfg["alignment"]["epochs"]:
            step_in_epoch = 0
            for batch in self.loader:
                step_in_epoch += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                safe_out = self.model(
                    input_ids=batch["safe_input_ids"],
                    attention_mask=batch["safe_attention_mask"],
                    labels=batch["safe_labels"],
                )
                stability_loss = safe_out.loss
                self._capture_layer_outputs()
                _ = self.model(
                    input_ids=batch["harmful_input_ids"],
                    attention_mask=batch["harmful_attention_mask"],
                    labels=batch["harmful_labels"],
                )
                harmful_hidden = list(self.saved_outputs)
                self._clear_hooks()
                ascent = self._layerwise_ascent(
                    harmful_hidden,
                    batch["harmful_labels"],
                    self.model.get_output_embeddings(),
                    batch["shared_mask"],
                )
                noise = self._layerwise_noise(harmful_hidden, batch["shared_mask"])
                total_loss = (stability_loss + alpha * noise - beta * torch.log(ascent + 1e-8)) / grad_accum
                total_loss.backward()
                if step_in_epoch % grad_accum == 0:
                    opt.step()
                    opt.zero_grad()
                metrics["steps"] += 1
                metrics["last_loss"] = float(total_loss.item() * grad_accum)
            epoch += 1
        artifact = save_artifact(self.model, self.tokenizer, self.cfg, "repnoise")
        save_metrics(self.cfg["alignment"]["output_dir"], metrics)
        return artifact
