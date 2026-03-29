import torch
from torch.optim import AdamW

from beq.artifacts.io import save_artifact, save_metrics
from beq.core.models import build_model, build_tokenizer, get_transformer_layers
from beq.data.datasets import PromptResponseDataset, build_loader, collate_prompt_response


class VaccineAligner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.saved_outputs = []
        self.hooks = []

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
        self.layers = get_transformer_layers(self.model)

    def _clear_capture(self):
        self.saved_outputs = []
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _capture_hooks(self):
        self._clear_capture()

        def make_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    x = out[0]
                else:
                    x = out
                x.retain_grad()
                self.saved_outputs.append((idx, x))
                return out

            return hook

        for i, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_hook(make_hook(i)))

    def _make_perturbations(self, rho):
        grads = [self.saved_outputs[i][1].grad for i in range(len(self.saved_outputs))]
        norm_sq = None
        for g in grads:
            if g is not None:
                val = (g.float() * g.float()).sum()
                norm_sq = val if norm_sq is None else norm_sq + val
        if norm_sq is None:
            return {}
        norm = torch.sqrt(norm_sq) + 1e-12
        perturb = {}
        for idx, out in self.saved_outputs:
            g = out.grad
            if g is not None:
                perturb[idx] = (rho * g / norm).detach()
        return perturb

    def _register_add_hooks(self, perturb):
        handles = []

        def make_hook(idx):
            def hook(module, inp, out):
                if idx not in perturb:
                    return out
                if isinstance(out, tuple):
                    first = out[0] + perturb[idx].to(out[0].dtype)
                    return (first,) + out[1:]
                return out + perturb[idx].to(out.dtype)

            return hook

        for i, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(make_hook(i)))
        return handles

    def train(self):
        self.model.train()
        opt = AdamW(
            self.model.parameters(),
            lr=self.cfg["alignment"]["lr"],
            weight_decay=self.cfg["alignment"]["weight_decay"],
        )
        device = next(self.model.parameters()).device
        rho = self.cfg["method_params"]["vaccine"]["rho"]
        grad_accum = self.cfg["alignment"].get("grad_accum", 1)
        metrics = {"method": "vaccine", "steps": 0, "last_loss": None}
        epoch = 0
        while epoch < self.cfg["alignment"]["epochs"]:
            step_in_epoch = 0
            for batch in self.loader:
                step_in_epoch += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                self._capture_hooks()
                out1 = self.model(**batch)
                first_loss = out1.loss
                first_loss.backward(retain_graph=False)
                perturb = self._make_perturbations(rho)
                self._clear_capture()
                opt.zero_grad()
                add_handles = self._register_add_hooks(perturb)
                out2 = self.model(**batch)
                robust_loss = out2.loss / grad_accum
                robust_loss.backward()
                for h in add_handles:
                    h.remove()
                if step_in_epoch % grad_accum == 0:
                    opt.step()
                    opt.zero_grad()
                metrics["steps"] += 1
                metrics["last_loss"] = float(robust_loss.item() * grad_accum)
            epoch += 1
        artifact = save_artifact(self.model, self.tokenizer, self.cfg, "vaccine")
        save_metrics(self.cfg["alignment"]["output_dir"], metrics)
        return artifact
