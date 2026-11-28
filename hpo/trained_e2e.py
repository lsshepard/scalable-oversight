"""Supervisors trained end to end: the language model itself is adapted (LoRA) on the payments.

Same game and cost as hpo/trained.py.  Each supervisor has its own low-rank adapter on a
shared frozen language model and its own report head on the hidden states at the end of
every ledger line.  In a step, both first produce reports without gradients; then each is
updated on its own realised cost against the other's report.  No label enters except
through the stake.  Needs a GPU for anything beyond a smoke test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .mechanism import Reward
from .trained import LMAX, Head, _line_tokens, cost_t, report_metrics


@dataclass
class E2EConfig:
    steps: int = 1500
    batch: int = 16                  # originals per step (two views each)
    lr_adapter: float = 2e-4
    lr_head: float = 2e-3
    rank: int = 8
    eval_every: int = 150
    n_eval: int = 400
    seed: int = 0
    layers: Optional[Sequence[int]] = None      # hidden-state layers used as features
    supervised: bool = False                    # train on the true count instead (ceiling)
    estimator: str = "analytic"                 # "analytic": gradient of the realised cost in the own
                                                # report; "reinforce": each supervisor submits a noisy
                                                # report and learns only from the payment it receives
    sigma: float = 0.05                         # exploration noise of the submitted reports ("reinforce")


class NormedHead(nn.Module):
    """Report head with a normalisation in front (features move while the adapter trains)."""
    def __init__(self, d: int, seed: int):
        super().__init__()
        self.nx, self.nh = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = Head(d, d, seed=seed)

    def forward(self, X, mask, H):
        return self.head(self.nx(X), mask, self.nh(H))


def tokenize(texts: List[str], tok):
    """ids, attention mask and, per text, the index of the last token of the header and of
    each ledger line (padded to LMAX + 1)."""
    enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping").tolist()
    last = torch.zeros(len(texts), LMAX + 1, dtype=torch.long)
    for b, t in enumerate(texts):
        idx, _ = _line_tokens(t, offsets[b])
        last[b, :len(idx)] = torch.tensor(idx)
    return enc["input_ids"], enc["attention_mask"], last


def train_e2e(model_name: str, train: dict, test: dict, rw: Reward, cfg: E2EConfig,
              device: str = "cuda", log=print) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(cfg.seed)
    tiny = model_name.startswith("tiny:")            # "tiny:<name>": random 2-layer model, for smoke tests
    model_name = model_name[5:] if tiny else model_name
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if tiny:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model_name)
        c.hidden_size, c.num_hidden_layers, c.num_attention_heads, c.intermediate_size = 64, 2, 4, 128
        c.num_key_value_heads = 2
        base = AutoModelForCausalLM.from_config(c).to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    lcfg = LoraConfig(r=cfg.rank, lora_alpha=2 * cfg.rank, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])
    model = get_peft_model(base, lcfg, adapter_name="s0")
    model.add_adapter("s1", lcfg)
    nl, hid = base.config.num_hidden_layers, base.config.hidden_size
    layers = list(cfg.layers or [2 * nl // 3, nl])
    d = hid * len(layers)
    heads = [NormedHead(d, 1000 * cfg.seed + i).to(device) for i in range(2)]

    def params(i):
        return [p for n, p in model.named_parameters() if f".s{i}." in n]
    for i in range(2):                               # both adapters stay trainable throughout
        for p in params(i):
            p.requires_grad_(True)
    opt = torch.optim.AdamW([dict(params=params(0) + params(1), lr=cfg.lr_adapter),
                             dict(params=[p for h in heads for p in h.parameters()], lr=cfg.lr_head)], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=0.05 * cfg.lr_adapter)

    def prep(data, n=None):
        n = n or len(data["T"])
        V = data["D"].shape[1]
        views = []
        for v in range(V):
            ids, att, last = tokenize([row[v] for row in data["texts"][:n]], tok)
            views.append((ids, att, last, torch.as_tensor(data["status"][:n, v] >= 0)))
        t = {k: torch.as_tensor(data[k][:n], dtype=torch.float32) for k in ("T", "D", "K")}
        return views, t, V - 1

    def reports(i, view, idx, grad: bool):
        ids, att, last, mask = (x[idx].to(device) for x in view)
        keep = int(att.sum(1).max())
        model.set_adapter(f"s{i}")
        for p in params(0) + params(1):              # set_adapter freezes the other adapter; undo
            p.requires_grad_(True)
        with torch.set_grad_enabled(grad):
            out = model(input_ids=ids[:, :keep], attention_mask=att[:, :keep], output_hidden_states=True)
            hs = torch.cat([out.hidden_states[l] for l in layers], -1).float()
            feats = hs.gather(1, last.unsqueeze(-1).expand(-1, -1, hs.shape[-1]))
            return heads[i](feats[:, 1:], mask.float(), feats[:, 0])

    tr_views, tr, v2 = prep(train)
    te_views, te, _ = prep(test, cfg.n_eval)
    te_data = {k: v[:cfg.n_eval] for k, v in test.items() if k != "texts"}
    gen = torch.Generator().manual_seed(cfg.seed)
    n = len(tr["T"])
    trace = []

    @torch.no_grad()
    def evaluate():
        F = [[], []]
        for b0 in range(0, len(te["T"]), 64):
            idx = torch.arange(b0, min(b0 + 64, len(te["T"])))
            for i in range(2):
                F[i].append(reports(i, te_views[0 if i == 0 else v2], idx, False).cpu())
        F = [torch.cat(f) for f in F]
        return report_metrics(F, te["T"], [te["D"][:, 0], te["D"][:, v2]], [te["K"][:, 0], te["K"][:, v2]], te_data, v2, rw)

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            model.eval()
            ev = evaluate(); ev["step"] = step
            trace.append(ev)
            log(f"    step {step:5d}  concealed/dirty {ev['concealed_dirty']:+.3f}  share {ev['concealed_share']:.3f}  "
                f"clean {ev['under_clean']:+.3f}  charge {ev['charge']:.3f}  w(system) {ev['w_system']:+.2f}  "
                f"w(ins) {ev['w_inserted_plain']:+.2f}  w(ins,tell) {ev['w_inserted_tell']:+.2f}  rmse {ev['rmse_count']:.2f}")
            model.train()
        if step == cfg.steps:
            break
        idx = torch.randint(0, n, (cfg.batch,), generator=gen)
        T = tr["T"][idx].to(device)
        D = [tr["D"][idx, 0].to(device), tr["D"][idx, v2].to(device)]
        K = [tr["K"][idx, 0].to(device), tr["K"][idx, v2].to(device)]
        other = [reports(i, tr_views[0 if i == 0 else v2], idx, False).detach() for i in range(2)]
        if cfg.estimator == "reinforce":   # submitted (noisy) reports of both supervisors
            sub = [other[i] + cfg.sigma * torch.randn(other[i].shape, generator=gen).to(device) for i in range(2)]
        opt.zero_grad(set_to_none=True)
        for i in range(2):
            F = reports(i, tr_views[0 if i == 0 else v2], idx, True)
            if cfg.supervised:
                loss = ((F - K[i]) ** 2).mean()
            elif cfg.estimator == "reinforce":
                # Score-function (REINFORCE) estimate from the payment alone. Baseline: the payment
                # the un-noised report would have earned against the same submitted partner
                # report, as in hpo/trained.py.
                pay = -cost_t(rw, sub[i], sub[1 - i], D[i], D[1 - i], T)[0]
                base = -cost_t(rw, F.detach(), sub[1 - i], D[i], D[1 - i], T)[0]
                logp = -((sub[i] - F) ** 2) / (2 * cfg.sigma ** 2)
                loss = -((pay - base) * logp).mean()
            else:
                loss = cost_t(rw, F, other[1 - i], D[i], D[1 - i], T)[0].mean()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params(0) + params(1), 1.0)
        opt.step(); sched.step()
    return dict(trace=trace, final=trace[-1], layers=layers)
