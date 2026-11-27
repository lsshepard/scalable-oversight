"""Supervisors that are trained on their payments instead of being prompted.

A supervisor is a small report head on per-line features of its view.  The features are
either a language model's activations at the end of each ledger line (the model reads the
rendered text; see `lm_features`) or, for checks, a synthetic encoding with perfect
perception (`synthetic_features`).  Two supervisors with separate parameters are trained
at the same time, each by stochastic gradient steps on its OWN realised cost

    cost_i = beta (V_i - V_j)^2 + lam max(Delta_i - F_i, 0)^2 + eps F_i - b 1{T > 0} (T - V_i),

V = F - Delta, with the other supervisor's report taken as given (gradient play).  No
payment rule is ever stated to them and no label of the truth enters the cost except
through the private stake b.  `hpo.equilibrium.solve_equilibrium` gives the exact
equilibrium of the same game for a supervisor that perceives its view perfectly.

Report head (permutation invariant over lines):
    F(view) = w . sum_l phi(x_l) + rho(sum_l phi(x_l), psi(header))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .equilibrium import Potential, solve_equilibrium
from .ledger import Ledger, LedgerConfig, make_original, plant
from .mechanism import Reward
from .model import ExactModel

STATUS = {"ok": 0, "genuine": 1, "planted": 2}
LMAX = 32          # 20 base lines + 2 system errors + 8 insertions, with slack


# ------------------------------------------------------------------------------ data
def sample_views(cfg: LedgerConfig, n: int, seed: int, plain: bool = False) -> dict:
    """n originals, each with two privately perturbed views (or, with `plain`, the
    unperturbed original shown to both supervisors).

    Returns texts [n][V] and integer arrays: T [n]; D, K, R, M [n, V]; per-line status
    and tell [n, V, LMAX] (status -1 = padding).  V = 1 with `plain`, else 2."""
    rng = np.random.default_rng([seed, 7])
    V = 1 if plain else 2
    out = dict(T=np.zeros(n, np.int64), D=np.zeros((n, V), np.int64), K=np.zeros((n, V), np.int64),
               R=np.zeros((n, V), np.int64), M=np.zeros((n, V), np.int64),
               status=-np.ones((n, V, LMAX), np.int64), tell=np.zeros((n, V, LMAX), bool), texts=[])
    for i in range(n):
        orig = make_original(cfg, rng)
        out["T"][i] = orig.count("genuine")
        row = []
        for v in range(V):
            view, delta = (orig, 0) if plain else plant(orig, cfg, rng)
            assert len(view.lines) <= LMAX
            st = [STATUS[ln.status] for ln in view.lines]
            out["status"][i, v, :len(st)] = st
            out["tell"][i, v, :len(st)] = [ln.tell for ln in view.lines]
            out["D"][i, v] = delta
            out["K"][i, v] = sum(s > 0 for s in st)
            out["R"][i, v] = sum(ln.tell for ln in view.lines)
            out["M"][i, v] = view.ledger_id % 2
            row.append(view.render())
        out["texts"].append(row)
    return out


def synthetic_features(data: dict, noise: float = 0.0, seed: int = 0, extra: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Perfect perception: each line carries [is wrong, has the tell] and the header
    carries the parity of the ledger number, plus `extra` uninformative noise columns."""
    rng = np.random.default_rng([seed, 11])
    st, tell = data["status"], data["tell"]
    n, V, L = st.shape
    X = np.zeros((n, V, L, 2 + extra), np.float32)
    X[..., 0] = st > 0
    X[..., 1] = tell
    X[..., 2:] = rng.standard_normal((n, V, L, extra))
    if noise:
        X[..., :2] += noise * rng.standard_normal((n, V, L, 2))
    X[st < 0] = 0.0
    H = np.zeros((n, V, 1 + extra), np.float32)
    H[..., 0] = data["M"]
    H[..., 1:] = rng.standard_normal((n, V, extra))
    return X, H


def _line_tokens(text: str, offsets: Sequence[Tuple[int, int]]) -> Tuple[List[int], List[List[int]]]:
    """For each row of the text (header first): index of its last token, and for ledger
    lines the indices of the tokens of the stated result."""
    last, result = [], []
    pos = 0
    ends = {}
    for t, (s, e) in enumerate(offsets):
        if e > s:
            ends[e] = t
    starts = [s for s, e in offsets]
    for j, row in enumerate(text.split("\n")):
        end = pos + len(row)
        last.append(ends[end])
        if j > 0:
            r0 = pos + row.index("=") + 1
            while text[r0] == " ":
                r0 += 1
            result.append([t for t, (s, e) in enumerate(offsets) if e > s and s >= r0 and e <= end])
        pos = end + 1
    return last, result


@torch.no_grad()
def lm_features(texts: List[str], tok, model, layers: Sequence[int], batch: int = 8,
                device: str = "cpu", progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Run each view through the language model.  Returns

    X [n, LMAX, len(layers) * hidden + 3]  per ledger line: hidden state at the line's last
        token in each requested layer, then three numbers from the model's own next-token
        probabilities for the stated result (sum, minimum and first log-probability);
    H [n, len(layers) * hidden]            the same hidden states at the end of the header.
    """
    hid = model.config.hidden_size
    dl = len(layers) * hid
    X = np.zeros((len(texts), LMAX, dl + 3), np.float16)
    H = np.zeros((len(texts), dl), np.float16)
    order = np.argsort([len(t) for t in texts])
    for b0 in range(0, len(texts), batch):
        idx = order[b0:b0 + batch]
        chunk = [texts[i] for i in idx]
        enc = tok(chunk, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = torch.cat([out.hidden_states[l].float() for l in layers], -1)       # [B, S, dl]
        ids = enc["input_ids"]
        for b, i in enumerate(idx):
            last, result = _line_tokens(chunk[b], offsets[b])
            feats = hs[b, last].cpu().numpy()
            H[i] = feats[0]
            X[i, :len(last) - 1, :dl] = feats[1:]
            flat = [t for toks in result for t in toks]
            lp = torch.log_softmax(out.logits[b, [t - 1 for t in flat]].float(), -1)
            lp = lp[torch.arange(len(flat)), ids[b, flat]].cpu().numpy()
            at = 0
            for j, toks in enumerate(result):
                seg = lp[at:at + len(toks)]
                at += len(toks)
                X[i, j, dl:] = [seg.sum(), seg.min(), seg[0]]
        if progress and (b0 // batch) % 10 == 0:
            print(f"  features {min(b0 + batch, len(texts))}/{len(texts)}", flush=True)
    return X, H


# ------------------------------------------------------------------------------ head
class Head(nn.Module):
    def __init__(self, d_line: int, d_head: int, k: int = 8, hidden: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.phi = nn.Sequential(nn.Linear(d_line, hidden), nn.GELU(), nn.Linear(hidden, k))
        self.psi = nn.Linear(d_head, 4)
        self.lin = nn.Linear(k, 1)
        self.rho = nn.Sequential(nn.Linear(k + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        for p in self.parameters():                      # reproducible, separate inits
            if p.dim() > 1:
                nn.init.normal_(p, 0.0, 1.0 / math.sqrt(p.shape[1]), generator=g)
            else:
                nn.init.zeros_(p)
        with torch.no_grad():                            # start near a report of zero
            self.lin.weight.mul_(0.1); self.rho[-1].weight.mul_(0.1)

    def forward(self, X, mask, H):
        s = (self.phi(X) * mask.unsqueeze(-1)).sum(-2)
        return (self.lin(s) + self.rho(torch.cat([s, self.psi(H)], -1))).squeeze(-1)


def cost_t(rw: Reward, F_i, F_j, D_i, D_j, T):
    """hpo.mechanism.cost in torch; returns (total, agreement part, floor part)."""
    V_i, V_j = F_i - D_i, F_j - D_j
    agree = rw.beta * (V_i - V_j) ** 2
    floor = rw.lam * torch.clamp(D_i - F_i, min=0.0) ** 2
    return agree + floor + rw.eps * F_i - rw.b * (T > 0) * (T - V_i), agree, floor


@dataclass
class TrainConfig:
    steps: int = 4000
    batch: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.0
    estimator: str = "analytic"      # "analytic": gradient of the realised cost in the own report;
                                     # "reinforce": score-function estimate from sampled reports
    sigma: float = 0.3               # exploration noise of the sampled reports ("reinforce")
    eval_every: int = 250
    seed: int = 0
    supervised: bool = False         # train on the true count instead (perception ceiling)
    hidden: int = 64
    k: int = 8


def _prep(data: dict, X: np.ndarray, H: np.ndarray, device: str, stats=None):
    """Tensors for one dataset.  X [n, V, L, d], H [n, V, dh]; features are standardised
    with the statistics of the training set (passed on to the test set)."""
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    Ht = torch.as_tensor(np.asarray(H), dtype=torch.float32, device=device)
    mask = torch.as_tensor(data["status"] >= 0, device=device)
    if stats is None:
        flat = Xt[mask]
        stats = (flat.mean(0), flat.std(0).clamp_min(1e-3), Ht.mean((0, 1)), Ht.std((0, 1)).clamp_min(1e-3))
    Xt.sub_(stats[0]).div_(stats[1]).mul_(mask.unsqueeze(-1))
    Ht = (Ht - stats[2]) / stats[3]
    t = {k: torch.as_tensor(data[k], dtype=torch.float32, device=device) for k in ("T", "D", "K", "R", "M")}
    return dict(X=Xt, H=Ht, mask=mask.float(), **t), stats


def report_metrics(F, T, D, K, data: dict, v2: int, rw: Reward) -> dict:
    """Behaviour of a pair from its reports F = [F_0, F_1] on a set of originals.
    T [n]; D, K lists of two [n] tensors; `data` the arrays from `sample_views`;
    v2 = index of the second supervisor's view (0 when both see the same ledger)."""
    dirty = T > 0
    out = dict(cost=0.0, charge=0.0, agree=0.0, floor=0.0)
    V = [F[i] - D[i] for i in range(2)]
    for i in range(2):
        c, a, f = cost_t(rw, F[i], F[1 - i], D[i], D[1 - i], T)
        out["cost"] += float(c.mean()) / 2
        out["agree"] += float(a.mean()) / 2; out["floor"] += float(f.mean()) / 2
    out["charge"] = out["agree"] + out["floor"]
    Vm = 0.5 * (V[0] + V[1])
    out["concealed_dirty"] = float((T - Vm)[dirty].mean())            # mean errors withheld, dirty originals
    out["net_share"] = float((T - Vm)[dirty].sum() / T[dirty].sum())  # the same as a share of the errors present
    out["under_clean"] = float((-Vm)[~dirty].mean())                  # mean under-report, clean originals
    cap = sum(torch.minimum(torch.clamp(T - V[i], min=0.0), T)[dirty].sum() for i in range(2))
    out["concealed_share"] = float(cap / (2 * T[dirty].sum()))
    out["rmse_count"] = float(torch.sqrt(sum(((F[i] - K[i]) ** 2).mean() for i in range(2)) / 2))
    # how much one line of each kind moves the report (least squares over views)
    views = [0, v2]
    A, y = [], []
    for i in range(2):
        st, tl = data["status"][:, views[i]], data["tell"][:, views[i]]
        cols = [np.ones(len(st)), (st == 0).sum(1), (st == 1).sum(1), ((st == 2) & ~tl).sum(1), ((st == 2) & tl).sum(1)]
        A.append(np.stack(cols, 1).astype(float)); y.append(F[i].float().cpu().numpy())
    A, y = np.concatenate(A), np.concatenate(y)
    keep = A.std(0) > 0
    keep[0] = True
    coef = np.full(5, np.nan)
    coef[keep] = np.linalg.lstsq(A[:, keep], y, rcond=None)[0]
    out.update(w_const=float(coef[0]), w_ok=float(coef[1]), w_system=float(coef[2]),
               w_inserted_plain=float(coef[3]), w_inserted_tell=float(coef[4]))
    # mean shading by what the view shows, for comparison with the exact equilibrium
    g = {}
    for i in range(2):
        key = np.stack([data["K"][:, views[i]], data["R"][:, views[i]], data["M"][:, views[i]]], 1)
        sh = (F[i] - K[i]).float().cpu().numpy()
        for row, val in zip(map(tuple, key), sh):
            g.setdefault(row, []).append(float(val))
    out["shading"] = {f"{k},{r},{m}": (float(np.mean(v)), len(v)) for (k, r, m), v in sorted(g.items())}
    return out


@torch.no_grad()
def evaluate(heads, ds: dict, data: dict, rw: Reward) -> dict:
    """Behaviour of the trained pair on held-out ledgers (deterministic reports)."""
    v2 = ds["X"].shape[1] - 1
    F = [heads[0](ds["X"][:, 0], ds["mask"][:, 0], ds["H"][:, 0]),
         heads[1](ds["X"][:, v2], ds["mask"][:, v2], ds["H"][:, v2])]
    return report_metrics(F, ds["T"], [ds["D"][:, 0], ds["D"][:, v2]], [ds["K"][:, 0], ds["K"][:, v2]], data, v2, rw)


def train_pair(train: dict, test: dict, feats_train, feats_test, rw: Reward, tc: TrainConfig,
               device: str = "cpu", log=None) -> dict:
    """Train two supervisors against each other on `rw`; returns the evaluation trace.
    `feats_*` = (X, H) arrays for the datasets from `sample_views`."""
    torch.manual_seed(tc.seed)
    ds, stats = _prep(train, feats_train[0], feats_train[1], device)
    dt, _ = _prep(test, feats_test[0], feats_test[1], device, stats)
    d_line, d_head = ds["X"].shape[-1], ds["H"].shape[-1]
    heads = [Head(d_line, d_head, tc.k, tc.hidden, seed=1000 * tc.seed + i).to(device) for i in range(2)]
    opt = torch.optim.Adam([p for h in heads for p in h.parameters()], lr=tc.lr, weight_decay=tc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tc.steps, eta_min=tc.lr * 0.05)
    n, v2 = ds["X"].shape[0], ds["X"].shape[1] - 1
    gen = torch.Generator(device="cpu").manual_seed(tc.seed)
    trace = []
    for step in range(tc.steps + 1):
        if step % tc.eval_every == 0 or step == tc.steps:
            ev = evaluate(heads, dt, test, rw)
            ev["step"] = step
            trace.append(ev)
            if log:
                log(f"    step {step:5d}  concealed/dirty {ev['concealed_dirty']:+.3f}  share {ev['concealed_share']:.3f}  "
                    f"clean {ev['under_clean']:+.3f}  charge {ev['charge']:.3f}  w(system) {ev['w_system']:+.2f}  "
                    f"w(ins) {ev['w_inserted_plain']:+.2f}  w(ins,tell) {ev['w_inserted_tell']:+.2f}  rmse {ev['rmse_count']:.2f}")
        if step == tc.steps:
            break
        idx = torch.randint(0, n, (tc.batch,), generator=gen).to(device)
        F = [heads[0](ds["X"][idx, 0], ds["mask"][idx, 0], ds["H"][idx, 0]),
             heads[1](ds["X"][idx, v2], ds["mask"][idx, v2], ds["H"][idx, v2])]
        T, D = ds["T"][idx], [ds["D"][idx, 0], ds["D"][idx, v2]]
        if tc.supervised:
            K = [ds["K"][idx, 0], ds["K"][idx, v2]]
            loss = sum(((F[i] - K[i]) ** 2).mean() for i in range(2))
        elif tc.estimator == "analytic":
            loss = sum(cost_t(rw, F[i], F[1 - i].detach(), D[i], D[1 - i], T)[0].mean() for i in range(2))
        else:
            # Each supervisor submits a noisy report and sees only its payment.  Baseline: its
            # payment for the un-noised report against the same partner report.
            noise = [tc.sigma * torch.randn(tc.batch, device=device) for _ in range(2)]
            A = [(F[i] + noise[i]).detach() for i in range(2)]
            loss = 0.0
            for i in range(2):
                pay = -cost_t(rw, A[i], A[1 - i], D[i], D[1 - i], T)[0]
                base = -cost_t(rw, F[i].detach(), A[1 - i], D[i], D[1 - i], T)[0]
                logp = -((A[i] - F[i]) ** 2) / (2 * tc.sigma ** 2)
                loss = loss - ((pay - base) * logp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
    return dict(trace=trace, final=trace[-1])


# ------------------------------------------------------------------ exact predictions
def exact_prediction(cfg: LedgerConfig, rw: Reward, plain: bool = False) -> dict:
    """Equilibrium of the same game for supervisors that perceive (K, R, M) exactly."""
    if plain:
        cfg = LedgerConfig(**{**cfg.__dict__, "plant_values": (0,), "recognizability": 0.0})
    m = ExactModel(cfg)
    pot = Potential(m, rw)
    if rw.b > 0 and rw.lam == 0:
        return dict(bounded=False)
    res = solve_equilibrium(m, rw)
    g = 0.5 * (res.g1 + res.g2)
    conc = pot.concealment(g)
    # charge parts at equilibrium, and the capped concealed share
    neg = np.maximum(-(g[None, :] + m.Tvals[:, None]), 0.0)
    floor = rw.lam * float((m.joint * neg ** 2).sum())
    agree = rw.beta * float(2 * (m.Ps @ g ** 2) - 2 * g @ (m.J @ g))
    Tm = m.Tvals[:, None]
    cap = np.minimum(np.maximum(-g[None, :], 0.0), Tm)
    share = float((m.joint * cap).sum() / (m.joint * Tm).sum())
    mean_T_dirty = float((m.pT * m.Tvals)[1:].sum() / (1 - m.p0))
    return dict(bounded=True, converged=res.converged, concealed_dirty=conc["dirty"], under_clean=conc["clean"],
                concealed_share=share, net_share=conc["dirty"] / mean_T_dirty, agree=agree, floor=floor,
                charge=agree + floor,
                shading={f"{int(k)},{int(r)},{int(mm)}": float(v) for k, r, mm, v in zip(m.K, m.R, m.M, g)})
