"""Synthetic ledgers: originals, privately planted views, and the view parser.

An *original* is a transcript of simple arithmetic steps such as ``07. 37 + 48 = 85``.
It carries T in {0, 1, 2} genuine errors.  A *view* is the original after a code
transformation applied ``Delta ~ Poisson(plant_mean)`` further errors drawn from the
SAME error generator (:func:`apply_error`) that produced the genuine ones, so a plant
without a tell is indistinguishable from a genuine error by construction.

Two optional leaks can be switched on:

``recognizability`` (p)
    each planted error independently carries a stylistic *tell* (a doubled space after
    the equals sign) with probability p.  Genuine errors never carry it.
``marker``
    a feature of the benign content (parity of the ledger id in the header, which the
    perturbation never touches) is present iff the original is dirty.  With
    ``marker=False`` the same feature is an independent fair coin.

Ground truth (which lines are genuine / planted) lives only in :class:`Line.status`
and is never rendered into the text a supervisor sees.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import List, Tuple

import numpy as np
from scipy.stats import poisson

OPS = ("+", "-", "*")
ERROR_OFFSETS = (-10, -3, -2, -1, 1, 2, 3, 10)
SMALL_OFFSETS = (-3, -2, -1, 1, 2, 3)     # tell_kind="offset": every error except a recognizable plant
TEN_OFFSETS = (-10, 10)                   # tell_kind="offset": a recognizable plant
TELL = "=  "  # doubled space after '='; a clean line renders "= "


@dataclass(frozen=True)
class LedgerConfig:
    n_min: int = 12                       # base number of steps, uniform on [n_min, n_max]
    n_max: int = 20
    p_genuine: Tuple[float, ...] = (0.80, 0.15, 0.05)   # P(T = 0, 1, 2)
    plant_mean: float = 1.5               # Poisson mean of the number of plants
    plant_max: int = 8                    # Poisson truncated (and renormalised) here
    recognizability: float = 0.0          # P(a plant carries a tell)
    insert_prob: float = 0.5              # error generator: insert a wrong line vs corrupt one
    marker: bool = False                  # benign-content marker present iff dirty
    # --- added for the "predicted exploits" experiment; the defaults keep the old behaviour ---
    plant_values: Tuple[int, ...] = ()    # if non-empty: number of plants uniform on these values
    tell_kind: str = "space"              # "space": a recognizable plant has a doubled space after '='
                                          # "offset": a recognizable plant is off by exactly 10 and
                                          #           every other error is off by 1, 2 or 3
    rerandomize_id: bool = False          # each view gets a fresh random ledger id (destroys the marker)

    def plant_pmf(self) -> np.ndarray:
        if self.plant_values:
            pmf = np.zeros(self.plant_max + 1)
            pmf[list(self.plant_values)] = 1.0 / len(self.plant_values)
            return pmf
        pmf = poisson.pmf(np.arange(self.plant_max + 1), self.plant_mean)
        return pmf / pmf.sum()


@dataclass
class Line:
    a: int
    op: str
    b: int
    shown: int
    status: str = "ok"      # "ok" | "genuine" | "planted"   (evaluation only)
    tell: bool = False


@dataclass
class Ledger:
    ledger_id: int
    lines: List[Line]
    tell_kind: str = "space"

    def render(self) -> str:
        """The text a supervisor sees.  Line numbers are assigned at render time so
        that inserted lines leave no numbering gap."""
        out = [f"# ledger {self.ledger_id:06d}"]
        for i, ln in enumerate(self.lines, 1):
            eq = TELL if (ln.tell and self.tell_kind == "space") else "= "
            out.append(f"{i:02d}. {ln.a} {ln.op} {ln.b} {eq}{ln.shown}")
        return "\n".join(out)

    def count(self, status: str) -> int:
        return sum(ln.status == status for ln in self.lines)


def _true_value(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b if op == "-" else a * b


def _fresh_line(rng: np.random.Generator) -> Line:
    op = OPS[rng.integers(3)]
    if op == "*":
        a, b = int(rng.integers(2, 13)), int(rng.integers(2, 13))
    else:
        a, b = int(rng.integers(10, 100)), int(rng.integers(10, 100))
    return Line(a, op, b, _true_value(a, op, b))


def apply_error(lines: List[Line], rng: np.random.Generator, status: str,
                tell: bool, insert_prob: float, offsets: Tuple[int, ...] = ERROR_OFFSETS) -> None:
    """THE error generator, shared by genuine errors and plants (in place).

    With probability ``insert_prob`` a new wrong line is inserted at a uniform slot;
    otherwise a uniformly chosen currently-correct line has its result corrupted.
    Only correct lines are corrupted, so every call adds exactly one error.
    """
    off = int(offsets[rng.integers(len(offsets))])
    ok = [i for i, ln in enumerate(lines) if ln.status == "ok"]
    if rng.random() < insert_prob or not ok:
        ln = _fresh_line(rng)
        ln.shown += off
        ln.status, ln.tell = status, tell
        lines.insert(int(rng.integers(len(lines) + 1)), ln)
    else:
        ln = lines[ok[rng.integers(len(ok))]]
        ln.shown += off
        ln.status, ln.tell = status, tell


def make_original(cfg: LedgerConfig, rng: np.random.Generator) -> Ledger:
    n = int(rng.integers(cfg.n_min, cfg.n_max + 1))
    lines = [_fresh_line(rng) for _ in range(n)]
    t = int(rng.choice(len(cfg.p_genuine), p=cfg.p_genuine))
    small = cfg.tell_kind == "offset"
    for _ in range(t):
        apply_error(lines, rng, "genuine", False, cfg.insert_prob,
                    SMALL_OFFSETS if small else ERROR_OFFSETS)
    lid = int(rng.integers(0, 500000)) * 2          # even id
    if cfg.marker:
        lid += int(t > 0)                           # odd iff dirty
    else:
        lid += int(rng.integers(2))                 # independent coin
    return Ledger(lid, lines, cfg.tell_kind)


def plant(original: Ledger, cfg: LedgerConfig, rng: np.random.Generator) -> Tuple[Ledger, int]:
    """Designer-side perturbation.  Returns (view, Delta); Delta stays private."""
    lines = [replace(ln) for ln in original.lines]
    delta = int(rng.choice(cfg.plant_max + 1, p=cfg.plant_pmf()))
    for _ in range(delta):
        tell = bool(rng.random() < cfg.recognizability)
        offsets = ERROR_OFFSETS if cfg.tell_kind != "offset" else (TEN_OFFSETS if tell else SMALL_OFFSETS)
        apply_error(lines, rng, "planted", tell, cfg.insert_prob, offsets)
    lid = original.ledger_id
    if cfg.rerandomize_id:
        lid = int(rng.integers(0, 1000000))         # fresh number, parity is a fair coin
    return Ledger(lid, lines, cfg.tell_kind), delta


# ---------------------------------------------------------------------------------
# Supervisor side: everything below uses only the rendered text.
# ---------------------------------------------------------------------------------
_LINE_RE = re.compile(r"^\d+\. (-?\d+) ([+\-*]) (-?\d+) =( {1,2})(-?\d+)$")
_HEAD_RE = re.compile(r"^# ledger (\d+)$")


@dataclass(frozen=True)
class ViewSummary:
    n_lines: int
    n_errors: int     # K: lines whose arithmetic is wrong
    n_tells: int      # R: wrong lines carrying the stylistic tell
    marker: int       # parity of the ledger id
    n_ten: int = 0    # wrong lines that are off by exactly 10 (the tell when tell_kind="offset")


def parse_view(text: str) -> ViewSummary:
    """Check every line of a view.  This is all a rule-based supervisor can know."""
    rows = text.split("\n")
    marker = int(_HEAD_RE.match(rows[0]).group(1)) % 2
    k = r = ten = 0
    for row in rows[1:]:
        m = _LINE_RE.match(row)
        if m is None:
            raise ValueError(f"unparseable ledger line: {row!r}")
        a, op, b, gap, shown = m.groups()
        true = _true_value(int(a), op, int(b))
        if true != int(shown):
            k += 1
            r += len(gap) == 2
            ten += abs(true - int(shown)) == 10
    return ViewSummary(len(rows) - 1, k, r, marker, ten)


FIELDS = ("T", "K1", "R1", "M1", "L1", "D1", "K2", "R2", "M2", "L2", "D2")


def sample_text_batch(cfg: LedgerConfig, n: int, rng: np.random.Generator,
                      keep_text: int = 0) -> dict:
    """Generate n originals with two independently planted views each, render them to
    text, and parse the text back.  Returns integer arrays:

    T        genuine errors (ground truth, evaluation only)
    D1, D2   plants in each view (designer's private knowledge)
    K*, R*, M*, L*   what each supervisor can read off its own view

    ``keep_text`` > 0 additionally keeps the first few (text1, text2) pairs.
    """
    out = {f: np.empty(n, dtype=np.int64) for f in FIELDS}
    texts = []
    for i in range(n):
        orig = make_original(cfg, rng)
        out["T"][i] = orig.count("genuine")
        for j in ("1", "2"):
            view, delta = plant(orig, cfg, rng)
            text = view.render()
            s = parse_view(text)
            out["K" + j][i], out["R" + j][i] = s.n_errors, s.n_tells
            out["M" + j][i], out["L" + j][i], out["D" + j][i] = s.marker, s.n_lines, delta
            if i < keep_text:
                texts.append(text)
    if keep_text:
        out["texts"] = texts
    return out


def sample_summary_batch(cfg: LedgerConfig, n: int, rng: np.random.Generator) -> dict:
    """Fast vectorised sampler of the same summary distribution, skipping the text.
    (tests/test_ledger.py checks that it matches :func:`sample_text_batch`.)"""
    out = {}
    t = rng.choice(len(cfg.p_genuine), size=n, p=cfg.p_genuine)
    base = rng.integers(cfg.n_min, cfg.n_max + 1, size=n)
    g_ins = rng.binomial(t, cfg.insert_prob)
    out["T"] = t
    coin = rng.integers(2, size=n)
    for j in ("1", "2"):
        d = rng.choice(cfg.plant_max + 1, size=n, p=cfg.plant_pmf())
        out["D" + j] = d
        out["K" + j] = t + d
        out["R" + j] = rng.binomial(d, cfg.recognizability)
        out["M" + j] = (t > 0).astype(np.int64) if cfg.marker else coin
        out["L" + j] = base + g_ins + rng.binomial(d, cfg.insert_prob)
    return out
