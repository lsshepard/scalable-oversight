import numpy as np

from hpo.ledger import (LedgerConfig, make_original, parse_view, plant,
                        sample_summary_batch, sample_text_batch)
from hpo.model import ExactModel


def test_parse_recovers_counts_and_hides_truth():
    cfg = LedgerConfig(recognizability=0.5)
    rng = np.random.default_rng(1)
    for _ in range(300):
        orig = make_original(cfg, rng)
        view, delta = plant(orig, cfg, rng)
        text = view.render()
        s = parse_view(text)
        assert s.n_errors == orig.count("genuine") + delta == view.count("genuine") + view.count("planted")
        assert s.n_tells == sum(ln.tell for ln in view.lines) <= delta
        assert s.n_lines == len(view.lines)
        assert "genuine" not in text and "planted" not in text
        assert parse_view(orig.render()).n_errors == orig.count("genuine")


def test_views_share_original_and_plants_are_independent():
    cfg = LedgerConfig()
    b = sample_text_batch(cfg, 3000, np.random.default_rng(2))
    assert np.array_equal(b["K1"] - b["D1"], b["T"])
    assert np.array_equal(b["K2"] - b["D2"], b["T"])
    assert abs(np.corrcoef(b["D1"], b["D2"])[0, 1]) < 0.06
    assert abs(b["D1"].mean() - cfg.plant_pmf() @ np.arange(9)) < 0.08


def test_marker_iff_dirty_and_coin_otherwise():
    b = sample_text_batch(LedgerConfig(marker=True), 1500, np.random.default_rng(3))
    assert np.array_equal(b["M1"], (b["T"] > 0).astype(int))
    assert np.array_equal(b["M1"], b["M2"])
    b = sample_text_batch(LedgerConfig(marker=False), 4000, np.random.default_rng(3))
    dirty = b["T"] > 0
    assert abs(b["M1"][dirty].mean() - b["M1"][~dirty].mean()) < 0.08


def test_text_pipeline_matches_exact_model_and_fast_sampler():
    cfg = LedgerConfig(recognizability=0.4)
    m = ExactModel(cfg)
    n = 20000
    bt = sample_text_batch(cfg, n, np.random.default_rng(4))
    bs = sample_summary_batch(cfg, n, np.random.default_rng(5))
    for b in (bt, bs):
        freq = np.zeros(len(m.states))
        for k, r in zip(b["K1"], b["R1"]):
            freq[m.index[(int(k), int(r), 0)]] += 1
        freq /= n
        # every state frequency within 5 binomial standard errors of the exact value
        se = np.sqrt(m.Ps * (1 - m.Ps) / n)
        assert np.all(np.abs(freq - m.Ps) < 5 * se + 1e-4)
        # posterior P(dirty | K, R) matches on well-populated states
        for (k, r, _), i in m.index.items():
            sel = (b["K1"] == k) & (b["R1"] == r)
            if sel.sum() >= 400:
                emp = (b["T"][sel] > 0).mean()
                assert abs(emp - m.p_dirty[i]) < 5 * np.sqrt(0.25 / sel.sum())
    assert abs(bt["L1"].mean() - bs["L1"].mean()) < 0.15


def test_plants_without_tell_are_indistinguishable_by_length():
    """Given the error count, ledger length carries no information about dirtiness."""
    cfg = LedgerConfig(recognizability=0.0)
    b = sample_text_batch(cfg, 30000, np.random.default_rng(6))
    for k in (1, 2):
        sel = b["K1"] == k
        lo, hi = sel & (b["L1"] <= 16), sel & (b["L1"] > 16)
        d_lo, d_hi = (b["T"][lo] > 0).mean(), (b["T"][hi] > 0).mean()
        se = np.sqrt(0.25 / lo.sum() + 0.25 / hi.sum())
        assert abs(d_lo - d_hi) < 4 * se
