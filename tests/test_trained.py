"""Supervisors trained on payments (hpo/trained.py): the training loop reaches the exact
equilibrium when perception is perfect."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hpo.ledger import LedgerConfig  # noqa: E402
from hpo.mechanism import Reward  # noqa: E402
from hpo.trained import TrainConfig, exact_prediction, sample_views, synthetic_features, train_pair  # noqa: E402

P = (0.6, 0.3, 0.1)


def _run(cfg, rw, plain=False, steps=1200, n=5000, seed=0):
    torch.set_num_threads(1)
    train, test = sample_views(cfg, n, seed, plain), sample_views(cfg, 2000, seed + 1, plain)
    res = train_pair(train, test, synthetic_features(train, seed=seed), synthetic_features(test, seed=seed + 1),
                     rw, TrainConfig(steps=steps, eval_every=steps, seed=seed, lr=3e-3))
    return res["final"], exact_prediction(cfg, rw, plain)


def test_sample_views_is_consistent():
    d = sample_views(LedgerConfig(p_genuine=P, recognizability=0.5), 300, 3)
    wrong = (d["status"] > 0).sum(-1)
    assert np.array_equal(wrong, d["K"]) and np.array_equal(d["K"], d["T"][:, None] + d["D"])
    assert np.array_equal(d["tell"].sum(-1), d["R"]) and (d["R"] <= d["D"]).all()
    assert not d["tell"][d["status"] != 2].any()                     # only insertions carry the tell
    plain = sample_views(LedgerConfig(p_genuine=P), 200, 3, plain=True)
    assert plain["D"].shape == (200, 1) and (plain["D"] == 0).all() and np.array_equal(plain["K"][:, 0], plain["T"])


def test_exact_prediction_plain_agreement():
    """Identical copies: the pair hides everything; the floor only stops negative reports,
    so the report on a dirty ledger sits b / (2 lam) below zero."""
    ex = exact_prediction(LedgerConfig(p_genuine=P), Reward(beta=1, lam=1, b=0.5), plain=True)
    mean_T_dirty = (1 * 0.3 + 2 * 0.1) / 0.4
    assert ex["concealed_dirty"] == pytest.approx(mean_T_dirty + 0.25, abs=1e-4)
    assert ex["concealed_share"] == pytest.approx(1.0, abs=1e-6)
    assert not exact_prediction(LedgerConfig(p_genuine=P), Reward(beta=1, lam=0, b=0.5))["bounded"]


def test_training_matches_equilibrium_under_the_mechanism():
    fin, ex = _run(LedgerConfig(p_genuine=P), Reward(beta=1, lam=1, b=0.5))
    assert fin["concealed_dirty"] == pytest.approx(ex["concealed_dirty"], abs=0.06)
    assert fin["concealed_share"] == pytest.approx(ex["concealed_share"], abs=0.05)
    assert fin["w_system"] == pytest.approx(fin["w_inserted_plain"], abs=0.05) and fin["w_system"] > 0.85


def test_recognizable_insertions_are_learned_and_system_errors_dropped():
    fin, ex = _run(LedgerConfig(p_genuine=P, recognizability=1.0), Reward(beta=1, lam=1, b=0.5))
    assert ex["concealed_share"] == pytest.approx(1.0, abs=1e-6)
    assert fin["concealed_share"] > 0.95 and fin["w_inserted_tell"] > 0.9 and fin["w_system"] < 0.1


def test_plain_agreement_training_hides_everything():
    fin, ex = _run(LedgerConfig(p_genuine=P), Reward(beta=1, lam=1, b=0.5), plain=True)
    assert fin["concealed_share"] > 0.97 and fin["charge"] < 0.1
