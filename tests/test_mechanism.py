import numpy as np

from hpo.equilibrium import (Potential, exploitability, gradient_play, hide_one_threshold,
                             integer_br_iteration, integer_cost_table, solve_equilibrium)
from hpo.ledger import LedgerConfig, sample_summary_batch, sample_text_batch
from hpo.mechanism import Reward, cost
from hpo.model import ExactModel
from hpo.strategies import MENU, deviation_gains, evaluate_menu
from hpo.supervisor import LLMSupervisor, RuleSupervisor
import pytest

CFG = LedgerConfig(recognizability=0.5)
MODEL = ExactModel(CFG)


def test_zero_cost_profiles():
    b = sample_summary_batch(CFG, 20000, np.random.default_rng(0))
    names, C = evaluate_menu(b, MODEL, Reward(1, 0))
    i = names.index
    assert C[i("honest"), i("honest")] == 0 and C[i("shift_minus_one"), i("shift_minus_one")] == 0
    assert 0 < C[i("hide_one"), i("hide_one")] < 0.4       # only the K=0 corner bites
    names, C = evaluate_menu(b, MODEL, Reward(0, 1))
    assert C[i("report_max"), i("report_max")] == 0 and C[i("always_zero"), i("always_zero")] > 1
    names, C = evaluate_menu(b, MODEL, Reward(1, 1))
    assert C[i("honest"), i("honest")] == 0 and C[i("over_one"), i("over_one")] == 0
    assert C[i("hide_one"), i("hide_one")] > 0.1
    G = deviation_gains(C)
    assert np.all(G[i("honest")] <= 1e-12)


def test_monte_carlo_matches_exact_cost():
    rw = Reward(1.0, 0.7, eps=0.03, b=0.2)
    b = sample_summary_batch(CFG, 400000, np.random.default_rng(1))
    pot = Potential(MODEL, rw)
    for name in ("hide_one", "posterior_shade_50", "hide_unrecognized"):
        F1 = MENU[name](b["K1"], b["R1"], b["M1"], MODEL)
        F2 = MENU[name](b["K2"], b["R2"], b["M2"], MODEL)
        mc = cost(rw, F1, F2, b["D1"], b["D2"], b["T"]).mean()
        g = MENU[name](MODEL.K.astype(int), MODEL.R.astype(int), MODEL.M.astype(int), MODEL) - MODEL.K
        assert abs(mc - pot.player_cost(g, g)) < 0.01


def test_gradient_is_correct_and_potential_matches_player_gradient():
    rw = Reward(1.3, 0.6, eps=0.05, b=0.3)
    pot = Potential(MODEL, rw)
    rng = np.random.default_rng(2)
    g1, g2 = rng.standard_normal(pot.S), rng.standard_normal(pot.S)
    v, d1, d2 = pot.value_grad(g1, g2)
    h = 1e-6
    for k in rng.choice(pot.S, 8, replace=False):
        e = np.zeros(pot.S); e[k] = h
        fd_pot = (pot.value_grad(g1 + e, g2)[0] - pot.value_grad(g1 - e, g2)[0]) / (2 * h)
        fd_own = (pot.player_cost(g1 + e, g2) - pot.player_cost(g1 - e, g2)) / (2 * h)
        assert abs(fd_pot - d1[k]) < 1e-6 and abs(fd_own - d1[k]) < 1e-6


def test_truth_is_equilibrium_without_stake_and_play_converges_with_stake():
    eq = solve_equilibrium(MODEL, Reward(1, 1))
    assert eq.psi < 1e-12
    rw = Reward(1, 1, b=0.2)
    eq = solve_equilibrium(MODEL, rw)
    pot = Potential(MODEL, rw)
    assert pot.concealment(eq.g1)["dirty"] > 0.01
    assert exploitability(MODEL, rw, 0.5 * (eq.g1 + eq.g2)) < 1e-9
    assert exploitability(MODEL, rw, np.zeros(pot.S)) > 1e-4
    assert exploitability(MODEL, Reward(1, 1), np.zeros(pot.S)) < 1e-12
    g1, g2, _ = gradient_play(MODEL, rw, seed=0, iters=20000)
    assert abs(pot.concealment(g1)["dirty"] - pot.concealment(eq.g1)["dirty"]) < 1e-3


def test_integer_hide_one_threshold():
    m = ExactModel(LedgerConfig(recognizability=0.0))
    thr = hide_one_threshold(m, 1.0)
    assert thr["binding_k"] == 1
    hide = np.maximum(m.K - 1, 0)
    for lam, expect in ((thr["lam_crit"] - 0.02, True), (thr["lam_crit"] + 0.02, False)):
        *_, stable = integer_br_iteration(m, Reward(1.0, lam), hide)
        assert stable is expect
    # closed form for the two candidate reports at K = 1
    Q = integer_cost_table(m, Reward(1.0, 0.5), hide, np.arange(0.0, 14))
    k1 = m.index[(1, 0, 0)]
    c1, q0 = m.c_k()[1], m.q0
    assert np.isclose(Q[k1, 0], 0.5 * c1 + c1 * q0)
    assert np.isclose(Q[k1, 1], 1 - c1 * q0)


def test_supervisor_interface():
    b = sample_text_batch(CFG, 5, np.random.default_rng(3), keep_text=5)
    sup = RuleSupervisor(MENU["honest"], MODEL)
    assert sup.report(b["texts"][0]) == b["K1"][0]
    llm = LLMSupervisor()
    with pytest.raises(NotImplementedError):
        llm.report(b["texts"][0])
    assert LLMSupervisor.parse_reply("thinking...\nREPORT: 2.5") == 2.5
    assert LLMSupervisor(lambda s, u: "REPORT: 3", collusive=True).report("x") == 3.0
