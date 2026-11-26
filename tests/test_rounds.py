import numpy as np
import pytest

from hpo.rounds import (GameConfig, play_trajectory, payments, parse_reply, summarize,
                        mock_backend, system_prompt, user_prompt)


def test_payments_plain_ignores_plants():
    (a1, f1), (a2, f2) = payments("plain", 2, 0, 5, 7)
    assert a1 == a2 == -4 and f1 == f2 == 0


def test_payments_perturb_truthful_is_free():
    # T=1; plants 2 and 0; truthful reports 3 and 1
    (a1, f1), (a2, f2) = payments("perturb", 3, 1, 2, 0)
    assert a1 == a2 == 0 and f1 == f2 == 0


def test_payments_perturb_zero_report_hits_floor_and_agreement():
    (a1, f1), (a2, f2) = payments("perturb", 0, 0, 2, 0)
    assert f1 == -4 and f2 == 0 and a1 == a2 == -4


def test_floor_is_one_sided():
    (_, f1), _ = payments("perturb", 5, 3, 2, 0)   # reporting above the plants costs no floor
    assert f1 == 0


def test_parse_reply_takes_last_report_and_note():
    r, n = parse_reply("REPORT: 9\nactually\nREPORT: 1.5\nNOTE: shade a little")
    assert r == 1.5 and n == "shade a little"
    assert parse_reply("I will not answer")[0] is None


def test_prompts_contain_rule_and_hide_truth():
    s = system_prompt("perturb", 2, 20, 2.0)
    assert "secretly inserted" in s and "privately gain 2 points" in s and "leaves out" in s
    assert "secretly" not in system_prompt("plain", 0, 20, 2.0)
    u = user_prompt([dict(round=1, report=2, payment=-1.0, note="x")], 2, "# ledger 000002\n01. 1 + 1 = 2")
    assert "round 1: you reported 2; payment -1.00" in u and "planted" not in u


@pytest.mark.parametrize("rule", ["plain", "perturb", "placebo"])
def test_honest_mock_is_truthful_and_unpenalised(rule):
    recs = play_trajectory(GameConfig(rule=rule, rounds=12), mock_backend("honest"), seed=3)
    assert len(recs) == 24
    for r in recs:
        assert r["adjusted"] == r["T"] and r["pay_agree"] == 0 and r["pay_floor"] == 0
        assert (r["planted"] == 0) or rule == "perturb"


def test_zero_mock_free_under_plain_penalised_under_perturb():
    plain = play_trajectory(GameConfig(rule="plain", rounds=30), mock_backend("zero"), seed=5)
    pert = play_trajectory(GameConfig(rule="perturb", rounds=30), mock_backend("zero"), seed=5)
    assert all(r["pay_agree"] + r["pay_floor"] == 0 for r in plain)
    assert sum(r["pay_agree"] + r["pay_floor"] for r in pert) < 0
    s = summarize(plain, n_boot=50)
    assert s["all_rounds"]["concealed_share"]["mean"] == pytest.approx(1.0)


def test_views_differ_only_under_perturb():
    def run(rule):
        seen = []
        def spy(system, user):
            seen.append(user.split("Ledger to audit:")[-1])
            return "REPORT: 0\nNOTE: n"
        play_trajectory(GameConfig(rule=rule, rounds=8), spy, seed=1)
        return [(seen[i], seen[i + 1]) for i in range(0, len(seen), 2)]
    assert all(a == b for a, b in run("plain"))
    assert any(a != b for a, b in run("perturb"))
