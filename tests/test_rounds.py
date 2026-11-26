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


# ------------------------------------------------- void rounds, retries, metering, CLI rails
import json
import pathlib
import subprocess
import sys

from hpo.rounds import BudgetExceeded, Meter, Reply, metered


def test_void_round_is_shown_as_void_not_as_a_zero_payment():
    honest, n, seen = mock_backend("honest"), {"k": 0}, []
    def flaky(system, user):                      # reviewer 2 gives no report in round 1
        seen.append(user); n["k"] += 1
        return "I will not answer." if n["k"] == 2 else honest(system, user)
    recs = play_trajectory(GameConfig(rounds=2, parse_retries=0), flaky, seed=1)
    assert "no payment (round void)" in seen[2] and "payment 0.00" not in seen[2]
    assert "you gave no valid report" in seen[3]
    first = [r for r in recs if r["round"] == 1]
    assert [r["parsed"] for r in first] == [True, False] and all(r["void"] for r in first)
    assert all(np.isnan(r["pay_agree"]) for r in first)
    assert not any(r["void"] for r in recs if r["round"] == 2)


def test_missing_report_is_asked_again_and_the_failed_attempt_is_kept():
    honest, n = mock_backend("honest"), {"k": 0}
    def flaky(system, user):
        n["k"] += 1
        return Reply("I refuse.", stop_reason="refusal") if n["k"] == 1 else honest(system, user)
    a, b = play_trajectory(GameConfig(rounds=1), flaky, seed=1)
    assert a["parsed"] and a["attempts"] == 2 and not a["void"] and a["pay_agree"] == 0
    assert a["failed_replies"] == [dict(text="I refuse.", stop_reason="refusal")]
    assert b["attempts"] == 1
    s = summarize([a, b], last_k=1, n_boot=20)
    assert s["first_attempt_failure_rate"] == 0.5 and s["refusal_rate"] == 0.5
    assert s["parse_failure_rate"] == 0 and s["void_rate"] == 0


def test_reply_metadata_is_recorded_and_metered():
    honest = mock_backend("honest")
    def backend(system, user):
        return Reply(honest(system, user), thinking="t", stop_reason="end_turn",
                     in_tokens=100, out_tokens=10)
    meter = Meter(price_in=2.0, price_out=10.0, max_usd=1.0)
    recs = play_trajectory(GameConfig(rounds=3), metered(backend, meter), seed=2)
    assert all(r["thinking"] == "t" and r["in_tokens"] == 100 for r in recs)
    assert meter.calls == 6 and meter.usd == pytest.approx((600 * 2 + 60 * 10) / 1e6)
    s = summarize(recs, last_k=3, n_boot=20)
    assert s["in_tokens"] == 600 and s["out_tokens"] == 60
    assert s["all_rounds"]["exact_count"]["mean"] == 1.0


def test_spending_cap_stops_before_the_next_call():
    def backend(system, user):                    # one dollar per call
        return Reply("REPORT: 0\nNOTE: n", in_tokens=1_000_000)
    meter = Meter(price_in=1.0, price_out=1.0, max_usd=3.0)
    with pytest.raises(BudgetExceeded):
        play_trajectory(GameConfig(rounds=10), metered(backend, meter), seed=0)
    assert meter.calls == 3
    assert Meter(max_usd=1.0).usd is None         # no prices: cost unknown, cap not enforced


def test_prompt_fingerprint_changes_with_wording(monkeypatch):
    import hpo.rounds as R
    before = R.prompt_fingerprint()
    monkeypatch.setattr(R, "FORMAT", R.FORMAT + " ")
    assert R.prompt_fingerprint() != before


def _cli(tmp_path, *extra):
    script = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "exp6_llm_rounds.py"
    return subprocess.run(
        [sys.executable, str(script), "elicit", "--backend", "mock-adaptive", "--levels", "1",
         "--trajectories", "2", "--rounds", "3", "--out", str(tmp_path), *extra],
        capture_output=True, text=True)


def test_cli_refuses_to_mix_configurations_in_one_folder(tmp_path):
    assert _cli(tmp_path).returncode == 0
    arm = tmp_path / "mock-adaptive" / "plain_level1.jsonl"
    saved = arm.read_text()
    assert len(saved.splitlines()) == 12
    assert "privately gain 0.5 points" in arm.with_name("plain_level1.prompt.txt").read_text()
    assert json.loads((tmp_path / "mock-adaptive" / "config.json").read_text())["stake"] == 0.5
    for changed in (("--stake", "0.9"), ("--rounds", "5")):
        again = _cli(tmp_path, *changed)
        assert again.returncode != 0 and "different configuration" in again.stderr
        assert arm.read_text() == saved            # nothing reused, nothing deleted
    assert _cli(tmp_path).returncode == 0          # same settings: resumes, runs nothing new
    assert arm.read_text() == saved


def test_level_5_assigns_a_role_without_saying_what_to_report():
    s = system_prompt("plain", 5, 10, 0.5)
    assert "privately gain 0.5 points" in s and "cares about nothing except total points" in s
    assert "under-report" not in s and "settle with the other reviewer" not in s
