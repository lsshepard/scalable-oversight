import numpy as np
import pytest

from hpo.ledger import LedgerConfig
from hpo.model import ExactModel


@pytest.mark.parametrize("p", [0.0, 0.3, 0.8, 1.0])
def test_probabilities_and_gamma(p):
    m = ExactModel(LedgerConfig(recognizability=p))
    assert np.isclose(m.Ps.sum(), 1) and np.allclose(m.PTs.sum(0), 1)
    full = np.sort(np.linalg.eigvals(m.view_operator()).real)[::-1]
    assert np.isclose(full[0], 1)
    assert np.isclose(1 - full[1], m.gamma, atol=1e-9)
    assert np.allclose(m.view_operator().sum(1), 1)
    assert 0 <= m.sigma2 <= m.p0 * (1 - m.p0) + 1e-12


def test_perfect_recognizability_reveals_everything():
    m = ExactModel(LedgerConfig(recognizability=1.0))
    assert np.isclose(m.sigma2, m.p0 * (1 - m.p0))
    assert abs(m.gamma) < 1e-9


def test_marker_reveals_dirtiness():
    m = ExactModel(LedgerConfig(marker=True))
    assert np.isclose(m.sigma2, m.p0 * (1 - m.p0))
    assert abs(m.gamma) < 1e-9


def test_sigma_gamma_monotone_in_p():
    ms = [ExactModel(LedgerConfig(recognizability=p)) for p in (0, .2, .4, .6, .8, 1)]
    assert np.all(np.diff([m.sigma2 for m in ms]) > 0)
    assert np.all(np.diff([m.gamma for m in ms]) < 0)


def test_c_k_closed_form():
    m = ExactModel(LedgerConfig())
    pD, pT = m.pD, m.pT
    c1 = pT[0] * pD[1] / (pT[0] * pD[1] + pT[1] * pD[0])
    assert np.isclose(m.c_k()[1], c1)
    assert np.isclose(m.c_k()[0], 1.0)
