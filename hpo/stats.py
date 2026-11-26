"""Small statistics helpers."""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


def mean_ci(x, axis=0, level=0.95):
    """Mean and half-width of a Student-t confidence interval over ``axis``.

    Used for across-seed aggregation (the unit of replication is the seed).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    m = x.mean(axis=axis)
    if n < 2:
        return m, np.full_like(m, np.nan)
    se = x.std(axis=axis, ddof=1) / np.sqrt(n)
    return m, _st.t.ppf(0.5 + level / 2, n - 1) * se
