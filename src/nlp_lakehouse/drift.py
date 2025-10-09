from typing import Sequence
import numpy as np

def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.size == 0 or a.size == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(e, quantiles))
    if cuts.size < 2:
        return 0.0
    e_hist, _ = np.histogram(e, bins=cuts)
    a_hist, _ = np.histogram(a, bins=cuts)
    e_pct = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
