"""SoftVoteEnsemble class — kept in its own module so pickling works
regardless of whether the surrounding code runs as `__main__` or imported.

When `python src/models.py` runs, models.py is `__main__`. Anything pickled
from there gets stored with module=`__main__`, which then fails to unpickle
from any other entry point. By defining SoftVoteEnsemble here, the class is
always at `ensemble.SoftVoteEnsemble` (when src/ is on sys.path) regardless
of who imported it.

Usage in a consumer (e.g. simulate.py):

    import sys
    sys.path.insert(0, "src")
    import joblib
    ensemble = joblib.load("models/ensemble.pkl")
    probs = ensemble.predict_proba(X)        # (n, 3) - away_win, draw, home_win
"""
from __future__ import annotations

import numpy as np


class SoftVoteEnsemble:
    """Picklable soft-voting ensemble for use by simulate.py.

    Holds references to three calibrated models + a scaler, and exposes a
    single predict_proba(X) that internally:
      - scales X for LR (the only one that needs scaled features)
      - leaves X unscaled for RF and XGB
      - averages the three predict_proba outputs (equal weight by default)
    """

    def __init__(self, lr, rf, xgb, scaler, weights=None):
        self.lr = lr
        self.rf = rf
        self.xgb = xgb
        self.scaler = scaler
        self.weights = weights
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X) -> np.ndarray:
        probs = [
            self.lr.predict_proba(self.scaler.transform(X)),
            self.rf.predict_proba(X),
        ]
        if self.xgb is not None:
            probs.append(self.xgb.predict_proba(X))
        arr = np.stack(probs, axis=0)
        if self.weights is None:
            return arr.mean(axis=0)
        w = np.asarray(self.weights, dtype=float)
        w = w / w.sum()
        return (arr * w[:, None, None]).sum(axis=0)

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
