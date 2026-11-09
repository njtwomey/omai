"""Decision-threshold tuning against sklearn.model_selection: `TunedThresholdClassifierCV` (the threshold that maximises
a metric of the labels over a stratified k-fold, with the fold curves interpolated onto a common grid) and
`FixedThresholdClassifier`, both over GaussianNB so that the probabilities are deterministic."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.model_selection import FixedThresholdClassifier, StratifiedKFold, TunedThresholdClassifierCV
from sklearn.naive_bayes import GaussianNB

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def data(seed: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    g = rng(seed)
    n0, n1 = 70, 30
    x = np.vstack([g.normal(loc=0.0, size=(n0, 2)), g.normal(loc=1.2, scale=1.3, size=(n1, 2))])
    y = np.concatenate([np.zeros(n0), np.ones(n1)])
    held = g.normal(loc=0.6, scale=1.5, size=(25, 2))
    return x, y, held


def cases() -> list[Case]:
    out: list[Case] = []
    for seed, scoring, folds, thresholds in [
        (1, "balanced_accuracy", 5, 100),
        (2, "f1", 4, 60),
        (3, "accuracy", 3, 100),
        (4, "recall", 5, 40),
        (5, "jaccard", 5, 100),
    ]:
        x, y, held = data(seed)
        tuned = TunedThresholdClassifierCV(
            GaussianNB(),
            scoring=scoring,
            cv=StratifiedKFold(folds),
            thresholds=thresholds,
            refit=True,
        ).fit(x, y)
        out.append(
            case(
                f"tuned/{scoring}/{seed}",
                "tuned",
                [x, y, held],
                {
                    "threshold": np.array(tuned.best_threshold_),
                    "score": np.array(tuned.best_score_),
                    "predict": tuned.predict(held).astype(np.float64),
                },
                rtol=1e-7,
                atol=1e-9,
                scoring=scoring,
                folds=folds,
                thresholds=thresholds,
            )
        )
    for seed, threshold in [(1, 0.3), (2, 0.8), (3, 0.5)]:
        x, y, held = data(seed)
        fixed = FixedThresholdClassifier(GaussianNB(), threshold=threshold).fit(x, y)
        out.append(
            case(
                f"fixed/{threshold}",
                "fixed",
                [x, y, held],
                {"predict": fixed.predict(held).astype(np.float64)},
                rtol=1e-12,
                atol=1e-12,
                threshold=threshold,
            )
        )
    return out
