"""Linear classifiers against scikit-learn and statsmodels: binary logistic regression with L2, L1 and elastic-net
penalties, multinomial logistic regression, the ridge classifier and probit regression."""

import numpy as np
import sklearn
import statsmodels
import statsmodels.api as sm
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray, rng
from sklearn.linear_model import LogisticRegression, RidgeClassifier

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__}


def problem(seed: int, n: int = 80, k: int = 2) -> tuple[FloatArray, FloatArray]:
    """Gaussian classes with shifted means in three dimensions."""
    g = rng(seed)
    y = np.arange(n) % k
    centres = g.normal(size=(k, 3)) * 1.2
    x = centres[y] + g.normal(size=(n, 3))
    return x, y.astype(np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    x, y = problem(3)
    for name, options, tol in [
        ("l2", {"C": 0.5, "l1_ratio": 0.0}, 1e-7),
        ("l1", {"C": 0.3, "l1_ratio": 1.0, "solver": "saga"}, 1e-5),
        ("elasticnet", {"C": 0.3, "l1_ratio": 0.5, "solver": "saga"}, 1e-5),
    ]:
        lr = LogisticRegression(tol=1e-12, max_iter=100000, random_state=0, **options).fit(x, y)  # pyright: ignore[reportArgumentType]
        out.append(
            case(
                f"logistic/{name}",
                "logistic",
                [x, y],
                {"coef": lr.coef_[0], "intercept": lr.intercept_, "proba": lr.predict_proba(x[:5])},
                rtol=tol,
                atol=tol,
                C=float(options["C"]),
                penalty={0.0: "l2", 1.0: "l1"}.get(float(options["l1_ratio"]), "elasticnet"),
                l1Ratio=float(options.get("l1_ratio", 0.5)),
            )
        )
    x3, y3 = problem(4, 90, 3)
    lr = LogisticRegression(C=1.0, tol=1e-12, max_iter=100000).fit(x3, y3)
    out.append(
        case(
            "logistic/multinomial",
            "logistic",
            [x3, y3],
            {"coef": lr.coef_, "intercept": lr.intercept_, "proba": lr.predict_proba(x3[:5])},
            rtol=1e-6,
            atol=1e-7,
            C=1.0,
            penalty="l2",
            l1Ratio=0.5,
        )
    )
    rc = RidgeClassifier(alpha=2.0).fit(x3, y3)
    out.append(
        case(
            "ridgeClassifier",
            "ridgeClassifier",
            [x3, y3],
            {"coef": rc.coef_, "intercept": rc.intercept_},
            rtol=1e-10,
            atol=1e-10,
        )
    )
    probit = sm.GLM(
        y, np.column_stack([x, np.ones(len(y))]), family=sm.families.Binomial(sm.families.links.Probit())
    ).fit(tol=1e-13)
    out.append(case("probit", "probit", [x, y], {"params": probit.params, "bse": probit.bse}, rtol=1e-6, atol=1e-8))
    return out
