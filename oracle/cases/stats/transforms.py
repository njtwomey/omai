"""Power transforms against scipy.stats (boxcox, yeojohnson and their log-likelihoods and lambdas), scipy.special
(inv_boxcox) and sklearn (the Yeo-Johnson inverse and StandardScaler)."""

import numpy as np
import scipy
import scipy.stats as st
import sklearn
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer, StandardScaler
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "sklearn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(77)
    x = g.lognormal(0.5, 0.7, size=40)
    z = g.normal(0.5, 1.5, size=40) ** 3 / 4
    out: list[Case] = []
    for lam in [-1.0, 0.0, 0.5, 2.0]:
        out.append(case(f"boxcoxLlf/{lam}", "boxcoxLlf", [x, x], np.asarray(st.boxcox_llf(lam, x)), lam=lam))
        out.append(case(f"boxcox/{lam}", "boxcoxFixed", [x, x], np.asarray(st.boxcox(x, lmbda=lam)), lam=lam))
        out.append(
            case(f"yeojohnsonLlf/{lam}", "yeojohnsonLlf", [z, z], np.asarray(st.yeojohnson_llf(lam, z)), lam=lam)
        )
        out.append(
            case(f"yeojohnson/{lam}", "yeojohnsonFixed", [z, z], np.asarray(st.yeojohnson(z, lmbda=lam)), lam=lam)
        )
        y = st.boxcox(x, lmbda=lam)
        out.append(case(f"invBoxcox/{lam}", "invBoxcox", [y, y], inv_boxcox(y, lam), lam=lam))
    # scipy's brent stops at xtol 1.48e-8 in lambda; the transforms inherit that accuracy.
    _, lmax, ci = st.boxcox(x, alpha=0.1)
    out.append(case("boxcox/lambda", "boxcoxLambda", [x, x], np.asarray(lmax), rtol=1e-6))
    out.append(case("boxcox/interval", "boxcoxInterval", [x, x], np.asarray(ci), rtol=1e-6))
    out.append(case("yeojohnson/lambda", "yeojohnsonLambda", [z, z], np.asarray(st.yeojohnson_normmax(z)), rtol=1e-6))
    pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(z[:, None])
    pt.lambdas_ = np.array([0.7])
    y = st.yeojohnson(z, lmbda=0.7)
    out.append(case("invYeojohnson", "invYeojohnson", [y, y], pt.inverse_transform(y[:, None])[:, 0], lam=0.7))
    M = g.normal(size=(10, 3)) * np.array([1.0, 5.0, 0.1]) + np.array([2.0, -1.0, 0.0])
    out.append(case("standardise", "standardise", [M, M], StandardScaler().fit_transform(M)))
    return out
