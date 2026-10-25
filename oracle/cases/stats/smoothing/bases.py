"""Smoothing bases against patsy (bs, cr, cc and their tensor product te, which follow R's splines and mgcv) and
scipy.interpolate.BSpline (P-spline design matrices). Natural splines are compared by their column space with
patsy's cr, which spans the same natural cubic splines on the same knots."""

from importlib.metadata import version

import numpy as np
import patsy
import scipy
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from scipy.interpolate import BSpline

REFERENCE: dict[str, str] = {"patsy": version("patsy"), "scipy": scipy.__version__}


def design(formula: str, data: dict[str, object]) -> np.ndarray:
    return np.asarray(
        patsy.dmatrix(  # pyright: ignore[reportAttributeAccessIssue]
            formula, data
        ),
        dtype=float,
    )


def cases() -> list[Case]:
    g = rng(321)
    x = np.sort(g.uniform(0, 10, 40))
    z = g.uniform(0, 1, 40)
    out: list[Case] = []
    for df, intercept in [(5, False), (6, True), (4, False)]:
        formula = f"bs(x, df={df}, include_intercept={intercept}) - 1"
        out.append(
            case(
                f"bs/{df}/{intercept}",
                "bs",
                [x, x],
                design(formula, {"x": x}),
                rtol=1e-10,
                atol=1e-12,
                df=df,
                intercept=intercept,
            )
        )
    knots = [2.0, 5.0, 7.5]
    out.append(
        case(
            "bs/knots/quadratic",
            "bsKnots",
            [x, np.asarray(knots)],
            design("bs(x, knots=(2.0, 5.0, 7.5), degree=2) - 1", {"x": x}),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    for k in [4, 7]:
        out.append(case(f"cr/{k}", "cr", [x, x], design(f"cr(x, df={k}) - 1", {"x": x}), rtol=1e-10, atol=1e-12, k=k))
        out.append(
            case(f"cc/{k}", "cc", [z, z], design(f"cc(x, df={k}) - 1", {"x": z}), rtol=1e-10, atol=1e-12, k=k + 1)
        )
    te = design("te(cr(x, df=3), cc(z, df=4)) - 1", {"x": x, "z": z})
    out.append(case("te/cr-cc", "te", [x, z], te, rtol=1e-10, atol=1e-12))
    # The natural spline space: patsy's cr design on the knots that ns(df=5, intercept) uses.
    inner = np.percentile(x, [25.0, 50.0, 75.0])
    allk = np.concatenate([[x.min()], inner, [x.max()]])
    crx = np.asarray(
        patsy.dmatrix(  # pyright: ignore[reportAttributeAccessIssue]
            "cr(x, knots=k, lower_bound=lo, upper_bound=hi) - 1", {"x": x, "k": inner, "lo": x.min(), "hi": x.max()}
        ),
        dtype=float,
    )
    out.append(case("ns/span", "nsSpan", [x, crx], np.asarray(0.0), rtol=0, atol=1e-9, knots=allk))
    lo, hi, nseg, deg = x.min(), x.max(), 7, 3
    dx = (hi - lo) / nseg
    t = lo + (np.arange(nseg + 2 * deg + 1) - deg) * dx
    ps = BSpline.design_matrix(x, t, deg, extrapolate=True).toarray()
    out.append(case("ps", "ps", [x, x], ps, rtol=1e-10, atol=1e-12, k=nseg + deg))
    return out
