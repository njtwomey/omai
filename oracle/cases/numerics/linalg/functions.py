"""Matrix functions and matrix equations against scipy.linalg (and sklearn's Johnson-Lindenstrauss bound)."""

import numpy as np
import scipy
import scipy.linalg as sl
import sklearn
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng, spd

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(29)
    out: list[Case] = []
    for scale in [0.01, 0.5, 3.0, 30.0]:
        a = g.normal(size=(4, 4)) * scale
        out.append(case(f"expm/{scale}", "expm", [a], sl.expm(a), rtol=1e-10, atol=1e-13))
    a, e = g.normal(size=(3, 3)), g.normal(size=(3, 3))
    _, frechet = sl.expm_frechet(a, e)
    out.append(case("expmFrechet", "expmFrechet", [a, e], frechet, rtol=1e-10, atol=1e-13))
    pos = spd(g, 4) / 4 + g.normal(size=(4, 4)) * 0.1  # eigenvalues comfortably in the right half-plane
    out.append(case("sqrtm", "sqrtm", [pos], np.real(sl.sqrtm(pos)), rtol=1e-9, atol=1e-12))
    out.append(case("logm", "logm", [pos], np.real(sl.logm(pos)), rtol=1e-9, atol=1e-12))
    m, n = 3, 2
    sa, sb, sq = g.normal(size=(m, m)), g.normal(size=(n, n)), g.normal(size=(m, n))
    out.append(case("solveSylvester", "solveSylvester", [sa, sb, sq], sl.solve_sylvester(sa, sb, sq), rtol=1e-9))
    stable = g.normal(size=(3, 3)) - 3 * np.eye(3)
    q = spd(g, 3)
    out.append(
        case(
            "solveContinuousLyapunov",
            "solveContinuousLyapunov",
            [stable, q],
            sl.solve_continuous_lyapunov(stable, q),
            rtol=1e-9,
        )
    )
    contraction = g.normal(size=(3, 3)) * 0.3
    out.append(
        case(
            "solveDiscreteLyapunov",
            "solveDiscreteLyapunov",
            [contraction, q],
            sl.solve_discrete_lyapunov(contraction, q),
            rtol=1e-9,
        )
    )
    ca, cb = g.normal(size=(3, 3)), g.normal(size=(3, 2))
    cq, cr = spd(g, 3), spd(g, 2)
    out.append(
        case("solveCare", "solveCare", [ca, cb, cq, cr], sl.solve_continuous_are(ca, cb, cq, cr), rtol=1e-8, atol=1e-10)
    )
    out.append(
        case(
            "solveDare",
            "solveDare",
            [ca * 0.5, cb, cq, cr],
            sl.solve_discrete_are(ca * 0.5, cb, cq, cr),
            rtol=1e-8,
            atol=1e-10,
        )
    )
    for points, eps in [(1000, 0.1), (1e6, 0.5), (50, 0.9)]:
        out.append(
            case(
                f"johnsonLindenstraussDim/{points}/{eps}",
                "johnsonLindenstraussDim",
                [],
                np.array(float(johnson_lindenstrauss_min_dim(int(points), eps=eps))),
                n=points,
                eps=eps,
            )
        )
    return out
