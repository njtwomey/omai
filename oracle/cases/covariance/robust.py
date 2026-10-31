"""The minimum covariance determinant and the elliptic envelope against scikit-learn on data with planted outliers: the
reweighted and raw location and covariance, the support and the squared Mahalanobis distances. The planted outliers are
far enough that the global minimum of the determinant is unique, so scikit-learn's random starts and ours find it."""

from importlib.metadata import version

import numpy as np
import numpy.typing as npt
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng
from sklearn.covariance import EllipticEnvelope, MinCovDet

REFERENCE: dict[str, str] = {"scikit-learn": version("scikit-learn"), "numpy": np.__version__}

TOL = {"rtol": 1e-8, "atol": 1e-10}


def planted(seed: int) -> npt.NDArray[np.float64]:
    """Ninety correlated normal rows and ten far ones."""
    g = rng(seed)
    a = np.array([[1.0, 0.4, 0.1], [0.0, 0.8, -0.3], [0.0, 0.0, 0.6]])
    clean = g.normal(size=(90, 3)) @ a + np.array([1.0, -2.0, 0.5])
    far = g.normal(size=(10, 3)) * 0.5 + np.array([9.0, 8.0, -7.0])
    return np.concatenate([clean, far])


def settled(x: npt.NDArray[np.float64], fractions: list[float | None]) -> bool:
    """Whether the determinant's minimum is found from every one of five random starts, for every support size: the
    minimum over subsets of the clean rows has several nearby local minima, and a golden needs the global one."""
    for fraction in fractions:
        found = [MinCovDet(support_fraction=fraction, random_state=r).fit(x).raw_location_ for r in range(5)]
        if any(not np.allclose(found[0], f, rtol=0, atol=1e-12) for f in found[1:]):
            return False
    return True


def cases() -> list[Case]:
    fractions: list[float | None] = [None, 0.9]
    x = planted(next(s for s in range(41, 400) if settled(planted(s), fractions)))
    out: list[Case] = []
    for name, fraction in [("default", None), ("fraction0.9", 0.9)]:
        mcd = MinCovDet(support_fraction=fraction, random_state=0).fit(x)
        out.append(
            case(
                f"mcd/{name}",
                "mcd",
                [x],
                {
                    "location": np.asarray(mcd.location_),
                    "covariance": np.asarray(mcd.covariance_),
                    "rawLocation": np.asarray(mcd.raw_location_),
                    "rawCovariance": np.asarray(mcd.raw_covariance_),
                    "support": np.asarray(mcd.support_, dtype=np.float64),
                    "mahalanobis": np.asarray(mcd.mahalanobis(x)),
                },
                **TOL,
                fraction=-1.0 if fraction is None else fraction,
            )
        )
    for contamination in [0.1, 0.2]:
        env = EllipticEnvelope(contamination=contamination, random_state=0).fit(x)
        out.append(
            case(
                f"envelope/c{contamination}",
                "envelope",
                [x],
                {
                    "decision": np.asarray(env.decision_function(x)),
                    "predict": np.asarray(env.predict(x), dtype=np.float64),
                    "offset": np.asarray(float(env.offset_)),
                },
                **TOL,
                contamination=contamination,
            )
        )
    return out
