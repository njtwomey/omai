"""Explicit feature maps against scikit-learn's kernel_approximation, given the same random draws: RBFSampler
(random_weights_, random_offset_), Nystroem (component_indices_, normalization_), SkewedChi2Sampler and
PolynomialCountSketch (indexHash_, bitHash_), whose transforms whitebox-js reproduces from those draws, and the
deterministic AdditiveChi2Sampler. The randomised constructors themselves are tested statistically in TypeScript."""

import numpy as np
import sklearn
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    Nystroem,
    PolynomialCountSketch,
    RBFSampler,
    SkewedChi2Sampler,
)
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng, uniform

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def cases() -> list[Case]:
    g = rng(51)
    x = uniform(g, (9, 3))
    h = uniform(g, (6, 4), 0.0, 2.0)
    h[0, 1] = 0.0
    h[4, 3] = 0.0
    out: list[Case] = []
    for gamma, n in [(0.7, 20), (2.0, 7)]:
        s = RBFSampler(gamma=gamma, n_components=n, random_state=0).fit(x)
        out.append(
            case(
                f"rbf-sampler-{gamma}-{n}",
                "rbfSampler",
                [x, s.random_weights_, s.random_offset_],
                s.transform(x),
                rtol=1e-12,
                atol=1e-14,
            )
        )
    for kernel, params in [("rbf", {"gamma": 0.5}), ("polynomial", {"gamma": 0.3, "degree": 2, "coef0": 1.0})]:
        ny = Nystroem(kernel=kernel, n_components=5, random_state=0, **params).fit(x)
        out.append(
            case(
                f"nystroem-{kernel}",
                "nystroem",
                [x],
                {"normalization": ny.normalization_, "transform": ny.transform(x)},
                rtol=1e-9,
                atol=1e-11,
                kernel=kernel,
                indices=[int(i) for i in ny.component_indices_],
                **params,
            )
        )
    for steps in [1, 2, 3]:
        out.append(
            case(
                f"additive-chi2-{steps}",
                "additiveChi2",
                [h],
                AdditiveChi2Sampler(sample_steps=steps).fit_transform(h),
                rtol=1e-12,
                atol=1e-14,
                steps=steps,
            )
        )
    out.append(
        case(
            "additive-chi2-interval",
            "additiveChi2",
            [h],
            AdditiveChi2Sampler(sample_steps=4, sample_interval=0.3).fit_transform(h),
            rtol=1e-12,
            atol=1e-14,
            steps=4,
            interval=0.3,
        )
    )
    sk = SkewedChi2Sampler(skewedness=0.5, n_components=15, random_state=0).fit(h)
    out.append(
        case(
            "skewed-chi2",
            "skewedChi2",
            [h, sk.random_weights_, sk.random_offset_],
            sk.transform(h),
            rtol=1e-12,
            atol=1e-14,
            skewedness=0.5,
        )
    )
    for degree, coef0, gamma in [(2, 0.0, 1.0), (3, 1.5, 0.8)]:
        ps = PolynomialCountSketch(gamma=gamma, degree=degree, coef0=coef0, n_components=16, random_state=0).fit(x)
        out.append(
            case(
                f"polynomial-count-sketch-{degree}",
                "polynomialCountSketch",
                [x],
                ps.transform(x),
                rtol=1e-10,
                atol=1e-12,
                degree=degree,
                coef0=coef0,
                gamma=gamma,
                indexHash=np.asarray(ps.indexHash_).tolist(),
                bitHash=np.asarray(ps.bitHash_).tolist(),
            )
        )
    return out
