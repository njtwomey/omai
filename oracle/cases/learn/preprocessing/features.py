"""Feature construction against sklearn.preprocessing: polynomial features, B-spline features with every
extrapolation, binning (uniform, quantile, k-means) with inverses, the binariser and function transformers."""

from typing import Any

import numpy as np
import sklearn
from mdml_oracle.fixture import Case, case, encode, encode_all, encode_params
from mdml_oracle.helpers import rng
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    KBinsDiscretizer,
    PolynomialFeatures,
    SplineTransformer,
)

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def param_case(
    name: str, fn: str, inputs: list[object], output: object, params: dict[str, object], rtol: float = 1e-12
) -> Case:
    """A case whose parameters come as a dictionary."""
    return Case(
        name=name,
        fn=fn,
        inputs=encode_all(inputs),
        params=encode_params(params),
        output=encode(output),
        rtol=rtol,
        atol=1e-12,
    )


def cases() -> list[Case]:
    g = rng(4333)
    x = g.normal(size=(25, 3))
    z = np.vstack([g.normal(size=(4, 3)), [[-4.0, 0.0, 5.0]]])
    out: list[Case] = []
    poly: list[tuple[str, dict[str, Any], dict[str, object]]] = [
        ("2", {"degree": 2}, {"degree": 2}),
        ("3-nobias", {"degree": 3, "include_bias": False}, {"degree": 3, "includeBias": False}),
        ("interaction", {"degree": 3, "interaction_only": True}, {"degree": 3, "interactionOnly": True}),
        ("range", {"degree": (2, 3)}, {"minDegree": 2, "degree": 3}),
    ]
    for label, kw, params in poly:
        est = PolynomialFeatures(**kw).fit(x)
        names = ",".join(str(s) for s in est.get_feature_names_out())
        out.append(
            param_case(
                f"polynomial/{label}", "polynomial", [x, z], {"transform": est.transform(z), "names": names}, params
            )
        )
    for extrapolation in ["constant", "linear", "continue", "periodic"]:
        for degree in [1, 2, 3]:
            for knots in ["uniform", "quantile"]:
                if extrapolation == "periodic" and knots == "quantile":
                    continue
                # sklearn 1.9 bumps `degree` in place for linear extrapolation of degree 1, which shifts the range of
                # every later feature (a bug); one feature keeps the comparison to the correct behaviour.
                xs, zs = (x[:, :1], z[:, :1]) if (extrapolation, degree) == ("linear", 1) else (x, z)
                est = SplineTransformer(n_knots=5, degree=degree, knots=knots, extrapolation=extrapolation).fit(xs)
                params: dict[str, object] = {
                    "degree": degree,
                    "knots": knots,
                    "extrapolation": extrapolation,
                    "includeBias": True,
                }
                out.append(
                    param_case(
                        f"spline/{extrapolation}/{degree}/{knots}",
                        "spline",
                        [xs, zs],
                        {"transform": est.transform(zs), "train": est.transform(xs)},
                        params,
                        rtol=1e-10,
                    )
                )
    est = SplineTransformer(n_knots=4, degree=2, include_bias=False).fit(x)
    out.append(
        param_case(
            "spline/nobias",
            "spline",
            [x, z],
            {"transform": est.transform(z)},
            {"degree": 2, "knots": "uniform", "extrapolation": "constant", "includeBias": False, "nKnots": 4},
            rtol=1e-10,
        )
    )
    xb = np.column_stack([x[:, 0], np.round(x[:, 1] * 2) / 2, np.exp(x[:, 2])])
    for strategy in ["uniform", "quantile", "kmeans"]:
        for encoding in ["ordinal", "onehot-dense"]:
            est = KBinsDiscretizer(
                n_bins=4, encode=encoding, strategy=strategy, quantile_method="averaged_inverted_cdf"
            )
            est.fit(xb)
            t = est.transform(xb[:8])
            out.append(
                param_case(
                    f"kbins/{strategy}/{encoding}",
                    "kbins",
                    [xb, xb[:8]],
                    {"transform": t, "inverse": est.inverse_transform(t), "edges": np.concatenate(est.bin_edges_)},
                    {"strategy": strategy, "encode": "onehot" if encoding != "ordinal" else "ordinal"},
                    rtol=1e-10,
                )
            )
    est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile", quantile_method="linear").fit(xb)
    out.append(
        param_case(
            "kbins/quantile-linear",
            "kbins",
            [xb, xb[:8]],
            {"transform": est.transform(xb[:8]), "edges": np.concatenate(est.bin_edges_)},
            {"strategy": "quantile", "encode": "ordinal", "nBins": 3, "quantileMethod": "linear"},
        )
    )
    out.append(
        case(
            "binarizer", "binarizer", [x, z], {"transform": Binarizer(threshold=0.3).fit(x).transform(z)}, threshold=0.3
        )
    )
    ft = FunctionTransformer(np.log1p, np.expm1).fit(np.abs(x))
    out.append(
        case(
            "function",
            "function",
            [np.abs(x), np.abs(z)],
            {"transform": ft.transform(np.abs(z)), "inverse": ft.inverse_transform(np.abs(z))},
        )
    )
    return out
