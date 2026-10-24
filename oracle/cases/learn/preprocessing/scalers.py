"""Scalers and transformers against sklearn.preprocessing: fitted parameters, transforms of new data and inverses."""

from typing import Any

import numpy as np
import sklearn
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def scaler_case(
    name: str,
    fn: str,
    est: Any,  # noqa: ANN401  (sklearn estimators are untyped)
    x: FloatArray,
    z: FloatArray,
    w: FloatArray | None = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    **hyper: object,
) -> Case:
    """Fit on x (weighted by w), then record the transform of z, its inverse and the transform of x."""
    if w is None:
        est.fit(x)
    else:
        est.fit(x, sample_weight=w)
    t = est.transform(z)
    outputs: dict[str, object] = {"transform": t, "inverse": est.inverse_transform(t), "train": est.transform(x)}
    if hasattr(est, "lambdas_"):
        outputs["lambdas"] = np.asarray(est.lambdas_)
    inputs = [x, z] if w is None else [x, z, w]
    return case(name, fn, inputs, outputs, rtol=rtol, atol=atol, **hyper)


def cases() -> list[Case]:
    g = rng(4330)
    x = g.normal(size=(30, 3)) * np.array([1.0, 5.0, 0.1]) + np.array([0.0, 3.0, -1.0])
    x[:, 2] = np.round(x[:, 2], 1)
    z = g.normal(size=(8, 3)) * np.array([1.5, 6.0, 0.2]) + np.array([0.0, 3.0, -1.0])
    xnan = x.copy()
    xnan[[2, 7, 11], [0, 1, 0]] = np.nan
    w = g.uniform(0.2, 2.0, size=30)
    const = np.column_stack([x[:, 0], np.full(30, 2.5)])
    pos = np.exp(x[:, :2] / 3)
    zpos = np.exp(z[:, :2] / 3)
    out: list[Case] = []
    for wm in [True, False]:
        for ws in [True, False]:
            out.append(
                scaler_case(
                    f"standard/{wm}-{ws}",
                    "standard",
                    StandardScaler(with_mean=wm, with_std=ws),
                    x,
                    z,
                    withMean=wm,
                    withStd=ws,
                )
            )
    out.append(scaler_case("standard/weighted", "standard", StandardScaler(), x, z, w, withMean=True, withStd=True))
    out.append(scaler_case("standard/nan", "standard", StandardScaler(), xnan, z, withMean=True, withStd=True))
    out.append(
        scaler_case("standard/constant", "standard", StandardScaler(), const, const[:5], withMean=True, withStd=True)
    )
    out.append(scaler_case("minmax", "minmax", MinMaxScaler(), x, z, lo=0.0, hi=1.0, clip=False))
    out.append(
        scaler_case("minmax/range", "minmax", MinMaxScaler(feature_range=(-1, 2)), xnan, z, lo=-1.0, hi=2.0, clip=False)
    )
    est = MinMaxScaler(clip=True).fit(x)
    out.append(case("minmax/clip", "minmaxClip", [x, z], {"transform": est.transform(z)}, lo=0.0, hi=1.0, clip=True))
    out.append(scaler_case("maxabs", "maxabs", MaxAbsScaler(), xnan, z))
    for rng_q, unit in [((25.0, 75.0), False), ((10.0, 90.0), True)]:
        out.append(
            scaler_case(
                f"robust/{rng_q[0]}-{unit}",
                "robust",
                RobustScaler(quantile_range=rng_q, unit_variance=unit),
                xnan,
                z,
                lo=rng_q[0],
                hi=rng_q[1],
                unitVariance=unit,
            )
        )
    for norm in ["l1", "l2", "max"]:
        est = Normalizer(norm=norm).fit(x)
        out.append(
            case(
                f"normalizer/{norm}",
                "normalizer",
                [x, np.vstack([z, np.zeros(3)])],
                {"transform": est.transform(np.vstack([z, np.zeros(3)]))},
                norm=norm,
            )
        )
    for dist in ["uniform", "normal"]:
        for nq in [10, 30]:
            est = QuantileTransformer(n_quantiles=nq, output_distribution=dist)
            out.append(scaler_case(f"quantile/{dist}-{nq}", "quantile", est, x, z, nQuantiles=nq, distribution=dist))
        est = QuantileTransformer(n_quantiles=20, output_distribution=dist)
        out.append(scaler_case(f"quantile/{dist}-nan", "quantile", est, xnan, z, nQuantiles=20, distribution=dist))
    # Brent's search for lambda stops at xtol 1.48e-8 (scipy), so the transforms agree to about 1e-7.
    for std in [True, False]:
        est = PowerTransformer(method="yeo-johnson", standardize=std)
        out.append(
            scaler_case(
                f"power/yeo-johnson-{std}",
                "power",
                est,
                x,
                z,
                rtol=1e-6,
                atol=1e-8,
                method="yeo-johnson",
                standardize=std,
            )
        )
        est = PowerTransformer(method="box-cox", standardize=std)
        out.append(
            scaler_case(
                f"power/box-cox-{std}", "power", est, pos, zpos, rtol=1e-6, atol=1e-8, method="box-cox", standardize=std
            )
        )
    return out
