"""Categorical encoders against sklearn.preprocessing: one-hot (drop, unknowns, infrequent groups), ordinal, label,
label and multi-label binarisers, and the cross-fitted target encoder."""

from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
from whitebox_oracle.fixture import Case, case, encode, encode_all, encode_params
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}


def as_float(a: Any) -> FloatArray:  # noqa: ANN401  (sklearn returns object arrays with None for unknowns)
    """An inverse transform as floats, None (unknown or infrequent) as NaN."""
    arr = np.asarray(a, dtype=object)
    return np.array([[np.nan if v is None or isinstance(v, str) else float(v) for v in row] for row in arr])


def param_case(name: str, fn: str, inputs: list[object], output: object, params: dict[str, object]) -> Case:
    """A case whose parameters come as a dictionary."""
    return Case(name=name, fn=fn, inputs=encode_all(inputs), params=encode_params(params), output=encode(output))


def cases() -> list[Case]:
    g = rng(4332)
    x = np.column_stack([g.choice([1, 2, 3, 5], size=30, p=[0.5, 0.3, 0.15, 0.05]), g.choice([0, 1], size=30)])
    x = x.astype(np.float64)
    z = np.array([[1, 0], [5, 1], [3, 1], [2, 0]], dtype=np.float64)
    zu = np.array([[4, 0], [2, 1]], dtype=np.float64)
    xnan = x.copy()
    xnan[[3, 8], 0] = np.nan
    out: list[Case] = []
    configs: list[tuple[str, dict[str, Any], dict[str, object]]] = [
        ("default", {}, {}),
        ("first", {"drop": "first"}, {"drop": "first"}),
        ("if_binary", {"drop": "if_binary"}, {"drop": "if_binary"}),
        ("minFrequency", {"min_frequency": 3}, {"minFrequency": 3}),
        ("minFrequencyFraction", {"min_frequency": 0.2}, {"minFrequency": 0.2}),
        ("maxCategories", {"max_categories": 3}, {"maxCategories": 3}),
        ("maxCategories-first", {"max_categories": 2, "drop": "first"}, {"maxCategories": 2, "drop": "first"}),
    ]
    for label, kw, params in configs:
        enc = OneHotEncoder(sparse_output=False, **kw).fit(x)
        t = enc.transform(z)
        names = ",".join(
            str(s).replace("infrequent_sklearn", "infrequent").replace(".0", "") for s in enc.get_feature_names_out()
        )
        output = {"transform": t, "inverse": as_float(enc.inverse_transform(t)), "names": names}
        out.append(param_case(f"oneHot/{label}", "oneHot", [x, z], output, params))
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(x)
    t = enc.transform(zu)
    out.append(
        case(
            "oneHot/ignore",
            "oneHot",
            [x, zu],
            {"transform": t, "inverse": as_float(enc.inverse_transform(t))},
            handleUnknown="ignore",
        )
    )
    enc = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist", min_frequency=3).fit(x)
    t = enc.transform(zu)
    out.append(
        case(
            "oneHot/infrequent_if_exist",
            "oneHot",
            [x, zu],
            {"transform": t},
            handleUnknown="infrequent_if_exist",
            minFrequency=3,
        )
    )
    enc = OneHotEncoder(sparse_output=False).fit(xnan)
    znan = np.array([[np.nan, 1], [2, 0]])
    t = enc.transform(znan)
    out.append(
        case("oneHot/nan", "oneHot", [xnan, znan], {"transform": t, "inverse": as_float(enc.inverse_transform(t))})
    )
    ordinal: list[tuple[str, dict[str, Any], dict[str, object]]] = [
        ("default", {}, {}),
        (
            "unknown",
            {"handle_unknown": "use_encoded_value", "unknown_value": -1},
            {"handleUnknown": "use_encoded_value", "unknownValue": -1},
        ),
        ("minFrequency", {"min_frequency": 3}, {"minFrequency": 3}),
    ]
    for label, kw, params in ordinal:
        enc = OrdinalEncoder(**kw).fit(x)
        zz = zu if label == "unknown" else z
        t = enc.transform(zz)
        output = {"transform": t, "inverse": as_float(enc.inverse_transform(t))}
        out.append(param_case(f"ordinal/{label}", "ordinal", [x, zz], output, params))
    enc = OrdinalEncoder(encoded_missing_value=-2).fit(xnan)
    t = enc.transform(znan)
    out.append(case("ordinal/missing", "ordinal", [xnan, znan], {"transform": t}, encodedMissingValue=-2))
    labels = np.array([10.0, 3.0, 7.0, 3.0, 10.0])
    le = LabelEncoder().fit(labels)
    out.append(
        case(
            "label",
            "label",
            [labels, np.array([7.0, 10.0])],
            {"transform": le.transform([7.0, 10.0]).astype(np.float64), "classes": le.classes_},
        )
    )
    for ys in [labels, np.array([4.0, 9.0, 4.0])]:
        lb = LabelBinarizer().fit(ys)
        t = lb.transform(ys).astype(np.float64)
        out.append(
            case(
                f"labelBinarizer/{len(lb.classes_)}",
                "labelBinarizer",
                [ys, ys],
                {"transform": t, "inverse": lb.inverse_transform(t).astype(np.float64)},
            )
        )
    sets = [[1, 4], [2], [], [4, 2, 1]]
    mlb = MultiLabelBinarizer().fit(sets)
    out.append(
        case(
            "multiLabel",
            "multiLabel",
            [np.zeros(1), np.zeros(1)],
            {"transform": mlb.transform([[4], [1, 2]]).astype(np.float64), "classes": mlb.classes_},
            sets=sets,
            query=[[4], [1, 2]],
        )
    )
    xc = g.choice([0.0, 1.0, 2.0, 3.0], size=(24, 2), p=[0.4, 0.3, 0.2, 0.1])
    zc = np.array([[0.0, 3.0], [2.0, 1.0], [9.0, 0.0]])
    yc = 2 * xc[:, 0] - xc[:, 1] + g.normal(size=24)
    yb = (yc > np.median(yc)).astype(np.float64)
    ym = np.digitize(yc, np.quantile(yc, [1 / 3, 2 / 3])).astype(np.float64)
    for label, ys, kind, cv in [
        ("continuous", yc, "continuous", KFold(3)),
        ("binary", yb, "binary", StratifiedKFold(3)),
        ("multiclass", ym, "multiclass", StratifiedKFold(3)),
    ]:
        for smooth in ["auto", 2.0]:
            # sklearn's signature types smooth as str and cv as int; both accept more.
            options: dict[str, Any] = {"smooth": smooth, "target_type": kind, "cv": cv}
            enc = TargetEncoder(**options)
            ft = enc.fit_transform(xc, ys)
            out.append(
                case(
                    f"target/{label}/{smooth}",
                    "target",
                    [xc, ys, zc],
                    {"fitTransform": ft, "transform": enc.transform(zc)},
                    smooth=smooth,
                    targetType=kind,
                    stratified=kind != "continuous",
                )
            )
    return out
