"""Vendored classification datasets against their sources: scikit-learn's loaders and the pinned CSV files."""

from typing import cast

import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.utils import Bunch
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.vendor import csv_rows, fetch_pinned

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "numpy": np.__version__}
PENGUINS_URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
PENGUINS_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"
GERMAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
GERMAN_SHA256 = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"


def cases() -> list[Case]:
    out: list[Case] = []
    for name, loader in [("iris", load_iris), ("wine", load_wine), ("breastCancer", load_breast_cancer)]:
        bunch = cast(Bunch, loader())
        out.append(case(f"{name}/x", name, [], np.asarray(bunch.data), rtol=0, atol=0, part="x"))
        out.append(case(f"{name}/y", name, [], np.asarray(bunch.target), rtol=0, atol=0, part="y"))
    digits = cast(Bunch, load_digits())
    out.append(case("digits/pixel-sums", "digits", [], np.asarray(digits.data).sum(axis=0), rtol=0, atol=0, part="x"))
    out.append(case("digits/y", "digits", [], np.asarray(digits.target), rtol=0, atol=0, part="y"))
    rows = csv_rows(fetch_pinned(PENGUINS_URL, PENGUINS_SHA256))
    cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    complete = [r for r in rows if all(r[c] not in ("", "NA") for c in cols)]
    x = np.asarray([[float(r[c]) for c in cols] for r in complete])
    species = sorted({r["species"] for r in rows})
    y = np.asarray([species.index(r["species"]) for r in complete])
    out.append(case("penguins/means", "penguinsClassification", [], x.mean(axis=0), rtol=1e-12, part="x"))
    out.append(case("penguins/y", "penguinsClassification", [], y, rtol=0, atol=0, part="y"))
    # German credit: duration, amount and age (attributes 2, 5 and 13), the women (attribute 9 is A92 or A95) and the
    # good risks (class 1), read straight from the source.
    german = [ln.split() for ln in fetch_pinned(GERMAN_URL, GERMAN_SHA256).decode("ascii").splitlines() if ln.strip()]
    numeric = np.asarray([[float(f[1]), float(f[4]), float(f[12])] for f in german])
    female = np.asarray([1.0 if f[8] in ("A92", "A95") else 0.0 for f in german])
    good = np.asarray([1 if f[20] == "1" else 0 for f in german])
    out.append(case("germanCredit/numeric", "germanCredit", [], numeric, rtol=0, atol=0, part="numeric"))
    out.append(case("germanCredit/female", "germanCredit", [], female, rtol=0, atol=0, part="female"))
    out.append(case("germanCredit/y", "germanCredit", [], good, rtol=0, atol=0, part="y"))
    return out
