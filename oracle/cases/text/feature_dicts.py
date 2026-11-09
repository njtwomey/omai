"""DictVectorizer and FeatureHasher against sklearn.feature_extraction: the vocabulary, column order and matrix of
records with numbers, booleans, strings and lists of strings (sorted or not, with other separators), the transform of
new records with unseen features, and the hashed matrix of records, pairs and bare strings with and without sign
alternation."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from sklearn.feature_extraction import DictVectorizer, FeatureHasher

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__}

RECORDS: list[dict[str, object]] = [
    {"city": "Paris", "temp": 12.5, "tags": ["rain", "wind"], "indoor": True},
    {"city": "Rome", "temp": 21.0, "tags": ["sun"], "indoor": False},
    {"city": "Paris", "temp": -3.0, "pop": 2200000},
    {"temp": 0.0, "tags": []},
]
NEW: list[dict[str, object]] = [
    {"city": "Oslo", "temp": 5.0, "tags": ["rain", "snow"]},
    {"city": "Rome", "unseen": 9.0, "pop": 7},
]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, sort, separator in [("sorted", True, "="), ("first-seen", False, "="), ("separator", True, "#")]:
        vec = DictVectorizer(sparse=False, sort=sort, separator=separator)
        matrix = vec.fit_transform(RECORDS)
        out.append(
            case(
                f"dict/{name}",
                "dict",
                [],
                {
                    "matrix": matrix,
                    "names": "|".join(vec.get_feature_names_out()),
                    "transformed": vec.transform(NEW),
                },
                records=RECORDS,
                new=NEW,
                sort=sort,
                separator=separator,
            )
        )
    samples_dict: list[dict[str, object]] = [
        {"dog": 1, "cat": 2, "elephant": 4},
        {"dog": 2, "run": 5},
        {"colour": "red", "size": 3.5, "wide": True},
        {"zero": 0, "kept": 2},
    ]
    samples_pair: list[list[tuple[str, object]]] = [[("dog", 1), ("cat", 2)], [("shape", "round"), ("size", 0.5)]]
    samples_string: list[list[str]] = [["dog", "cat", "dog"], ["run", "walk"]]
    for width in [8, 64]:
        for alternate in [True, False]:
            for kind, samples in [("dict", samples_dict), ("pair", samples_pair), ("string", samples_string)]:
                hasher = FeatureHasher(n_features=width, input_type=kind, alternate_sign=alternate)
                matrix = hasher.transform(samples).toarray()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
                out.append(
                    case(
                        f"hash/{kind}/{width}/{'signed' if alternate else 'plain'}",
                        "hash",
                        [],
                        {"matrix": np.asarray(matrix)},
                        samples=samples,
                        width=width,
                        alternate=alternate,
                        kind=kind,
                    )
                )
    return out
