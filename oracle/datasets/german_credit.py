"""The Statlog (German credit) data from the UCI repository (CC BY 4.0): 1000 loan applicants, good or bad credit risk.

Twelve of the twenty attributes are kept, coded as numbers (ordinal codes in the order the documentation lists them),
with `female` derived from attribute 9 (personal status and sex), the protected attribute of fairness audits.
"""

from mdml_oracle.vendor import fetch_pinned, render_classification

TARGET = "packages/engine/src/lib/datasets/_data/german-credit.data.ts"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
SHA256 = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"

FEATURES = [
    "checking",
    "duration",
    "amount",
    "savings",
    "employment",
    "installmentRate",
    "age",
    "existingCredits",
    "dependents",
    "ownsHome",
    "foreignWorker",
    "female",
]


def code(value: str, prefix: str) -> float:
    """The position of a qualitative code in its attribute's list (`A12` under `A1` is 1)."""
    return float(int(value[len(prefix) :]) - 1)


def row(fields: list[str]) -> list[float]:
    """The kept attributes of one applicant, as numbers."""
    return [
        code(fields[0], "A1"),
        float(fields[1]),
        float(fields[4]),
        code(fields[5], "A6"),
        code(fields[6], "A7"),
        float(fields[7]),
        float(fields[12]),
        float(fields[15]),
        float(fields[17]),
        1.0 if fields[14] == "A152" else 0.0,
        1.0 if fields[19] == "A201" else 0.0,
        1.0 if fields[8] in ("A92", "A95") else 0.0,
    ]


def render() -> str:
    lines = [ln.split() for ln in fetch_pinned(URL, SHA256).decode("ascii").splitlines() if ln.strip()]
    x = [row(f) for f in lines]
    # Class 1 is good credit, 2 bad: good is the positive outcome (index 1).
    y = [1 if f[20] == "1" else 0 for f in lines]
    return render_classification(
        "GERMAN_CREDIT",
        "german_credit",
        f"the UCI Statlog (German credit) data ({URL}, sha256 {SHA256[:12]})",
        x,
        y,
        FEATURES,
        ["bad", "good"],
    )
