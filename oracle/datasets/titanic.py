"""Survival of the 2201 people aboard the Titanic, by class, sex and age (R's datasets::Titanic, via Rdatasets).

The source is a 4 by 2 by 2 by 2 contingency table; it is vendored as its 32 counts and expanded to rows in TypeScript.
"""

from omai_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/titanic.data.ts"
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Titanic.csv"
SHA256 = "ca983380c99d10ece64011fef5e07150ce49949d52b517e3fc55c7715938d179"


def render() -> str:
    rows = csv_rows(fetch_pinned(URL, SHA256))
    return render_table(
        "TITANIC",
        "titanic",
        f"R datasets::Titanic via Rdatasets ({URL}, sha256 {SHA256[:12]})",
        rows,
        ["Freq"],
        ["Class", "Sex", "Age", "Survived"],
    )
