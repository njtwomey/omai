"""Motor Trend road tests of 32 cars (Henderson and Velleman 1981), R's datasets::mtcars via Rdatasets."""

from whitebox_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/mtcars.data.ts"
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
SHA256 = "4d6dfa9b029dbcc15333a6e3598803fae8ae7c4d08605040b1706e6367695058"


def render() -> str:
    rows = csv_rows(fetch_pinned(URL, SHA256))
    for r in rows:
        r["model"] = r["rownames"]
    return render_table(
        "MTCARS",
        "mtcars",
        f"R datasets::mtcars via Rdatasets ({URL}, sha256 {SHA256[:12]})",
        rows,
        ["mpg", "cyl", "disp", "hp", "drat", "wt", "qsec", "vs", "am", "gear", "carb"],
        ["model"],
    )
