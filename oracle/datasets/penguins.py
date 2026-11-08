"""The Palmer penguins data (Gorman, Williams and Fraser 2014; Horst, Hill and Gorman 2020), CC0."""

from omai_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/penguins.data.ts"
URL = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"


def render() -> str:
    rows = csv_rows(fetch_pinned(URL, SHA256))
    return render_table(
        "PENGUINS",
        "penguins",
        f"palmerpenguins ({URL}, sha256 {SHA256[:12]})",
        rows,
        ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "year"],
        ["species", "island", "sex"],
    )
