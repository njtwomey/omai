"""Annual flow of the river Nile at Aswan, 1871 to 1970, R's datasets::Nile (Cobb 1978; Durbin and Koopman 2001)."""

from omai_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/nile.data.ts"
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Nile.csv"
SHA256 = "d0452bea38c61e796a4eeb950bf91d20fb5c7f13d5822eadf5990fe54f9c8d07"


def render() -> str:
    rows = csv_rows(fetch_pinned(URL, SHA256))
    return render_table(
        "NILE",
        "nile",
        f"R datasets::Nile via Rdatasets ({URL}, sha256 {SHA256[:12]})",
        rows,
        ["time", "value"],
        [],
    )
