"""Anscombe's quartet (Anscombe 1973), R's datasets::anscombe via Rdatasets."""

from omai_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/anscombe.data.ts"
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/anscombe.csv"
SHA256 = "02a87e7348a05d2fc235e79e6bd89a9e279110d9398e98e32975d82323eb535c"


def render() -> str:
    rows = csv_rows(fetch_pinned(URL, SHA256))
    return render_table(
        "ANSCOMBE",
        "anscombe",
        f"R datasets::anscombe via Rdatasets ({URL}, sha256 {SHA256[:12]})",
        rows,
        ["x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"],
        [],
    )
