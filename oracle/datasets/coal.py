"""Dates of British coal-mining disasters 1851 to 1962 (Jarrett 1979), boot::coal via Rdatasets."""

from omai_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/coal.data.ts"
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/boot/coal.csv"
SHA256 = "8c8c9f98dca57efc85c8a2e6d73a19f414fcc5e6ece32b1135e95dd176e422f8"


def render() -> str:
    rows = csv_rows(fetch_pinned(URL, SHA256))
    return render_table(
        "COAL",
        "coal",
        f"boot::coal via Rdatasets ({URL}, sha256 {SHA256[:12]})",
        rows,
        ["date"],
        [],
    )
