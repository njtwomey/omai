"""Old Faithful geyser eruptions (Hardle 1991), R's datasets::faithful via Rdatasets."""

from whitebox_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/faithful.data.ts"
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/faithful.csv"
SHA256 = "5043db1e2c51c8e8fd67e0868c768ae589770cc76ad0ac0c5b7afd1fca31fc57"


def render() -> str:
    rows = csv_rows(fetch_pinned(URL, SHA256))
    return render_table(
        "FAITHFUL",
        "faithful",
        f"R datasets::faithful via Rdatasets ({URL}, sha256 {SHA256[:12]})",
        rows,
        ["eruptions", "waiting"],
        [],
    )
