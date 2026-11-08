"""Monthly international airline passengers 1949 to 1960 (Box and Jenkins), R's datasets::AirPassengers."""

from omai_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/air-passengers.data.ts"
URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv"
SHA256 = "bdb98adbd418a6de6842a742e0602f363c3b841c26677e7e61a1e055e9509bd8"


def render() -> str:
    rows = csv_rows(fetch_pinned(URL, SHA256))
    return render_table(
        "AIR_PASSENGERS",
        "air_passengers",
        f"R datasets::AirPassengers via Rdatasets ({URL}, sha256 {SHA256[:12]})",
        rows,
        ["time", "value"],
        [],
    )
