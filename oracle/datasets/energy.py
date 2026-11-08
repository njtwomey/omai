"""The UCI energy efficiency data (Tsanas and Xifara 2012), CC BY 4.0: 768 simulated buildings, heating and cooling
loads.

The repository publishes a spreadsheet; this reads the CSV rendering of the same table that it serves with the
dataset.
"""

from omai_oracle.uci import BASE, table_rows
from omai_oracle.vendor import fetch_pinned, render_regression

TARGET = "packages/engine/src/lib/datasets/_data/energy.data.ts"
URL = f"{BASE}/242/data.csv"
SHA256 = "db44dbe453acd464b5cf65be2fb01a28aa9c5b2630300e65fbe28cde35f5d96f"
FEATURES = [
    "relativeCompactness", "surfaceArea", "wallArea", "roofArea", "overallHeight", "orientation", "glazingArea",
    "glazingAreaDistribution",
]  # fmt: skip


def render() -> str:
    rows = table_rows(fetch_pinned(URL, SHA256), skip=1)
    x = [[float(v) for v in r[:8]] for r in rows]
    y = [[float(v) for v in r[8:10]] for r in rows]
    return render_regression(
        "ENERGY",
        "energy",
        f"UCI dataset 242 ({URL}, sha256 {SHA256[:12]})",
        x,
        y,
        FEATURES,
        ["heatingLoad", "coolingLoad"],
    )
