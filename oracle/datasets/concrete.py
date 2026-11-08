"""The UCI concrete compressive strength data (Yeh 1998), CC BY 4.0: 1030 concrete mixes and their strength in MPa.

The repository publishes a spreadsheet; this reads the CSV rendering of the same table that it serves with the
dataset.
"""

from omai_oracle.uci import BASE, table_rows
from omai_oracle.vendor import fetch_pinned, render_regression

TARGET = "packages/engine/src/lib/datasets/_data/concrete.data.ts"
URL = f"{BASE}/165/data.csv"
SHA256 = "8d4b15b6fc68cd932d745cbd663d5ceae66dd54422e99c1e4865f2936ab7e2af"
FEATURES = [
    "cement",
    "blastFurnaceSlag",
    "flyAsh",
    "water",
    "superplasticizer",
    "coarseAggregate",
    "fineAggregate",
    "age",
]


def render() -> str:
    rows = table_rows(fetch_pinned(URL, SHA256), skip=1)
    x = [[float(v) for v in r[:8]] for r in rows]
    y = [float(r[8]) for r in rows]
    return render_regression(
        "CONCRETE",
        "concrete",
        f"UCI dataset 165 ({URL}, sha256 {SHA256[:12]})",
        x,
        y,
        FEATURES,
        ["compressiveStrength"],
    )
