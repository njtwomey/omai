"""CDC 2000 growth charts: LMS parameters of weight-for-age and stature-for-age, 2 to 20 years (public domain)."""

from omai_oracle.vendor import csv_rows, fetch_pinned, render_table

TARGET = "packages/engine/src/lib/datasets/_data/growth-charts.data.ts"
WEIGHT_URL = "https://www.cdc.gov/growthcharts/data/zscore/wtage.csv"
WEIGHT_SHA256 = "3406c9d125bcb69c062a9e84eb8c0209bfe9346542bdc1d308643750dcc241b7"
STATURE_URL = "https://www.cdc.gov/growthcharts/data/zscore/statage.csv"
STATURE_SHA256 = "45130d2a9d7c50c54a47e7ba626b66c61d4554bc2d901198cedd9419a53f7251"


def render() -> str:
    rows: list[dict[str, str]] = []
    for measure, url, sha in [("weight", WEIGHT_URL, WEIGHT_SHA256), ("stature", STATURE_URL, STATURE_SHA256)]:
        for r in csv_rows(fetch_pinned(url, sha)):
            # The files repeat their header between the sexes.
            if r["Sex"] in ("1", "2"):
                rows.append({"measure": measure, "sex": "male" if r["Sex"] == "1" else "female", **r})
    return render_table(
        "GROWTH",
        "growth_charts",
        f"CDC growth charts ({WEIGHT_URL} sha256 {WEIGHT_SHA256[:12]}, {STATURE_URL} sha256 {STATURE_SHA256[:12]})",
        rows,
        ["Agemos", "L", "M", "S"],
        ["measure", "sex"],
    )
