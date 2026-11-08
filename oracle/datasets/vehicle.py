"""The UCI Statlog vehicle silhouettes data (Siebert 1987), CC BY 4.0: 846 silhouettes, 18 shape features, four vehicle
types.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/vehicle.data.ts"
ID, SLUG = 149, "statlog+vehicle+silhouettes"
SHA256 = "66ca807d5be5e3403cf035a579e85113880c3019d6cde825670deff05a4a71e0"
PARTS = [f"xa{c}.dat" for c in "abcdefghi"]
CLASSES = ["bus", "opel", "saab", "van"]
FEATURES = [
    "compactness", "circularity", "distanceCircularity", "radiusRatio", "praxisAspectRatio", "maxLengthAspectRatio",
    "scatterRatio", "elongatedness", "praxisRectangularity", "lengthRectangularity", "scaledVarianceMajor",
    "scaledVarianceMinor", "scaledRadiusOfGyration", "skewnessMajor", "skewnessMinor", "kurtosisMinor",
    "kurtosisMajor", "hollowsRatio",
]  # fmt: skip


def render() -> str:
    rows = [r for part in PARTS for r in table_rows(zip_member(ID, SLUG, SHA256, part), sep=None)]
    x = [[float(v) for v in r[:18]] for r in rows]
    y = [CLASSES.index(r[18]) for r in rows]
    return render_classification(
        "VEHICLE", "vehicle", source_note(ID, SLUG, SHA256, ["xaa.dat to xai.dat"]), x, y, FEATURES, CLASSES
    )
