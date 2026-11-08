"""The UCI wine quality data (Cortez et al. 2009), CC BY 4.0: the red vinho verde wines, 11 measurements and a taste
score.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_regression

TARGET = "packages/engine/src/lib/datasets/_data/wine-quality-red.data.ts"
ID, SLUG = 186, "wine+quality"
SHA256 = "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4"
FILE = "winequality-red.csv"
FEATURES = [
    "fixedAcidity", "volatileAcidity", "citricAcid", "residualSugar", "chlorides", "freeSulfurDioxide",
    "totalSulfurDioxide", "density", "pH", "sulphates", "alcohol",
]  # fmt: skip


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), sep=";", skip=1)
    x = [[float(v) for v in r[:11]] for r in rows]
    y = [float(r[11]) for r in rows]
    return render_regression(
        "WINE_QUALITY_RED", "wine_quality_red", source_note(ID, SLUG, SHA256, [FILE]), x, y, FEATURES, ["quality"]
    )
