"""The UCI balance scale data (Siegler 1976), CC BY 4.0: all 625 weight and distance settings, tips left, balances or
right.
"""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/balance-scale.data.ts"
ID, SLUG = 12, "balance+scale"
SHA256 = "4f8ec31ea68c2f9e63f397b4c24cc1e345022b6886ab8f1bd69dd359b230125e"
FILE = "balance-scale.data"
SIDES = ["L", "B", "R"]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    x = [[float(v) for v in r[1:5]] for r in rows]
    y = [SIDES.index(r[0]) for r in rows]
    return render_classification(
        "BALANCE_SCALE", "balance_scale", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["leftWeight", "leftDistance", "rightWeight", "rightDistance"], ["left", "balanced", "right"],
    )  # fmt: skip
