"""The UCI glass identification data (German 1987), CC BY 4.0: 214 glass samples of six types, nine oxide measures."""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/glass.data.ts"
ID, SLUG = 42, "glass+identification"
SHA256 = "499c07a17115bf63100bb19cc1fdefad7f464bf5dc2b168930fa2071fd6b6e65"
FILE = "glass.data"
# Types 1, 2, 3, 5, 6, 7 occur (there is no type 4); the class index is the position in this list.
TYPES = ["1", "2", "3", "5", "6", "7"]
CLASSES = [
    "building windows (float processed)",
    "building windows (non-float processed)",
    "vehicle windows (float processed)",
    "containers",
    "tableware",
    "headlamps",
]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    # The first column is a running id and is dropped.
    x = [[float(v) for v in r[1:10]] for r in rows]
    y = [TYPES.index(r[10]) for r in rows]
    return render_classification(
        "GLASS", "glass", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], CLASSES,
    )  # fmt: skip
