"""The UCI hepatitis data (Cestnik et al. 1987), CC BY 4.0: 155 patients, 19 attributes, survived or died.

The file marks 167 values missing with `?`; they stay missing (`null`). The yes/no attributes keep the published
codes (1 no, 2 yes; sex 1 male, 2 female); the class (1 die, 2 live) becomes text.
"""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/hepatitis.data.ts"
ID, SLUG = 46, "hepatitis"
SHA256 = "8573c2898714fb32ef8162291dcabd483a1ba9701123cab262f5acace3112d87"
FILE = "hepatitis.data"
NAMES = [
    "age", "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "liverBig", "liverFirm",
    "spleenPalpable", "spiders", "ascites", "varices", "bilirubin", "alkPhosphate", "sgot", "albumin", "protime",
    "histology",
]  # fmt: skip


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [(n, "n", [r[i + 1] for r in rows]) for i, n in enumerate(NAMES)]
    columns.append(("class", "c", ["DIE" if r[0] == "1" else "LIVE" for r in rows]))
    return render_ordered_table("HEPATITIS", "hepatitis", source_note(ID, SLUG, SHA256, [FILE]), columns)
