"""The UCI Haberman survival data (Haberman 1976), CC BY 4.0: 306 breast-cancer patients, five-year survival."""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/haberman.data.ts"
ID, SLUG = 43, "haberman+s+survival"
SHA256 = "1001bac29022350e279b064003c426c8ed60f4d76ab4916fb05baf886c8e3ac7"
FILE = "haberman.data"


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    x = [[float(v) for v in r[:3]] for r in rows]
    y = [int(r[3]) - 1 for r in rows]  # 1: survived five years or more, 2: died within five years
    return render_classification(
        "HABERMAN", "haberman", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["age", "yearOfOperation", "positiveNodes"], ["survived", "died"],
    )  # fmt: skip
