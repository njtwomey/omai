"""The UCI tic-tac-toe endgame data (Aha 1991), CC BY 4.0: all 958 final boards where x moved first, won by x or not."""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/tic-tac-toe.data.ts"
ID, SLUG = 101, "tic+tac+toe+endgame"
SHA256 = "c615fcd01bc23ab99daa36bbc30d897d53f7d3e151257880cada888c5451a586"
FILE = "tic-tac-toe.data"
NAMES = [
    "topLeft", "topMiddle", "topRight", "middleLeft", "middleMiddle", "middleRight", "bottomLeft", "bottomMiddle",
    "bottomRight",
]  # fmt: skip


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [(n, "c", [r[i] for r in rows]) for i, n in enumerate(NAMES)]
    columns.append(("class", "c", [r[9] for r in rows]))
    return render_ordered_table("TIC_TAC_TOE", "tic_tac_toe", source_note(ID, SLUG, SHA256, [FILE]), columns)
