"""The UCI lenses data (Cendrowska 1987), CC BY 4.0: all 24 combinations of four patient attributes, three fitting
decisions.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/lenses.data.ts"
ID, SLUG = 58, "lenses"
SHA256 = "782e06afbf115ee0802c17a87c6840619450a97d8c66d59f8eb9577989cd7d7b"
FILE = "lenses.data"


def render() -> str:
    # Each line is `id age prescription astigmatic tearRate class` (the id is a running number and is dropped).
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), sep=None)
    x = [[float(v) for v in r[1:5]] for r in rows]
    y = [int(r[5]) - 1 for r in rows]
    return render_classification(
        "LENSES", "lenses", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["age", "prescription", "astigmatic", "tearRate"], ["hard", "soft", "none"],
    )  # fmt: skip
