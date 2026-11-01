"""The UCI wholesale customers data (Cardoso 2014), CC BY 4.0: yearly spending of 440 clients of a Portuguese
wholesaler.
"""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/wholesale.data.ts"
ID, SLUG = 292, "wholesale+customers"
SHA256 = "647e6a61683ed23f48c7b04e1e2be78835eca8758b32267603fc42a1479dbfb8"
FILE = "Wholesale customers data.csv"


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), skip=1)
    # Channel (1 hotel, restaurant or cafe; 2 retail) is the class; region and the six spendings are features.
    x = [[float(v) for v in r[1:8]] for r in rows]
    y = [int(r[0]) - 1 for r in rows]
    return render_classification(
        "WHOLESALE", "wholesale", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["region", "fresh", "milk", "grocery", "frozen", "detergentsPaper", "delicatessen"], ["horeca", "retail"],
    )  # fmt: skip
