"""The UCI yeast data (Nakai), CC BY 4.0: 1484 proteins in ten localisation sites, eight scores."""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/yeast.data.ts"
ID, SLUG = 110, "yeast"
SHA256 = "8fe9a4490edfa67532f1dee7ec11f1804437ad43f9f8b7536e32d5be8bce0bb5"
FILE = "yeast.data"
SITES = ["CYT", "ERL", "EXC", "ME1", "ME2", "ME3", "MIT", "NUC", "POX", "VAC"]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), sep=None)
    # The first column is the protein's sequence name (an identifier) and is dropped.
    x = [[float(v) for v in r[1:9]] for r in rows]
    y = [SITES.index(r[9]) for r in rows]
    return render_classification(
        "YEAST", "yeast", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"], SITES,
    )  # fmt: skip
