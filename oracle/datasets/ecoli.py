"""The UCI E. coli data (Nakai), CC BY 4.0: 336 proteins in eight localisation sites, seven scores."""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/ecoli.data.ts"
ID, SLUG = 39, "ecoli"
SHA256 = "a6580298b76828674a5197152da7a99f4bd09bc589dbf1142900f1698882ec32"
FILE = "ecoli.data"
SITES = ["cp", "im", "imL", "imS", "imU", "om", "omL", "pp"]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), sep=None)
    # The first column is the protein's sequence name (an identifier) and is dropped.
    x = [[float(v) for v in r[1:8]] for r in rows]
    y = [SITES.index(r[8]) for r in rows]
    return render_classification(
        "ECOLI", "ecoli", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"], SITES,
    )  # fmt: skip
