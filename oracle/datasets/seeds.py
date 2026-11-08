"""The UCI seeds data (Charytanowicz et al. 2010), CC BY 4.0: 210 wheat kernels of three varieties, seven geometry
measures.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/seeds.data.ts"
ID, SLUG = 236, "seeds"
SHA256 = "5033bd71b24592fa70d9b3cc5be02332dc6d03d0b27e6655e6d553f39afab323"
FILE = "seeds_dataset.txt"


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), sep=None)
    x = [[float(v) for v in r[:7]] for r in rows]
    y = [int(r[7]) - 1 for r in rows]
    return render_classification(
        "SEEDS", "seeds", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["area", "perimeter", "compactness", "kernelLength", "kernelWidth", "asymmetry", "grooveLength"],
        ["Kama", "Rosa", "Canadian"],
    )  # fmt: skip
