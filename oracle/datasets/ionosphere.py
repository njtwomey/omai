"""The UCI ionosphere data (Sigillito et al. 1989), CC BY 4.0: 351 radar returns, 34 numbers each, good or bad."""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/ionosphere.data.ts"
ID, SLUG = 52, "ionosphere"
SHA256 = "4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860"
FILE = "ionosphere.data"


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    x = [[float(v) for v in r[:34]] for r in rows]
    y = [1 if r[34] == "g" else 0 for r in rows]
    names = [f"pulse{p + 1}_{part}" for p in range(17) for part in ("real", "imaginary")]
    return render_classification(
        "IONOSPHERE", "ionosphere", source_note(ID, SLUG, SHA256, [FILE]), x, y, names, ["bad", "good"]
    )
