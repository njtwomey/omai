"""The UCI abalone data (Nash et al. 1994), CC BY 4.0: 4177 abalone, their sex, measurements and ring count."""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/abalone.data.ts"
ID, SLUG = 1, "abalone"
SHA256 = "755a6a67c5b266961a3f149ea13be2cfb6e6c727e48cee3c83bc0b4526210ee4"
FILE = "abalone.data"
NAMES = ["sex", "length", "diameter", "height", "wholeWeight", "shuckedWeight", "visceraWeight", "shellWeight", "rings"]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [(n, "c" if n == "sex" else "n", [r[i] for r in rows]) for i, n in enumerate(NAMES)]
    return render_ordered_table("ABALONE", "abalone", source_note(ID, SLUG, SHA256, [FILE]), columns)
