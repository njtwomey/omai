"""The UCI banknote authentication data (Lohweg 2012), CC BY 4.0: wavelet features of 1372 banknote images."""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/banknote.data.ts"
ID, SLUG = 267, "banknote+authentication"
SHA256 = "1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227"
FILE = "data_banknote_authentication.txt"


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    x = [[float(v) for v in r[:4]] for r in rows]
    y = [int(r[4]) for r in rows]
    return render_classification(
        "BANKNOTE", "banknote", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["variance", "skewness", "curtosis", "entropy"], ["0", "1"],
    )  # fmt: skip
