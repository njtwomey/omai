"""The UCI sonar data (Gorman and Sejnowski 1988), CC BY 4.0: 208 sonar returns in 60 energy bands, rock or mine."""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/sonar.data.ts"
ID, SLUG = 151, "connectionist+bench+sonar+mines+vs+rocks"
SHA256 = "088b0b6813fb5ab84736bc2a1e3d6bb886e317520d2f2c04152e2ed65af7a18d"
FILE = "sonar.all-data"


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    x = [[float(v) for v in r[:60]] for r in rows]
    y = [1 if r[60] == "M" else 0 for r in rows]
    return render_classification(
        "SONAR", "sonar", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        [f"band{i + 1}" for i in range(60)], ["rock", "mine"],
    )  # fmt: skip
