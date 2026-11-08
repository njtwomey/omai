"""The UCI Auto MPG data (Quinlan 1993), CC BY 4.0: 398 cars of model years 1970 to 1982, fuel economy and design.

Six horsepower values are missing in the file (marked `?`) and stay missing (`null`).
"""

from omai_oracle.uci import source_note, zip_member
from omai_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/auto-mpg.data.ts"
ID, SLUG = 9, "auto+mpg"
SHA256 = "5fc1507522b73b8050d3aa45df51eefe1b666ae5f428a5f1347b6483339f1003"
FILE = "auto-mpg.data"
NUMERIC = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "modelYear", "origin"]


def render() -> str:
    # Eight whitespace-separated numbers, a tab, then the quoted car name.
    rows = []
    for line in zip_member(ID, SLUG, SHA256, FILE).decode("latin-1").splitlines():
        if not line.strip():
            continue
        numbers, name = line.split("\t", 1)
        rows.append(([("" if v == "?" else v) for v in numbers.split()], name.strip().strip('"')))
    columns = [(n, "n", [r[0][i] for r in rows]) for i, n in enumerate(NUMERIC)]
    columns.append(("carName", "c", [r[1] for r in rows]))
    return render_ordered_table("AUTO_MPG", "auto_mpg", source_note(ID, SLUG, SHA256, [FILE]), columns)
