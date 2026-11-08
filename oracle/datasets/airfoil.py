"""The UCI airfoil self-noise data (Brooks et al. 1989), CC BY 4.0: 1503 NASA wind-tunnel tests and their sound
pressure level.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_regression

TARGET = "packages/engine/src/lib/datasets/_data/airfoil.data.ts"
ID, SLUG = 291, "airfoil+self+noise"
SHA256 = "5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c"
FILE = "airfoil_self_noise.dat"


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), sep=None)
    x = [[float(v) for v in r[:5]] for r in rows]
    y = [float(r[5]) for r in rows]
    return render_regression(
        "AIRFOIL", "airfoil", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["frequency", "angleOfAttack", "chordLength", "freeStreamVelocity", "suctionSideDisplacementThickness"],
        ["soundPressureLevel"],
    )  # fmt: skip
