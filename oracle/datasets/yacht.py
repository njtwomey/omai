"""The UCI yacht hydrodynamics data (Gerritsma et al. 1981), CC BY 4.0: 308 hull designs and their residuary
resistance.
"""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_regression

TARGET = "packages/engine/src/lib/datasets/_data/yacht.data.ts"
ID, SLUG = 243, "yacht+hydrodynamics"
SHA256 = "aa52b68f88c4bb552187a53ef4c5753fa178f6a36035a3771c5bc04e078487ac"
FILE = "yacht_hydrodynamics.data"


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), sep=None)
    x = [[float(v) for v in r[:6]] for r in rows]
    y = [float(r[6]) for r in rows]
    return render_regression(
        "YACHT", "yacht", source_note(ID, SLUG, SHA256, [FILE]), x, y,
        ["longitudinalPosition", "prismaticCoefficient", "lengthDisplacementRatio", "beamDraughtRatio",
         "lengthBeamRatio", "froudeNumber"],
        ["residuaryResistance"],
    )  # fmt: skip
