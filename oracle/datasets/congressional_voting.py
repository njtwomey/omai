"""The UCI congressional voting records (Schlimmer 1987), CC BY 4.0: the 1984 House votes of 435 members on 16 issues.

The `?` entries (not yea or nay: absent, paired or abstaining, not necessarily unknown) are kept as missing
(`null`).
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/congressional-voting.data.ts"
ID, SLUG = 105, "congressional+voting+records"
SHA256 = "ffec9d9328f28f151d95d2f8d36fd94ed8d2b0cdd0c9bd284bd8e65031a5e6a6"
FILE = "house-votes-84.data"
NAMES = [
    "handicappedInfants", "waterProjectCostSharing", "adoptionOfTheBudgetResolution", "physicianFeeFreeze",
    "elSalvadorAid", "religiousGroupsInSchools", "antiSatelliteTestBan", "aidToNicaraguanContras", "mxMissile",
    "immigration", "synfuelsCorporationCutback", "educationSpending", "superfundRightToSue", "crime", "dutyFreeExports",
    "exportAdministrationActSouthAfrica",
]  # fmt: skip


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [(n, "c", [r[i + 1] for r in rows]) for i, n in enumerate(NAMES)]
    columns.append(("party", "c", [r[0] for r in rows]))
    return render_ordered_table(
        "CONGRESSIONAL_VOTING", "congressional_voting", source_note(ID, SLUG, SHA256, [FILE]), columns
    )
