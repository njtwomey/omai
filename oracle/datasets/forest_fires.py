"""The UCI forest fires data (Cortez and Morais 2007), CC BY 4.0: 517 fires in Montesinho park, Portugal, and the area
burnt.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/forest-fires.data.ts"
ID, SLUG = 162, "forest+fires"
SHA256 = "abd40d6142b5e30fea73ff7292f91dcdd8b64bf3b3f36e20f96c37f5d14f06fb"
FILE = "forestfires.csv"
KINDS = {"month": "c", "day": "c"}


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE), skip=1)
    header = zip_member(ID, SLUG, SHA256, FILE).decode("latin-1").splitlines()[0].split(",")
    columns = [(n, KINDS.get(n, "n"), [r[i] for r in rows]) for i, n in enumerate(header)]
    return render_ordered_table("FOREST_FIRES", "forest_fires", source_note(ID, SLUG, SHA256, [FILE]), columns)
