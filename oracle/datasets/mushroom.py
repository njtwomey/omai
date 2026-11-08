"""The UCI mushroom data (Schlimmer 1981), CC BY 4.0, as a stratified subsample of 2000 of its 8124 mushrooms.

The full file is 370 KB of text before it is vendored as arrays, over the size a vendored dataset may take, so the
sample keeps the share of edible and poisonous (about 52 and 48 %) and the original order of the rows; which rows is
fixed by the SHA-256 of `mushroom:<row index>` (see `stratified_subsample`). The attributes keep their one-letter
codes; the 2480 missing `stalk-root` values of the full file (marked `?`) stay missing (`null`) in the sample.
"""

from omai_oracle.uci import source_note, stratified_subsample, table_rows, zip_member
from omai_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/mushroom.data.ts"
ID, SLUG = 73, "mushroom"
SHA256 = "face32f32647e0d939f6233f36dd30dd5d619ae9f3f9b8e10bea4ac7e1f60b1a"
FILE = "agaricus-lepiota.data"
SAMPLE = 2000
NAMES = [
    "capShape", "capSurface", "capColor", "bruises", "odor", "gillAttachment", "gillSpacing", "gillSize", "gillColor",
    "stalkShape", "stalkRoot", "stalkSurfaceAboveRing", "stalkSurfaceBelowRing", "stalkColorAboveRing",
    "stalkColorBelowRing", "veilType", "veilColor", "ringNumber", "ringType", "sporePrintColor", "population",
    "habitat",
]  # fmt: skip


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    keep = stratified_subsample([r[0] for r in rows], SAMPLE, "mushroom")
    sample = [rows[i] for i in keep]
    columns = [(n, "c", [r[i + 1] for r in sample]) for i, n in enumerate(NAMES)]
    columns.append(("class", "c", [r[0] for r in sample]))
    note = source_note(ID, SLUG, SHA256, [FILE]) + f", a stratified subsample of {SAMPLE} of {len(rows)} rows"
    return render_ordered_table("MUSHROOM", "mushroom", note, columns)
