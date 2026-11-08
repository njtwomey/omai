"""The UCI adult (census income) data (Kohavi 1996), CC BY 4.0, as a stratified subsample of 2500 of the 32561 rows of
`adult.data`.

The full file is 4 MB, so the sample keeps the share of each income class (about 76 % at most 50K and 24 % above)
and the original order of the rows; which rows is fixed by the SHA-256 of `adult:<row index>` (see
`stratified_subsample`). The `?` entries of `workclass`, `occupation` and `nativeCountry` stay missing (`null`).
`adult.test` is not used.
"""

from omai_oracle.uci import source_note, stratified_subsample, table_rows, zip_member
from omai_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/adult.data.ts"
ID, SLUG = 2, "adult"
SHA256 = "7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb"
FILE = "adult.data"
SAMPLE = 2500
NAMES = [
    ("age", "n"), ("workclass", "c"), ("fnlwgt", "n"), ("education", "c"), ("educationNum", "n"),
    ("maritalStatus", "c"), ("occupation", "c"), ("relationship", "c"), ("race", "c"), ("sex", "c"),
    ("capitalGain", "n"), ("capitalLoss", "n"), ("hoursPerWeek", "n"), ("nativeCountry", "c"), ("income", "c"),
]  # fmt: skip


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    keep = stratified_subsample([r[14] for r in rows], SAMPLE, "adult")
    sample = [rows[i] for i in keep]
    columns = [(n, k, [r[i] for r in sample]) for i, (n, k) in enumerate(NAMES)]
    note = source_note(ID, SLUG, SHA256, [FILE]) + f", a stratified subsample of {SAMPLE} of {len(rows)} rows"
    return render_ordered_table("ADULT", "adult", note, columns)
