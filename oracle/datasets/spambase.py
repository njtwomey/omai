"""The UCI spambase data (Hopkins et al. 1999), CC BY 4.0, as a stratified subsample of 1000 of its 4601 e-mails.

The full file is 700 KB of text, over the size a vendored dataset may take, so the sample keeps the spam share
(about 39 %) and the original order of the rows; which rows is fixed by the SHA-256 of `spambase:<row index>` (see
`stratified_subsample`), so regeneration gives the same rows.
"""

from omai_oracle.uci import source_note, stratified_subsample, table_rows, zip_member
from omai_oracle.vendor import render_classification

TARGET = "packages/engine/src/lib/datasets/_data/spambase.data.ts"
ID, SLUG = 94, "spambase"
SHA256 = "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636"
FILE = "spambase.data"
SAMPLE = 1000
WORDS = [
    "make",
    "address",
    "all",
    "3d",
    "our",
    "over",
    "remove",
    "internet",
    "order",
    "mail",
    "receive",
    "will",
    "people",
    "report",
    "addresses",
    "free",
    "business",
    "email",
    "you",
    "credit",
    "your",
    "font",
    "000",
    "money",
    "hp",
    "hpl",
    "george",
    "650",
    "lab",
    "labs",
    "telnet",
    "857",
    "data",
    "415",
    "85",
    "technology",
    "1999",
    "parts",
    "pm",
    "direct",
    "cs",
    "meeting",
    "original",
    "project",
    "re",
    "edu",
    "table",
    "conference",
]
CHARS = ["semicolon", "parenthesis", "bracket", "exclamation", "dollar", "hash"]
FEATURES = (
    [f"word_freq_{w}" for w in WORDS]
    + [f"char_freq_{c}" for c in CHARS]
    + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
)


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    keep = stratified_subsample([r[57] for r in rows], SAMPLE, "spambase")
    x = [[float(v) for v in rows[i][:57]] for i in keep]
    y = [int(rows[i][57]) for i in keep]
    note = source_note(ID, SLUG, SHA256, [FILE]) + f", a stratified subsample of {SAMPLE} of {len(rows)} rows"
    return render_classification("SPAMBASE", "spambase", note, x, y, FEATURES, ["nonspam", "spam"])
