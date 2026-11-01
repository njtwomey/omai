"""The UCI bike sharing data (Fanaee-T and Gama 2013), CC BY 4.0: the daily file, 731 days of a bike-share system in
Washington, D.C.

The running index `instant` is dropped; the date stays as a text column, and `casual` and `registered` (which sum to
the target `cnt`) stay as published.
"""

from mdml_oracle.uci import source_note, zip_member
from mdml_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/bike-sharing.data.ts"
ID, SLUG = 275, "bike+sharing+dataset"
SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
FILE = "day.csv"


def render() -> str:
    raw = zip_member(ID, SLUG, SHA256, FILE).decode("latin-1").splitlines()
    header = raw[0].strip().split(",")
    rows = [ln.strip().split(",") for ln in raw[1:] if ln.strip()]
    columns = [(n, "c" if n == "dteday" else "n", [r[i] for r in rows]) for i, n in enumerate(header) if n != "instant"]
    return render_ordered_table("BIKE_SHARING", "bike_sharing", source_note(ID, SLUG, SHA256, [FILE]), columns)
