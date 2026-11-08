"""The UCI servo data (Quinlan 1993), CC BY 4.0: 167 servomechanism set-ups and their rise time.

Motor and screw are categorical (letters A to E); the two gains and the rise time are numbers.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/servo.data.ts"
ID, SLUG = 87, "servo"
SHA256 = "02822bd9f00d8af7394163649e6398711da0a0b02170e380fd4d15b14e8a5659"
FILE = "servo.data"
NAMES = [("motor", "c"), ("screw", "c"), ("pgain", "n"), ("vgain", "n"), ("riseTime", "n")]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [(n, k, [r[i] for r in rows]) for i, (n, k) in enumerate(NAMES)]
    return render_ordered_table("SERVO", "servo", source_note(ID, SLUG, SHA256, [FILE]), columns)
