"""The UCI zoo data (Forsyth 1990), CC BY 4.0: 101 animals, 15 yes/no attributes and a leg count, seven classes.

The yes/no attributes keep the published 0 and 1; the class (1 to 7) becomes its name, as listed in the file's
documentation. The animal name is a text column.
"""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/zoo.data.ts"
ID, SLUG = 111, "zoo"
SHA256 = "d49013253628277182bb172f5ab1bf3befad090b54bbb7e52733c9c9534e240a"
FILE = "zoo.data"
NAMES = [
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes",
    "venomous", "fins", "legs", "tail", "domestic", "catsize",
]  # fmt: skip
TYPES = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "invertebrate"]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [("animal", "c", [r[0] for r in rows])]
    columns += [(n, "n", [r[i + 1] for r in rows]) for i, n in enumerate(NAMES)]
    columns.append(("type", "c", [TYPES[int(r[17]) - 1] for r in rows]))
    return render_ordered_table("ZOO", "zoo", source_note(ID, SLUG, SHA256, [FILE]), columns)
