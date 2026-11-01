"""The UCI car evaluation data (Bohanec and Rajkovic 1988), CC BY 4.0: all 1728 combinations of six attributes, four
acceptability classes.
"""

from mdml_oracle.uci import source_note, table_rows, zip_member
from mdml_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/car-evaluation.data.ts"
ID, SLUG = 19, "car+evaluation"
SHA256 = "1559d51dcf327f4f8c71b711ceed7fd95a382fc8d8e1998667f4f23b82860403"
FILE = "car.data"
NAMES = ["buying", "maintenance", "doors", "persons", "luggageBoot", "safety", "class"]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [(n, "c", [r[i] for r in rows]) for i, n in enumerate(NAMES)]
    return render_ordered_table("CAR_EVALUATION", "car_evaluation", source_note(ID, SLUG, SHA256, [FILE]), columns)
