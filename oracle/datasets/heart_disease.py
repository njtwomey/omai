"""The UCI heart disease data, Cleveland Clinic file (Detrano et al. 1989), CC BY 4.0: 303 patients, 13 attributes.

Six values are missing (four of `ca`, two of `thal`, marked `?`) and stay missing (`null`). The categorical
attributes keep the numeric codes of the published file; the target `num` (0 for no disease, 1 to 4 for its degree)
is text.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/heart-disease.data.ts"
ID, SLUG = 45, "heart+disease"
SHA256 = "b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc"
FILE = "processed.cleveland.data"
NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [(n, "n", [r[i] for r in rows]) for i, n in enumerate(NAMES)]
    columns.append(("num", "c", [str(int(float(r[13]))) for r in rows]))
    return render_ordered_table("HEART_DISEASE", "heart_disease", source_note(ID, SLUG, SHA256, [FILE]), columns)
