"""The UCI student performance data (Cortez and Silva 2008), CC BY 4.0: the mathematics file, 395 Portuguese secondary
school students, 33 attributes.

The repository states the data are anonymised. Only the mathematics file (`student-mat.csv`) is used, with the
attributes as published, `G3` (the final grade) last.
"""

from mdml_oracle.uci import nested_zip_member, source_note
from mdml_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/student-math.data.ts"
ID, SLUG = 320, "student+performance"
SHA256 = "82ae9d66437b9808df42e8c89d2bb179c46e9cfbcf06f38abc1d20b3b747e177"
INNER, FILE = "student.zip", "student-mat.csv"
NUMERIC = {
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "G1", "G2", "G3",
}  # fmt: skip


def render() -> str:
    lines = nested_zip_member(ID, SLUG, SHA256, INNER, FILE).decode("latin-1").splitlines()
    header = [h.strip('"') for h in lines[0].split(";")]
    rows = [[v.strip('"') for v in ln.split(";")] for ln in lines[1:] if ln.strip()]
    columns = [(n, "n" if n in NUMERIC else "c", [r[i] for r in rows]) for i, n in enumerate(header)]
    return render_ordered_table(
        "STUDENT_MATH", "student_math", source_note(ID, SLUG, SHA256, [f"{INNER}/{FILE}"]), columns
    )
