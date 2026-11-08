"""The UCI dermatology data (Guvenir et al. 1998), CC BY 4.0: 366 patients, 34 clinical and histopathological
attributes, six diseases.

The eight missing ages (marked `?`) stay missing (`null`). The class (1 to 6) becomes the disease's name.
"""

from omai_oracle.uci import source_note, table_rows, zip_member
from omai_oracle.vendor import render_ordered_table

TARGET = "packages/engine/src/lib/datasets/_data/dermatology.data.ts"
ID, SLUG = 33, "dermatology"
SHA256 = "c86e681b3bd02d838b3602d88700b007952bb57eb80162c7ed8966d426f5c6c8"
FILE = "dermatology.data"
CLINICAL = [
    "erythema", "scaling", "definiteBorders", "itching", "koebnerPhenomenon", "polygonalPapules", "follicularPapules",
    "oralMucosalInvolvement", "kneeAndElbowInvolvement", "scalpInvolvement", "familyHistory",
]  # fmt: skip
HISTO = [
    "melaninIncontinence", "eosinophilsInInfiltrate", "pnlInfiltrate", "fibrosisPapillaryDermis", "exocytosis",
    "acanthosis", "hyperkeratosis", "parakeratosis", "clubbingReteRidges", "elongationReteRidges",
    "thinningSuprapapillaryEpidermis", "spongiformPustule", "munroMicroabcess", "focalHypergranulosis",
    "disappearanceGranularLayer", "vacuolisationBasalLayer", "spongiosis", "sawToothRetes", "follicularHornPlug",
    "perifollicularParakeratosis", "inflammatoryMononuclearInfiltrate", "bandLikeInfiltrate",
]  # fmt: skip
NAMES = CLINICAL + HISTO + ["age"]
DISEASES = [
    "psoriasis", "seboreic dermatitis", "lichen planus", "pityriasis rosea", "chronic dermatitis",
    "pityriasis rubra pilaris",
]  # fmt: skip


def render() -> str:
    rows = table_rows(zip_member(ID, SLUG, SHA256, FILE))
    columns = [(n, "n", [r[i] for r in rows]) for i, n in enumerate(NAMES)]
    columns.append(("disease", "c", [DISEASES[int(r[34]) - 1] for r in rows]))
    return render_ordered_table("DERMATOLOGY", "dermatology", source_note(ID, SLUG, SHA256, [FILE]), columns)
