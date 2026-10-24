"""The Porter stemmer (nltk's three modes) and the Snowball English stemmer against nltk, on words chosen to reach
every rule of both algorithms."""

from importlib.metadata import version

from nltk.stem import PorterStemmer, SnowballStemmer
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"nltk": version("nltk")}

TEXT: str = """
caresses ponies ties caress cats feed agreed plastered bled motoring sing conflated troubled sized hopping tanned
falling hissing fizzed failing filing happy sky relational conditional rational valenci hesitanci digitizer
conformabli radicalli differentli vileli analogousli vietnamization predication operator feudalism decisiveness
hopefulness callousness formaliti sensitiviti sensibiliti triplicate formative formalize electriciti electrical
hopeful goodness revival allowance inference airliner gyroscopic adjustable defensible irritant replacement
adjustment dependent adoption homologou communism activate angulariti homologous effective bowdlerize probate rate
cease controll roll generalizations generously running dog's dogs' communication communicate arsenal skies skis
dying lying tying idly gently ugly early only singly news howe atlas cosmos bias andes innings outings cannings
herrings earrings proceeds proceeded proceeding exceeds exceeded exceeding succeeds succeeded succeeding inning
consign consigned consigning consignment consist consisted consistency consistent consistently consisting consists
consolation consolations consolatory console consoled consoles consolidate consolidated consolidating consoling
knack knackeries knacks knag knave knaves knavish kneaded kneading knee kneel kneeled kneeling kneels knees knell
knelt knew knick knif knife knight knightly knights knit knits knitted knitting knives knob knobs knock knocked
knocker knockers knocking knocks knopp knot knots y yy yes yay boy toy employs saying played crying flies dries
tried studied cried denied ied ies ays oyster psychology archaeology theology logic ologies fulli lessli
carelessly hopelessly useless gratefully beautifully eventually specially probably beautiful nationalism
'tis o'clock ’quoted’ ‘rock aed eed bleed speed exceedingly agreedly ringingly markedly abatement
"""
WORDS: list[str] = [str(w) for w in TEXT.split()]


def cases() -> list[Case]:
    out: list[Case] = []
    for mode, name in [
        (PorterStemmer.NLTK_EXTENSIONS, "nltk"),
        (PorterStemmer.MARTIN_EXTENSIONS, "martin"),
        (PorterStemmer.ORIGINAL_ALGORITHM, "original"),
    ]:
        stemmer = PorterStemmer(mode)
        out.append(case(f"porter/{name}", "porterStem", [WORDS], [stemmer.stem(w) for w in WORDS], mode=name))
    snowball = SnowballStemmer("english")
    out.append(case("snowball/english", "snowballStem", [WORDS], [snowball.stem(w) for w in WORDS]))
    return out
