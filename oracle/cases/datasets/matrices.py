"""Matrix generators against sklearn.datasets through their invariants (singular values, block structure)."""

import numpy as np
import sklearn
from omai_oracle.fixture import Case, case
from sklearn.datasets import make_low_rank_matrix

REFERENCE: dict[str, str] = {"scikit-learn": sklearn.__version__, "numpy": np.__version__}


def cases() -> list[Case]:
    out: list[Case] = []
    for n, d, rank, tail in [(30, 12, 3, 0.5), (12, 20, 5, 0.1), (15, 15, 2, 0.9)]:
        x = make_low_rank_matrix(n_samples=n, n_features=d, effective_rank=rank, tail_strength=tail, random_state=0)
        s = np.linalg.svd(x, compute_uv=False)
        out.append(
            case(
                f"lowRank/{n}x{d}/r{rank}/t{tail}",
                "makeLowRankMatrix",
                [],
                np.asarray(s),
                rtol=1e-10,
                atol=1e-12,
                n=n,
                d=d,
                rank=rank,
                tail=tail,
            )
        )
    return out
