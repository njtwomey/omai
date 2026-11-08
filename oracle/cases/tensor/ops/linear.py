"""Products and contractions against numpy."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng, uniform

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def cases() -> list[Case]:
    g = rng(700)
    a = uniform(g, (3, 4))
    b = uniform(g, (4, 2))
    batch_a = uniform(g, (2, 3, 4))
    batch_b = uniform(g, (4, 5))
    v = uniform(g, (4,))
    w = uniform(g, (4,))
    t1 = uniform(g, (2, 3, 4))
    t2 = uniform(g, (3, 4, 5))
    sq = uniform(g, (4, 4))
    return [
        case("matmul/2d", "matmul", [a, b], a @ b),
        case("matmul/batched-broadcast", "matmul", [batch_a, batch_b], batch_a @ batch_b),
        case("matmul/matrix-vector", "matmul", [a, v], a @ v),
        case("matmul/vector-matrix", "matmul", [v, b], v @ b),
        case("matmul/vector-vector", "matmul", [v, w], v @ w),
        case("dot", "dot", [v, w], np.dot(v, w)),
        case("vdot", "vdot", [a, a], np.vdot(a, a)),
        case("outer", "outer", [v, w[:3]], np.outer(v, w[:3])),
        case("inner", "inner", [a, v], np.inner(a, v)),
        case("tensordot/2", "tensordot", [t1, t2], np.tensordot(t1, t2, 2), axes=2),
        case("tensordot/explicit", "tensordot", [t1, t2], np.tensordot(t1, t2, ([1], [0])), axes=[[1], [0]]),
        case("einsum/ij,jk->ik", "einsum", [a, b], np.einsum("ij,jk->ik", a, b), spec="ij,jk->ik"),
        case(
            "einsum/bij,jk->bik",
            "einsum",
            [batch_a, batch_b],
            np.einsum("bij,jk->bik", batch_a, batch_b),
            spec="bij,jk->bik",
        ),
        case("einsum/ii->", "einsum", [sq], np.einsum("ii->", sq), spec="ii->"),
        case("einsum/ii->i", "einsum", [sq], np.einsum("ii->i", sq), spec="ii->i"),
        case("einsum/ij->j", "einsum", [a], np.einsum("ij->j", a), spec="ij->j"),
        case("einsum/i,j->ij", "einsum", [v, w], np.einsum("i,j->ij", v, w), spec="i,j->ij"),
        case("kron", "kron", [a[:2, :2], b[:2]], np.kron(a[:2, :2], b[:2])),
        case("diagonal/0", "diagonal", [a], np.diagonal(a)),
        case("diagonal/offset", "diagonal", [a], np.diagonal(a, 1), offset=1),
        case("diagonal/negative-offset", "diagonal", [a], np.diagonal(a, -1), offset=-1),
        case("trace", "trace", [sq], np.trace(sq)),
    ]
