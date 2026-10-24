"""Counting, enumeration and ranking against sympy (exact combinatorial numbers, permutation ranks) and Python's
itertools and math. Exact integers are encoded as comma-separated strings."""

import itertools
import math
from collections.abc import Iterator
from typing import cast

import numpy as np
import sympy
from sympy.combinatorics import Permutation
from sympy.functions.combinatorial.numbers import partition, stirling, subfactorial
from sympy.utilities.iterables import partitions
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"sympy": sympy.__version__}


def text(values: list[int]) -> str:
    return ",".join(str(v) for v in values)


def ints(xs: list[int]) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return np.array(xs, dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    out.append(
        case(
            "factorial", "factorial", [ints(list(range(0, 31, 5)))], text([math.factorial(n) for n in range(0, 31, 5)])
        )
    )
    pairs = [(10, 3), (52, 5), (100, 50), (7, 0), (5, 9), (60, 30)]
    out.append(
        case("binomial", "binomial", [ints([v for p in pairs for v in p])], text([math.comb(n, k) for n, k in pairs]))
    )
    for ks in [[1, 4, 4, 2], [3, 3, 3], [10, 0, 5], [2]]:
        coef = cast(dict[tuple[int, ...], int], sympy.multinomial_coefficients(len(ks), sum(ks)))[tuple(ks)]
        out.append(case(f"multinomial/{ks}", "multinomial", [ints(ks)], str(int(coef))))
    for n, r in [(4, 2), (3, 3), (5, 1), (3, 0)]:
        perms = [list(p) for p in itertools.permutations(range(n), r)]
        out.append(
            case(
                f"permutations/{n}/{r}",
                "permutations",
                [ints([n, r])],
                np.array(perms, dtype=np.float64).reshape(len(perms), r),
            )
        )
        combs = [list(c) for c in itertools.combinations(range(n), r)]
        out.append(
            case(
                f"combinations/{n}/{r}",
                "combinations",
                [ints([n, r])],
                np.array(combs, dtype=np.float64).reshape(len(combs), r),
            )
        )
    for perm in [[1, 0, 2], [3, 1, 4, 0, 2], [0, 1, 2, 3], [6, 5, 4, 3, 2, 1, 0], [2, 7, 0, 5, 1, 6, 3, 4]]:
        out.append(case(f"rank/{perm}", "permutationRank", [ints(perm)], str(int(Permutation(perm).rank()))))
    for n, r in [(4, 23), (6, 100), (8, 12345)]:
        p = Permutation.unrank_lex(n, r)
        out.append(case(f"unrank/{n}/{r}", "permutationUnrank", [ints([n, r])], ints(list(p.array_form))))
    for n, k in [(6, 3), (8, 4), (10, 2)]:
        combs = list(itertools.combinations(range(n), k))
        for idx in [0, len(combs) // 3, len(combs) - 1]:
            out.append(case(f"crank/{n}/{k}/{idx}", "combinationRank", [ints([n, *combs[idx]])], str(idx)))
            out.append(case(f"cunrank/{n}/{k}/{idx}", "combinationUnrank", [ints([n, k, idx])], ints(list(combs[idx]))))
    out.append(
        case(
            "derangements",
            "derangements",
            [ints(list(range(0, 21, 4)))],
            text([int(subfactorial(n)) for n in range(0, 21, 4)]),
        )
    )
    grid = [(5, 2), (10, 4), (12, 7), (20, 10), (3, 5), (0, 0)]
    flat = ints([v for p in grid for v in p])
    out.append(case("stirling2", "stirling2", [flat], text([int(stirling(n, k, kind=2)) for n, k in grid])))
    out.append(case("stirling1", "stirling1", [flat], text([int(stirling(n, k, kind=1)) for n, k in grid])))
    out.append(
        case(
            "stirling1/signed",
            "stirling1Signed",
            [flat],
            text([int(stirling(n, k, kind=1, signed=True)) for n, k in grid]),
        )
    )
    out.append(case("bell", "bell", [ints(list(range(0, 26, 5)))], text([int(sympy.bell(n)) for n in range(0, 26, 5)])))
    out.append(
        case(
            "catalan", "catalan", [ints(list(range(0, 31, 6)))], text([int(sympy.catalan(n)) for n in range(0, 31, 6)])
        )
    )
    out.append(
        case(
            "partitionCount",
            "partitionCount",
            [ints([0, 1, 5, 50, 100, 250])],
            text([int(partition(n)) for n in [0, 1, 5, 50, 100, 250]]),
        )
    )
    for n in [6, 9]:
        parts = sorted(
            (
                sorted([k for k, m in p.items() for _ in range(m)], reverse=True)
                for p in cast(Iterator[dict[int, int]], partitions(n))
            ),
            reverse=True,
        )
        width = max(len(p) for p in parts)
        padded = np.array([p + [0] * (width - len(p)) for p in parts], dtype=np.float64)
        out.append(case(f"partitions/{n}", "integerPartitions", [ints([n])], padded))
    families = [[1, 2, 3, 4], [3, 4, 5], [4, 5, 6, 7, 1], [1, 7, 9]]
    union = set().union(*map(set, families))
    out.append(
        case(
            "union",
            "inclusionExclusion",
            [np.array([v for f in families for v in [len(f), *f]], dtype=np.float64)],
            np.array(float(len(union))),
        )
    )
    tri = [[math.comb(n, k) for k in range(n + 1)] for n in range(12)]
    out.append(case("pascal", "pascal", [ints([12])], np.array([v for r in tri for v in r], dtype=np.float64)))
    tri2 = [[math.comb(n, k) % 3 for k in range(n + 1)] for n in range(40)]
    out.append(
        case("pascal/mod3", "pascalMod3", [ints([40])], np.array([v for r in tri2 for v in r], dtype=np.float64))
    )
    return out
