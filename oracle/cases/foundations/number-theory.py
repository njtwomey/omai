"""Elementary number theory against sympy.ntheory (exact integer arithmetic). Integers are encoded as float64 (exact
below 2^53) and huge ones as strings."""

import math
from typing import cast

import numpy as np
import sympy
from sympy.core.intfunc import igcdex
from sympy.ntheory.continued_fraction import continued_fraction, continued_fraction_convergents
from sympy.ntheory.modular import crt, solve_congruence
from whitebox_oracle.fixture import Case, case

REFERENCE: dict[str, str] = {"sympy": sympy.__version__}


def floats(xs: list[int]) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    return np.array(xs, dtype=np.float64)


def cases() -> list[Case]:
    out: list[Case] = []
    pairs = [(252, 105), (240, 46), (17, 5), (1071, 462), (-84, 36), (0, 9), (123456789, 987654321)]
    for a, b in pairs:
        x, y, g = cast(tuple[int, int, int], igcdex(a, b))
        out.append(case(f"gcd/{a}/{b}", "gcd", [floats([a, b])], floats([math.gcd(a, b)])))
        out.append(case(f"bezout/{a}/{b}", "bezout", [floats([a, b])], floats([g, x, y])))
    out.append(case("lcm", "lcm", [floats([4, 6, 10, 15])], floats([int(sympy.ilcm(4, 6, 10, 15))])))
    for a, m in [(3, 11), (10, 17), (123, 4567), (-5, 13), (99991, 2**31 - 1)]:
        out.append(case(f"inverse/{a}/{m}", "modInverse", [floats([a, m])], floats([int(sympy.mod_inverse(a, m))])))
    for b, e, m in [(4, 13, 497), (2, 10**15, 10**9 + 7), (123456789, 987654321, 2**52 - 47), (7, 0, 13)]:
        out.append(case(f"pow/{b}/{e}/{m}", "modPow", [floats([b, e, m])], floats([pow(b, e, m)])))
    out.append(case("sieve/200", "sieve", [floats([200])], floats(list(sympy.primerange(0, 201)))))
    candidates = [0, 1, 2, 561, 1105, 7919, 2147483647, 2**53 - 111, 2**53 - 113, 3215031751, 999999000001, 10**15 + 37]
    out.append(
        case("isPrime", "isPrime", [floats(candidates)], floats([1 if sympy.isprime(n) else 0 for n in candidates]))
    )
    for n in [360, 1001, 8051, 600851475143, 2**40 - 1, 999999000001 * 7, 2**52 + 1, 97 * 1000003 * 1000033]:
        fac = cast(dict[int, int], sympy.factorint(n))
        flat = [v for p in sorted(fac) for v in (p, fac[p])]
        out.append(case(f"factorise/{n}", "factorise", [floats([n])], floats(flat)))
        out.append(case(f"phi/{n}", "eulerPhi", [floats([n])], floats([int(sympy.totient(n))])))
    for n in [28, 360, 997, 1]:
        out.append(case(f"divisors/{n}", "divisors", [floats([n])], floats([int(d) for d in sympy.divisors(n)])))
    for residues, moduli in [([2, 3, 2], [3, 5, 7]), ([1, 4, 6], [5, 7, 11]), ([3, 5], [10**6 + 3, 10**6 + 33])]:
        r = crt(moduli, residues)
        assert r is not None
        out.append(case(f"crt/{moduli}", "crt", [floats(residues), floats(moduli)], floats([int(r[0]), int(r[1])])))
    for residues, moduli in [([2, 4], [6, 8]), ([1, 2], [4, 6])]:
        r = solve_congruence(*zip(residues, moduli, strict=True))
        want = floats([int(r[0]), int(r[1])]) if r is not None else floats([])
        out.append(case(f"congruence/{moduli}/{residues}", "crt", [floats(residues), floats(moduli)], want))
    for p, q in [(415, 93), (355, 113), (-17, 5), (1, 7), (89, 55)]:
        cf = [int(a) for a in continued_fraction(sympy.Rational(p, q))]
        out.append(case(f"cf/{p}/{q}", "continuedFraction", [floats([p, q])], floats(cf)))
        conv = [cast(sympy.Rational, c) for c in continued_fraction_convergents(cf)]
        flat = [v for c in conv for v in (int(c.p), int(c.q))]
        out.append(case(f"convergents/{p}/{q}", "convergents", [floats(cf)], floats(flat)))
    for n in [0, 1, 2, 10, 50, 78, 100, 300, -7, -8]:
        out.append(
            case(
                f"fibonacci/{n}",
                "fibonacci",
                [floats([n])],
                str(int(sympy.fibonacci(n)) if n >= 0 else (-1) ** (-n + 1) * int(sympy.fibonacci(-n))),
            )
        )
        lucas = int(sympy.lucas(n)) if n >= 0 else (-1) ** (-n) * int(sympy.lucas(-n))
        out.append(case(f"lucas/{n}", "lucas", [floats([n])], str(lucas)))
    return out
