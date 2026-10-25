"""Finite probability against sympy.stats (dice, coins, finite random variables, conditioning) and continuous
transformations and sums against scipy.stats."""

from typing import cast

import numpy as np
import scipy
import scipy.stats as st
import sympy
import sympy.stats as ss
from mdml_oracle.fixture import Case, case
from sympy.logic.boolalg import Boolean
from sympy.stats.frv import FiniteDensity

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "sympy": sympy.__version__}


def value(e: object) -> float:
    return float(sympy.N(cast(sympy.Basic, e), 20))


def eq(a: sympy.Basic, b: int) -> Boolean:
    return cast(Boolean, sympy.Eq(a, b))


def pmf(expr: sympy.Expr) -> tuple[list[float], list[float]]:
    d = cast(dict[sympy.Basic, sympy.Basic], cast(FiniteDensity, ss.density(expr)).dict)
    keys = sorted(d, key=value)
    return [value(k) for k in keys], [value(d[k]) for k in keys]


def cases() -> list[Case]:
    out: list[Case] = []
    x, y, z = ss.Die("X", 6), ss.Die("Y", 6), ss.Die("Z", 6)
    out.append(case("dice/sumAbove7", "diceEvent", [], np.array(value(ss.P(x + y > 7)))))
    out.append(
        case("dice/conditional", "diceConditional", [], np.array(value(ss.P(sympy.Eq(x + y, 8), sympy.Eq(x, 3)))))
    )
    out.append(case("dice/conditionalEven", "diceConditionalEven", [], np.array(value(ss.P(x + y > 8, x > 3)))))
    out.append(case("dice/expectProduct", "diceExpectProduct", [], np.array(value(ss.E(x * y)))))
    out.append(case("dice/varianceSum", "diceVarianceSum", [], np.array(value(ss.variance(x + y)))))
    out.append(case("dice/covariance", "diceCovariance", [], np.array(value(ss.covariance(x, x + y)))))
    values, probs = pmf(x + y + z)
    out.append(case("dice/sum3", "diceSum3", [], np.array([values, probs])))
    joint = ss.P(sympy.And(x > 4, x + y > 9))
    out.append(
        case(
            "dice/independence",
            "diceIndependence",
            [],
            np.array([value(ss.P(x > 4)), value(ss.P(x + y > 9)), value(joint)]),
        )
    )

    coins = [ss.Bernoulli(f"C{i}", sympy.Rational(1, 3)) for i in range(4)]
    values, probs = pmf(cast(sympy.Expr, sum(coins)))
    out.append(case("coins/sum", "coinSum", [], np.array([values, probs])))
    out.append(case("coins/binomial", "binomialConvolution", [], st.binom.pmf(np.arange(9), 8, 0.3)))

    # Bayes with a joint finite variable on 2 h + d: hypothesis h, datum d.
    for i, (prior, likelihood) in enumerate([([0.5, 0.5], [0.3, 0.8]), ([0.2, 0.5, 0.3], [0.9, 0.4, 0.1])]):
        weights = {}
        for h, (p, q) in enumerate(zip(prior, likelihood, strict=True)):
            weights[2 * h + 1] = sympy.nsimplify(p) * sympy.nsimplify(q)
            weights[2 * h] = sympy.nsimplify(p) * (1 - sympy.nsimplify(q))
        v = ss.FiniteRV(f"V{i}", weights)
        observed = sympy.Or(*[eq(v, 2 * h + 1) for h in range(len(prior))])
        posterior = [value(ss.P(sympy.Eq(v, 2 * h + 1), observed)) for h in range(len(prior))]
        out.append(case(f"bayes/{i}", "bayes", [np.array(prior), np.array(likelihood)], np.array(posterior)))
    disease = ss.FiniteRV(
        "D",
        {
            3: sympy.Rational(1, 100) * sympy.Rational(99, 100),
            2: sympy.Rational(1, 100) * sympy.Rational(1, 100),
            1: sympy.Rational(99, 100) * sympy.Rational(5, 100),
            0: sympy.Rational(99, 100) * sympy.Rational(95, 100),
        },
    )
    # D = 2 * ill + positive.
    positive = sympy.Or(eq(disease, 3), eq(disease, 1))
    negative = sympy.Or(eq(disease, 2), eq(disease, 0))
    ppv = value(ss.P(disease > 2, positive))
    npv = value(ss.P(disease < 2, negative))
    out.append(case("diagnostic", "diagnostic", [np.array([0.01, 0.99, 0.95])], np.array([ppv, npv])))

    ys = np.array([0.2, 0.7, 1.0, 2.5, 6.0])
    out.append(case("transform/lognormal", "lognormal", [ys], st.lognorm.pdf(ys, 1.0), rtol=1e-12))
    out.append(case("transform/scaled", "scaledExponential", [ys], st.expon.pdf(ys, scale=0.5), rtol=1e-12))
    s = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])
    out.append(case("sumDensity/normal", "normalSum", [s], st.norm.pdf(s, scale=np.sqrt(2)), rtol=1e-8, atol=1e-12))
    s2 = np.array([0.5, 1.0, 2.0, 4.0, 7.0])
    out.append(case("sumDensity/gamma", "exponentialSum", [s2], st.gamma.pdf(s2, 2), rtol=1e-8, atol=1e-12))
    return out
