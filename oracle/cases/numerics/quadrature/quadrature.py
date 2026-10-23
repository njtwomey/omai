"""Quadrature against numpy.polynomial, scipy.special.roots_* and scipy.integrate, on a shared table of integrands."""

from collections.abc import Callable

import numpy as np
import scipy
import scipy.integrate as si
import scipy.special as sp
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng, softplus

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__}

type Integrand = Callable[[float], float]

# name -> (f, a, b); the TypeScript test defines the same integrands.
INTEGRANDS: dict[str, tuple[Integrand, float, float]] = {
    "poly": (lambda x: x**3 - 2 * x + 1, 0.0, 2.0),
    "exp": (np.exp, 0.0, 1.0),
    "gauss": (lambda x: np.exp(-x * x), -np.inf, np.inf),
    "decay": (lambda x: np.exp(-x), 0.0, np.inf),
    "lorentz": (lambda x: 1 / (1 + x * x), -np.inf, 0.0),
    "sqrt": (np.sqrt, 0.0, 1.0),
    "log": (np.log, 0.0, 1.0),
    "oscillatory": (lambda x: x * np.cos(20 * x), 0.0, np.pi),
    "kink": (lambda x: abs(x - 0.3), 0.0, 1.0),
}

FINITE: list[str] = ["poly", "exp", "sqrt", "oscillatory"]


def square(x: float) -> float:
    return x * x


def normal_pdf(x: float) -> float:
    return float(np.exp(-0.5 * ((x - 0.5) / 1.5) ** 2) / (1.5 * np.sqrt(2 * np.pi)))


def dbl_integrand(y: float, x: float) -> float:
    return float(x * y * y + np.sin(x))


def tpl_integrand(z: float, y: float, x: float) -> float:
    return x + y * z


def gauss3(x0: float, x1: float, x2: float) -> float:
    return float(np.exp(-(x0**2 + x1**2 + x2**2)))


def scalar(f: Integrand, a: float, b: float, **kwargs: object) -> float:
    value, _ = si.quad(f, a, b, epsabs=1e-13, epsrel=1e-13, limit=200, **kwargs)  # type: ignore[arg-type]
    return float(value)


def cases() -> list[Case]:
    out: list[Case] = []
    for n in [1, 2, 3, 5, 10, 20, 40]:
        x, w = np.polynomial.legendre.leggauss(n)
        out.append(case(f"gaussLegendre/{n}", "gaussLegendre", [], np.stack([x, w]), rtol=1e-12, atol=1e-14, n=n))
        x, w = np.polynomial.hermite.hermgauss(n)
        out.append(case(f"gaussHermite/{n}", "gaussHermite", [], np.stack([x, w]), rtol=1e-10, atol=1e-13, n=n))
        x, w = np.polynomial.laguerre.laggauss(n)
        out.append(case(f"gaussLaguerre/{n}", "gaussLaguerre", [], np.stack([x, w]), rtol=1e-10, atol=1e-13, n=n))
    for n, alpha, beta in [(3, 0.5, 0.5), (6, -0.5, 1.5), (10, 2.0, 0.0), (5, -0.5, -0.5)]:
        x, w = sp.roots_jacobi(n, alpha, beta)
        out.append(
            case(
                f"gaussJacobi/{n}/{alpha}/{beta}",
                "gaussJacobi",
                [],
                np.stack([x, w]),
                rtol=1e-10,
                atol=1e-13,
                n=n,
                alpha=alpha,
                beta=beta,
            )
        )

    g = rng(31)
    for size in [7, 8]:
        y = g.normal(size=size)
        xs = np.sort(g.uniform(0, 3, size=size))
        out.append(
            case(f"trapezoid/dx/{size}", "trapezoid", [y], np.array(np.trapezoid(y, dx=0.5)), rtol=1e-12, dx=0.5)
        )
        out.append(case(f"trapezoid/x/{size}", "trapezoid", [y, xs], np.array(np.trapezoid(y, x=xs)), rtol=1e-12))
        out.append(case(f"simpson/dx/{size}", "simpson", [y], np.array(si.simpson(y, dx=0.5)), rtol=1e-12, dx=0.5))
        out.append(case(f"simpson/x/{size}", "simpson", [y, xs], np.array(si.simpson(y, x=xs)), rtol=1e-12))
        out.append(
            case(
                f"cumulativeTrapezoid/x/{size}",
                "cumulativeTrapezoid",
                [y, xs],
                si.cumulative_trapezoid(y, x=xs, initial=0),
                rtol=1e-12,
            )
        )
    y2 = g.normal(size=(3, 6))
    out.append(case("simpson/axis0", "simpson", [y2], si.simpson(y2, axis=0), rtol=1e-12, axis=0, dx=1.0))
    out.append(case("trapezoid/axis1", "trapezoid", [y2], np.trapezoid(y2, axis=1), rtol=1e-12, axis=1, dx=1.0))

    for name, (f, a, b) in INTEGRANDS.items():
        points = [0.3] if name == "kink" else None
        exact = scalar(f, a, b, points=points) if points else scalar(f, a, b)
        out.append(case(f"quad/{name}", "quad", [], np.array(exact), rtol=1e-9, atol=1e-10, fn_name=name))
        if name in FINITE:
            fixed, _ = si.fixed_quad(np.vectorize(f), a, b, n=8)
            out.append(
                case(f"fixedQuad/{name}", "fixedQuad", [], np.array(fixed), rtol=1e-12, atol=1e-13, fn_name=name)
            )
        if np.isfinite(a) and np.isfinite(b) and name != "kink":
            out.append(case(f"tanhSinh/{name}", "tanhSinh", [], np.array(exact), rtol=1e-9, atol=1e-10, fn_name=name))
        # Simpson samples the endpoints, where log is singular.
        if np.isfinite(a) and np.isfinite(b) and name not in ("kink", "log"):
            out.append(
                case(
                    f"adaptiveSimpson/{name}",
                    "adaptiveSimpson",
                    [],
                    np.array(exact),
                    rtol=1e-7,
                    atol=1e-8,
                    fn_name=name,
                )
            )

    # E[f(X)] for X ~ N(0.5, 1.5^2).
    expectations: dict[str, Integrand] = {
        "square": square,
        "cos": np.cos,
        "softplus": lambda x: float(softplus(np.array(x))),
    }
    for name, h in expectations.items():
        expected = scalar(lambda x, h=h: h(x) * normal_pdf(x), -np.inf, np.inf)
        out.append(
            case(f"normalExpectation/{name}", "normalExpectation", [], np.array(expected), rtol=1e-9, fn_name=name)
        )

    dbl, _ = si.dblquad(dbl_integrand, 0, 2, lambda x: x / 2, lambda x: 1 + x, epsabs=1e-13, epsrel=1e-13)
    out.append(case("dblquad/variable", "dblquad", [], np.array(dbl), rtol=1e-10))
    tpl, _ = si.tplquad(tpl_integrand, 0, 1, 0, lambda x: 1 - x, 0, lambda x, y: 1 - x - y, epsabs=1e-13, epsrel=1e-13)
    out.append(case("tplquad/simplex", "tplquad", [], np.array(tpl), rtol=1e-10))
    nq, _ = si.nquad(
        gauss3,
        [[0, 1], [0, 1], [0, 1]],
        opts={"epsabs": 1e-13, "epsrel": 1e-13},
    )
    out.append(case("nquad/gauss3", "nquad", [], np.array(nq), rtol=1e-10))
    return out
