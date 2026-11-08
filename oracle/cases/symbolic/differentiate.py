"""Derivatives against SymPy's `diff` (printed form and values), mixed partials, higher orders, gradients, Jacobians,
Hessians and implicit differentiation (`idiff`, compared by value: SymPy simplifies to a different but equal form)."""

import sympy as sp
from omai_oracle.fixture import Case
from omai_oracle.symbolic import SYMBOLS, expr_case, parse

REFERENCE: dict[str, str] = {"sympy": sp.__version__}

x, y = SYMBOLS["x"], SYMBOLS["y"]

DIFF: list[str] = [
    "x**2*sin(x)",
    "exp(x)*cos(x)",
    "log(x)/x",
    "sqrt(x**2 + 1)",
    "tan(x)",
    "atan(x)",
    "asin(x)",
    "acos(x)",
    "sinh(x)*cosh(x)",
    "tanh(x)",
    "x**x",
    "2**x",
    "sec(x)",
    "cot(x)",
    "csc(x)",
    "asinh(x)",
    "acosh(x)",
    "atanh(x)",
    "acot(x)",
    "erf(x)",
    "exp(-x**2/2)",
    "log(sin(x))",
    "sin(cos(tan(x)))",
    "(x + 1)**10",
    "1/(x**2 + 1)",
    "x*log(x) - x",
    "exp(x)/(1 + exp(x))",
    "a*x**3 + b*x**2 + c*x",
    "x**(3/2)",
    "cos(x)**3",
]

PARTIAL: list[tuple[str, list[tuple[str, int]]]] = [
    ("x**3*y**2", [("x", 1), ("y", 1)]),
    ("sin(x*y)", [("x", 1), ("y", 1)]),
    ("exp(x*y)", [("x", 2)]),
    ("x**5", [("x", 3)]),
    ("x**2*y + y**3", [("y", 2)]),
]

IDIFF: list[tuple[str, int]] = [
    ("x**2 + y**2 - 4", 1),
    ("x**2 + y**2 - 4", 2),
    ("x*y - 1", 1),
    ("y**3 + x*y - x**2", 1),
]


def cases() -> list[Case]:
    out: list[Case] = []
    for i, s in enumerate(DIFF):
        out.append(expr_case(f"diff/{i}", "diff", [s, "x"], sp.diff(parse(s), x)))
    for i, (s, spec) in enumerate(PARTIAL):
        args: list[sp.Basic] = []
        for name, k in spec:
            args.extend([SYMBOLS[name]] * k)
        out.append(
            expr_case(
                f"partial/{i}",
                "diffMany",
                [s, ",".join(f"{n}:{k}" for n, k in spec)],
                sp.diff(parse(s), *args),
            )
        )
    f = parse("x**2*y + sin(y)*x")
    for i, g in enumerate([sp.diff(f, x), sp.diff(f, y)]):
        out.append(expr_case(f"gradient/{i}", "gradient", [str(f), str(i)], g))
    for i, row in enumerate(sp.hessian(f, (x, y)).tolist()):
        for j, h in enumerate(row):
            out.append(expr_case(f"hessian/{i}{j}", "hessian", [str(f), f"{i},{j}"], h))
    fs = [parse("x*y"), parse("x**2 + y"), parse("sin(x)*y")]
    jac = sp.Matrix(fs).jacobian([x, y]).tolist()
    for i, row in enumerate(jac):
        for j, h in enumerate(row):
            out.append(expr_case(f"jacobian/{i}{j}", "jacobian", [";".join(map(str, fs)), f"{i},{j}"], h))
    for i, (eq, n) in enumerate(IDIFF):
        d = sp.idiff(parse(eq), y, x, n)
        if d is None:
            raise ValueError(f"idiff gave no result for {eq}")
        out.append(expr_case(f"idiff/{i}", "implicitDiff", [eq, str(n)], d, exact=False))
    return out
