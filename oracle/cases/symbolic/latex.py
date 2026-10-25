"""LaTeX against SymPy's `latex()` with default settings: fractions, roots, powers of functions, named and
undefined functions, Greek and subscripted symbols, relations, piecewise, integrals, derivatives, big-O, matrices."""

import sympy as sp
from mdml_oracle.fixture import Case
from mdml_oracle.symbolic import expr_case, parse, parse_any

REFERENCE: dict[str, str] = {"sympy": sp.__version__}

INPUTS: list[str] = [
    "x**2 + 2*x + 1",
    "x**2*cos(x) + x*sin(x) + exp(x) + 3",
    "2*x/3",
    "-x/2",
    "1/(x + 1)",
    "sin(x)**2",
    "Abs(x)",
    "atan(x)",
    "sqrt(2)/2",
    "2*3**x",
    "(x + 1)**(1/3)",
    "x**(-3/2)",
    "x**(2/3)",
    "log(x)**2",
    "exp(-x)*x",
    "pi*x",
    "I*x + 1",
    "oo",
    "-oo",
    "Piecewise((x, x > 0), (-x, True))",
    "Eq(x, 1)",
    "x >= 1",
    "floor(x) + ceiling(y)",
    "gamma(x)**2",
    "erf(x)",
    "factorial(x)",
    "sec(x) + csc(x) + cot(x)",
    "asin(x) + acos(x) + acot(x)",
    "sinh(x) + cosh(x) + tanh(x) + asinh(x)",
    "x*y/z**2",
    "(x + 1)/(x - 1)",
    "-x/(y + 1)",
    "sqrt(x*y)",
    "(x - 1)*(x + 1)**2",
    "x*(y - x)",
    "3*sqrt(2)*x",
    "-x*y/3",
    "sin(x)/(2*y)",
    "Integral(exp(x**2), x)",
    "Integral(x + 1, (x, 0, 1))",
    "Derivative(f(x), x)",
    "Derivative(f(x), (x, 2))",
    "O(x**6)",
    "x - x**3/6 + O(x**5)",
    "alpha + beta_1 + x1",
    "theta_hat",
    "Gamma + omega",
    "re(x) + im(x)",
    "sign(x)",
    "exp(x)**y",
    "(x**2)**y",
    "x**(y + 1)",
    "foo(x)",
    "f(x, y)",
]


def cases() -> list[Case]:
    out: list[Case] = []
    local: dict[str, object] = {"f": sp.Function("f"), "foo": sp.Function("foo")}
    for name in ["alpha", "beta_1", "x1", "theta_hat", "Gamma", "omega"]:
        local[name] = sp.Symbol(name)
    for i, s in enumerate(INPUTS):
        e = parse_any(s, local)
        if not isinstance(e, sp.Basic):
            raise TypeError(f"{s!r} did not parse to an expression")
        out.append(expr_case(f"latex/{i}", "latex", [s], e))
    m = sp.Matrix([[1, parse("x")], [parse("y"), 2]])
    out.append(expr_case("latex/matrix", "latexMatrix", ["1,x;y,2"], sp.Symbol("M"), matrix=sp.latex(m)))
    return out
