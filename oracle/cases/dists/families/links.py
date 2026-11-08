"""GLM links (dists/families/links.ts) against statsmodels.genmod.families.links."""

import numpy as np
import statsmodels
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import rng
from statsmodels.genmod.families import links as L

REFERENCE: dict[str, str] = {"statsmodels": statsmodels.__version__}


def cases() -> list[Case]:
    g = rng(48)
    probs = g.uniform(0.02, 0.98, 7)
    pos = np.exp(g.normal(size=7) * 0.7)
    eta = g.normal(size=7)
    eta_neg = -np.exp(g.normal(size=7))
    links: list[tuple[str, L.Link, np.ndarray, np.ndarray]] = [
        ("identity", L.Identity(), pos, eta),
        ("log", L.Log(), pos, eta),
        ("logit", L.Logit(), probs, eta),
        ("probit", L.Probit(), probs, eta),
        ("cauchit", L.Cauchy(), probs, eta),
        ("cloglog", L.CLogLog(), probs, eta),
        ("loglog", L.LogLog(), probs, eta),
        ("inverse", L.InversePower(), pos, np.abs(eta) + 0.1),
        ("inverseSquared", L.InverseSquared(), pos, np.abs(eta) + 0.1),
        ("sqrt", L.Sqrt(), pos, np.abs(eta)),
        ("power", L.Power(1.7), pos, np.abs(eta)),
        ("negativeBinomial", L.NegativeBinomial(0.6), pos, eta_neg),
    ]
    out: list[Case] = []
    for name, link, mu, z in links:
        out += [
            case(f"{name}/link", f"{name}.link", [mu], link(mu), rtol=1e-12),
            case(f"{name}/inverse", f"{name}.inverse", [z], link.inverse(z), rtol=1e-12),
            case(f"{name}/derivative", f"{name}.derivative", [mu], link.deriv(mu), rtol=1e-12),
            case(f"{name}/inverseDerivative", f"{name}.inverseDerivative", [z], link.inverse_deriv(z), rtol=1e-12),
            case(f"{name}/secondDerivative", f"{name}.secondDerivative", [mu], link.deriv2(mu), rtol=1e-10),
        ]
    return out
