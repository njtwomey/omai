"""Named diffusions against scipy.stats: Ornstein-Uhlenbeck (norm), geometric Brownian motion (lognorm), CIR (a scaled
ncx2, with a gamma stationary law) and Bessel (the square root of a scaled ncx2) transition densities and moments;
Vasicek and CIR bond prices by integrating their Riccati equations with scipy.integrate.solve_ivp; Merton's
jump-diffusion price and log-return density as Poisson mixtures (scipy.stats.poisson, norm)."""

import numpy as np
import scipy
import scipy.stats as st
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray
from scipy.integrate import solve_ivp

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "numpy": np.__version__}


def riccati_bond(r0: float, t: float, kappa: float, theta: float, sigma: float, cir: bool) -> float:
    """P(0, T) = exp(A(T) - B(T) r0) with B' = 1 - kappa B - sigma^2 B^2 / 2 (CIR) or B' = 1 - kappa B (Vasicek),
    A' = -kappa theta B (+ sigma^2 B^2 / 2 for Vasicek), integrated in the time to maturity."""

    def rhs(_s: float, y: FloatArray) -> FloatArray:
        b, _a = y
        if cir:
            return np.array([1 - kappa * b - 0.5 * sigma**2 * b**2, -kappa * theta * b])
        return np.array([1 - kappa * b, -kappa * theta * b + 0.5 * sigma**2 * b**2])

    sol = solve_ivp(rhs, (0.0, t), np.array([0.0, 0.0]), rtol=1e-12, atol=1e-14)
    b, a = sol.y[:, -1]
    return float(np.exp(a - b * r0))


def merton_price(s: float, k: float, t: float, r: float, sigma: float, lam: float, m: float, d: float) -> float:
    """The Poisson-weighted sum of Black-Scholes call prices."""
    kk = np.exp(m + d**2 / 2) - 1
    lp = lam * (1 + kk)
    total = 0.0
    for n in range(80):
        rn = r - lam * kk + n * np.log(1 + kk) / t
        vol = np.sqrt(sigma**2 + n * d**2 / t)
        d1 = (np.log(s / k) + (rn + vol**2 / 2) * t) / (vol * np.sqrt(t))
        d2 = d1 - vol * np.sqrt(t)
        bs = s * st.norm.cdf(d1) - k * np.exp(-rn * t) * st.norm.cdf(d2)
        total += float(st.poisson.pmf(n, lp * t)) * float(bs)
    return total


def merton_density(x: FloatArray, t: float, mu: float, sigma: float, lam: float, m: float, d: float) -> FloatArray:
    """The Poisson mixture of normal densities of the log return."""
    kk = np.exp(m + d**2 / 2) - 1
    out = np.zeros_like(x)
    for n in range(80):
        mean = (mu - lam * kk - sigma**2 / 2) * t + n * m
        sd = np.sqrt(sigma**2 * t + n * d**2)
        out += st.poisson.pmf(n, lam * t) * st.norm.pdf(x, mean, sd)
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    dt = 0.7
    x = 0.8
    ys = np.array([0.05, 0.3, 0.8, 1.4, 2.5])
    theta, mu, sigma = 1.3, 0.5, 0.6
    mean = mu + (x - mu) * np.exp(-theta * dt)
    sd = np.sqrt(sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta))
    ou = {"theta": theta, "mu": mu, "sigma": sigma, "x": x, "dt": dt}
    out.append(case("ou/transition", "ou-transition", [ys], st.norm(mean, sd).logpdf(ys), rtol=1e-12, **ou))
    out.append(case("ou/mean", "ou-mean", [], np.array(mean), **ou))
    out.append(case("ou/variance", "ou-variance", [], np.array(sd**2), **ou))
    stat = st.norm(mu, sigma / np.sqrt(2 * theta))
    out.append(case("ou/stationary", "ou-stationary", [ys], stat.logpdf(ys), rtol=1e-12, **ou))

    gmu, gsigma = 0.1, 0.4
    law = st.lognorm(s=gsigma * np.sqrt(dt), scale=x * np.exp((gmu - gsigma**2 / 2) * dt))
    gbm = {"mu": gmu, "sigma": gsigma, "x": x, "dt": dt}
    out.append(case("gbm/transition", "gbm-transition", [ys], law.logpdf(ys), rtol=1e-12, **gbm))
    out.append(case("gbm/mean", "gbm-mean", [], np.array(law.mean()), rtol=1e-12, **gbm))
    out.append(case("gbm/variance", "gbm-variance", [], np.array(law.var()), rtol=1e-10, **gbm))

    for kappa, ctheta, csigma in [(1.5, 0.6, 0.5), (0.8, 1.0, 1.2)]:
        c = csigma**2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
        df = 4 * kappa * ctheta / csigma**2
        nc = x * np.exp(-kappa * dt) / c
        ncx = st.ncx2(df, nc)
        p = {"kappa": kappa, "theta": ctheta, "sigma": csigma, "x": x, "dt": dt}
        tag = f"{kappa}/{csigma}"
        logpdf = ncx.logpdf(ys / c) - np.log(c)
        out.append(case(f"cir/transition/{tag}", "cir-transition", [ys], logpdf, rtol=1e-9, atol=1e-12, **p))
        out.append(case(f"cir/mean/{tag}", "cir-mean", [], np.array(ncx.mean() * c), rtol=1e-12, **p))
        out.append(case(f"cir/variance/{tag}", "cir-variance", [], np.array(ncx.var() * c**2), rtol=1e-10, **p))
        gam = st.gamma(2 * kappa * ctheta / csigma**2, scale=csigma**2 / (2 * kappa))
        out.append(case(f"cir/stationary/{tag}", "cir-stationary", [ys], gam.logpdf(ys), rtol=1e-10, **p))
        for r0, maturity in [(0.03, 1.0), (0.08, 5.0)]:
            q = {"kappa": kappa, "theta": ctheta * 0.05, "sigma": csigma * 0.1, "r0": r0, "maturity": maturity}
            cir_p = riccati_bond(r0, maturity, kappa, ctheta * 0.05, csigma * 0.1, cir=True)
            vas_p = riccati_bond(r0, maturity, kappa, ctheta * 0.05, csigma * 0.1, cir=False)
            out.append(case(f"cir/bond/{tag}/{maturity}", "cir-bond", [], np.array(cir_p), rtol=1e-9, **q))
            out.append(case(f"vasicek/bond/{tag}/{maturity}", "vasicek-bond", [], np.array(vas_p), rtol=1e-9, **q))

    for delta in [2.0, 3.0, 4.5]:
        ncx = st.ncx2(delta, x**2 / dt)
        dens = ncx.logpdf(ys**2 / dt) + np.log(2 * ys / dt)
        out.append(
            case(f"bessel/transition/{delta}", "bessel-transition", [ys], dens, rtol=1e-9, delta=delta, x=x, dt=dt)
        )

    merton = {"mu": 0.08, "sigma": 0.25, "rate": 0.8, "jumpMean": -0.1, "jumpStd": 0.15}
    for s, k, t, r in [(100.0, 100.0, 1.0, 0.05), (90.0, 110.0, 0.5, 0.02), (120.0, 100.0, 2.0, 0.01)]:
        price = merton_price(s, k, t, r, 0.25, 0.8, -0.1, 0.15)
        out.append(
            case(
                f"merton/price/{s}/{k}/{t}",
                "merton-price",
                [],
                np.array(price),
                rtol=1e-10,
                spot=s,
                strike=k,
                maturity=t,
                r=r,
                **merton,
            )
        )
    xs = np.linspace(-1.0, 0.8, 7)
    dens = merton_density(xs, 1.5, 0.08, 0.25, 0.8, -0.1, 0.15)
    out.append(case("merton/density", "merton-density", [xs], dens, rtol=1e-10, t=1.5, **merton))
    return out
