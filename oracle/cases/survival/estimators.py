"""Survival analysis against statsmodels (Kaplan-Meier, weighted log-rank tests, Cox PHReg) and scipy (accelerated
failure time models by maximum likelihood, written from scipy.stats log densities), and Harrell's concordance index
from its definition.
"""

import numpy as np
import scipy
import statsmodels
from mdml_oracle.fixture import Case, case
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.tools.numdiff import approx_fprime, approx_hess3

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "statsmodels": statsmodels.__version__}

F64 = NDArray[np.float64]


def _data(seed: int, n: int, ties: bool) -> tuple[F64, F64, F64]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    t = rng.weibull(1.5, size=n) * np.exp(-0.5 * x[:, 0] + 0.3 * x[:, 1]) * 5
    c = rng.exponential(7.0, size=n)
    time = np.minimum(t, c)
    event = (t <= c).astype(float)
    if ties:
        time = np.round(time, 1) + 0.1
    return time.astype(np.float64), event, x


def _concordance(time: F64, event: F64, score: F64) -> list[float]:
    concordant = discordant = tied = 0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[i] < time[j] or (time[i] == time[j] and event[j] == 0):
                if score[i] > score[j]:
                    concordant += 1
                elif score[i] < score[j]:
                    discordant += 1
                else:
                    tied += 1
    comparable = concordant + discordant + tied
    return [(concordant + tied / 2) / comparable, concordant, discordant, tied, comparable]


def _aft(dist: str, x: F64, logt: F64, event: F64, scaled: bool) -> list[float]:
    n, p = x.shape
    design = np.column_stack([np.ones(n), x])
    k = p + 1

    def logpdf_logsf(z: F64) -> tuple[F64, F64]:
        if dist == "weibull":
            return stats.gumbel_l.logpdf(z), stats.gumbel_l.logsf(z)
        if dist == "lognormal":
            return stats.norm.logpdf(z), stats.norm.logsf(z)
        return stats.logistic.logpdf(z), stats.logistic.logsf(z)

    def loglik(theta: F64) -> float:
        s = float(theta[k]) if scaled else 0.0
        z = (logt - design @ theta[:k]) / np.exp(s)
        logf, logs = logpdf_logsf(z)
        return float(np.sum(event * (logf - s - logt) + (1 - event) * logs))

    def negative(theta: F64) -> float:
        return -loglik(theta)

    theta = np.zeros(k + (1 if scaled else 0))
    theta[:k] = np.linalg.lstsq(design, logt, rcond=None)[0]
    if scaled:
        theta[k] = np.log(np.std(logt - design @ theta[:k]))
    theta = minimize(negative, theta, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 20000}).x
    for _ in range(40):
        g = approx_fprime(theta, negative, centered=True)
        h = approx_hess3(theta, negative)
        step = np.linalg.solve(h, g)
        theta = theta - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    h = approx_hess3(theta, negative)
    se = np.sqrt(np.diag(np.linalg.inv(h)))
    return [*theta.tolist(), *se.tolist(), loglik(theta)]


def cases() -> list[Case]:
    out: list[Case] = []
    for name, seed, n, ties in [("plain", 11, 60, False), ("ties", 12, 70, True)]:
        time, event, x = _data(seed, n, ties)
        sf = SurvfuncRight(time, event)
        out.append(
            case(
                f"km/{name}",
                "km",
                [time, event],
                np.concatenate([sf.surv_times, sf.surv_prob, sf.surv_prob_se, sf.n_risk, sf.n_events]),
                rtol=1e-10,
            )
        )
        group3 = np.digitize(x[:, 1], [-0.4, 0.4]).astype(float)
        group2 = (x[:, 1] > 0).astype(float)
        for label, group in [("two", group2), ("three", group3)]:
            for weight, kwargs in [
                ("logrank", {}),
                ("gehan-breslow", {"weight_type": "gb"}),
                ("tarone-ware", {"weight_type": "tw"}),
                ("fleming-harrington", {"weight_type": "fh", "fh_p": 1}),
            ]:
                chi2, p = survdiff(time, event, group, **kwargs)  # pyright: ignore[reportArgumentType]
                out.append(
                    case(
                        f"logrank/{name}/{label}/{weight}",
                        "logrank",
                        [time, event, group],
                        np.asarray([chi2, p]),
                        rtol=1e-8,
                        weight=weight,
                    )
                )
        strata = (x[:, 0] > 0).astype(float)
        for tie in ["breslow", "efron"]:
            fit = PHReg(time, x, status=event, ties=tie).fit()
            times, cumhaz, _ = fit.baseline_cumulative_hazard[0]
            out.append(
                case(
                    f"cox/{name}/{tie}",
                    "cox",
                    [time, event, x],
                    np.concatenate([fit.params, fit.bse, [fit.llf]]),
                    rtol=1e-6,
                    atol=1e-8,
                    ties=tie,
                )
            )
            out.append(
                case(
                    f"cox/{name}/{tie}/baseline",
                    "coxBaseline",
                    [time, event, x],
                    # statsmodels reports the cumulative hazard just before each time, so its entries 1.. are H(t_k) for k < m - 1.
                    np.concatenate([times[:-1], cumhaz[1:]]),
                    rtol=1e-6,
                    atol=1e-8,
                    ties=tie,
                )
            )
        stratified = PHReg(time, x[:, [1]], status=event, strata=strata, ties="efron").fit()
        out.append(
            case(
                f"cox/{name}/strata",
                "coxStrata",
                [time, event, x[:, [1]], strata],
                np.concatenate([stratified.params, stratified.bse, [stratified.llf]]),
                rtol=1e-6,
                atol=1e-8,
            )
        )
        # Harrell's C for a risk score and its reverse.
        score = x[:, 0] * -0.5 + 0.1 * x[:, 1]
        out.append(
            case(
                f"concordance/{name}",
                "concordance",
                [time, event, score],
                np.asarray(_concordance(time, event, score)),
                rtol=1e-12,
            )
        )
    time, event, x = _data(21, 80, False)
    for dist in ["weibull", "lognormal", "loglogistic"]:
        out.append(
            case(
                f"aft/{dist}",
                "aft",
                [time, event, x],
                np.asarray(_aft(dist, x, np.log(time), event, True)),
                rtol=1e-5,
                atol=1e-6,
                distribution=dist,
            )
        )
    out.append(
        case(
            "aft/exponential",
            "aft",
            [time, event, x],
            np.asarray(_aft("weibull", x, np.log(time), event, False)),
            rtol=1e-5,
            atol=1e-6,
            distribution="exponential",
        )
    )
    return out
