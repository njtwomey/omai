"""Diffusion-model schedules and coefficients against the formulas of their reference implementations in numpy: DDPM's
linear betas and cumulative products (Ho et al.), the cosine betas of improved DDPM (the `betas_for_alpha_bar` of
diffusers), the sigmoid schedule of Jabri et al. (as in denoising-diffusion-pytorch), the VP-SDE signal scale of Song
et al., and the EDM preconditioning of Karras et al."""

import numpy as np
from omai_oracle.fixture import Case, case
from omai_oracle.helpers import FloatArray

REFERENCE: dict[str, str] = {"numpy": np.__version__}


def linear_betas(n: int, start: float, end: float) -> FloatArray:
    return np.linspace(start, end, n, dtype=np.float64)


def cosine_betas(n: int, s: float = 0.008, max_beta: float = 0.999) -> FloatArray:
    def alpha_bar(t: float) -> float:
        return float(np.cos((t + s) / (1 + s) * np.pi / 2) ** 2)

    return np.array([min(1 - alpha_bar((i + 1) / n) / alpha_bar(i / n), max_beta) for i in range(n)])


def cosine_alpha_bar(t: FloatArray, s: float = 0.008) -> FloatArray:
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return np.asarray(f / (np.cos(s / (1 + s) * np.pi / 2) ** 2), dtype=np.float64)


def sigmoid_alpha_bar(t: FloatArray, start: float, end: float, tau: float) -> FloatArray:
    def sig(x: FloatArray | float) -> FloatArray:
        return np.asarray(1 / (1 + np.exp(-np.asarray(x))), dtype=np.float64)

    v_start, v_end = sig(start / tau), sig(end / tau)
    return np.asarray((v_end - sig((t * (end - start) + start) / tau)) / (v_end - v_start), dtype=np.float64)


def vp_alpha(t: FloatArray, beta_min: float, beta_max: float) -> FloatArray:
    return np.asarray(np.exp(-0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min), dtype=np.float64)


def edm_coefficients(sigma: FloatArray, sigma_data: float) -> FloatArray:
    total = sigma**2 + sigma_data**2
    return np.stack(
        [sigma_data**2 / total, sigma * sigma_data / np.sqrt(total), 1 / np.sqrt(total), np.log(sigma) / 4], axis=1
    )


def cases() -> list[Case]:
    betas = linear_betas(1000, 1e-4, 0.02)
    grid = np.arange(1, 1001) / 1000
    t = np.linspace(0.02, 0.97, 25)
    sigma = np.exp(np.linspace(-3, 3, 9))
    return [
        case("discrete/alpha-bar", "discreteAlphaBar", [betas], np.cumprod(1 - betas)),
        case("discrete/grid", "discreteGrid", [grid], np.cumprod(1 - betas)),
        case("cosine/betas", "cosineBetas", [], cosine_betas(300), n=300),
        case("cosine/alpha-bar", "cosineAlphaBar", [t], cosine_alpha_bar(t)),
        case("sigmoid/alpha-bar", "sigmoidAlphaBar", [t], sigmoid_alpha_bar(t, -3.0, 3.0, 1.0)),
        case(
            "sigmoid/shifted",
            "sigmoidAlphaBar",
            [t],
            sigmoid_alpha_bar(t, -2.0, 4.0, 0.7),
            start=-2.0,
            end=4.0,
            tau=0.7,
        ),
        case("vp/alpha", "vpAlpha", [t], vp_alpha(t, 0.1, 20.0)),
        case("vp/alpha-wide", "vpAlpha", [t], vp_alpha(t, 0.05, 10.0), betaMin=0.05, betaMax=10.0),
        case("edm/preconditioning", "edmPreconditioning", [sigma], edm_coefficients(sigma, 0.5), sigmaData=0.5),
    ]
