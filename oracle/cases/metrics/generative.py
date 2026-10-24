"""Generative-model metrics from features: the Frechet distance as pytorch-fid computes it (scipy.linalg.sqrtm), the
KID polynomial MMD of torchmetrics, precision, recall, density and coverage from the prdc package; the Inception score,
CLIPScore and Vendi score from their definitions.
"""

import contextlib
import io
from importlib.metadata import version

import numpy as np
import scipy
import scipy.linalg as sl
import torch
from prdc import compute_prdc
from torchmetrics.image.kid import poly_mmd
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {
    "scipy": scipy.__version__,
    "prdc": version("prdc"),
    "torchmetrics": version("torchmetrics"),
}


def frechet(a: FloatArray, b: FloatArray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.asarray(sl.sqrtm(s1 @ s2)).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def inception(p: FloatArray, splits: int) -> tuple[float, float]:
    scores: list[float] = []
    n = len(p)
    for s in range(splits):
        part = p[s * n // splits : (s + 1) * n // splits]
        marginal = part.mean(0)
        kl = (part * (np.log(part) - np.log(marginal))).sum(1).mean()
        scores.append(float(np.exp(kl)))
    return float(np.mean(scores)), float(np.std(scores))


def vendi(x: FloatArray, q: float) -> float:
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    lam = np.linalg.eigvalsh(xn @ xn.T / len(x))
    lam = lam[lam > 0]
    if q == 1:
        return float(np.exp(-(lam * np.log(lam)).sum()))
    return float((lam**q).sum() ** (1 / (1 - q)))


def cases() -> list[Case]:
    g = rng(70)
    real = g.normal(size=(40, 5))
    fake = 0.9 * g.normal(size=(40, 5)) + 0.3  # torchmetrics' MMD assumes equal set sizes
    logits = g.normal(size=(30, 6)) * 2
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    img, txt = g.normal(size=(8, 4)), g.normal(size=(8, 4))
    with contextlib.redirect_stdout(io.StringIO()):
        prdc = compute_prdc(real, fake, nearest_k=4)
    kid = float(poly_mmd(torch.tensor(real), torch.tensor(fake), degree=3, gamma=None, coef=1.0))
    mean_is, std_is = inception(probs, 3)
    cos = (img * txt).sum(1) / (np.linalg.norm(img, axis=1) * np.linalg.norm(txt, axis=1))
    out: list[Case] = [
        case("inceptionScore", "inceptionScore", [probs, probs], np.asarray(mean_is), splits=3),
        case("inceptionScore/std", "inceptionScoreStd", [probs, probs], np.asarray(std_is), splits=3),
        case(
            "frechetDistance", "frechetDistance", [real, fake], np.asarray(frechet(real, fake)), rtol=1e-9, atol=1e-10
        ),
        case(
            "kernelInceptionDistance", "kernelInceptionDistance", [real, fake], np.asarray(kid), subsets=1, rtol=1e-10
        ),
        case("clipScore", "clipScore", [img, txt], np.asarray(float(np.maximum(100 * cos, 0).mean()))),
        case("vendiScore", "vendiScore", [real, real], np.asarray(vendi(real, 1.0)), rtol=1e-10),
        case("vendiScore/q2", "vendiScore", [real, real], np.asarray(vendi(real, 2.0)), q=2.0, rtol=1e-10),
    ]
    for key in ["precision", "recall", "density", "coverage"]:
        out.append(case(f"prdc/{key}", "prdc", [real, fake], np.asarray(float(prdc[key])), field=key, nearestK=4))
    return out
