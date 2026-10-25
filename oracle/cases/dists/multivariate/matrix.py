"""Matrix distributions (dists/multivariate/matrix.ts) against scipy.stats and torch.distributions."""

import numpy as np
import scipy
import torch
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import rng, spd
from mdml_oracle.torch_helpers import to_numpy, to_torch
from scipy import special, stats

REFERENCE: dict[str, str] = {"scipy": scipy.__version__, "torch": torch.__version__}


def cases() -> list[Case]:
    g = rng(45)
    out: list[Case] = []
    p = 3
    sigma = spd(g, p) / 3
    df = np.array(5.5)
    w = stats.wishart(df=5.5, scale=sigma)
    xs = np.stack([spd(g, p) for _ in range(4)])
    out += [
        case("wishart/logProb", "wishart.logProb", [df, sigma, xs], w.logpdf(np.moveaxis(xs, 0, -1)), rtol=1e-10),
        case("wishart/mean", "wishart.mean", [df, sigma], w.mean()),
        case("wishart/variance", "wishart.variance", [df, sigma], w.var()),
        case("wishart/mode", "wishart.mode", [df, sigma], w.mode()),
        case("wishart/entropy", "wishart.entropy", [df, sigma], np.array(w.entropy()), rtol=1e-10),
    ]
    psi = spd(g, p) / 2
    iw = stats.invwishart(df=7.0, scale=psi)
    dfi = np.array(7.0)
    out += [
        case(
            "inverseWishart/logProb",
            "inverseWishart.logProb",
            [dfi, psi, xs],
            iw.logpdf(np.moveaxis(xs, 0, -1)),
            rtol=1e-10,
        ),
        case("inverseWishart/mean", "inverseWishart.mean", [dfi, psi], iw.mean()),
        case("inverseWishart/variance", "inverseWishart.variance", [dfi, psi], iw.var()),
        case("inverseWishart/mode", "inverseWishart.mode", [dfi, psi], iw.mode()),
    ]
    # scipy's invwishart.entropy disagrees with Monte Carlo (it is not the entropy of its own logpdf), so the reference
    # is the change of variables X = W^{-1} from scipy's Wishart: H(X) = H(W) - (p + 1) E[log |W|].
    w_inv = stats.wishart(df=7.0, scale=np.linalg.inv(psi))
    e_log_det = special.digamma(3.5 + (1 - np.arange(1, p + 1)) / 2).sum() + p * np.log(2) - np.linalg.slogdet(psi)[1]
    iw_entropy = np.array(w_inv.entropy() - (p + 1) * e_log_det)
    out.append(case("inverseWishart/entropy", "inverseWishart.entropy", [dfi, psi], iw_entropy, rtol=1e-10))
    # LKJ over Cholesky factors against torch.
    for i, eta in enumerate([0.7, 1.0, 3.5]):
        lkj = torch.distributions.LKJCholesky(4, to_torch(np.array(eta)))
        torch.manual_seed(i)
        ls = to_numpy(lkj.sample(torch.Size([5])))
        out.append(
            case(
                f"lkjCholesky/{i}/logProb",
                "lkjCholesky.logProb",
                [np.array(eta), ls],
                to_numpy(lkj.log_prob(to_torch(ls))),
                rtol=1e-10,
                dim=4,
            )
        )
    out.append(
        case(
            "multigammaln",
            "multigammaln",
            [np.array([2.5, 4.0, 7.3])],
            special.multigammaln(np.array([2.5, 4.0, 7.3]), 3),
            rtol=1e-12,
            p=3,
        )
    )
    return out
