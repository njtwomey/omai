"""Differential privacy against diffprivlib (classic and analytic Gaussian calibration, exponential-mechanism
probabilities, the budget accountant's composition) and opacus (RDP of the subsampled Gaussian and its conversion to
(epsilon, delta)). diffprivlib's models package does not import with this scikit-learn, so its mechanisms and accountant
are loaded without the package's top-level imports."""

import importlib.util
import sys
import types
from typing import Any

import numpy as np
import opacus
from mdml_oracle.fixture import Case, case
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

_spec = importlib.util.find_spec("diffprivlib")
if _spec is not None and _spec.submodule_search_locations is not None and "diffprivlib" not in sys.modules:
    _pkg = types.ModuleType("diffprivlib")
    _pkg.__path__ = list(_spec.submodule_search_locations)
    sys.modules["diffprivlib"] = _pkg
mechanisms: Any = importlib.import_module("diffprivlib.mechanisms")
accountant: Any = importlib.import_module("diffprivlib.accountant")

REFERENCE: dict[str, str] = {"diffprivlib": "0.6.6", "opacus": opacus.__version__}

ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))


def cases() -> list[Case]:
    out: list[Case] = []
    for eps, delta in [(0.1, 1e-5), (0.5, 1e-6), (1.0, 1e-5), (3.0, 1e-3), (8.0, 1e-8)]:
        scale = mechanisms.GaussianAnalytic(epsilon=eps, delta=delta, sensitivity=1.0)._scale
        out.append(
            case(
                f"analytic/{eps}/{delta}",
                "analytic",
                [],
                {"scale": np.array(scale)},
                epsilon=eps,
                delta=delta,
                # phi(-x) = (1 + erf(-x / sqrt 2)) / 2 cancels badly for large epsilon in diffprivlib's own formula, so
                # the reference is only good to about 1e-7 there.
                rtol=1e-10 if eps < 5 else 1e-6,
                atol=0.0,
            )
        )
    for eps, delta in [(0.1, 1e-5), (0.5, 1e-6), (0.9, 1e-3)]:
        scale = mechanisms.Gaussian(epsilon=eps, delta=delta, sensitivity=2.0)._scale
        out.append(
            case(f"classic/{eps}", "classic", [], {"scale": np.array(scale)}, epsilon=eps, delta=delta, sensitivity=2.0)
        )
    utility = [0.5, 2.0, 1.0, 3.5]
    for monotonic in [False, True]:
        cdf = mechanisms.Exponential(epsilon=1.2, sensitivity=1.5, utility=utility, monotonic=monotonic)._probabilities
        out.append(
            case(
                f"exponential/{monotonic}",
                "exponential",
                [np.array(utility)],
                {"probabilities": np.diff(np.concatenate([[0.0], cdf]))},
                epsilon=1.2,
                sensitivity=1.5,
                monotonic=monotonic,
                rtol=1e-12,
                atol=1e-14,
            )
        )
    spends = [(0.5, 1e-6), (0.3, 0.0), (1.0, 1e-5), (0.2, 0.0)] * 5
    for slack in [0.0, 1e-5, 1e-3]:
        acc = accountant.BudgetAccountant(epsilon=float("inf"), delta=1.0, slack=slack)
        total = acc.total(spent_budget=spends, slack=slack)
        out.append(
            case(
                f"composition/{slack}",
                "composition",
                [np.array([e for e, _ in spends]), np.array([d for _, d in spends])],
                {"epsilon": np.array(total[0]), "delta": np.array(total[1])},
                slack=slack,
                rtol=1e-12,
                atol=1e-15,
            )
        )
    for q, sigma, steps in [
        (0.01, 1.1, 1000),
        (0.05, 0.8, 200),
        (0.002, 2.0, 5000),
        (1.0, 1.5, 10),
        # The standard settings: Abadi et al. 2016's MNIST (lot 600 of 60000, sigma 4, 10000 steps), the TensorFlow
        # Privacy MNIST tutorial (batch 256, sigma 1.1, 60 epochs) and Opacus's MNIST example (rate 0.004, sigma 1).
        (0.01, 4.0, 10000),
        (256 / 60000, 1.1, 14062),
        (0.004, 1.0, 2500),
    ]:
        rdp = np.asarray(compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=ORDERS))
        eps, order = get_privacy_spent(orders=ORDERS, rdp=rdp.tolist(), delta=1e-5)
        out.append(
            case(
                f"rdp/{q}/{sigma}/{steps}",
                "rdp",
                [],
                {"rdp": rdp, "epsilon": np.array(eps), "order": np.array(float(order))},
                q=q,
                sigma=sigma,
                steps=steps,
                # The fractional-order series is truncated at log terms below -30 and uses log_ndtr: about 1e-9.
                rtol=1e-8,
                atol=1e-12,
            )
        )
    return out
