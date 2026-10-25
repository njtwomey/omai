"""Policy-gradient objectives against torch (float64): PPO's clipped surrogate as stable-baselines3 writes it, the
KL-penalised surrogate with torch.distributions.kl_divergence for categorical and diagonal Gaussian policies, and
TRPO's natural-gradient step on the Fisher matrix of a softmax policy (torch.autograd.functional.hessian of the mean
KL), whose conjugate-gradient solution must match numpy.linalg.solve."""

import numpy as np
import torch
from mdml_oracle.fixture import Case, case
from mdml_oracle.helpers import FloatArray, rng
from mdml_oracle.torch_helpers import to_numpy, to_torch
from torch.distributions import Categorical, Independent, Normal, kl_divergence

REFERENCE: dict[str, str] = {"torch": torch.__version__, "numpy": np.__version__}


def clip_objective(log_prob: FloatArray, old: FloatArray, adv: FloatArray, clip_range: float) -> float:
    """The negative of stable-baselines3's PPO policy_loss."""
    ratio = torch.exp(to_torch(log_prob) - to_torch(old))
    a = to_torch(adv)
    loss = -torch.min(a * ratio, a * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
    return -float(loss.item())


def categorical_kl_case(g: np.random.Generator, beta: float) -> tuple[list[FloatArray], float, FloatArray]:
    old = g.normal(size=(8, 3))
    new = old + 0.3 * g.normal(size=(8, 3))
    actions = g.integers(0, 3, size=8).astype(np.float64)
    adv = g.normal(size=8)
    p_old, p_new = Categorical(logits=to_torch(old)), Categorical(logits=to_torch(new))
    a = to_torch(actions)
    kl = kl_divergence(p_old, p_new)
    ratio = torch.exp(p_new.log_prob(a) - p_old.log_prob(a))
    objective = (ratio * to_torch(adv)).mean() - beta * kl.mean()
    return [old, new, actions, adv], float(objective.item()), to_numpy(kl)


def gaussian_kl_case(g: np.random.Generator, beta: float) -> tuple[list[FloatArray], float, FloatArray]:
    mean_old, log_std_old = g.normal(size=(6, 2)), 0.3 * g.normal(size=(6, 2))
    mean_new, log_std_new = mean_old + 0.2 * g.normal(size=(6, 2)), log_std_old + 0.1 * g.normal(size=(6, 2))
    actions = g.normal(size=(6, 2))
    adv = g.normal(size=6)
    p_old = Independent(Normal(to_torch(mean_old), torch.exp(to_torch(log_std_old))), 1)
    p_new = Independent(Normal(to_torch(mean_new), torch.exp(to_torch(log_std_new))), 1)
    a = to_torch(actions)
    kl = kl_divergence(p_old, p_new)
    ratio = torch.exp(p_new.log_prob(a) - p_old.log_prob(a))
    objective = (ratio * to_torch(adv)).mean() - beta * kl.mean()
    return [mean_old, log_std_old, mean_new, log_std_new, actions, adv], float(objective.item()), to_numpy(kl)


def fisher_and_gradient(g: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """The Fisher matrix (Hessian of the mean KL at the old parameters) and the surrogate gradient of a linear softmax
    policy with logits x @ theta, from torch autograd."""
    x = to_torch(g.normal(size=(10, 2)))
    theta0 = to_torch(0.5 * g.normal(size=(2, 3)))
    actions = torch.tensor(g.integers(0, 3, size=10))
    adv = to_torch(g.normal(size=10))
    old = Categorical(logits=x @ theta0)

    def mean_kl(flat: torch.Tensor) -> torch.Tensor:
        return kl_divergence(old, Categorical(logits=x @ flat.reshape(2, 3))).mean()

    def surrogate(flat: torch.Tensor) -> torch.Tensor:
        new = Categorical(logits=x @ flat.reshape(2, 3))
        return (torch.exp(new.log_prob(actions) - old.log_prob(actions)) * adv).mean()

    flat0 = theta0.reshape(-1).clone().requires_grad_(True)
    fisher = torch.autograd.functional.hessian(mean_kl, flat0)
    gradient = torch.autograd.grad(surrogate(flat0), flat0)[0]
    return to_numpy(fisher), to_numpy(gradient)


def cases() -> list[Case]:
    g = rng(91)
    out: list[Case] = []
    for i, clip_range in enumerate((0.1, 0.2, 0.3)):
        old = g.normal(size=16)
        log_prob = old + 0.3 * g.normal(size=16)
        adv = g.normal(size=16)
        out.append(
            case(
                f"ppoClip/{i}",
                "ppoClipObjective",
                [log_prob, old, adv],
                np.asarray(clip_objective(log_prob, old, adv, clip_range)),
                clipRange=clip_range,
            )
        )
    for i, beta in enumerate((0.5, 2.0)):
        inputs, objective, kl = categorical_kl_case(g, beta)
        out.append(case(f"ppoKl/categorical/{i}", "ppoKlCategorical", inputs, np.asarray(objective), beta=beta))
        out.append(case(f"kl/categorical/{i}", "klCategorical", inputs[:2], kl))
        inputs, objective, kl = gaussian_kl_case(g, beta)
        out.append(case(f"ppoKl/gaussian/{i}", "ppoKlGaussian", inputs, np.asarray(objective), beta=beta))
        out.append(case(f"kl/gaussian/{i}", "klGaussian", inputs[:4], kl))
    for i, (max_kl, damping) in enumerate(((0.01, 0.1), (0.05, 0.01))):
        fisher, gradient = fisher_and_gradient(g)
        a = fisher + damping * np.eye(len(gradient))
        direction = np.linalg.solve(a, gradient)
        step = np.sqrt(2 * max_kl / float(direction @ a @ direction)) * direction
        inputs = [fisher, gradient]
        out.append(
            case(f"trpo/{i}/direction", "trpoDirection", inputs, direction, atol=1e-10, maxKl=max_kl, damping=damping)
        )
        out.append(
            case(f"trpo/{i}/step", "trpoStep", inputs, step, atol=1e-10, rtol=1e-10, maxKl=max_kl, damping=damping)
        )
    return out
