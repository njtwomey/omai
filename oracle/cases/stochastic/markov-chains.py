"""Discrete-time Markov chains against numpy, scipy and NetworkX: stationary laws (the unit left eigenvector of
scipy.linalg.eig), n-step laws (numpy.linalg.matrix_power), absorption (numpy.linalg.inv of I - Q), hitting
probabilities and times (numpy.linalg.solve), spectra (numpy.linalg.eigvals), mixing times (matrix powers),
time reversals, and communicating and closed classes (NetworkX strongly connected and attracting components)."""

import networkx as nx
import numpy as np
import scipy
import scipy.linalg as sla
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import FloatArray, rng

REFERENCE: dict[str, str] = {"numpy": np.__version__, "scipy": scipy.__version__, "networkx": nx.__version__}


def random_chain(g: np.random.Generator, n: int, sparsity: float = 0.0) -> FloatArray:
    """A random stochastic matrix; with sparsity, some entries are zeroed (keeping the diagonal)."""
    m = g.uniform(0.1, 1.0, size=(n, n))
    if sparsity > 0:
        m = m * (g.uniform(size=(n, n)) > sparsity) + np.eye(n) * 0.2
    return m / m.sum(axis=1, keepdims=True)


def stationary(p: FloatArray) -> FloatArray:
    w, v = sla.eig(p.T)
    k = int(np.argmin(np.abs(w - 1)))
    pi = np.real(v[:, k])
    return pi / pi.sum()


def reversible_chain(g: np.random.Generator, n: int) -> FloatArray:
    """A random walk on a weighted undirected graph (always reversible)."""
    w = g.uniform(0.1, 1.0, size=(n, n))
    w = w + w.T
    return w / w.sum(axis=1, keepdims=True)


def mixing(p: FloatArray, eps: float) -> int:
    pi = stationary(p)
    m = np.eye(p.shape[0])
    for t in range(10000):
        if np.max(0.5 * np.abs(m - pi).sum(axis=1)) <= eps:
            return t
        m = m @ p
    return -1


def cases() -> list[Case]:
    g = rng(808)
    out: list[Case] = []
    for n in [3, 5]:
        for sparsity in [0.0, 0.4]:
            p = random_chain(g, n, sparsity)
            tag = f"{n}/{sparsity}"
            out.append(case(f"stationary/{tag}", "stationary", [p], stationary(p), rtol=1e-10, atol=1e-12))
            p0 = np.eye(n)[0]
            for k in [1, 7]:
                law = p0 @ np.linalg.matrix_power(p, k)
                out.append(case(f"after/{tag}/{k}", "after", [p, p0], law, rtol=1e-12, atol=1e-14, steps=k))
            moduli = np.sort(np.abs(np.linalg.eigvals(p)))[::-1]
            out.append(case(f"moduli/{tag}", "moduli", [p], moduli, rtol=1e-9, atol=1e-12))
            pi = stationary(p)
            reversal = (p.T * pi[None, :]) / pi[:, None]
            out.append(case(f"reversal/{tag}", "reversal", [p], reversal, rtol=1e-10, atol=1e-12))
            for eps in [0.25, 0.01]:
                out.append(case(f"mixing/{tag}/{eps}", "mixing", [p], np.array(float(mixing(p, eps))), epsilon=eps))
            out.append(
                case(f"hitting-times/{tag}", "hitting-times", [p], hitting_times(p, [0]), rtol=1e-10, target=[0])
            )
    for n in [4, 6]:
        p = reversible_chain(g, n)
        w = np.sort(np.real(np.linalg.eigvals(p)))[::-1]
        out.append(case(f"gap/{n}", "gap", [p], np.array(1 - w[1]), rtol=1e-9))
    # An absorbing chain: two absorbing states and four transient ones.
    p = np.zeros((6, 6))
    p[0, 0] = p[5, 5] = 1
    inner = g.uniform(0.1, 1, size=(4, 6))
    p[1:5] = inner / inner.sum(axis=1, keepdims=True)
    transient, absorbing = [1, 2, 3, 4], [0, 5]
    q = p[np.ix_(transient, transient)]
    r = p[np.ix_(transient, absorbing)]
    fundamental = np.linalg.inv(np.eye(4) - q)
    steps = fundamental @ np.ones(4)
    out.append(case("absorption/fundamental", "absorption", [p], fundamental, rtol=1e-10, part="fundamental"))
    out.append(case("absorption/probabilities", "absorption", [p], fundamental @ r, rtol=1e-10, part="probabilities"))
    out.append(case("absorption/steps", "absorption", [p], steps, rtol=1e-10, part="expectedSteps"))
    variance = (2 * fundamental - np.eye(4)) @ steps - steps**2
    out.append(case("absorption/variance", "absorption", [p], variance, rtol=1e-9, part="varianceSteps"))
    out.append(
        case(
            "hitting-probabilities", "hitting-probabilities", [p], hitting_probabilities(p, [5]), rtol=1e-10, target=[5]
        )
    )
    # Classification against NetworkX: a chain with two closed classes and transient states.
    c = np.array(
        [
            [0.5, 0.5, 0, 0, 0, 0],
            [0.3, 0.7, 0, 0, 0, 0],
            [0.2, 0, 0.3, 0.5, 0, 0],
            [0, 0, 0.4, 0.2, 0, 0.4],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    graph = nx.DiGraph([(i, j) for i in range(6) for j in range(6) if c[i, j] > 0])
    classes = sorted(sorted(int(v) for v in comp) for comp in nx.strongly_connected_components(graph))
    closed = sorted(sorted(int(v) for v in comp) for comp in nx.attracting_components(graph))
    labels = np.zeros(6)
    for k, comp in enumerate(classes):
        for v in comp:
            labels[v] = k
    recurrent = np.array([float(any(v in comp for comp in closed)) for v in range(6)])
    out.append(case("classes/labels", "class-of", [c], labels))
    out.append(case("classes/recurrent", "recurrent", [c], recurrent))
    return out


def hitting_probabilities(p: FloatArray, target: list[int]) -> FloatArray:
    """h = 1 on the target, h = P h on the states that can reach it, 0 elsewhere (numpy.linalg.solve)."""
    n = p.shape[0]
    graph = nx.DiGraph([(i, j) for i in range(n) for j in range(n) if p[i, j] > 0])
    graph.add_nodes_from(range(n))
    can = set(target)
    for t in target:
        can |= set(nx.ancestors(graph, t))
    free = [i for i in range(n) if i in can and i not in target]
    h = np.zeros(n)
    h[target] = 1
    if free:
        a = np.eye(len(free)) - p[np.ix_(free, free)]
        b = p[np.ix_(free, target)].sum(axis=1)
        h[free] = np.linalg.solve(a, b)
    return h


def hitting_times(p: FloatArray, target: list[int]) -> FloatArray:
    """k = 0 on the target and (I - P_UU) k = 1 on the rest (an irreducible chain hits surely)."""
    n = p.shape[0]
    rest = [i for i in range(n) if i not in target]
    k = np.zeros(n)
    k[rest] = np.linalg.solve(np.eye(len(rest)) - p[np.ix_(rest, rest)], np.ones(len(rest)))
    return k
