"""Linear-chain CRFs against pytorch-crf and sklearn-crfsuite.

pytorch-crf (with zero start and end transitions, which CRFsuite does not have) gives exact log-likelihoods and Viterbi
paths for random emission and transition scores; the marginals are the one-hot labels minus the gradient of the
log-likelihood with respect to the emissions (torch autograd), since log Z is the score minus the log-likelihood.

sklearn-crfsuite trains a CRF by L-BFGS with an L2 penalty on attribute dictionaries built from feature templates
(the current word, the previous and next words, suffixes and a bias), with every state and transition feature, to
a tight tolerance; the cases keep its weights (which CRFsuite reports to 6 decimals), its marginals and its
predictions."""

from importlib.metadata import version
from typing import cast

import numpy as np
import sklearn_crfsuite
import torch
from torchcrf import CRF
from whitebox_oracle.fixture import Case, case
from whitebox_oracle.helpers import rng

REFERENCE: dict[str, str] = {"pytorch-crf": version("pytorch-crf"), "sklearn-crfsuite": version("sklearn-crfsuite")}

SENTENCES: list[list[tuple[str, str]]] = [
    [("the", "D"), ("dog", "N"), ("runs", "V")],
    [("a", "D"), ("cat", "N"), ("sleeps", "V"), ("now", "A")],
    [("the", "D"), ("cat", "N"), ("runs", "V"), ("fast", "A")],
    [("dogs", "N"), ("run", "V")],
    [("a", "D"), ("big", "J"), ("dog", "N"), ("barks", "V")],
    [("the", "D"), ("small", "J"), ("cat", "N"), ("sleeps", "V"), ("soundly", "A")],
    [("cats", "N"), ("sleep", "V"), ("now", "A")],
]


def item_features(words: list[str], t: int) -> dict[str, float]:
    """Templates: bias, w[0], w[-1]|w[0], w[1] and the 2-letter suffix (CRFsuite-style attribute names)."""
    w = words[t]
    out = {"bias": 1.0, f"w[0]={w}": 1.0, f"suf2={w[-2:]}": 1.0}
    if t > 0:
        out[f"w[-1]|w[0]={words[t - 1]}|{w}"] = 1.0
    if t + 1 < len(words):
        out[f"w[1]={words[t + 1]}"] = 1.0
    return out


def cases() -> list[Case]:
    out: list[Case] = []
    g = rng(61)
    for t, (n, k) in enumerate([(5, 3), (7, 4), (1, 2), (9, 2)]):
        emissions = np.round(g.normal(size=(n, k)), 6)
        transitions = np.round(g.normal(size=(k, k)), 6)
        tags = g.integers(0, k, size=n)
        model = CRF(k, batch_first=True).double()
        with torch.no_grad():
            model.transitions.copy_(torch.tensor(transitions))
            model.start_transitions.zero_()
            model.end_transitions.zero_()
        e = torch.tensor(emissions[None, :, :], requires_grad=True)
        y = torch.tensor(tags[None, :])
        ll = model(e, y, reduction="sum")
        (grad,) = torch.autograd.grad(ll, e)
        onehot = np.eye(k)[tags]
        marginals = onehot - grad.detach().numpy()[0]
        path = model.decode(e.detach())[0]
        out.append(
            case(
                f"exact/{t}",
                "exact",
                [emissions, transitions, tags],
                np.concatenate([[float(ll.item())], marginals.ravel()]),
                rtol=1e-10,
                atol=1e-12,
                viterbi=[int(s) for s in path],
            )
        )
    X = [[item_features([w for w, _ in s], t) for t in range(len(s))] for s in SENTENCES]
    Y = [[tag for _, tag in s] for s in SENTENCES]
    for c2 in [0.1, 1.0]:
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.0,
            c2=c2,
            max_iterations=10000,
            epsilon=1e-14,
            delta=1e-14,
            period=50,
            all_possible_transitions=True,
            all_possible_states=True,
        )
        crf.fit(X, Y)
        state_features = cast(dict[tuple[str, str], float], crf.state_features_)
        transition_features = cast(dict[tuple[str, str], float], crf.transition_features_)
        classes = cast(list[str], crf.classes_)
        state = {f"{a}|{label}": float(w) for (a, label), w in state_features.items()}
        transition = {f"{a}|{b}": float(w) for (a, b), w in transition_features.items()}
        predicted = cast(list[list[dict[str, float]]], crf.predict_marginals(X))
        marginals = [[[float(m[label]) for label in classes] for m in seq] for seq in predicted]
        test = [[item_features(["the", "big", "cat", "sleeps"], t) for t in range(4)]]
        out.append(
            case(
                f"crfsuite/c2={c2}",
                "crfsuite",
                [],
                np.array(0.0),
                x=X,
                y=Y,
                c2=c2,
                labels=[str(c) for c in classes],
                state=state,
                transition=transition,
                marginals=marginals,
                test=test,
                prediction=[str(p) for p in cast(list[list[str]], crf.predict(test))[0]],
            )
        )
    return out
