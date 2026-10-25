"""Parameter importance against Optuna's PED-ANOVA evaluator (`PedAnovaImportanceEvaluator`), which is deterministic
given the trials: Scott-rule Parzen estimators on a 50-cell grid per parameter, compared by Pearson divergence between
the top-quantile and region trials. Histories cover reals, log reals, integers, a stepped real and an option, a
conditional parameter (absent from some trials), a maximised objective and a local region."""

import math
import warnings
from typing import Literal

import optuna
from mdml_oracle.fixture import Case, Json, case
from mdml_oracle.helpers import rng
from optuna.distributions import BaseDistribution, CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.importance import PedAnovaImportanceEvaluator
from optuna.trial import FrozenTrial, create_trial

REFERENCE: dict[str, str] = {"optuna": optuna.__version__}

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

optuna.logging.set_verbosity(optuna.logging.ERROR)

SPACE: dict[str, BaseDistribution] = {
    "x": FloatDistribution(-5.0, 5.0),
    "lr": FloatDistribution(1e-4, 1.0, log=True),
    "k": IntDistribution(1, 20),
    "kl": IntDistribution(1, 512, log=True),
    "s": FloatDistribution(0.0, 5.0, step=0.25),
    "act": CategoricalDistribution(["a", "b", "c"]),
}
DEPTH = IntDistribution(1, 8)


def history(seed: int, n: int, conditional: bool) -> list[FrozenTrial]:
    g = rng(seed)
    out: list[FrozenTrial] = []
    for _ in range(n):
        p: dict[str, float | int | str] = {
            "x": float(g.uniform(-5, 5)),
            "lr": float(math.exp(g.uniform(math.log(1e-4), 0.0))),
            "k": int(g.integers(1, 21)),
            "kl": int(min(512, math.floor(math.exp(g.uniform(0.0, math.log(513.0)))))),
            "s": float(0.25 * g.integers(0, 21)),
            "act": ["a", "b", "c"][int(g.integers(0, 3))],
        }
        space = dict(SPACE)
        if conditional and p["act"] == "c":
            p["depth"] = int(g.integers(1, 9))
            space["depth"] = DEPTH
        value = (float(p["x"]) - 1) ** 2 + 0.5 * math.log10(float(p["lr"]) / 0.01) ** 2 + (0 if p["act"] == "b" else 1)
        value += 0.05 * int(p.get("depth", 0))
        out.append(create_trial(params=p, distributions=space, value=value))
    return out


def trial_json(t: FrozenTrial) -> dict[str, Json]:
    return {
        "number": t.number,
        "params": {k: v for k, v in t.params.items()},
        "state": "complete",
        "values": [float(v) for v in t.values],
        "intermediate": [],
    }


def ped_case(
    name: str,
    seed: int,
    n: int,
    direction: Literal["minimize", "maximize"],
    conditional: bool,
    target: float = 0.1,
    region: float = 1.0,
    local: bool = True,
) -> Case:
    trials = history(seed, n, conditional)
    if direction == "maximize":
        trials = [
            create_trial(params=t.params, distributions=t.distributions, value=-float(t.values[0])) for t in trials
        ]
    study = optuna.create_study(direction=direction)
    study.add_trials(trials)
    evaluator = PedAnovaImportanceEvaluator(target_quantile=target, region_quantile=region, evaluate_on_local=local)
    importances = optuna.importance.get_param_importances(study, evaluator=evaluator, normalize=False)
    out: dict[str, Json] = {k: float(v) for k, v in importances.items()}
    return case(
        name,
        "pedAnova",
        [[trial_json(t) for t in study.trials]],
        out,
        rtol=1e-9,
        atol=1e-12,
        direction=direction,
        conditional=conditional,
        targetQuantile=target,
        regionQuantile=region,
        evaluateOnLocal=local,
    )


def cases() -> list[Case]:
    return [
        ped_case("pedAnova/default", 0, 60, "minimize", False),
        ped_case("pedAnova/maximize", 1, 60, "maximize", False),
        ped_case("pedAnova/conditional", 2, 80, "minimize", True),
        ped_case("pedAnova/region", 3, 100, "minimize", False, target=0.1, region=0.5),
        ped_case("pedAnova/global", 4, 50, "minimize", False, target=0.2, local=False),
    ]
