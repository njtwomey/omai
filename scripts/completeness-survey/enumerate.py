"""Enumerate the public API of the reference libraries that are installed, one JSON file per library.

Run from the repository root with the oracle environment (read only; nothing is installed or downloaded):

    mise exec -- uv run --group oracle-core --group oracle-torch --group oracle-prob --group oracle-rl \
        python scripts/completeness-survey/enumerate.py [library ...]

With library names it enumerates only those (the second pass added the libraries from `opacus` on). Without
arguments it redoes everything.

The covariance family (third pass) also reads libraries that are NOT installed, from their source: download each with
`pip download --no-deps <name>` (a wheel, or an sdist for skggm) into a scratch directory, unpack every archive into its
own folder (`<dir>/pyriemann/pyriemann`, `<dir>/skggm/skggm-0.2.8/inverse_covariance`, ...) and point `OMAI_SURVEY_SRC`
at `<dir>`. Nothing is imported or installed: `src_api` reads `__all__` and the top-level definitions with `ast`.
Without `OMAI_SURVEY_SRC` those jobs are skipped.

Output: docs/completeness/data/ref-<library>.json, `{library, version, source, entries: [{area, name, kind, doc}]}`.
Libraries that are not installed are skipped here and come from `memory.json` (written by hand from knowledge of
their public API) in `match.ts`. Modules are probed through `importlib` and `getattr`, so the libraries need no
type information.
"""

# ruff: noqa: SIM905
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import pkgutil
import sys
import warnings
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[2] / "docs" / "completeness" / "data"

Rec = dict[str, str]


def mod(name: str) -> Any:  # noqa: ANN401
    return importlib.import_module(name)


def first_line(obj: object) -> str:
    for line in (inspect.getdoc(obj) or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:110]
    return ""


def kind_of(obj: object) -> str:
    if inspect.isclass(obj):
        return "class"
    return "function" if callable(obj) else "object"


def rec(area: str, name: str, obj: object, kind: str | None = None) -> Rec:
    return {"area": area, "name": name, "kind": kind or kind_of(obj), "doc": first_line(obj)}


def from_all(modname: str, area: str | None = None, keep: set[str] | None = None) -> list[Rec]:
    """Public callables of a module (`__all__` if it has one), defined by the same top-level package."""
    try:
        m = mod(modname)
    except Exception as exc:  # noqa: BLE001
        print(f"skip {modname}: {exc}", file=sys.stderr)
        return []
    names: list[str] = list(getattr(m, "__all__", None) or [n for n in dir(m) if not n.startswith("_")])
    root = modname.split(".")[0]
    out: list[Rec] = []
    for n in names:
        if keep is not None and n not in keep:
            continue
        try:
            obj = getattr(m, n, None)
        except Exception:  # noqa: BLE001
            continue
        if obj is None or inspect.ismodule(obj) or not callable(obj):
            continue
        owner = str(getattr(obj, "__module__", "") or "")
        if not hasattr(m, "__all__") and not owner.startswith(modname):
            continue  # no __all__: keep only what the module defines itself, not what it imports
        if not owner.startswith(root) and root not in ("numpyro", "arviz", "gymnasium", "torch", "arviz_base"):
            continue
        out.append(rec(area or modname, n, obj))
    return out


def sort_key(r: Rec) -> tuple[str, str]:
    return (r["area"], r["name"])


def dump(lib: str, recs: list[Rec], dist: str | None = None, ver: str | None = None) -> None:
    if ver is None:
        try:
            ver = version(dist or lib)
        except PackageNotFoundError:
            ver = "?"
    seen: set[tuple[str, str]] = set()
    uniq: list[Rec] = []
    for r in recs:
        k = (r["area"], r["name"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    uniq.sort(key=sort_key)
    text = json.dumps({"library": lib, "version": ver, "source": "introspection", "entries": uniq}, indent=0)
    (OUT / f"ref-{lib}.json").write_text(text + "\n")
    print(f"{lib} {ver}: {len(uniq)}")


def modules(prefix: str, names: Sequence[str]) -> list[Rec]:
    recs: list[Rec] = []
    for n in names:
        recs += from_all(f"{prefix}.{n}", f"{prefix}.{n}")
    return recs


def src_root() -> Path | None:
    value = os.environ.get("OMAI_SURVEY_SRC")
    return Path(value) if value else None


def src_version(folder: Path) -> str:
    """The version of an unpacked wheel (`*.dist-info/METADATA`) or sdist (`PKG-INFO`)."""
    for meta in [*folder.glob("*.dist-info/METADATA"), *folder.glob("*/PKG-INFO"), *folder.glob("PKG-INFO")]:
        for line in meta.read_text(encoding="utf8", errors="replace").splitlines():
            if line.startswith("Version:"):
                return line.split(":", 1)[1].strip()
    return "?"


def src_api(folder: str, package_dir: str, targets: Sequence[str]) -> tuple[list[Rec], str]:
    """Public names of a library that is not installed, read from its unpacked source with `ast`.

    `folder` is the unpacked archive under `OMAI_SURVEY_SRC`, `package_dir` the package inside it (relative), `targets`
    dotted sub-packages or modules to list (`skfolio.moments`). A target's area is its dotted name. Names come from the
    `__all__` of a package's `__init__.py`, or from the public top-level classes and functions of a module; the kind
    and first docstring line come from the definition wherever in the package it is.
    """
    root = src_root()
    if root is None:
        return [], "?"
    base = root / folder / package_dir
    defs: dict[str, tuple[str, str]] = {}
    trees: dict[Path, ast.Module] = {}
    for path in sorted(base.rglob("*.py")):
        if any(part in ("tests", "test") for part in path.relative_to(base).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf8"))
        except SyntaxError:
            continue
        trees[path] = tree
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                defs.setdefault(node.name, (kind, (doc[0] if doc else "")[:110]))
    recs: list[Rec] = []
    for target in targets:
        rel = Path(*target.split(".")[1:]) if "." in target else Path()
        init = base / rel / "__init__.py"
        module = base / rel.with_suffix(".py") if rel.parts else None
        names: list[str] = []
        if init.exists() and init in trees:
            for node in trees[init].body:
                is_all = isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
                )
                if is_all and isinstance(node, ast.Assign) and isinstance(node.value, (ast.List, ast.Tuple)):
                    names = [
                        e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]
        elif module is not None and module in trees:
            names = [
                n.name
                for n in trees[module].body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and not n.name.startswith("_")
            ]
        for name in names:
            kind, doc = defs.get(name, ("function", ""))
            recs.append({"area": target, "name": name, "kind": kind, "doc": doc})
    return recs, src_version(root / folder)


def sklearn() -> list[Rec]:
    mod("sklearn.experimental.enable_halving_search_cv")
    mod("sklearn.experimental.enable_iterative_imputer")
    names = (
        "base calibration cluster compose covariance cross_decomposition datasets decomposition "
        "discriminant_analysis dummy ensemble feature_extraction feature_extraction.text feature_extraction.image "
        "feature_selection gaussian_process gaussian_process.kernels impute inspection isotonic "
        "kernel_approximation kernel_ridge linear_model manifold metrics metrics.pairwise mixture model_selection "
        "multiclass multioutput naive_bayes neighbors neural_network pipeline preprocessing random_projection "
        "semi_supervised svm tree utils frozen"
    ).split()
    keep_utils = {
        "resample",
        "shuffle",
        "compute_class_weight",
        "compute_sample_weight",
        "gen_batches",
        "murmurhash3_32",
    }
    recs = modules("sklearn", names)
    return [r for r in recs if r["area"] != "sklearn.utils" or r["name"] in keep_utils]


def scipy() -> list[Rec]:
    names = (
        "stats optimize linalg signal interpolate integrate spatial spatial.distance spatial.transform special "
        "sparse sparse.linalg sparse.csgraph fft cluster.hierarchy cluster.vq ndimage stats.qmc stats.contingency "
        "signal.windows odr differentiate io"
    ).split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recs = modules("scipy", names)
        st = mod("scipy.stats")
        for n in dir(st):
            o = getattr(st, n)
            if isinstance(o, (st.rv_continuous, st.rv_discrete)):
                recs.append(rec("scipy.stats.distributions", n, o, "distribution"))
    return recs


def statsmodels() -> list[Rec]:
    names = (
        "api tsa.api stats.api stats.diagnostic stats.outliers_influence stats.multitest stats.proportion "
        "stats.power stats.weightstats stats.anova stats.contingency_tables stats.stattools "
        "stats.correlation_tools regression.linear_model regression.mixed_linear_model "
        "regression.quantile_regression regression.recursive_ls regression.rolling regression.dimred "
        "genmod.generalized_linear_model genmod.families genmod.families.links "
        "genmod.generalized_estimating_equations discrete.discrete_model discrete.count_model "
        "duration.hazard_regression duration.survfunc nonparametric.api tsa.statespace.sarimax "
        "tsa.statespace.structural tsa.statespace.dynamic_factor tsa.statespace.varmax tsa.holtwinters "
        "tsa.vector_ar.var_model tsa.vector_ar.vecm tsa.arima.model tsa.stattools tsa.seasonal "
        "tsa.forecasting.theta tsa.forecasting.stl tsa.regime_switching.markov_regression "
        "tsa.regime_switching.markov_autoregression tsa.ardl tsa.filters.hp_filter tsa.filters.bk_filter "
        "tsa.filters.cf_filter tsa.filters.filtertools tsa.arima_process tsa.ar_model "
        "tsa.exponential_smoothing.ets tsa.stl gam.api imputation.mice multivariate.pca multivariate.factor "
        "multivariate.manova multivariate.cancorr miscmodels.ordinal_model robust.robust_linear_model "
        "robust.norms othermod.betareg emplike.api sandbox.stats.multicomp treatment.treatment_effects "
        "distributions.empirical_distribution graphics.api"
    ).split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return modules("statsmodels", names)


def torch() -> list[Rec]:
    nn = mod("torch.nn")
    recs: list[Rec] = []
    for n in dir(nn):
        o = getattr(nn, n)
        if inspect.isclass(o) and issubclass(o, nn.Module) and not n.startswith("_") and n != "Module":
            recs.append(rec("torch.nn.modules", n, o, "class"))
    for m in (
        "torch.nn.functional torch.nn.init torch.nn.utils torch.nn.utils.rnn torch.nn.utils.parametrizations "
        "torch.nn.utils.parametrize"
    ).split():
        recs += from_all(m, m)
    opt = mod("torch.optim")
    for n in dir(opt):
        o = getattr(opt, n)
        if inspect.isclass(o) and issubclass(o, opt.Optimizer) and n != "Optimizer":
            recs.append(rec("torch.optim", n, o, "class"))
    sch = mod("torch.optim.lr_scheduler")
    for n in dir(sch):
        o = getattr(sch, n)
        is_sched = n.endswith(("LR", "Restarts", "Plateau"))
        if inspect.isclass(o) and not n.startswith("_") and n != "LRScheduler" and is_sched:
            recs.append(rec("torch.optim.lr_scheduler", n, o, "class"))
    swa = mod("torch.optim.swa_utils")
    recs += [rec("torch.optim.swa_utils", n, getattr(swa, n), "class") for n in ("AveragedModel", "SWALR", "update_bn")]
    dist = mod("torch.distributions")
    for n in dist.__all__:
        o = getattr(dist, n, None)
        if o is not None and not inspect.ismodule(o):
            recs.append(rec("torch.distributions", n, o))
    recs += from_all("torch.distributions.constraints") + from_all("torch.distributions.transforms")
    for m in (
        "torch.autograd torch.autograd.functional torch.func torch.utils.data torch.linalg torch.fft torch.special"
    ).split():
        recs += from_all(m, m)
    return recs


def torchvision() -> list[Rec]:
    recs: list[Rec] = []
    tv_t = mod("torchvision.transforms.v2")
    for n in dir(tv_t):
        o = getattr(tv_t, n)
        if inspect.isclass(o) and not n.startswith("_"):
            recs.append(rec("torchvision.transforms.v2", n, o, "class"))
    tv_d = mod("torchvision.datasets")
    for n in getattr(tv_d, "__all__", []):
        recs.append(rec("torchvision.datasets", n, getattr(tv_d, n), "dataset"))
    return recs


def torchmetrics() -> list[Rec]:
    tm = mod("torchmetrics")
    recs = [rec("torchmetrics", n, getattr(tm, n)) for n in tm.__all__ if not inspect.ismodule(getattr(tm, n, None))]
    return recs + from_all("torchmetrics.functional")


def numpyro() -> list[Rec]:
    names = (
        "numpyro numpyro.distributions numpyro.infer numpyro.infer.autoguide numpyro.infer.reparam numpyro.handlers "
        "numpyro.contrib.control_flow numpyro.optim numpyro.diagnostics numpyro.infer.util "
        "numpyro.distributions.transforms numpyro.distributions.constraints"
    ).split()
    recs: list[Rec] = []
    for m in names:
        recs += from_all(m, m)
    dd = mod("numpyro.distributions")
    for n in dir(dd):
        o = getattr(dd, n)
        if inspect.isclass(o) and not n.startswith("_") and issubclass(o, dd.Distribution):
            recs.append(rec("numpyro.distributions", n, o, "distribution"))
    return recs


def arviz() -> list[Rec]:
    recs: list[Rec] = []
    for m in ("arviz", "arviz_base", "arviz_stats", "arviz_plots"):
        recs += from_all(m, m)
    return recs


def pgmpy() -> list[Rec]:
    warnings.simplefilter("ignore")
    names = (
        "models factors factors.discrete factors.continuous factors.hybrid inference estimators sampling metrics "
        "base readwrite utils causal_discovery ci_tests"
    ).split()
    return modules("pgmpy", names)


def hmmlearn() -> list[Rec]:
    return from_all("hmmlearn.hmm") + from_all("hmmlearn.vhmm")


def gymnasium() -> list[Rec]:
    gym = mod("gymnasium")
    recs: list[Rec] = []
    for eid, spec in sorted(gym.registry.items()):
        ep = str(spec.entry_point)
        fam = "other"
        for key in ("mujoco", "box2d", "toy_text", "classic_control"):
            if key in ep:
                fam = key
        if ep.startswith("ale") or "ALE" in eid:
            fam = "atari"
        recs.append({"area": f"gymnasium.envs.{fam}", "name": eid, "kind": "environment", "doc": ep.split(":")[-1]})
    for m in (
        "gymnasium.wrappers",
        "gymnasium.wrappers.vector",
        "gymnasium.spaces",
        "gymnasium.vector",
        "gymnasium.utils",
    ):
        recs += from_all(m, m)
    return recs


def fairlearn() -> list[Rec]:
    return modules("fairlearn", "metrics reductions postprocessing preprocessing adversarial datasets".split())


def classes_of(m: Any, area: str) -> list[Rec]:  # noqa: ANN401
    found: list[Rec] = []
    for n in dir(m):
        o = getattr(m, n)
        if not n.startswith("_") and inspect.isclass(o):
            found.append(rec(area, n, o, "class"))
    return found


def shap() -> list[Rec]:
    s = mod("shap")
    recs: list[Rec] = []
    for n in dir(s):
        o = getattr(s, n)
        if not n.startswith("_") and (inspect.isclass(o) or inspect.isfunction(o)):
            recs.append(rec("shap", n, o))
    return (
        recs + classes_of(mod("shap.explainers"), "shap.explainers") + classes_of(mod("shap.maskers"), "shap.maskers")
    )


def captum() -> list[Rec]:
    recs: list[Rec] = []
    for m in ("captum.attr", "captum.metrics", "captum.robust", "captum.concept"):
        recs += from_all(m, m)
    return recs


def skimage() -> list[Rec]:
    names = (
        "color exposure feature filters measure morphology restoration segmentation transform util draw data "
        "metrics registration graph"
    ).split()
    return modules("skimage", names)


def nltk() -> list[Rec]:
    names = "tokenize stem tag parse chunk classify cluster metrics translate sentiment lm probability".split()
    return modules("nltk", names)


def gensim() -> list[Rec]:
    recs: list[Rec] = []
    for m in ("models", "corpora", "similarities", "matutils", "parsing.preprocessing", "topic_coherence"):
        try:
            mm = mod(f"gensim.{m}")
        except Exception:  # noqa: BLE001
            continue
        for n in getattr(mm, "__all__", None) or dir(mm):
            o = getattr(mm, n, None)
            if inspect.isclass(o) and str(getattr(o, "__module__", "")).startswith("gensim"):
                recs.append(rec(f"gensim.{m}", n, o, "class"))
    return recs


def networkx() -> list[Rec]:
    nx = mod("networkx")
    areas = (
        "centrality community components shortest_paths flow tree traversal matching coloring link_analysis cycles "
        "dag isomorphism clique bipartite similarity connectivity approximation link_prediction"
    ).split()
    recs: list[Rec] = []
    pairs = [(a, mod(f"networkx.algorithms.{a}")) for a in areas] + [("generators", mod("networkx.generators"))]
    pairs.append(("algorithms", nx))
    for area, m in pairs:
        for n in dir(m):
            o = getattr(m, n, None)
            if n.startswith("_") or not (inspect.isfunction(o) or inspect.isclass(o)):
                continue
            if area == "algorithms" and not str(getattr(o, "__module__", "")).startswith("networkx.algorithms"):
                continue
            recs.append(rec(f"networkx.{area}", n, o))
    return recs


def optuna() -> list[Rec]:
    return modules("optuna", "samplers pruners visualization importance terminator storages".split())


def imblearn() -> list[Rec]:
    return modules("imblearn", "over_sampling under_sampling combine ensemble metrics pipeline".split())


def opacus() -> list[Rec]:
    names = (
        "accountants optimizers grad_sample layers schedulers validators utils.batch_memory_manager data_loader".split()
    )
    recs = from_all("opacus", "opacus") + modules("opacus", names)
    return recs


def modal() -> list[Rec]:
    return modules(
        "modAL",
        "models acquisition uncertainty disagreement density batch expected_error multilabel utils.selection".split(),
    )


def gpytorch() -> list[Rec]:
    names = (
        "kernels likelihoods means mlls models variational priors constraints distributions lazy metrics optim "
        "settings utils functions"
    ).split()
    return modules("gpytorch", names)


def ast_scan(package: str, skip: Sequence[str] = ("tests", "test", "conftest")) -> list[Rec]:
    """Public classes and functions of a package read from its source files with `ast`, without importing it."""
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return []
    root = Path(spec.origin).parent
    recs: list[Rec] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in skip or part.startswith("_") and part != "__init__.py" for part in rel.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf8"))
        area = ".".join([package, *rel.with_suffix("").parts]).removesuffix(".__init__")
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
                doc = (ast.get_docstring(node) or "").strip().splitlines()
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                recs.append({"area": area, "name": node.name, "kind": kind, "doc": (doc[0] if doc else "")[:110]})
    return recs


def skorch() -> list[Rec]:
    return ast_scan("skorch")


def lime() -> list[Rec]:
    names = "lime_tabular lime_text lime_image lime_base submodular_pick discretize explanation".split()
    return (
        classes_of(mod("lime.lime_tabular"), "lime.lime_tabular")
        + classes_of(mod("lime.lime_text"), "lime.lime_text")
        + modules("lime", names)
    )


def librosa() -> list[Rec]:
    names = "core feature effects onset beat segment decompose filters sequence util".split()
    recs = from_all("librosa", "librosa")
    for n in names:
        recs += from_all(f"librosa.{n}", f"librosa.{n}")
    return recs


def control() -> list[Rec]:
    return from_all("control", "control")


def filterpy() -> list[Rec]:
    recs: list[Rec] = []
    for m in ("kalman", "monte_carlo", "hinfinity", "leastsq", "memory", "discrete_bayes", "stats", "gh", "common"):
        recs += classes_of(mod(f"filterpy.{m}"), f"filterpy.{m}")
        for n in dir(mod(f"filterpy.{m}")):
            o = getattr(mod(f"filterpy.{m}"), n)
            if inspect.isfunction(o) and not n.startswith("_") and str(o.__module__).startswith("filterpy"):
                recs.append(rec(f"filterpy.{m}", n, o))
    return recs


def pymoo() -> list[Rec]:
    recs: list[Rec] = []
    for m in (
        "algorithms.moo",
        "algorithms.soo.nonconvex",
        "algorithms.soo.convex",
        "operators.crossover",
        "operators.mutation",
        "operators.selection",
        "operators.sampling",
        "indicators",
        "decomposition",
        "mcdm",
        "termination",
        "visualization",
        "constraints",
        "gradient",
        "problems.single",
        "problems.multi",
        "problems.many",
        "problems.constr",
    ):
        try:
            mm = mod(f"pymoo.{m}")
        except Exception:  # noqa: BLE001
            continue
        for sub in sorted(x.name for x in pkgutil.iter_modules(getattr(mm, "__path__", []))):
            try:
                recs += classes_of(mod(f"pymoo.{m}.{sub}"), f"pymoo.{m}")
            except Exception:  # noqa: BLE001
                continue
        recs += classes_of(mm, f"pymoo.{m}")
    return [r for r in recs if r["name"][0].isupper() or r["area"].startswith("pymoo.mcdm")]


def cma() -> list[Rec]:
    return (
        classes_of(mod("cma"), "cma")
        + from_all("cma", "cma")
        + from_all("cma.fitness_functions", "cma.fitness_functions")
    )


def causallearn() -> list[Rec]:
    recs: list[Rec] = []
    for m in (
        "search.ConstraintBased.PC",
        "search.ConstraintBased.FCI",
        "search.ScoreBased.GES",
        "search.ScoreBased.ExactSearch",
        "search.ScoreBased.GRaSP",
        "search.FCMBased.lingam",
        "search.FCMBased.ANM.ANM",
        "search.FCMBased.PNL.PNL",
        "search.PermutationBased.GRaSP",
        "search.HiddenCausal.GIN.GIN",
        "search.Granger.Granger",
        "utils.cit",
        "utils.KCI.KCI",
        "utils.GraphUtils",
        "utils.PCUtils.BackgroundKnowledge",
        "utils.DAG2CPDAG",
        "score.LocalScoreFunction",
        "graph.GeneralGraph",
    ):
        try:
            mm = mod(f"causallearn.{m}")
        except Exception:  # noqa: BLE001
            continue
        for n in dir(mm):
            o = getattr(mm, n)
            if (
                not n.startswith("_")
                and (inspect.isclass(o) or inspect.isfunction(o))
                and str(getattr(o, "__module__", "")).startswith("causallearn")
            ):
                recs.append(rec(f"causallearn.{m}", n, o))
    return recs


def mapie() -> list[Rec]:
    names = (
        "regression classification calibration risk_control metrics.regression metrics.classification "
        "metrics.calibration conformity_scores subsample exchangeability_testing"
    ).split()
    recs: list[Rec] = []
    for n in names:
        try:
            mm = mod(f"mapie.{n}")
        except Exception:  # noqa: BLE001
            continue
        recs += classes_of(mm, f"mapie.{n}")
        for k in dir(mm):
            o = getattr(mm, k)
            if inspect.isfunction(o) and not k.startswith("_") and str(o.__module__).startswith("mapie"):
                recs.append(rec(f"mapie.{n}", k, o))
    return recs


def pot() -> list[Rec]:
    names = (
        "lp bregman da gromov sliced unbalanced smooth partial dr gaussian coot stochastic factored lowrank mapping "
        "solvers regpath weak gmm gnn datasets batch"
    ).split()
    recs = from_all("ot", "ot")
    for n in names:
        recs += from_all(f"ot.{n}", f"ot.{n}")
    return recs


def optax() -> list[Rec]:
    recs = (
        from_all("optax", "optax")
        + from_all("optax.losses", "optax.losses")
        + from_all("optax.projections", "optax.projections")
    )
    recs += from_all("optax.contrib", "optax.contrib") + from_all("optax.second_order", "optax.second_order")
    recs += from_all("optax.schedules", "optax.schedules") + from_all("optax.tree_utils", "optax.tree_utils")
    return recs


def blackjax() -> list[Rec]:
    bj = mod("blackjax")
    recs: list[Rec] = []
    for n in dir(bj):
        o = getattr(bj, n)
        if not n.startswith("_") and not inspect.ismodule(o) and (callable(o) or hasattr(o, "init")):
            recs.append(rec("blackjax", n, o, "algorithm"))
    for m in ("adaptation", "diagnostics", "optimizers", "vi", "smc", "sgmcmc", "mcmc"):
        recs += from_all(f"blackjax.{m}", f"blackjax.{m}")
    return recs


def stumpy() -> list[Rec]:
    return from_all("stumpy", "stumpy")


def ruptures() -> list[Rec]:
    recs = classes_of(mod("ruptures"), "ruptures") + from_all("ruptures", "ruptures")
    for m in ("detection", "costs", "metrics", "datasets", "show"):
        recs += classes_of(mod(f"ruptures.{m}"), f"ruptures.{m}")
        recs += from_all(f"ruptures.{m}", f"ruptures.{m}")
    return recs


def pysindy() -> list[Rec]:
    recs = classes_of(mod("pysindy"), "pysindy")
    for m in ("optimizers", "feature_library", "differentiation"):
        recs += classes_of(mod(f"pysindy.{m}"), f"pysindy.{m}")
    return recs


def shapely() -> list[Rec]:
    recs = from_all("shapely", "shapely")
    for m in ("ops", "affinity", "strtree", "prepared", "validation", "wkt", "wkb", "geometry", "plotting"):
        recs += from_all(f"shapely.{m}", f"shapely.{m}")
    return [r for r in recs if r["name"] not in ("ShapelyDeprecationWarning",)]


def cvxpy() -> list[Rec]:
    cp = mod("cvxpy")
    recs: list[Rec] = []
    for n in dir(cp):
        o = getattr(cp, n)
        if (
            not n.startswith("_")
            and (inspect.isclass(o) or inspect.isfunction(o))
            and str(getattr(o, "__module__", "")).startswith("cvxpy")
        ):
            recs.append(rec("cvxpy", n, o))
    return recs


def nashpy() -> list[Rec]:
    return ast_scan("nashpy")


def pingouin() -> list[Rec]:
    return from_all("pingouin", "pingouin")


def torchvision_models() -> list[Rec]:
    tvm = mod("torchvision.models")
    return [{"area": "torchvision.models", "name": n, "kind": "model", "doc": ""} for n in tvm.list_models()]


def crf_libraries() -> list[Rec]:
    recs: list[Rec] = []
    for m in ("sklearn_crfsuite", "sklearn_crfsuite.metrics", "sklearn_crfsuite.utils", "pycrfsuite", "torchcrf"):
        recs += classes_of(mod(m), m) + from_all(m, m)
    # `CRF` is the name of the class in two packages; keep both by qualifying the second one.
    return [{**r, "name": "torchcrf.CRF"} if r["area"] == "torchcrf" else r for r in recs]


def sympy() -> list[Rec]:
    sp = mod("sympy")
    recs: list[Rec] = []
    for n in dir(sp):
        o = getattr(sp, n)
        owner = str(getattr(o, "__module__", ""))
        if n.startswith("_") or not (inspect.isclass(o) or inspect.isfunction(o)) or not owner.startswith("sympy."):
            continue
        area = ".".join(owner.split(".")[:2])
        recs.append(rec(area, n, o))
    return recs


def openspiel() -> list[Rec]:
    ps = mod("pyspiel")
    recs = [{"area": "pyspiel.games", "name": n, "kind": "game", "doc": ""} for n in ps.registered_names()]
    for m in ("open_spiel.python.algorithms", "open_spiel.python.egt"):
        try:
            mm = mod(m)
        except Exception:  # noqa: BLE001
            continue
        for sub in sorted(x.name for x in pkgutil.iter_modules(getattr(mm, "__path__", []))):
            recs.append({"area": m, "name": sub, "kind": "module", "doc": ""})
    return recs


def datasketch() -> list[Rec]:
    return classes_of(mod("datasketch"), "datasketch")


def pysubgroup() -> list[Rec]:
    sg = mod("pysubgroup")
    return classes_of(sg, "pysubgroup") + [
        rec("pysubgroup", n, getattr(sg, n))
        for n in dir(sg)
        if inspect.isfunction(getattr(sg, n)) and not n.startswith("_")
    ]


def pyitlib() -> list[Rec]:
    return from_all("pyitlib.discrete_random_variable", "pyitlib")


def npeet() -> list[Rec]:
    return from_all("npeet.entropy_estimators", "npeet")


def sdeint() -> list[Rec]:
    return from_all("sdeint", "sdeint")


def numpy() -> list[Rec]:
    np = mod("numpy")
    recs: list[Rec] = []
    for n in dir(np):
        o = getattr(np, n)
        if not n.startswith("_") and callable(o) and not inspect.isclass(o):
            recs.append(rec("numpy", n, o))
    recs += from_all("numpy.linalg", "numpy.linalg") + from_all("numpy.fft", "numpy.fft")
    gen = np.random.Generator
    recs += [rec("numpy.random.Generator", n, getattr(gen, n), "method") for n in dir(gen) if not n.startswith("_")]
    return recs


# ---- the covariance family (third pass)


def sklearn_covariance() -> list[Rec]:
    return modules("sklearn", ["covariance"])


def statsmodels_cov() -> list[Rec]:
    names = (
        "stats.correlation_tools stats.covariance stats.sandwich_covariance stats.regularized_covariance "
        "stats.moment_helpers stats.multivariate stats.multivariate_tools stats.dist_dependence_measures "
        "multivariate.pca multivariate.factor"
    ).split()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return modules("statsmodels", names)


def scipy_cov() -> list[Rec]:
    """The matrix-variate and covariance-adjacent names of scipy (already counted in the scipy list)."""
    st = mod("scipy.stats")
    sl = mod("scipy.linalg")
    names = (
        "multivariate_normal multivariate_t wishart invwishart matrix_normal matrix_t random_correlation ortho_group "
        "special_ortho_group unitary_group dirichlet_multinomial"
    ).split()
    recs = [rec("scipy.stats", n, getattr(st, n), "distribution") for n in names if hasattr(st, n)]
    recs += [rec("scipy.linalg", n, getattr(sl, n)) for n in "sqrtm logm expm fractional_matrix_power cholesky".split()]
    recs += from_all(
        "scipy.stats", "scipy.stats", {"cov", "pearsonr", "spearmanr", "kendalltau", "chatterjeexi", "somersd"}
    )
    recs += [rec("scipy.spatial.distance", "mahalanobis", mod("scipy.spatial.distance").mahalanobis)]
    recs += [rec("scipy.cluster.vq", "whiten", mod("scipy.cluster.vq").whiten)]
    return recs


def pyriemann() -> tuple[list[Rec], str]:
    parts = (
        "estimation geometry.covariance geometry.base geometry.mean geometry.median geometry.distance "
        "geometry.geodesic "
        "geometry.tangentspace geometry.kernel geometry.ajd geometry.test preprocessing tangentspace "
        "spatialfilters "
        "artifact_detection datasets.sampling datasets.simulated utils.viz"
    ).split()
    recs, ver = src_api("pyriemann", "pyriemann", [f"pyriemann.{p}" for p in parts])
    return recs, ver


def skfolio() -> tuple[list[Rec], str]:
    parts = (
        "moments.covariance moments.variance moments.expected_returns prior distance uncertainty_set pre_selection "
        "model_selection utils.stats"
    ).split()
    return src_api("skfolio", "skfolio", [f"skfolio.{p}" for p in parts])


def pypfopt() -> tuple[list[Rec], str]:
    return src_api("pyportfolioopt", "pypfopt", ["pypfopt.risk_models", "pypfopt.plotting"])


def riskfolio() -> tuple[list[Rec], str]:
    return src_api(
        "riskfolio_lib",
        "riskfolio",
        ["riskfolio.AuxFunctions", "riskfolio.GerberStatistic", "riskfolio.ParamsEstimation", "riskfolio.DBHT"],
    )


def skggm() -> tuple[list[Rec], str]:
    parts = (
        "adaptive_graph_lasso inverse_covariance metrics model_average quic_graph_lasso rank_correlation "
        "profiling.monte_carlo_profile profiling.metrics profiling.graphs"
    ).split()
    return src_api("skggm", "skggm-0.2.8/inverse_covariance", [f"inverse_covariance.{p}" for p in parts])


def gglasso() -> tuple[list[Rec], str]:
    parts = (
        "problem solver.admm_solver solver.single_admm_solver solver.ppdna_solver helper.model_selection "
        "helper.data_generation"
    ).split()
    return src_api("gglasso", "gglasso", [f"gglasso.{p}" for p in parts])


def regain() -> tuple[list[Rec], str]:
    parts = (
        "covariance.graphical_lasso_ covariance.time_graphical_lasso_ covariance.kernel_time_graphical_lasso_ "
        "covariance.latent_graphical_lasso_ covariance.latent_time_graphical_lasso_ "
        "covariance.kernel_latent_time_graphical_lasso_ covariance.missing_graphical_lasso_ "
        "covariance.infimal_convolution_ "
        "bayesian.gwishart_inference bayesian.wishart_distribution_ bayesian.wishart_process_ "
        "generalized_linear_model.glm_gaussian generalized_linear_model.glm_ising generalized_linear_model.glm_poisson "
        "model_selection.stability_optimization scores discriminant_analysis datasets.gaussian"
    ).split()
    return src_api("regain", "regain", [f"regain.{p}" for p in parts])


def robpy() -> tuple[list[Rec], str]:
    parts = (
        "covariance.mcd covariance.cellmcd covariance.ogk covariance.kendall covariance.wrapping "
        "covariance.initial_ddcw "
        "outliers.ddc pca.robpca pca.spca utils.outlyingness utils.median regression.lts regression.mm regression.s "
        "univariate.onestep_m univariate.mcd univariate.qn univariate.tau"
    ).split()
    return src_api("robpy", "robpy", [f"robpy.{p}" for p in parts])


def rmt_denoise() -> tuple[list[Rec], str]:
    return src_api(
        "rmt_denoise", "rmt_denoise", ["rmt_denoise.mp_law", "rmt_denoise.generalized_cov", "rmt_denoise.core"]
    )


def geomstats() -> tuple[list[Rec], str]:
    parts = "spd_matrices full_rank_correlation_matrices positive_lower_triangular_matrices symmetric_matrices".split()
    return src_api("geomstats", "geomstats", [f"geomstats.geometry.{p}" for p in parts])


def pymanopt() -> tuple[list[Rec], str]:
    parts = "positive_definite psd positive fixed_rank".split()
    return src_api("pymanopt", "pymanopt", [f"pymanopt.manifolds.{p}" for p in parts])


def arch_cov() -> tuple[list[Rec], str]:
    return src_api("arch", "arch", ["arch.covariance.kernel"])


def nilearn_connectome() -> tuple[list[Rec], str]:
    return src_api(
        "nilearn", "nilearn", ["nilearn.connectome.connectivity_matrices", "nilearn.connectome.group_sparse_cov"]
    )


def mne_cov() -> tuple[list[Rec], str]:
    return src_api("mne", "mne", ["mne.cov"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Callable[[], list[Rec]], str | None]] = [
        ("sklearn", sklearn, "scikit-learn"),
        ("scipy", scipy, None),
        ("statsmodels", statsmodels, None),
        ("torch", torch, None),
        ("torchvision", torchvision, None),
        ("torchmetrics", torchmetrics, None),
        ("numpyro", numpyro, None),
        ("arviz", arviz, None),
        ("pgmpy", pgmpy, None),
        ("hmmlearn", hmmlearn, None),
        ("gymnasium", gymnasium, None),
        ("fairlearn", fairlearn, None),
        ("shap", shap, None),
        ("captum", captum, None),
        ("skimage", skimage, "scikit-image"),
        ("nltk", nltk, None),
        ("gensim", gensim, None),
        ("networkx", networkx, None),
        ("optuna", optuna, None),
        ("imblearn", imblearn, "imbalanced-learn"),
        ("opacus", opacus, None),
        ("modal", modal, "modAL-python"),
        ("gpytorch", gpytorch, None),
        ("skorch", skorch, None),
        ("lime", lime, None),
        ("librosa", librosa, None),
        ("control", control, None),
        ("filterpy", filterpy, None),
        ("pymoo", pymoo, None),
        ("cma", cma, None),
        ("causallearn", causallearn, "causal-learn"),
        ("mapie", mapie, None),
        ("pot", pot, "pot"),
        ("optax", optax, None),
        ("blackjax", blackjax, None),
        ("stumpy", stumpy, None),
        ("ruptures", ruptures, None),
        ("pysindy", pysindy, None),
        ("shapely", shapely, None),
        ("cvxpy", cvxpy, None),
        ("nashpy", nashpy, None),
        ("pingouin", pingouin, None),
        ("sympy", sympy, None),
        ("openspiel", openspiel, "open_spiel"),
        ("datasketch", datasketch, None),
        ("pysubgroup", pysubgroup, None),
        ("pyitlib", pyitlib, None),
        ("npeet", npeet, None),
        ("sdeint", sdeint, None),
        ("numpy", numpy, None),
        ("torchvision_models", torchvision_models, "torchvision"),
        ("crf_libraries", crf_libraries, "sklearn-crfsuite"),
        ("sklearn_covariance", sklearn_covariance, "scikit-learn"),
        ("statsmodels_cov", statsmodels_cov, "statsmodels"),
        ("scipy_cov", scipy_cov, "scipy"),
    ]
    sources: list[tuple[str, Callable[[], tuple[list[Rec], str]]]] = [
        ("pyriemann", pyriemann),
        ("skfolio", skfolio),
        ("pypfopt", pypfopt),
        ("riskfolio", riskfolio),
        ("skggm", skggm),
        ("gglasso", gglasso),
        ("regain", regain),
        ("robpy", robpy),
        ("rmt_denoise", rmt_denoise),
        ("geomstats", geomstats),
        ("pymanopt", pymanopt),
        ("arch_cov", arch_cov),
        ("nilearn_connectome", nilearn_connectome),
        ("mne_cov", mne_cov),
    ]
    only = set(sys.argv[1:])
    if src_root() is not None:
        for lib, src_fn in sources:
            if only and lib not in only:
                continue
            try:
                found, ver = src_fn()
                dump(lib, found, ver=ver)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {lib}: {exc!r}", file=sys.stderr)
    for lib, fn, dist in jobs:
        if only and lib not in only:
            continue
        try:
            dump(lib, fn(), dist)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {lib}: {exc!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
