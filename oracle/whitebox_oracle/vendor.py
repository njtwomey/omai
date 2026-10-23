"""Dataset vendoring: modules under oracle/datasets/ render static datasets as generated TypeScript (`*.data.ts`)."""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from .settings import ROOT

DATASETS_DIR = ROOT / "oracle" / "datasets"


class DatasetModule(Protocol):
    """The contract of a module under oracle/datasets/: where its data goes, and the file's contents."""

    TARGET: str

    def render(self) -> str: ...


def number_list(values: npt.ArrayLike) -> str:
    """A compact JSON array of numbers (integers without a trailing .0, floats in shortest round-trip form)."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    return "[" + ",".join(str(int(v)) if float(v).is_integer() else repr(float(v)) for v in flat) + "]"


def sha256_of(*arrays: npt.ArrayLike) -> str:
    """A checksum of the data as canonical JSON, recorded in the generated file."""
    payload = json.dumps([np.asarray(a, dtype=np.float64).ravel().tolist() for a in arrays], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load(path: Path) -> DatasetModule:
    """Import a dataset module from its file path."""
    spec = importlib.util.spec_from_file_location(f"whitebox_datasets.{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(DatasetModule, module)


def rendered() -> dict[Path, str]:
    """Every vendored file, keyed by its target path."""
    out: dict[Path, str] = {}
    for path in sorted(DATASETS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = load(path)
        out[ROOT / module.TARGET] = module.render()
    return out
