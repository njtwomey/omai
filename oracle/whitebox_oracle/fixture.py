"""The fixture format shared with the TypeScript tests (packages/engine/src/testing/golden.ts)."""

import math
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

SCHEMA = "whitebox.fixture/1"


def encode_number(x: float) -> float | str:
    """Encode non-finite numbers as strings, as the TypeScript side expects."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x)


def encode(value: Any) -> Any:
    """Encode numpy arrays (and nested containers of them) as tensor JSON."""
    if isinstance(value, np.ndarray | np.generic):
        arr = np.asarray(value)
        if arr.dtype == np.bool_:
            dtype, data = "bool", [int(v) for v in arr.ravel()]
        elif np.issubdtype(arr.dtype, np.integer):
            dtype, data = "int32", [int(v) for v in arr.ravel()]
        elif arr.dtype == np.float32:
            dtype, data = "float32", [encode_number(float(v)) for v in arr.ravel()]
        else:
            dtype, data = "float64", [encode_number(float(v)) for v in arr.astype(np.float64).ravel()]
        return {"dtype": dtype, "shape": list(arr.shape), "data": data}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [encode(v) for v in value]
    if isinstance(value, float):
        return encode_number(value)
    return value


class Case(BaseModel):
    """One reference computation: inputs, parameters and the expected output."""

    name: str
    fn: str
    inputs: list[Any] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    rtol: float = 1e-12
    atol: float = 1e-12


class Fixture(BaseModel):
    """A fixture file: the cases of one case module, with provenance."""

    schema_: str = Field(default=SCHEMA, alias="schema")
    source: str
    reference: dict[str, str]
    cases: list[Case]

    model_config = {"populate_by_name": True}


def case(
    name: str, fn: str, inputs: list[Any], output: Any, *, rtol: float = 1e-12, atol: float = 1e-12, **params: Any
) -> Case:
    """Build a case, encoding arrays."""
    return Case(
        name=name, fn=fn, inputs=encode(inputs), params=encode(params), output=encode(output), rtol=rtol, atol=atol
    )
