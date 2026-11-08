"""The fixture format shared with the TypeScript tests (packages/engine/src/testing/golden.ts)."""

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

SCHEMA = "omai.fixture/1"

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
"""An encoded JSON value: what `encode` produces and what a fixture stores for inputs, parameters and outputs."""


def encode_number(x: float) -> float | str:
    """Encode non-finite numbers as strings, as the TypeScript side expects."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x)


def encode_array(arr: npt.NDArray[np.generic]) -> dict[str, Json]:
    """Encode a numpy array as tensor JSON: dtype, shape and flat row-major data."""
    data: list[Json]
    if arr.dtype == np.bool_:
        dtype = "bool"
        data = [int(v) for v in arr.ravel()]
    elif np.issubdtype(arr.dtype, np.integer):
        dtype = "int32"
        data = [int(v) for v in arr.ravel()]
    elif arr.dtype == np.float32:
        dtype = "float32"
        data = [encode_number(float(v)) for v in arr.ravel()]
    else:
        dtype = "float64"
        data = [encode_number(float(v)) for v in arr.astype(np.float64).ravel()]
    return {"dtype": dtype, "shape": list(arr.shape), "data": data}


def encode_all(values: Sequence[object]) -> list[Json]:
    """Encode each element of a list or tuple."""
    return [encode(v) for v in values]


def encode_params(params: Mapping[str, object]) -> dict[str, Json]:
    """Encode each value of a mapping with string keys."""
    return {k: encode(v) for k, v in params.items()}


def encode(value: object) -> Json:
    """Encode numpy arrays (and nested containers of them) as tensor JSON; anything else non-JSON is a TypeError."""
    if isinstance(value, np.ndarray | np.generic):
        return encode_array(np.asarray(cast(npt.ArrayLike, value)))
    if isinstance(value, dict):
        return encode_params(cast(Mapping[str, object], value))
    if isinstance(value, list | tuple):
        return encode_all(cast(Sequence[object], value))
    if isinstance(value, float):
        return encode_number(value)
    if value is None or isinstance(value, bool | int | str):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} as fixture JSON")


class Case(BaseModel):
    """One reference computation: inputs, parameters and the expected output."""

    name: str
    fn: str
    inputs: list[Json] = Field(default_factory=list[Json])
    params: dict[str, Json] = Field(default_factory=dict[str, Json])
    output: Json = None
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
    name: str,
    fn: str,
    inputs: Sequence[object],
    output: object,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    **params: object,
) -> Case:
    """Build a case, encoding arrays."""
    return Case(
        name=name,
        fn=fn,
        inputs=encode_all(inputs),
        params=encode_params(params),
        output=encode(output),
        rtol=rtol,
        atol=atol,
    )
