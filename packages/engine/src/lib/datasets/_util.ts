// Shared helpers for the dataset generators: columns of draws, noise, column access and metadata stamps.
import type { DatasetMeta, GeneratorKnobs } from '../data/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import {
  add,
  concat,
  fromStorage,
  mul,
  reshape,
  sliceAxis,
  squeeze,
  type Tensor,
  tensor,
  type Value,
} from '../tensor/index.ts'

/** Column `j` of a matrix value, as a vector. */
export function column(x: Value, j: number): Value {
  return squeeze(sliceAxis(x, -1, j, j + 1), -1)
}

/** Columns of independent uniforms on the given ranges, as an `[n, d]` matrix. */
export function uniformColumns(k: Key, n: number, ranges: readonly (readonly [number, number])[]): Tensor {
  const d = ranges.length
  const data = new Float64Array(n * d)
  for (const [j, [low, high]] of ranges.entries()) {
    const c = uniform(foldIn(k, j), [n], { low, high }).data
    for (let i = 0; i < n; i++) data[i * d + j] = c[i] ?? 0
  }
  return fromStorage(data, [n, d], 'float64')
}

/** Add Gaussian noise of standard deviation `noise` to a value. */
export function withNoise(k: Key, clean: Value, noise: number): Tensor {
  return add(clean, mul(normal(k, clean.shape), noise)) as Tensor
}

/** A matrix from row-major numbers. */
export function matrix(data: ArrayLike<number>, n: number, d: number): Tensor {
  return fromStorage(Float64Array.from(data as ArrayLike<number>), [n, d], 'float64')
}

/** A vector from numbers. */
export function vector(data: ArrayLike<number>): Tensor {
  return fromStorage(Float64Array.from(data as ArrayLike<number>), [data.length], 'float64')
}

/** Class labels from numbers. */
export function labels(data: ArrayLike<number>): Tensor<'int32'> {
  return fromStorage(Int32Array.from(data as ArrayLike<number>), [data.length], 'int32')
}

/** Columns `[n]` stacked into an `[n, d]` matrix value. */
export function columns(cols: readonly Value[]): Value {
  return concat(
    cols.map((c) => reshape(c, [-1, 1])),
    1,
  )
}

/** Metadata for generated data: name, description and the recipe. */
export function stamp(name: string, description: string, knobs: GeneratorKnobs): Partial<DatasetMeta> {
  return { name, description, source: 'mdml-js generator', recipe: { generator: name, knobs } }
}

/** A constant row vector as a tensor. */
export function row(values: readonly number[]): Tensor {
  return tensor(values as number[])
}

/** Integer knob value. */
export function intKnob(knobs: GeneratorKnobs, name: string): number {
  return knobs[name] as number
}

/** Real knob value. */
export function realKnob(knobs: GeneratorKnobs, name: string): number {
  return knobs[name] as number
}

/** Split `n` items into `k` near-equal parts (the first `n mod k` get one more), as sklearn does for blobs. */
export function evenSplit(n: number, k: number): number[] {
  return Array.from({ length: k }, (_, i) => Math.floor(n / k) + (i < n % k ? 1 : 0))
}
