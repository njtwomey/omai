/**
 * Helpers shared by the stochastic modules, private to stochastic: validated time grids and their step sizes, concrete
 * numbers from scalars and traced values, and float64 vectors.
 */

import { type Estimate, estimate, MdmlError } from '../base/index.ts'
import {
  asTensor,
  concreteOrThrow,
  expandDims,
  fromStorage,
  isTraced,
  primalOf,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import type { TraceOptions } from './paths.ts'

/** A grid of times: an increasing list of numbers. */
export type TimesLike = readonly number[] | Float64Array

/** Validate a time grid: finite and strictly increasing, with at least one point. */
export function asTimes(times: TimesLike, where: string): Float64Array {
  const t = Float64Array.from(times)
  if (t.length === 0) throw new MdmlError('value', `${where}: the time grid is empty`)
  for (let i = 0; i < t.length; i++) {
    if (!Number.isFinite(t[i] ?? Number.NaN)) throw new MdmlError('value', `${where}: times must be finite`)
    if (i > 0 && !((t[i] ?? 0) > (t[i - 1] ?? 0))) {
      throw new MdmlError('value', `${where}: times must be strictly increasing`)
    }
  }
  return t
}

/** The step sizes of a grid. */
export function stepsOf(times: Float64Array): Float64Array {
  return Float64Array.from({ length: Math.max(0, times.length - 1) }, (_, i) => (times[i + 1] ?? 0) - (times[i] ?? 0))
}

/** A concrete number from a scalar (a number or a concrete scalar value). */
export function numberOf(x: TensorLike, where: string): number {
  if (typeof x === 'number') return x
  const t = concreteOrThrow(asTensor(x), where)
  if (t.data.length !== 1) throw new MdmlError('shape', `${where}: expected a scalar`)
  return t.data[0] ?? Number.NaN
}

/** The concrete numbers of a value (its primal when traced). */
export function primalNumbers(x: TensorLike): number[] {
  const v = asTensor(x)
  return toFlat(isTraced(v) ? primalOf(v) : v)
}

/** A float64 tensor from numbers. */
export function vector(data: ArrayLike<number>, shape: readonly number[] = [data.length]): Tensor<'float64'> {
  return fromStorage(Float64Array.from(data), [...shape], 'float64')
}

/** Validate a positive integer. */
export function positiveInteger(n: number, name: string, where: string): number {
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', `${where}: ${name} must be a positive integer`)
  return n
}

/** Validate a probability vector or stochastic matrix row (non-negative, summing to one within tol). */
export function checkProbabilities(p: ArrayLike<number>, where: string, tol = 1e-9): void {
  let s = 0
  for (let i = 0; i < p.length; i++) {
    const v = p[i] ?? Number.NaN
    if (!(v >= 0) || !Number.isFinite(v)) throw new MdmlError('value', `${where}: probabilities must be >= 0`)
    s += v
  }
  if (Math.abs(s - 1) > tol) throw new MdmlError('value', `${where}: probabilities must sum to 1 (got ${s})`)
}

/** The tracing fields of an options object, without undefined entries. */
export function traceOf(o: TraceOptions): TraceOptions {
  return {
    ...(o.tracer === undefined ? {} : { tracer: o.tracer }),
    ...(o.policy === undefined ? {} : { policy: o.policy }),
    ...(o.parent === undefined ? {} : { parent: o.parent }),
    ...(o.runId === undefined ? {} : { runId: o.runId }),
  }
}

/** The concrete value of a state for trace events, or undefined while it is traced. */
export function observable(x: Value): Value | undefined {
  return isTraced(x) ? undefined : x
}

/** Path values `[paths, n, ...state]` from paths or a value (a single path `[n]` gains the paths axis). */
export function valuesOf(x: { readonly times: Float64Array; readonly values: Value } | TensorLike): Value {
  const isPaths = typeof x === 'object' && x !== null && 'times' in x && 'values' in x
  const v = asTensor(isPaths ? (x as { values: Value }).values : (x as TensorLike))
  return v.shape.length === 1 ? expandDims(v, 0) : v
}

/** The sample mean of numbers with its standard error. */
export function sampleMean(xs: ArrayLike<number>): Estimate {
  const n = xs.length
  let m = 0
  for (let i = 0; i < n; i++) m += xs[i] ?? 0
  m /= n
  let v = 0
  for (let i = 0; i < n; i++) v += ((xs[i] ?? 0) - m) ** 2
  v /= Math.max(1, n - 1)
  return estimate(m, Math.sqrt(v / n))
}
