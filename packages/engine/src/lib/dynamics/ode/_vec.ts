// Shared state arithmetic for the ODE solvers. States are values of any shape; combinations of concrete tensors run
// on typed arrays (the fast, untraced path) and anything traced goes through tensor operations, so the same solver
// code is differentiable by differentiating through its steps.
import { MdmlError } from '../../base/index.ts'
import { nextAfter } from '../../numerics/approximation/index.ts'
import { dnrm2 } from '../../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  isTensor,
  mul,
  primalOf,
  type Shape,
  type Tensor,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'

/** A term c * v of a linear combination. */
export type Term = readonly [number, Value]

/** `base + sum c_i v_i`, skipping zero coefficients; typed-array arithmetic when every value is concrete. */
export function combine(base: Value | null, terms: readonly Term[]): Value {
  const live = terms.filter(([c]) => c !== 0)
  const all = base === null ? live.map(([, v]) => v) : [base, ...live.map(([, v]) => v)]
  const first = base ?? terms[0]?.[1]
  if (first === undefined) throw new MdmlError('value', 'combine: nothing to combine')
  if (all.length === 0) return mul(first, 0)
  if (all.every((v) => isTensor(v))) {
    const out =
      base === null ? new Float64Array((first as Tensor).data.length) : Float64Array.from((base as Tensor).data)
    for (const [c, v] of live) {
      const d = (v as Tensor).data
      for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) + c * (d[i] ?? 0)
    }
    return fromStorage(out, first.shape, 'float64')
  }
  let acc: Value | null = base
  for (const [c, v] of live) {
    const term = c === 1 ? v : mul(v, c)
    acc = acc === null ? term : add(acc, term)
  }
  return acc ?? mul(first, 0)
}

/** The concrete numbers of a value (its primal, flattened). */
export function numbersOf(v: Value): Float64Array {
  const t = primalOf(v)
  return t.data instanceof Float64Array ? t.data : Float64Array.from(t.data)
}

/** A float64 tensor of the given shape over the numbers. */
export function fromNumbers(data: ArrayLike<number>, shape: Shape): Tensor {
  return fromStorage(data instanceof Float64Array ? data : Float64Array.from(data), shape, 'float64')
}

/** A float64 value of a tensor-like (a copy of concrete inputs, so solvers never alias caller memory). */
export function floatState(y: TensorLike): Value {
  const v = asTensor(y)
  if (isTensor(v)) return fromNumbers(Float64Array.from(v.data), v.shape)
  return v.dtype === 'float64' ? v : mul(v, 1)
}

/** Root-mean-square of numbers (the norm scipy's solvers use). */
export function rms(x: ArrayLike<number>): number {
  return x.length === 0 ? 0 : dnrm2(x) / Math.sqrt(x.length)
}

/** A tolerance: one number for every component, or one per component. */
export type Tol = number | readonly number[]

/** The tolerance of component `i`. */
export function tolAt(tol: Tol, i: number): number {
  return typeof tol === 'number' ? tol : (tol[i] ?? tol[0] ?? 0)
}

/** The error scale `atol + rtol max(|y0|, |y1|)` per component. */
export function errorScale(y0: ArrayLike<number>, y1: ArrayLike<number>, rtol: Tol, atol: Tol): Float64Array {
  const out = new Float64Array(y0.length)
  for (let i = 0; i < out.length; i++) {
    out[i] = tolAt(atol, i) + Math.max(Math.abs(y0[i] ?? 0), Math.abs(y1[i] ?? 0)) * tolAt(rtol, i)
  }
  return out
}

/** RMS of `x / scale`. */
export function scaledRms(x: ArrayLike<number>, scale: ArrayLike<number>): number {
  const r = new Float64Array(x.length)
  for (let i = 0; i < r.length; i++) r[i] = (x[i] ?? 0) / (scale[i] ?? 1)
  return rms(r)
}

export { allFinite } from '../../base/index.ts'

/** The smallest step scipy allows at `t`: ten units in the last place. */
export function minStep(t: number, direction: number): number {
  const next = nextAfter(t, direction * Number.POSITIVE_INFINITY)
  return 10 * Math.abs(next - t)
}
