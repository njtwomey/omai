/**
 * The solvers of a smooth objective on a box, as a table of modes like `smoothSolvers`: L-BFGS-B (the default) and
 * SLSQP, which handle the bounds themselves, and every gradient-based generic solver of `smoothSolvers` (all but Newton) run on a smooth
 * reparametrisation of the box (a logistic map for a two-sided bound, an exponential for a one-sided one), whose
 * iterate is mapped back so that every mode yields the fitted vector in the problem's own coordinates. A method whose
 * parameters are bounded (the hyperparameters of a Gaussian process) takes `boundedSolvers` as its solver slot.
 */

import { OmaiError } from '../base/index.ts'
import { dict, real } from '../space/index.ts'
import { add, asTensor, exp, mul, sigmoid, type TensorLike, toFlat, toVector, type Value } from '../tensor/index.ts'
import { type Algorithm, type ModeSpec, type ModeTable, modes, stateFromJson, stateToJson } from '../trace/index.ts'
import { ftolKnob, given, gradTolKnob, numberKnob } from './_knobs.ts'
import { lbfgsbAlgorithm } from './minimize/lbfgsb.ts'
import type { Parameters } from './minimize/objective.ts'
import { slsqpAlgorithm } from './minimize/slsqp.ts'
import { type SmoothProblem, smoothSolvers, type VectorParams } from './solvers.ts'

/**
 * A smooth problem on a box: the objective, a start inside it, and the bounds, vectors of tensor-likes (infinities for
 * none; migration, D-009: they were `ArrayLike<number>`).
 */
export interface BoundedProblem extends SmoothProblem {
  readonly lower: TensorLike
  readonly upper: TensorLike
}

/** The reparametrisation of a box by unconstrained variables: the map and its inverse, as numbers and as tensors. */
export interface BoxMap {
  /** The unconstrained variables of a point of the box (a point on a bound is moved just inside). */
  toFree(x: ArrayLike<number>): Float64Array
  /** The point of the box of unconstrained variables. */
  fromFree(u: ArrayLike<number>): Float64Array
  /** {@link fromFree} on a differentiable value. */
  fromFreeValue(u: Value): Value
}

const sigmoidOf = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))

/**
 * The smooth map from unconstrained variables onto a box: `lower + (upper - lower) sigmoid(u)` where both bounds are
 * finite, `lower + exp(u)` or `upper - exp(u)` where one is, and `u` itself where neither is. Takes the bounds as
 * tensor-likes (migration, D-009: they were `ArrayLike<number>`); the returned map is a kernel on plain arrays, as the
 * generic solvers of the slot call it on every iteration.
 *
 * @param lower - Lower bounds, a vector (`-Infinity` for none).
 * @param upper - Upper bounds, a vector (`Infinity` for none).
 * @returns The map, its inverse and the map on differentiable values.
 * @throws OmaiError `shape` when the bounds differ in length or are not vectors, and `value` when a lower bound exceeds its upper bound.
 * @example
 * ```ts
 * import { boxMap } from 'omai-engine/optim'
 * const map = boxMap([0, 0], [1, Infinity])
 * Array.from(map.fromFree(map.toFree([0.25, 3]))) // => [0.25, 3]
 * ```
 */
export function boxMap(lowerBounds: TensorLike, upperBounds: TensorLike): BoxMap {
  const lower = toVector(lowerBounds, 'boxMap')
  const upper = toVector(upperBounds, 'boxMap')
  const n = lower.length
  if (upper.length !== n) throw new OmaiError('shape', 'boxMap: lower and upper must have the same length')
  // Per coordinate: 0 free, 1 two-sided, 2 lower only, 3 upper only.
  const kind = Int8Array.from({ length: n }, (_, i) => {
    const lo = lower[i] ?? -Infinity
    const hi = upper[i] ?? Infinity
    if (lo > hi) throw new OmaiError('value', `boxMap: lower bound ${lo} exceeds upper bound ${hi}`)
    const a = Number.isFinite(lo)
    const b = Number.isFinite(hi)
    return a && b ? 1 : a ? 2 : b ? 3 : 0
  })
  const lo = (i: number): number => lower[i] ?? -Infinity
  const hi = (i: number): number => upper[i] ?? Infinity
  const toFree = (x: ArrayLike<number>): Float64Array =>
    Float64Array.from({ length: n }, (_, i) => {
      const v = x[i] ?? 0
      const k = kind[i]
      if (k === 1) {
        const p = Math.min(Math.max((v - lo(i)) / (hi(i) - lo(i) || 1), 1e-9), 1 - 1e-9)
        return Math.log(p / (1 - p))
      }
      if (k === 2) return Math.log(Math.max(v - lo(i), 1e-9))
      if (k === 3) return Math.log(Math.max(hi(i) - v, 1e-9))
      return v
    })
  const fromFree = (u: ArrayLike<number>): Float64Array =>
    Float64Array.from({ length: n }, (_, i) => {
      const v = u[i] ?? 0
      const k = kind[i]
      if (k === 1) return lo(i) + (hi(i) - lo(i)) * sigmoidOf(v)
      if (k === 2) return lo(i) + Math.exp(v)
      if (k === 3) return hi(i) - Math.exp(v)
      return v
    })
  // x = base + wSig sigmoid(mSig u) + wExp exp(mExp u) + wId (mId u): each coordinate uses one term, the masks keep
  // the other terms finite.
  const vec = (f: (i: number) => number): Value => asTensor(Float64Array.from({ length: n }, (_, i) => f(i)))
  const base = vec((i) => (kind[i] === 1 || kind[i] === 2 ? lo(i) : kind[i] === 3 ? hi(i) : 0))
  const wSig = vec((i) => (kind[i] === 1 ? hi(i) - lo(i) : 0))
  const mSig = vec((i) => (kind[i] === 1 ? 1 : 0))
  const wExp = vec((i) => (kind[i] === 2 ? 1 : kind[i] === 3 ? -1 : 0))
  const mExp = vec((i) => (kind[i] === 2 || kind[i] === 3 ? 1 : 0))
  const mId = vec((i) => (kind[i] === 0 ? 1 : 0))
  return {
    toFree,
    fromFree,
    fromFreeValue: (u) =>
      add(add(add(base, mul(wSig, sigmoid(mul(mSig, u)))), mul(wExp, exp(mul(mExp, u)))), mul(mId, u)),
  }
}

const gradTol = gradTolKnob

const lbfgsb: ModeSpec<BoundedProblem, { readonly x: Float64Array }, VectorParams> = {
  description: 'L-BFGS-B: limited-memory BFGS with the bounds handled by a projected Cauchy search',
  space: dict({
    memory: real(1, 50, { default: 10, label: 'curvature pairs kept' }),
    ftol: ftolKnob,
    gradTol,
  }),
  make: (p, k) => {
    const memory = numberKnob(k, 'memory')
    return lbfgsbAlgorithm(p.objective, p.x0, {
      bounds: { lower: toVector(p.lower, 'boundedSolvers'), upper: toVector(p.upper, 'boundedSolvers') },
      ...given({
        memory: memory === undefined ? undefined : Math.round(memory),
        ftol: numberKnob(k, 'ftol'),
        gradTol: numberKnob(k, 'gradTol'),
      }),
    })
  },
  params: (s) => ({ x: Array.from(s.x) }),
}

const slsqp: ModeSpec<BoundedProblem, { readonly x: Float64Array }, VectorParams> = {
  description: 'sequential quadratic programming with the bounds in the quadratic subproblem (SLSQP)',
  space: dict({ ftol: real(1e-14, 1e-2, { log: true, default: 1e-6, label: 'objective change tolerance' }), gradTol }),
  make: (p, k) =>
    slsqpAlgorithm(p.objective, p.x0, {
      bounds: { lower: toVector(p.lower, 'boundedSolvers'), upper: toVector(p.upper, 'boundedSolvers') },
      ...given({ ftol: numberKnob(k, 'ftol'), gradTol: numberKnob(k, 'gradTol') }),
    }),
  params: (s) => ({ x: Array.from(s.x) }),
}

/** The problem of a generic solver over a bounded one: the objective composed with the box map, started at its image. */
function freeProblem(p: BoundedProblem, map: BoxMap): SmoothProblem {
  const x0 = Array.from(toFlat(asTensor(p.x0 as Parameters & Float64Array)))
  return {
    objective: (u, batch) => p.objective(map.fromFreeValue(u as Value), batch),
    x0: map.toFree(x0),
    ...(p.batches === undefined ? {} : { batches: p.batches }),
  }
}

/** What a generic solver's state looks like once its iterate is mapped back onto the box. */
interface MappedState {
  /** The iterate in the problem's own coordinates. */
  readonly x: Float64Array
  readonly value: number
  /** The gradient in the free coordinates the solver works in. */
  readonly grad: Float64Array
  readonly step: number
  readonly nfev: number
  readonly lr?: number
  readonly stop?: { readonly status: string }
  /** The solver's own state. */
  readonly inner: unknown
}

/** A generic solver's algorithm over the free coordinates, with the iterate of every state mapped back to the box. */
function mapped(inner: Algorithm<unknown>, map: BoxMap): Algorithm<MappedState> {
  const wrap = (state: unknown): MappedState => {
    const r = state as Omit<MappedState, 'inner'>
    return {
      x: map.fromFree(r.x),
      value: r.value,
      grad: r.grad,
      step: r.step,
      nfev: r.nfev ?? 0,
      ...(r.lr === undefined ? {} : { lr: r.lr }),
      ...(r.stop === undefined ? {} : { stop: r.stop }),
      inner: state,
    }
  }
  return {
    name: inner.name,
    config: inner.config,
    init: (k) => wrap(inner.init(k)),
    step: (s, ctx) => wrap(inner.step(s.inner, ctx)),
    ...(inner.done === undefined ? {} : { done: (s, t) => inner.done?.(s.inner, t) }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MappedState,
  }
}

const generic: Record<string, ModeSpec<BoundedProblem, MappedState, VectorParams>> = {}
for (const name of smoothSolvers.names) {
  // Newton needs second derivatives; the objectives of bounded problems here are given by their gradient.
  if (name === 'newton') continue
  const base = smoothSolvers.modeOf({ kind: name })
  generic[name] = {
    description: `${base.description ?? name}, on a smooth reparametrisation of the box`,
    space: base.space,
    make: (p, k, key) => {
      const map = boxMap(p.lower, p.upper)
      return mapped(base.make(freeProblem(p, map), k, key), map)
    },
    params: (s) => ({ x: Array.from(s.x) }),
  }
}

/**
 * The solvers of a smooth objective on a box as one slot: L-BFGS-B (default), SLSQP, and each first-order generic
 * solver of `smoothSolvers` (all but Newton, which needs second derivatives) on a reparametrisation of the box. The state of every mode carries the iterate `x` in the problem's
 * own coordinates (a generic mode's `grad` is in the free coordinates it works in).
 *
 * @index table The solvers of a smooth objective on a box.
 * @example
 * ```ts
 * import { boundedSolvers } from 'omai-engine/optim'
 * boundedSolvers.names.slice(0, 2) // => ['lbfgsb', 'slsqp']
 * boundedSolvers.modeOf({ kind: 'slsqp' }).name // => 'slsqp'
 * ```
 */
export const boundedSolvers: ModeTable<BoundedProblem, VectorParams> = modes<BoundedProblem, VectorParams>(
  { lbfgsb, slsqp, ...generic },
  { label: 'solver', description: 'how the objective is minimised on its box', default: 'lbfgsb' },
)
