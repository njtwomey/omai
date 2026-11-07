/**
 * The `Distribution` contract every family implements: a plain object of parameters (which may be traced, so everything
 * differentiates with respect to them) with pure methods for densities, CDFs, quantiles, moments, entropy and sampling,
 * built by `defineDistribution`. Also the shape conventions (batch and event shapes, as torch.distributions), supports
 * and membership tests, and parameter validation.
 *
 * @index structure The Distribution contract: density, CDF, quantile, moments and sampling.
 */

import { MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import {
  abs,
  all,
  any,
  asTensor,
  broadcastShapes,
  broadcastTo,
  diagonal,
  equal,
  exp,
  floor,
  greater,
  greaterEqual,
  isfinite,
  isnan,
  less,
  lessEqual,
  log,
  log1p,
  logicalAnd,
  neg,
  primalOf,
  shapesEqual,
  sliceAxis,
  sqrt,
  square,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  toFlat,
  tril,
  type Value,
  where,
} from '../tensor/index.ts'
import { concreteFlat } from './_validate.ts'

/**
 * The set a distribution's values lie in. Elementwise kinds describe each entry; vector kinds (`simplex`, `counts`,
 * `oneHot`, `sphere`, `ordered`) describe the last axis; matrix kinds (`positiveDefinite`, `lowerCholesky`,
 * `correlationCholesky`, `correlation`) describe the last two axes. `bijectors.bijectorForSupport` maps the real line
 * (or real vectors) onto each of them.
 */
export type Support =
  | { readonly kind: 'real' }
  | { readonly kind: 'positive' }
  | { readonly kind: 'nonNegative' }
  | { readonly kind: 'interval'; readonly low: Value; readonly high: Value }
  | { readonly kind: 'greaterThan'; readonly low: Value }
  | { readonly kind: 'lessThan'; readonly high: Value }
  | { readonly kind: 'unitInterval' }
  | { readonly kind: 'integerInterval'; readonly low: number; readonly high: number }
  | { readonly kind: 'boolean' }
  | { readonly kind: 'circular' }
  | { readonly kind: 'simplex' }
  | { readonly kind: 'counts'; readonly total: number }
  | { readonly kind: 'oneHot' }
  | { readonly kind: 'sphere' }
  | { readonly kind: 'ordered' }
  | { readonly kind: 'positiveDefinite' }
  | { readonly kind: 'lowerCholesky' }
  | { readonly kind: 'correlationCholesky' }
  | { readonly kind: 'correlation' }

/**
 * A probability distribution: a plain object with its parameters (values, possibly traced, so everything differentiates
 * with respect to them) and pure methods. Shapes follow torch.distributions: a sample of shape `s` has shape
 * `[...s, ...batchShape, ...eventShape]`.
 *
 * `rsample` draws samples that are differentiable functions of the parameters (pathwise gradients); `sample` is the
 * same draw with gradients stopped. Distributions without a reparameterisation (`hasRsample` false) throw from
 * `rsample`.
 *
 * @concept distributions
 */
export interface Distribution {
  readonly name: string
  readonly params: Readonly<Record<string, Value>>
  readonly support: Support
  readonly batchShape: readonly number[]
  readonly eventShape: readonly number[]
  readonly hasRsample: boolean
  logProb(x: TensorLike): Value
  prob(x: TensorLike): Value
  cdf(x: TensorLike): Value
  logCdf(x: TensorLike): Value
  sf(x: TensorLike): Value
  logSf(x: TensorLike): Value
  ppf(q: TensorLike): Value
  isf(q: TensorLike): Value
  sample(key: Key, shape?: readonly number[]): Value
  rsample(key: Key, shape?: readonly number[]): Value
  mean(): Value
  variance(): Value
  stddev(): Value
  mode(): Value
  entropy(): Value
}

/** What a distribution must define; everything else in {@link Distribution} has a default. */
export interface DistributionSpec {
  readonly name: string
  readonly params: Readonly<Record<string, Value>>
  readonly support: Support
  readonly batchShape: readonly number[]
  readonly eventShape?: readonly number[]
  logProb(x: Value): Value
  cdf?(x: Value): Value
  logCdf?(x: Value): Value
  sf?(x: Value): Value
  logSf?(x: Value): Value
  ppf?(q: Value): Value
  isf?(q: Value): Value
  /** Reparameterised sampling; when given, `sample` is its stop-gradient version. */
  rsample?(key: Key, shape: readonly number[]): Value
  /** Non-reparameterised sampling (discrete distributions). */
  sample?(key: Key, shape: readonly number[]): Value
  mean(): Value
  variance(): Value
  stddev?(): Value
  mode(): Value
  entropy(): Value
}

/**
 * Throw the error for a method a distribution does not provide (a `not-implemented` MdmlError naming it).
 *
 * @param name - The distribution's name.
 * @param what - The missing method, with any explanation.
 * @returns Never: it always throws.
 * @throws MdmlError with code `not-implemented`.
 * @example
 * ```ts
 * import { unavailable } from 'mdml-engine/dists'
 * unavailable('example', 'mode') // => throws not-implemented
 * ```
 */
export function unavailable(name: string, what: string): never {
  throw new MdmlError('not-implemented', `${name}: ${what} is not available for this distribution`)
}

/**
 * Build a distribution from its defining parts. Defaults: `prob = exp(logProb)`, `sf = 1 - cdf`, `logCdf = log(cdf)`,
 * `logSf = log1p(-cdf)`, `isf(q) = ppf(1 - q)`, `stddev = sqrt(variance)`, `sample = stopGradient(rsample)`. A NaN
 * argument to `logProb`, `cdf`, `sf`, `logCdf`, `logSf`, `ppf` or `isf` gives NaN (for an event, a NaN coordinate).
 *
 * @param spec - The defining parts.
 * @returns The distribution.
 * @concept distributions
 * @index distribution Define a new family from its log density, sampler and moments.
 * @example
 * ```ts
 * import { defineDistribution } from 'mdml-engine/dists'
 * import { tensor, neg, item } from 'mdml-engine/tensor'
 * const unitExponential = defineDistribution({
 *   name: 'unitExponential', params: {}, support: { kind: 'positive' }, batchShape: [],
 *   logProb: (x) => neg(x), mean: () => tensor(1), variance: () => tensor(1), mode: () => tensor(0),
 *   entropy: () => tensor(1),
 * })
 * item(unitExponential.prob(0)) // => 1
 * ```
 */
export function defineDistribution(spec: DistributionSpec): Distribution {
  const value = (x: TensorLike): Value => asTensor(x)
  // NaN in gives NaN out: a comparison against the support is false for NaN, and the edge value it then selects (0, 1,
  // -Infinity) would read as a probability. Event-valued arguments are NaN when any coordinate is.
  const eventRank = (spec.eventShape ?? []).length
  const nanIn = (x: Value, rank: number): Value => {
    const m = isnan(x)
    return rank === 0
      ? m
      : any(
          m,
          Array.from({ length: rank }, (_, i) => -1 - i),
        )
  }
  const keep = (x: Value, out: Value, rank = eventRank): Value => where(nanIn(x, rank), Number.NaN, out)
  // The limits of a univariate distribution function at an infinite argument, whatever the family's formula gives there:
  // `atMinus` below every value and `atPlus` above them.
  const limit = (x: Value, out: Value, atMinus: number, atPlus: number): Value =>
    eventRank === 0 ? where(isfinite(x), out, where(greater(x, 0), atPlus, atMinus)) : out
  // The density of a univariate law is zero at an infinite argument, whatever the family's formula gives there (some give
  // NaN from an Infinity - Infinity): the log density is -Infinity.
  const logProb = (x: Value): Value =>
    keep(x, eventRank === 0 ? where(isfinite(x), spec.logProb(x), Number.NEGATIVE_INFINITY) : spec.logProb(x))
  const cdf = (x: Value): Value => keep(x, limit(x, spec.cdf ? spec.cdf(x) : unavailable(spec.name, 'cdf'), 0, 1))
  const ppf = (q: Value): Value => keep(q, spec.ppf ? spec.ppf(q) : unavailable(spec.name, 'ppf'), 0)
  const rsample = (key: Key, shape: readonly number[] = []): Value =>
    spec.rsample
      ? spec.rsample(key, shape)
      : unavailable(
          spec.name,
          'rsample (no reparameterisation; use a relaxed distribution, straightThrough or a score-function estimator)',
        )
  return {
    name: spec.name,
    params: spec.params,
    support: spec.support,
    batchShape: spec.batchShape,
    eventShape: spec.eventShape ?? [],
    hasRsample: spec.rsample !== undefined,
    logProb: (x) => logProb(value(x)),
    prob: (x) => exp(logProb(value(x))),
    cdf: (x) => cdf(value(x)),
    logCdf: (x) =>
      spec.logCdf
        ? keep(value(x), limit(value(x), spec.logCdf(value(x)), Number.NEGATIVE_INFINITY, 0))
        : log(cdf(value(x))),
    sf: (x) => (spec.sf ? keep(value(x), limit(value(x), spec.sf(value(x)), 1, 0)) : sub(1, cdf(value(x)))),
    logSf: (x) => {
      if (spec.logSf) return keep(value(x), limit(value(x), spec.logSf(value(x)), 0, Number.NEGATIVE_INFINITY))
      return spec.sf ? log(keep(value(x), limit(value(x), spec.sf(value(x)), 1, 0))) : log1p(neg(cdf(value(x))))
    },
    ppf: (q) => ppf(value(q)),
    isf: (q) => (spec.isf ? keep(value(q), spec.isf(value(q)), 0) : ppf(sub(1, value(q)))),
    sample: (key, shape = []) => (spec.sample ? spec.sample(key, shape) : stopGradient(rsample(key, shape))),
    rsample,
    mean: spec.mean,
    variance: spec.variance,
    stddev: spec.stddev ?? ((): Value => sqrt(spec.variance())),
    mode: spec.mode,
    entropy: spec.entropy,
  }
}

/**
 * The common batch shape of a distribution's parameters (their broadcast shape).
 *
 * @param params - The parameters.
 * @returns The broadcast shape.
 * @example
 * ```ts
 * import { batchShapeOf } from 'mdml-engine/dists'
 * import { tensor } from 'mdml-engine/tensor'
 * batchShapeOf(tensor([1, 2, 3]), tensor(0)) // => [3]
 * ```
 */
export function batchShapeOf(...params: readonly Value[]): number[] {
  return params.reduce<number[]>((s, p) => broadcastShapes(s, p.shape), [])
}

/**
 * The shape of a draw of `sampleShape` samples: `[...sampleShape, ...batchShape, ...eventShape]`.
 *
 * @param dist - The distribution.
 * @param sampleShape - The sample shape.
 * @returns The full shape.
 * @example
 * ```ts
 * import { extendedShape, normal } from 'mdml-engine/dists'
 * extendedShape(normal([0, 1], 1), [5]) // => [5, 2]
 * ```
 */
export function extendedShape(dist: Distribution, sampleShape: readonly number[] = []): number[] {
  return [...sampleShape, ...dist.batchShape, ...dist.eventShape]
}

/**
 * A value broadcast to a batch shape (so moments of a distribution have its batch shape even when some parameters do
 * not vary). Differentiable.
 *
 * @param v - The value.
 * @param batchShape - The batch shape.
 * @returns `v` broadcast to `batchShape` (unchanged when it already has it).
 * @example
 * ```ts
 * import { broadcastBatch } from 'mdml-engine/dists'
 * import { tensor } from 'mdml-engine/tensor'
 * broadcastBatch(tensor(1), [2, 3]).shape // => [2, 3]
 * ```
 */
export function broadcastBatch(v: Value, batchShape: readonly number[]): Value {
  const shape = broadcastShapes(v.shape, batchShape)
  return shapesEqual(shape, v.shape) ? v : broadcastTo(v, shape)
}

/**
 * How many trailing axes of a value one element of a support occupies: 0 for elementwise supports, 1 for vectors
 * (simplex, counts, one-hot, sphere, ordered), 2 for matrices (positive definite, Cholesky, correlation).
 *
 * @param support - The support.
 * @returns The rank.
 * @example
 * ```ts
 * import { supportRank } from 'mdml-engine/dists'
 * supportRank({ kind: 'simplex' }) // => 1
 * ```
 */
export function supportRank(support: Support): number {
  switch (support.kind) {
    case 'simplex':
    case 'counts':
    case 'oneHot':
    case 'sphere':
    case 'ordered':
      return 1
    case 'positiveDefinite':
    case 'lowerCholesky':
    case 'correlationCholesky':
    case 'correlation':
      return 2
    default:
      return 0
  }
}

const isInteger = (x: Value): Value => equal(floor(x), x)
const TOL = 1e-6

/**
 * Whether values lie in a support, as a boolean value with the event axes of the support reduced (vector supports drop
 * the last axis, matrix supports the last two). Matrix checks use a tolerance of 1e-6 for symmetry, unit norms and unit
 * diagonals.
 *
 * @param support - The support.
 * @param x - Values.
 * @returns A boolean value.
 * @example
 * ```ts
 * import { inSupport } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(inSupport({ kind: 'positive' }, [-1, 0, 2])) // => [0, 0, 1]
 * toArray(inSupport({ kind: 'simplex' }, [[0.2, 0.8], [0.5, 0.6]])) // => [1, 0]
 * ```
 */
export function inSupport(support: Support, x: TensorLike): Value {
  const v = asTensor(x)
  const finite = isfinite(v)
  switch (support.kind) {
    case 'real':
      return equal(v, v)
    case 'positive':
      return greater(v, 0)
    case 'nonNegative':
      return greaterEqual(v, 0)
    case 'interval':
      return logicalAnd(greaterEqual(v, support.low), lessEqual(v, support.high))
    case 'greaterThan':
      return greater(v, support.low)
    case 'lessThan':
      return less(v, support.high)
    case 'unitInterval':
      return logicalAnd(greaterEqual(v, 0), lessEqual(v, 1))
    case 'integerInterval':
      return logicalAnd(logicalAnd(isInteger(v), greaterEqual(v, support.low)), lessEqual(v, support.high))
    case 'boolean':
      return logicalAnd(isInteger(v), logicalAnd(greaterEqual(v, 0), lessEqual(v, 1)))
    case 'circular':
      return logicalAnd(greaterEqual(v, -Math.PI), lessEqual(v, Math.PI))
    case 'simplex':
      return logicalAnd(all(greaterEqual(v, 0), -1), less(abs(sub(sum(v, -1), 1)), TOL))
    case 'counts':
      return logicalAnd(all(logicalAnd(isInteger(v), greaterEqual(v, 0)), -1), equal(sum(v, -1), support.total))
    case 'oneHot':
      return logicalAnd(all(logicalAnd(isInteger(v), greaterEqual(v, 0)), -1), equal(sum(v, -1), 1))
    case 'sphere':
      return less(abs(sub(sum(square(v), -1), 1)), TOL)
    case 'ordered': {
      const n = v.shape[v.shape.length - 1] ?? 0
      if (n < 2) return all(finite, -1)
      const step = sub(sliceAxis(v, -1, 1, n), sliceAxis(v, -1, 0, n - 1))
      return logicalAnd(all(greater(step, 0), -1), all(finite, -1))
    }
    case 'lowerCholesky':
    case 'correlationCholesky': {
      const lowerOk = all(all(equal(sub(v, tril(v)), 0), -1), -1)
      const diag = diagonal(v, 0, -2, -1)
      const positive = all(greater(diag, 0), -1)
      const base = logicalAnd(lowerOk, positive)
      if (support.kind === 'lowerCholesky') return base
      return logicalAnd(base, all(less(abs(sub(sum(square(v), -1), 1)), TOL), -1))
    }
    case 'positiveDefinite':
    case 'correlation':
      return positiveDefiniteMask(v, support.kind === 'correlation')
  }
}

function positiveDefiniteMask(v: Value, unitDiagonal: boolean): Value {
  const t = primalOf(v)
  const n = t.shape[t.shape.length - 1] ?? 0
  const batch = t.shape.slice(0, -2)
  const count = batch.reduce((a, b) => a * b, 1)
  const flat = toFlat(t)
  const out: boolean[] = []
  for (let b = 0; b < count; b++) {
    const m = flat.slice(b * n * n, (b + 1) * n * n)
    out.push(isPdArray(m, n, unitDiagonal))
  }
  return broadcastTo(asTensor(out.length === 1 && batch.length === 0 ? (out[0] ?? false) : out), batch)
}

function isPdArray(m: readonly number[], n: number, unitDiagonal: boolean): boolean {
  const l = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    if (unitDiagonal && Math.abs((m[i * n + i] ?? 0) - 1) > TOL) return false
    for (let j = 0; j < n; j++) if (Math.abs((m[i * n + j] ?? 0) - (m[j * n + i] ?? 0)) > TOL) return false
  }
  for (let j = 0; j < n; j++) {
    let d = m[j * n + j] ?? 0
    for (let k = 0; k < j; k++) d -= (l[j * n + k] ?? 0) ** 2
    if (!(d > 0)) return false
    const ljj = Math.sqrt(d)
    l[j * n + j] = ljj
    for (let i = j + 1; i < n; i++) {
      let s = m[i * n + j] ?? 0
      for (let k = 0; k < j; k++) s -= (l[i * n + k] ?? 0) * (l[j * n + k] ?? 0)
      l[i * n + j] = s / ljj
    }
  }
  return true
}

/** A constraint on a distribution parameter, checked by {@link validateParams}. */
export type ParameterConstraint =
  | 'real'
  | 'finite'
  | 'positive'
  | 'nonNegative'
  | 'unitInterval'
  | 'openUnitInterval'
  | 'integer'
  | 'nonNegativeInteger'
  | 'positiveInteger'

// `positive` and `nonNegative` exclude Infinity: an infinite scale, rate or df makes NaN at the first use.
const CHECKS: Readonly<Record<ParameterConstraint, (v: number) => boolean>> = {
  real: (v) => !Number.isNaN(v),
  finite: (v) => Number.isFinite(v),
  positive: (v) => v > 0 && v < Number.POSITIVE_INFINITY,
  nonNegative: (v) => v >= 0 && v < Number.POSITIVE_INFINITY,
  unitInterval: (v) => v >= 0 && v <= 1,
  openUnitInterval: (v) => v > 0 && v < 1,
  integer: (v) => Number.isInteger(v),
  nonNegativeInteger: (v) => Number.isInteger(v) && v >= 0,
  positiveInteger: (v) => Number.isInteger(v) && v > 0,
}

/**
 * Check distribution parameters against their constraints (torch's `validate_args`), throwing on the first violation.
 * Traced parameters are checked through their concrete values; values with none (inside `vmap`) are skipped.
 *
 * @param family - The distribution's name, for the message.
 * @param checks - Parameter name to `[value, constraint]`.
 * @returns Nothing; it throws on a violation.
 * @throws MdmlError with code `value` naming the parameter and the constraint.
 * @example
 * ```ts
 * import { validateParams } from 'mdml-engine/dists'
 * import { tensor } from 'mdml-engine/tensor'
 * validateParams('normal', { scale: [tensor(-1), 'positive'] }) // => throws value
 * ```
 */
export function validateParams(
  family: string,
  checks: Readonly<Record<string, readonly [Value, ParameterConstraint]>>,
): void {
  for (const [name, [v, constraint]] of Object.entries(checks)) {
    const values = concreteFlat(v)
    if (values === undefined) continue
    const ok = CHECKS[constraint]
    const bad = values.find((x) => !ok(x))
    if (bad !== undefined)
      throw new MdmlError('value', `${family}: ${name} must be ${constraint} (got ${bad})`, { parameter: name })
  }
}

/**
 * Sum a value over its last `n` axes (torch's `_sum_rightmost`): how log-densities of independent coordinates and
 * log-Jacobians of elementwise maps combine into one per event.
 *
 * @param x - The value.
 * @param n - How many trailing axes to sum (0 returns x).
 * @returns The sum.
 * @example
 * ```ts
 * import { sumRightmost } from 'mdml-engine/dists'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(sumRightmost([[1, 2], [3, 4]], 1)) // => [3, 7]
 * ```
 */
export function sumRightmost(x: TensorLike, n: number): Value {
  const v = asTensor(x)
  if (n <= 0) return v
  const rank = v.shape.length
  return sum(
    v,
    Array.from({ length: n }, (_, i) => rank - 1 - i),
  )
}
