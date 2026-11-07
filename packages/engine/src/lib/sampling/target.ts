/**
 * What samplers sample: the `LogDensity` target (an unnormalised log density with its gradient and the map back to
 * constrained parameters), built from a differentiable function, plain number functions or a distribution sampled on
 * unconstrained reals, and the `Proposal` used by importance, independence and rejection samplers. Also the standard
 * test targets Neal's funnel and Haario's banana.
 *
 * @index structure Log-density targets that every sampler runs on.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { bijectors, type Distribution, type Support } from '../dists/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import {
  add,
  concat,
  exp,
  fromStorage,
  item,
  mul,
  reshape,
  split,
  square,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'

/**
 * A target for Monte Carlo: an unnormalised log density on R^dim with its gradient, the map back to the model's
 * constrained space, and (when it was built from a differentiable function) that function, for samplers that need
 * Hessians. Samplers only ever see unconstrained coordinates: a target built from a constrained density carries the
 * change of variables in its log density, and `constrain` maps draws back.
 *
 * Every evaluation is total: a NaN log density reads as -Infinity, and a non-finite gradient marks the point as outside
 * the support (value -Infinity, gradient zero), so samplers reject it instead of propagating NaN.
 *
 * @concept log-density
 */
export interface LogDensity {
  /** The number of unconstrained coordinates. */
  readonly dim: number
  /** A name per coordinate (`x[0]`, `x[1]`, ... by default). */
  readonly names: readonly string[]
  /** Whether {@link LogDensity.valueAndGrad} is available. */
  readonly hasGradient: boolean
  /** The supports of the constrained coordinates, when the target was built from constrained densities. */
  readonly constraints?: readonly Support[]
  /** log p(x) up to a constant (NaN reads as -Infinity). */
  logDensity(x: ArrayLike<number>): number
  /** log p(x) and its gradient (from autodiff, or the hand-written gradient of a numeric target). */
  valueAndGrad(x: ArrayLike<number>): { readonly value: number; readonly grad: Float64Array }
  /** Unconstrained coordinates to the model's parameters (identity for targets on R^dim). */
  constrain(x: ArrayLike<number>): Float64Array
  /** The model's parameters to unconstrained coordinates. */
  unconstrain(y: ArrayLike<number>): Float64Array
  /** The differentiable log density on a `[dim]` value, when known (Riemannian HMC differentiates it twice). */
  readonly fn?: (x: Value) => Value
}

/** Options shared by the target constructors. */
export interface TargetOptions {
  /** A name per coordinate. */
  readonly names?: readonly string[]
}

function defaultNames(dim: number, names: readonly string[] | undefined): readonly string[] {
  if (names !== undefined && names.length !== dim) {
    throw new MdmlError('value', `target: ${names.length} names for ${dim} coordinates`)
  }
  return names ?? Array.from({ length: dim }, (_, i) => `x[${i}]`)
}

function checkDim(dim: number, name: string): void {
  if (!Number.isInteger(dim) || dim < 1) throw new MdmlError('value', `${name}: dim must be a positive integer`)
}

function point(x: ArrayLike<number>, dim: number, name: string): Value {
  if (x.length !== dim) throw new MdmlError('shape', `${name}: expected ${dim} coordinates, got ${x.length}`)
  return fromStorage(Float64Array.from(x), [dim], 'float64')
}

/** NaN reads as -Infinity; a non-finite gradient puts the point outside the support. */
function total(value: number, grad: Float64Array): { readonly value: number; readonly grad: Float64Array } {
  const v = Number.isNaN(value) ? Number.NEGATIVE_INFINITY : value
  let finite = Number.isFinite(v)
  for (let i = 0; finite && i < grad.length; i++) finite = Number.isFinite(grad[i] ?? Number.NaN)
  if (finite || v === Number.POSITIVE_INFINITY) return { value: v, grad }
  return { value: Number.NEGATIVE_INFINITY, grad: new Float64Array(grad.length) }
}

function identity(x: ArrayLike<number>): Float64Array {
  return Float64Array.from(x)
}

/**
 * A target from a differentiable log density of a `[dim]` vector; gradients come from reverse-mode autodiff, and the
 * function is kept for samplers that differentiate further (Riemannian HMC).
 *
 * @param fn - log p(x) for x of shape `[dim]`, returning a scalar value.
 * @param dim - The number of coordinates.
 * @param options - Coordinate names.
 * @returns The target.
 * @throws MdmlError with code `value` when `dim` is not a positive integer.
 * @concept log-density
 * @example
 * ```ts
 * import { target } from 'mdml-engine/sampling'
 * import { mul, square, sum } from 'mdml-engine/tensor'
 * const t = target((x) => mul(-0.5, sum(square(x))), 2)
 * t.logDensity([1, 2]) // => -2.5
 * Array.from(t.valueAndGrad([1, 2]).grad) // => [-1, -2]
 * ```
 */
export function target(fn: (x: Value) => Value, dim: number, options: TargetOptions = {}): LogDensity {
  checkDim(dim, 'target')
  const vg = valueAndGrad((x: Value) => fn(x))
  return {
    dim,
    names: defaultNames(dim, options.names),
    hasGradient: true,
    fn,
    logDensity: (x) => {
      const v = item(fn(point(x, dim, 'target')))
      return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
    },
    valueAndGrad: (x) => {
      const r = vg(point(x, dim, 'target'))
      return total(item(r.value), Float64Array.from(toFlat(r.grad)))
    },
    constrain: identity,
    unconstrain: identity,
  }
}

/** Options of {@link numericTarget}. */
export interface NumericTargetOptions extends TargetOptions {
  /** The gradient of the log density (without it, gradient-based samplers refuse the target). */
  readonly grad?: (x: Float64Array) => ArrayLike<number>
}

/**
 * A target from plain number functions: the fastest form, for hand-written densities and for tests. Without `grad`
 * only gradient-free samplers (Metropolis, slice, SMC with random-walk moves) apply.
 *
 * @param logDensity - log p(x) on a `Float64Array` of length `dim`.
 * @param dim - The number of coordinates.
 * @param options - The gradient and coordinate names.
 * @returns The target.
 * @throws MdmlError with code `value` when `dim` is not a positive integer.
 * @concept log-density
 * @example
 * ```ts
 * import { numericTarget } from 'mdml-engine/sampling'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })
 * t.valueAndGrad([2]).value // => -2
 * t.hasGradient // => true
 * ```
 */
export function numericTarget(
  logDensity: (x: Float64Array) => number,
  dim: number,
  options: NumericTargetOptions = {},
): LogDensity {
  checkDim(dim, 'numericTarget')
  const g = options.grad
  const at = (x: ArrayLike<number>): Float64Array => {
    if (x.length !== dim) throw new MdmlError('shape', `numericTarget: expected ${dim} coordinates, got ${x.length}`)
    return Float64Array.from(x)
  }
  return {
    dim,
    names: defaultNames(dim, options.names),
    hasGradient: g !== undefined,
    logDensity: (x) => {
      const v = logDensity(at(x))
      return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
    },
    valueAndGrad: (x) => {
      if (g === undefined) throw new MdmlError('value', 'numericTarget: this target has no gradient')
      const p = at(x)
      return total(logDensity(p), Float64Array.from(g(p)))
    },
    constrain: identity,
    unconstrain: identity,
  }
}

/**
 * log p(f(u)) + log |det df/du| as one scalar: the bijector's log-Jacobian is summed over every axis, so elementwise
 * (rank-0) bijectors on a vector work as well as vector bijectors (the target has no batch axes).
 */
function changeOfVariables(logDensity: (y: Value) => Value, bijector: bijectors.Bijector): (u: Value) => Value {
  return (u) => add(sum(logDensity(bijector.forward(u))), sum(bijector.forwardLogDetJacobian(u)))
}

function product(shape: readonly number[]): number {
  let n = 1
  for (const s of shape) n *= s
  return n
}

/**
 * A target from a distribution's log density, on unconstrained reals by default: the support's bijector
 * (`bijectors.bijectorForSupport`, NumPyro's `biject_to`) maps R^dim onto the support and its log-Jacobian is added, so
 * a gamma becomes a density on the log scale and a Dirichlet one on R^(K-1). `constrain` maps draws back. Event shapes
 * are flattened row-major; the distribution must have an empty batch shape.
 *
 * @param dist - The distribution (continuous, with an empty batch shape).
 * @param options - `unconstrained` (default true; false samples the raw density, for supports that are already R^n),
 *   and coordinate names.
 * @returns The target.
 * @throws MdmlError with code `shape` for a non-empty batch shape, and `not-implemented` for supports without a
 *   bijection from the reals (discrete supports, the sphere).
 * @concept log-density
 * @example
 * ```ts
 * import { targetFromDistribution } from 'mdml-engine/sampling'
 * import { exponential } from 'mdml-engine/dists'
 * const t = targetFromDistribution(exponential(1))
 * // u = log x: log p(e^u) + u = -1 at u = 0.
 * t.logDensity([0]) // => -1
 * Array.from(t.constrain([0])) // => [1]
 * ```
 */
export function targetFromDistribution(
  dist: Distribution,
  options: TargetOptions & { readonly unconstrained?: boolean } = {},
): LogDensity {
  if (dist.batchShape.length > 0) {
    throw new MdmlError('shape', `targetFromDistribution: batch shape [${dist.batchShape.join(', ')}] must be empty`)
  }
  const event = dist.eventShape
  if (options.unconstrained === false) {
    const dim = product(event)
    const fn = (x: Value): Value => dist.logProb(reshape(x, [...event]))
    return { ...target(fn, dim, options), constraints: [dist.support] }
  }
  const bijector = bijectors.bijectorForSupport(dist.support)
  const shape = bijector.inverseShape(event)
  const dim = product(shape)
  const density = changeOfVariables((y) => dist.logProb(y), bijector)
  const fn = (x: Value): Value => density(reshape(x, shape))
  return {
    ...target(fn, dim, options),
    constraints: [dist.support],
    constrain: (x) => Float64Array.from(toFlat(bijector.forward(reshape(point(x, dim, 'constrain'), shape)))),
    unconstrain: (y) => {
      const n = product(event)
      if (y.length !== n) throw new MdmlError('shape', `unconstrain: expected ${n} values, got ${y.length}`)
      return Float64Array.from(toFlat(bijector.inverse(reshape(Float64Array.from(y), [...event]))))
    },
  }
}

/**
 * A target from a log density on a constrained space, sampled on unconstrained reals: with one elementwise support per
 * coordinate (positive, intervals, bounds, the unit interval, circular) each coordinate gets its own bijector, or one
 * bijector (from `dists/bijectors`) maps the whole vector. The log-Jacobian is added, as in Stan and NumPyro.
 *
 * @param fn - log p(y) on the constrained space, for y of shape `[n]` (n is `dim` for elementwise supports, or the
 *   bijector's output size).
 * @param transform - A support per coordinate, or a bijector from R^dim.
 * @param dim - The number of unconstrained coordinates.
 * @param options - Coordinate names.
 * @returns The target.
 * @throws MdmlError with code `shape` when the number of supports differs from `dim`, and `value` for supports
 *   that are not elementwise.
 * @concept log-density
 * @example
 * ```ts
 * import { unconstrainedTarget } from 'mdml-engine/sampling'
 * import { neg, sum } from 'mdml-engine/tensor'
 * // An Exp(1) density on (0, inf), sampled as u = log y.
 * const t = unconstrainedTarget((y) => neg(sum(y)), [{ kind: 'positive' }], 1)
 * t.logDensity([0]) // => -1
 * Array.from(t.constrain([0])) // => [1]
 * ```
 */
export function unconstrainedTarget(
  fn: (y: Value) => Value,
  transform: readonly Support[] | bijectors.Bijector,
  dim: number,
  options: TargetOptions = {},
): LogDensity {
  checkDim(dim, 'unconstrainedTarget')
  const bijector = Array.isArray(transform)
    ? elementwise(transform as readonly Support[], dim)
    : (transform as bijectors.Bijector)
  const density = changeOfVariables(fn, bijector)
  const base = target((x) => density(x), dim, options)
  return {
    ...base,
    ...(Array.isArray(transform) ? { constraints: transform as readonly Support[] } : {}),
    constrain: (x) => Float64Array.from(toFlat(bijector.forward(point(x, dim, 'constrain')))),
    unconstrain: (y) => Float64Array.from(toFlat(bijector.inverse(Float64Array.from(y)))),
  }
}

const ELEMENTWISE = new Set(['real', 'positive', 'nonNegative', 'interval', 'greaterThan', 'lessThan', 'unitInterval'])

/** One rank-0 bijector per coordinate, applied to the split vector. */
function elementwise(supports: readonly Support[], dim: number): bijectors.Bijector {
  if (supports.length !== dim) {
    throw new MdmlError('shape', `unconstrainedTarget: ${supports.length} supports for ${dim} coordinates`)
  }
  for (const s of supports) {
    if (!ELEMENTWISE.has(s.kind) && s.kind !== 'circular') {
      throw new MdmlError('value', `unconstrainedTarget: the ${s.kind} support is not elementwise; pass a bijector`)
    }
  }
  const parts = supports.map((s) => bijectors.bijectorForSupport(s))
  const each = (x: TensorLike, f: (b: bijectors.Bijector, xi: Value) => Value): Value =>
    concat(split(x, dim).map((xi, i) => f(parts[i] as bijectors.Bijector, xi)))
  return bijectors.defineBijector({
    name: 'elementwise',
    inputRank: 1,
    outputRank: 1,
    forward: (x) => each(x, (b, xi) => b.forward(xi)),
    inverse: (y) => each(y, (b, yi) => b.inverse(yi)),
    forwardLogDetJacobian: (x) => sum(each(x, (b, xi) => b.forwardLogDetJacobian(xi))),
  })
}

/**
 * A distribution to draw proposals from, with its log density on the same coordinates as a target: the proposal of
 * independence Metropolis, importance sampling and rejection sampling, and the prior of SMC. `sampleMany` and
 * `logProbMany` work on row-major `[n, dim]` buffers.
 *
 * @concept proposal-distribution
 */
export interface Proposal {
  readonly dim: number
  sample(key: Key): Float64Array
  logProb(x: ArrayLike<number>): number
  sampleMany(key: Key, n: number): Float64Array
  logProbMany(xs: Float64Array): Float64Array
}

/**
 * A proposal from plain functions; the batched forms loop over the single ones with keys `foldIn(key, i)`.
 *
 * @param spec - The dimension, a sampler and a log density.
 * @returns The proposal.
 * @throws MdmlError with code `value` when `dim` is not a positive integer.
 * @concept proposal-distribution
 * @example
 * ```ts
 * import { proposal } from 'mdml-engine/sampling'
 * import { key, uniform } from 'mdml-engine/random'
 * const q = proposal({ dim: 1, sample: (k) => uniform(k, [1]).data, logProb: () => 0 })
 * q.sampleMany(key(0), 3).length // => 3
 * ```
 */
export function proposal(spec: {
  readonly dim: number
  readonly sample: (key: Key) => ArrayLike<number>
  readonly logProb: (x: Float64Array) => number
}): Proposal {
  checkDim(spec.dim, 'proposal')
  const { dim } = spec
  const sample = (k: Key): Float64Array => {
    const x = Float64Array.from(spec.sample(k))
    if (x.length !== dim) throw new MdmlError('shape', `proposal: a draw has ${x.length} values, need ${dim}`)
    return x
  }
  const logProb = (x: ArrayLike<number>): number => {
    const v = spec.logProb(Float64Array.from(x))
    return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  return {
    dim,
    sample,
    logProb,
    sampleMany: (k, n) => {
      const out = new Float64Array(n * dim)
      for (let i = 0; i < n; i++) out.set(sample(foldIn(k, i)), i * dim)
      return out
    },
    logProbMany: (xs) =>
      Float64Array.from({ length: xs.length / dim }, (_, i) => logProb(xs.subarray(i * dim, (i + 1) * dim))),
  }
}

/**
 * A proposal from a distribution with an empty batch shape: draws and densities on its (flattened) event, batched
 * through the distribution's own vectorised `sample` and `logProb`.
 *
 * @param dist - The distribution.
 * @returns The proposal.
 * @throws MdmlError with code `shape` for a non-empty batch shape.
 * @concept proposal-distribution
 * @example
 * ```ts
 * import { proposalFromDistribution } from 'mdml-engine/sampling'
 * import { normal } from 'mdml-engine/dists'
 * const q = proposalFromDistribution(normal(0, 1))
 * Math.abs(q.logProb([0]) + 0.5 * Math.log(2 * Math.PI)) < 1e-15 // => true
 * ```
 */
export function proposalFromDistribution(dist: Distribution): Proposal {
  if (dist.batchShape.length > 0) {
    throw new MdmlError('shape', `proposalFromDistribution: batch shape [${dist.batchShape.join(', ')}] must be empty`)
  }
  const event = [...dist.eventShape]
  const dim = product(event)
  const flat = (v: Value): Float64Array => Float64Array.from(toFlat(v))
  return {
    dim,
    sample: (k) => flat(dist.sample(k)),
    logProb: (x) => {
      const v = item(dist.logProb(reshape(Float64Array.from(x), event)))
      return Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
    },
    sampleMany: (k, n) => flat(dist.sample(k, [n])),
    logProbMany: (xs) => {
      const n = xs.length / dim
      const v = flat(dist.logProb(reshape(Float64Array.from(xs), [n, ...event])))
      return v.map((x) => (Number.isNaN(x) ? Number.NEGATIVE_INFINITY : x))
    },
  }
}

/** Options of {@link nealsFunnel}. */
export interface FunnelOptions {
  /** Sample the non-centred form: v = scale z_0, x_i = exp(v / 2) z_i with every z standard normal (default false). */
  readonly nonCentred?: boolean
  /** The standard deviation of the neck variable v (default 3, as Neal). */
  readonly scale?: number
}

/**
 * Neal's funnel in `dim` coordinates, v ~ N(0, scale^2) and x_i | v ~ N(0, e^v) for i = 1..dim-1: the standard target
 * whose curvature changes by orders of magnitude, where centred HMC diverges in the neck. The non-centred form samples
 * independent standard normals and `constrain` maps them to (v, x), which removes the funnel geometry exactly.
 *
 * @param dim - The number of coordinates (at least 2): v first, then the x_i.
 * @param options - Centred or non-centred form, and the scale of v.
 * @returns The target, with names `v, x[1], ...`.
 * @throws MdmlError with code `value` when `dim < 2`.
 * @concept neals-funnel
 * @reference R. M. Neal, Slice sampling (Annals of Statistics 2003), section 8.
 * @example
 * ```ts
 * import { nealsFunnel } from 'mdml-engine/sampling'
 * const f = nealsFunnel(2)
 * Math.abs(f.logDensity([0, 0]) - -0.5 * Math.log(2 * Math.PI * 9) + 0.5 * Math.log(2 * Math.PI)) < 1e-12 // => true
 * Array.from(nealsFunnel(2, { nonCentred: true }).constrain([1, 1])).map((v) => Math.round(v * 1e6) / 1e6) // => [3, 4.481689]
 * ```
 */
export function nealsFunnel(dim: number, options: FunnelOptions = {}): LogDensity {
  if (!(Number.isInteger(dim) && dim >= 2)) throw new MdmlError('value', 'nealsFunnel: dim must be at least 2')
  const scale = options.scale ?? 3
  const names = ['v', ...Array.from({ length: dim - 1 }, (_, i) => `x[${i + 1}]`)]
  const LOG2PI = Math.log(2 * Math.PI)
  if (options.nonCentred === true) {
    const fn = (z: Value): Value => sub(mul(-0.5, sum(square(z))), (dim * LOG2PI) / 2)
    const forward = (z: Float64Array): Float64Array => {
      const out = new Float64Array(dim)
      const v = scale * (z[0] ?? 0)
      out[0] = v
      for (let i = 1; i < dim; i++) out[i] = Math.exp(v / 2) * (z[i] ?? 0)
      return out
    }
    return {
      ...target(fn, dim, { names }),
      constrain: (z) => forward(Float64Array.from(z)),
      unconstrain: (y) => {
        const out = new Float64Array(dim)
        const v = y[0] ?? 0
        out[0] = v / scale
        for (let i = 1; i < dim; i++) out[i] = (y[i] ?? 0) / Math.exp(v / 2)
        return out
      },
    }
  }
  const fn = (x: Value): Value => {
    const [v, rest] = split(x, [1]) as [Value, Value]
    const vs = sum(v)
    // log N(v; 0, scale^2) + sum_i log N(x_i; 0, e^v).
    const neck = sub(mul(-0.5 / (scale * scale), square(vs)), 0.5 * LOG2PI + Math.log(scale))
    const body = sub(mul(-0.5, sum(mul(square(rest), exp(mul(-1, vs))))), mul(0.5 * (dim - 1), add(vs, LOG2PI)))
    return add(neck, body)
  }
  return target(fn, dim, { names })
}

/** Options of {@link banana}. */
export interface BananaOptions {
  /** The curvature b (default 0.03). */
  readonly curvature?: number
  /** The standard deviation of the first coordinate (default 10). */
  readonly scale?: number
}

/**
 * Haario's banana: x_1 ~ N(0, scale^2) and x_2 | x_1 ~ N(b x_1^2 - b scale^2, 1), a twisted Gaussian whose moments are
 * known exactly (E x = 0, Var x_1 = scale^2, Var x_2 = 1 + 2 b^2 scale^4), the classic test of adaptive Metropolis.
 *
 * @param options - Curvature b and the scale of x_1.
 * @returns The two-dimensional target.
 * @concept banana-distribution
 * @reference H. Haario, E. Saksman and J. Tamminen, Adaptive proposal distribution for random walk Metropolis (1999).
 * @example
 * ```ts
 * import { banana } from 'mdml-engine/sampling'
 * const b = banana({ curvature: 0.03 })
 * // At the mode (0, -3): log N(0; 0, 100) + log N(0; 0, 1).
 * Math.abs(b.logDensity([0, -3]) - (-Math.log(2 * Math.PI) - Math.log(10))) < 1e-12 // => true
 * ```
 */
export function banana(options: BananaOptions = {}): LogDensity {
  const b = options.curvature ?? 0.03
  const s = options.scale ?? 10
  const fn = (x: Value): Value => {
    const [x1, x2] = split(x, 2).map((v) => sum(v)) as [Value, Value]
    const r = sub(x2, sub(mul(b, square(x1)), b * s * s))
    return sub(add(mul(-0.5 / (s * s), square(x1)), mul(-0.5, square(r))), Math.log(2 * Math.PI) + Math.log(s))
  }
  return target(fn, 2, { names: ['x[0]', 'x[1]'] })
}
