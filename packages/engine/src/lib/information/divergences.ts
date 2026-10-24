import { grad } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { entr, klDiv, relEntr } from '../numerics/special/index.ts'
import {
  abs,
  add,
  div,
  equal,
  exp,
  greater,
  log,
  logicalAnd,
  max,
  mul,
  neg,
  promoteFloat,
  reshape,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import { float, inBase } from './_shared.ts'

/**
 * How the terms of a divergence are added up. Without `dx` or `x` the inputs are discrete (probability vectors or
 * positive measures) and the terms are summed; with them the inputs are densities sampled on a grid and the terms are
 * integrated by the trapezoidal rule (uniform spacing `dx`, or the grid points `x`).
 */
export interface GridOptions {
  /** The axis of the outcomes or grid (default -1). */
  readonly axis?: number
  /** Uniform grid spacing: integrate instead of summing. */
  readonly dx?: number
  /** Grid points (increasing) along the axis: integrate by the trapezoidal rule on them. */
  readonly x?: TensorLike
  /** Logarithm base for the divergences measured in nats (default e). */
  readonly base?: number
}

/** Orders this close to a removable singularity use the limiting formula. */
const NEAR = 1e-8

/** Trapezoid weights shaped to broadcast along `axis` of a value of the given shape (undefined: plain sum). */
function gridWeights(shape: readonly number[], options: GridOptions): Value | undefined {
  if (options.dx === undefined && options.x === undefined) return undefined
  const rank = shape.length
  const axis = (options.axis ?? -1) < 0 ? (options.axis ?? -1) + rank : (options.axis ?? -1)
  const n = shape[axis] ?? 0
  if (n < 2) throw new WhiteboxError('shape', 'divergences: a grid needs at least two points')
  const w = new Float64Array(n)
  if (options.x !== undefined) {
    const xs = toFlat(float(options.x))
    if (xs.length !== n) throw new WhiteboxError('shape', `divergences: ${xs.length} grid points for an axis of ${n}`)
    for (let i = 0; i + 1 < n; i++) {
      const h = (xs[i + 1] ?? 0) - (xs[i] ?? 0)
      if (!(h > 0)) throw new WhiteboxError('value', 'divergences: grid points must increase')
      w[i] = (w[i] ?? 0) + h / 2
      w[i + 1] = (w[i + 1] ?? 0) + h / 2
    }
  } else {
    const dx = options.dx ?? 1
    w.fill(dx)
    w[0] = dx / 2
    w[n - 1] = dx / 2
  }
  return reshape(
    float(w),
    shape.map((s, i) => (i === axis ? s : 1)),
  )
}

/** Sum (or integrate on the grid) the elementwise terms along the axis. */
function integrate(terms: Value, options: GridOptions): Value {
  const w = gridWeights(terms.shape, options)
  return sum(w === undefined ? terms : mul(terms, w), options.axis ?? -1)
}

function pair(p: TensorLike, q: TensorLike): [Value, Value] {
  const [a, b] = promoteFloat(p, q) as [Value, Value]
  return [a, b]
}

/** p^a q^(1 - a) elementwise, with the limits at zero (0 where either factor vanishes with a positive power). */
function powerMean(p: Value, q: Value, alpha: number): Value {
  const pp = greater(p, 0)
  const qp = greater(q, 0)
  const both = logicalAnd(pp, qp)
  const inner = exp(add(mul(log(where(both, p, 1)), alpha), mul(log(where(both, q, 1)), 1 - alpha)))
  // p = 0 < q: p^a is 0 for a > 0, 1 for a = 0, inf for a < 0; q = 0 < p mirrors with 1 - a.
  const pZero = alpha > 0 ? 0 : alpha === 0 ? q : Number.POSITIVE_INFINITY
  const qZero = alpha < 1 ? 0 : alpha === 1 ? p : Number.POSITIVE_INFINITY
  return where(both, inner, where(pp, qZero, where(qp, pZero, 0)))
}

/** The named generators of {@link fDivergence}. */
export type FGeneratorName =
  | 'kl'
  | 'reverseKl'
  | 'squaredHellinger'
  | 'totalVariation'
  | 'pearson'
  | 'neyman'
  | 'jensenShannon'
  | 'triangular'

/** A convex generator f with f(1) = 0, by name or as a function of the ratio t = p / q. */
export type FGenerator = FGeneratorName | ((t: Value) => Value)

const ratioSquared = (num: Value, den: Value): Value => {
  const positive = greater(den, 0)
  return where(positive, div(square(num), where(positive, den, 1)), where(equal(num, 0), 0, Number.POSITIVE_INFINITY))
}

const NAMED: Readonly<Record<FGeneratorName, (p: Value, q: Value) => Value>> = {
  kl: (p, q) => relEntr(p, q),
  reverseKl: (p, q) => relEntr(q, p),
  squaredHellinger: (p, q) => mul(square(sub(sqrt(p), sqrt(q))), 0.5),
  totalVariation: (p, q) => mul(abs(sub(p, q)), 0.5),
  pearson: (p, q) => ratioSquared(sub(p, q), q),
  neyman: (p, q) => ratioSquared(sub(p, q), p),
  jensenShannon: (p, q) => {
    const m = mul(add(p, q), 0.5)
    return mul(add(relEntr(p, m), relEntr(q, m)), 0.5)
  },
  triangular: (p, q) => ratioSquared(sub(p, q), add(p, q)),
}

/**
 * The f-divergence D_f(p || q) = sum q f(p / q) (Csiszar, Ali-Silvey) for a convex f with f(1) = 0, on probability
 * vectors or on densities sampled on a grid. The named generators handle zeros by their limits: `kl` (t log t),
 * `reverseKl` (-log t), `squaredHellinger` ((sqrt t - 1)^2 / 2, so 1 - the Bhattacharyya coefficient), `totalVariation`
 * (|t - 1| / 2), `pearson` and `neyman` chi-square, `jensenShannon` and `triangular` discrimination. A custom f is
 * evaluated as q f(p / q) and needs q > 0 everywhere.
 *
 * @param p - The first distribution (probabilities, or density values on the grid).
 * @param q - The second.
 * @param f - The generator, by name or as a function of the ratio.
 * @param options - Axis, grid (`dx` or `x`) and, for the logarithmic generators, the base.
 * @returns D_f(p || q), differentiable where the terms are smooth.
 * @concept f-divergence
 * @example
 * ```ts
 * import { fDivergence } from 'whitebox-engine/information'
 * import { item, log, mul, sub, type Value } from 'whitebox-engine/tensor'
 * item(fDivergence([0.5, 0.5], [1, 0], 'totalVariation')) // => 0.5
 * // A custom generator: f(t) = t log t is the KL divergence.
 * item(fDivergence([0.5, 0.5], [0.5, 0.5], (t: Value) => mul(t, log(t)))) // => 0
 * ```
 */
export function fDivergence(p: TensorLike, q: TensorLike, f: FGenerator, options: GridOptions = {}): Value {
  const [a, b] = pair(p, q)
  const terms = typeof f === 'function' ? mul(b, f(div(a, b))) : NAMED[f](a, b)
  const out = integrate(terms, options)
  const logarithmic = f === 'kl' || f === 'reverseKl' || f === 'jensenShannon'
  return logarithmic ? inBase(out, options.base) : out
}

/**
 * The Renyi divergence of order alpha, D_a(p || q) = log(sum p^a q^(1 - a)) / (a - 1): the KL divergence at a = 1,
 * -log q(support of p) at a = 0, log max p / q at a = Infinity; non-decreasing in a.
 *
 * @param p - The first distribution (probabilities, or density values on the grid).
 * @param q - The second.
 * @param alpha - The order, a >= 0 (Infinity allowed).
 * @param options - Axis, grid and base.
 * @returns D_a(p || q).
 * @concept renyi-divergence
 * @example
 * ```ts
 * import { renyiDivergence } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(renyiDivergence([1, 0], [0.5, 0.5], 2, { base: 2 })) // => 1
 * ```
 */
export function renyiDivergence(p: TensorLike, q: TensorLike, alpha: number, options: GridOptions = {}): Value {
  if (!(alpha >= 0)) throw new WhiteboxError('value', `renyiDivergence: alpha must be >= 0, got ${alpha}`)
  const [a, b] = pair(p, q)
  if (Math.abs(alpha - 1) < NEAR) return inBase(integrate(relEntr(a, b), options), options.base)
  if (alpha === Number.POSITIVE_INFINITY) {
    const support = greater(a, 0)
    const logRatio = sub(log(where(support, a, 1)), log(where(support, b, 1)))
    return inBase(max(where(support, logRatio, Number.NEGATIVE_INFINITY), options.axis ?? -1), options.base)
  }
  return inBase(div(log(integrate(powerMean(a, b, alpha), options)), alpha - 1), options.base)
}

/**
 * The Renyi entropy of order alpha, H_a(p) = log(sum p^a) / (1 - a): the Hartley entropy log |support| at a = 0, the
 * Shannon entropy at a = 1, the collision entropy -log sum p^2 at a = 2 and the min-entropy -log max p at a = Infinity;
 * non-increasing in a. On a grid it is the differential Renyi entropy. The input is used as given (not normalised).
 *
 * @param p - Probabilities (or density values on the grid).
 * @param alpha - The order, a >= 0 (Infinity allowed).
 * @param options - Axis, grid and base.
 * @returns H_a(p).
 * @concept renyi-entropy
 * @example
 * ```ts
 * import { renyiEntropy } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(renyiEntropy([0.25, 0.25, 0.25, 0.25], 2, { base: 2 })) // => 2
 * ```
 */
export function renyiEntropy(p: TensorLike, alpha: number, options: GridOptions = {}): Value {
  if (!(alpha >= 0)) throw new WhiteboxError('value', `renyiEntropy: alpha must be >= 0, got ${alpha}`)
  const a = float(p)
  if (Math.abs(alpha - 1) < NEAR) return inBase(integrate(entr(a), options), options.base)
  if (alpha === Number.POSITIVE_INFINITY) return inBase(neg(log(max(a, options.axis ?? -1))), options.base)
  const positive = greater(a, 0)
  const powered = alpha === 0 ? where(positive, 1, 0) : where(positive, exp(mul(log(where(positive, a, 1)), alpha)), 0)
  return inBase(div(log(integrate(powered, options)), 1 - alpha), options.base)
}

/**
 * The alpha-divergence of Amari and Cichocki between positive measures,
 * D_a(p || q) = sum (p^a q^(1 - a) - a p + (a - 1) q) / (a (a - 1)): the generalised KL divergence
 * sum (p log(p / q) - p + q) at a = 1, its reverse at a = 0, twice the squared Hellinger distance sum (sqrt p - sqrt q)^2
 * at a = 1/2 and half Pearson's chi-square at a = 2. Non-negative for every real a.
 *
 * @param p - The first measure (non-negative; need not be normalised).
 * @param q - The second.
 * @param alpha - The order (any real number).
 * @param options - Axis and grid.
 * @returns D_a(p || q) in nats.
 * @concept alpha-divergence
 * @example
 * ```ts
 * import { alphaDivergence } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(alphaDivergence([1, 0], [0, 1], 0.5)) // => 4
 * ```
 */
export function alphaDivergence(p: TensorLike, q: TensorLike, alpha: number, options: GridOptions = {}): Value {
  const [a, b] = pair(p, q)
  // Within 1e-8 of the removable singularities the closed form loses its digits to cancellation: use the limits.
  if (Math.abs(alpha - 1) < NEAR) return integrate(klDiv(a, b), options)
  if (Math.abs(alpha) < NEAR) return integrate(klDiv(b, a), options)
  const terms = add(sub(powerMean(a, b, alpha), mul(a, alpha)), mul(b, alpha - 1))
  return div(integrate(terms, options), alpha * (alpha - 1))
}

/**
 * The beta-divergence between positive measures (Basu et al., Eguchi and Kano; the loss of beta-NMF, as
 * sklearn.decomposition.NMF with `beta_loss`): sum (p^b + (b - 1) q^b - b p q^(b - 1)) / (b (b - 1)), with the
 * limits sum (p log(p / q) - p + q) at b = 1 (generalised KL) and sum (p / q - log(p / q) - 1) at b = 0 (Itakura-Saito);
 * half the squared Euclidean distance at b = 2.
 *
 * @param p - The data (non-negative).
 * @param q - The model (positive).
 * @param beta - The order (any real number).
 * @param options - Axis and grid.
 * @returns The divergence.
 * @concept beta-divergence
 * @example
 * ```ts
 * import { betaDivergence } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(betaDivergence([1, 2], [3, 2], 2)) // => 2
 * ```
 */
export function betaDivergence(p: TensorLike, q: TensorLike, beta: number, options: GridOptions = {}): Value {
  const [a, b] = pair(p, q)
  if (Math.abs(beta - 1) < NEAR) return integrate(klDiv(a, b), options)
  if (Math.abs(beta) < NEAR) {
    const ratio = div(a, b)
    return integrate(sub(sub(ratio, log(ratio)), 1), options)
  }
  if (beta === 2) return integrate(mul(square(sub(a, b)), 0.5), options)
  const power = (v: Value, e: number): Value => {
    const positive = greater(v, 0)
    return where(positive, exp(mul(log(where(positive, v, 1)), e)), e > 0 ? 0 : e === 0 ? 1 : Number.POSITIVE_INFINITY)
  }
  const terms = sub(add(power(a, beta), mul(power(b, beta), beta - 1)), mul(mul(a, power(b, beta - 1)), beta))
  return div(integrate(terms, options), beta * (beta - 1))
}

/** The named convex functions of {@link bregmanDivergence}. */
export type BregmanName = 'squaredEuclidean' | 'negativeEntropy' | 'itakuraSaito'

/**
 * The Bregman divergence D_phi(p, q) = phi(p) - phi(q) - <grad phi(q), p - q> of a strictly convex function phi. The
 * named functions are separable and work along an axis and on grids: `squaredEuclidean` (phi = |x|^2, giving
 * |p - q|^2), `negativeEntropy` (phi = sum x log x, giving the generalised KL divergence) and `itakuraSaito`
 * (phi = -sum log x). A custom phi maps a vector to a scalar; its gradient comes from reverse-mode differentiation, and
 * p and q are then single vectors.
 *
 * @param p - The first point.
 * @param q - The second.
 * @param phi - The convex function, by name or as a function of a vector.
 * @param options - Axis and grid (named functions only).
 * @returns D_phi(p, q) >= 0.
 * @concept bregman-divergence
 * @example
 * ```ts
 * import { bregmanDivergence } from 'whitebox-engine/information'
 * import { item, square, sum, type Value } from 'whitebox-engine/tensor'
 * item(bregmanDivergence([1, 2], [3, 2], 'squaredEuclidean')) // => 4
 * item(bregmanDivergence([1, 2], [3, 2], (x: Value) => sum(square(x)))) // => 4
 * ```
 */
export function bregmanDivergence(
  p: TensorLike,
  q: TensorLike,
  phi: BregmanName | ((x: Value) => Value),
  options: GridOptions = {},
): Value {
  const [a, b] = pair(p, q)
  if (typeof phi === 'function') {
    const g = grad(phi)(b)
    return sub(sub(phi(a), phi(b)), sum(mul(g, sub(a, b))))
  }
  if (phi === 'squaredEuclidean') return integrate(square(sub(a, b)), options)
  if (phi === 'negativeEntropy') return integrate(klDiv(a, b), options)
  const ratio = div(a, b)
  return integrate(sub(sub(ratio, log(ratio)), 1), options)
}
