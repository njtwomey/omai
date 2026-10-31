/**
 * Post-selection inference by the polyhedral lemma (Lee, Sun, Sun and Taylor 2016; Tibshirani, Taylor, Lockhart and
 * Tibshirani 2016). After a procedure chooses a model using the data, the usual t-tests and intervals for the chosen
 * coefficients are invalid: they ignore that the model was chosen because the coefficients looked large. When the
 * selection event is a polyhedron {y : A y <= b}, as it is for the lasso at a fixed penalty and for forward stepwise
 * regression, a linear contrast eta'y conditional on the selection is a truncated Gaussian whose truncation limits the
 * lemma gives in closed form, so p-values and confidence intervals for the selected coefficients are exact (with
 * known noise level).
 */

import { MdmlError } from '../base/index.ts'
import { dgemm, inv } from '../numerics/linalg/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import { logNdtr, ndtr } from '../numerics/special/index.ts'
import { lassoCoordinateDescent } from '../optim/sparse.ts'
import { fromStorage, item, primalOf, type TensorLike, toFlat } from '../tensor/index.ts'
import { matrixView, targetVector } from './validation.ts'

const flat = (v: Parameters<typeof primalOf>[0]): Float64Array => Float64Array.from(toFlat(primalOf(v)))
const Phi = (x: number): number => item(ndtr(x))
const logPhi = (x: number): number => item(logNdtr(x))

/** A polyhedron {y : A y <= b}, A row-major `[m, n]`. */
export interface Polyhedron {
  readonly a: Float64Array
  readonly b: Float64Array
  /** The number of constraints m. */
  readonly rows: number
  /** The dimension n of y. */
  readonly n: number
}

/**
 * The distribution function at x of a normal(mean, sd^2) variable truncated to [lower, upper], evaluated in the log scale
 * for the tails so that it stays accurate when the truncation interval lies many standard deviations from the mean (the
 * usual case after selecting a large effect).
 *
 * @param x - The point.
 * @param mean - The mean of the untruncated normal.
 * @param sd - Its standard deviation.
 * @param lower - The lower truncation limit (may be -Infinity).
 * @param upper - The upper truncation limit (may be Infinity).
 * @returns P(X <= x | lower <= X <= upper).
 * @concept selective-inference
 * @example
 * ```ts
 * import { truncatedNormalCdf } from 'mdml-engine/learn'
 * truncatedNormalCdf(0, 0, 1, -Infinity, Infinity) // => 0.5
 * truncatedNormalCdf(1, 0, 1, 0, 2) // => 0.7152
 * ```
 */
export function truncatedNormalCdf(x: number, mean: number, sd: number, lower: number, upper: number): number {
  if (!(sd > 0)) throw new MdmlError('value', 'truncatedNormalCdf: sd must be positive')
  if (!(lower < upper)) throw new MdmlError('value', 'truncatedNormalCdf: needs lower < upper')
  const a = (lower - mean) / sd
  const b = (upper - mean) / sd
  const z = Math.min(Math.max((x - mean) / sd, a), b)
  if (z <= a) return 0
  if (z >= b) return 1
  if (a > 0) {
    // Upper tail: Q(t) = Phi(-t); F = (Q(a) - Q(z)) / (Q(a) - Q(b)), as ratios of exponentials of log Q differences.
    const la = logPhi(-a)
    return Math.min(1, Math.max(0, Math.expm1(logPhi(-z) - la) / Math.expm1(logPhi(-b) - la)))
  }
  if (b < 0) {
    // Lower tail: divide through by Phi(b).
    const lb = logPhi(b)
    const la = logPhi(a)
    return Math.min(1, Math.max(0, (Math.exp(logPhi(z) - lb) - Math.exp(la - lb)) / (1 - Math.exp(la - lb))))
  }
  const pa = Phi(a)
  return Math.min(1, Math.max(0, (Phi(z) - pa) / (Phi(b) - pa)))
}

/** The truncation limits of the polyhedral lemma for a contrast. */
export interface PolyhedralBounds {
  /** V^-: the lower limit on eta'y given the rest of y. */
  readonly vMinus: number
  /** V^+: the upper limit. */
  readonly vPlus: number
  /** The smallest slack of constraints orthogonal to the contrast (must be non-negative). */
  readonly vZero: number
}

/**
 * The limits of the polyhedral lemma (Lee et al. 2016, lemma 5.1). For y ~ N(mu, sigma^2 I) and the event {A y <= b}, put
 * z = eta'y and c = eta / (eta'eta). Then the event is V^- <= z <= V^+ for limits that depend on y only through
 * the component y - c z, independent of z: V^- = max over alpha_j < 0 of (b_j - (A y)_j) / alpha_j + z and V^+ = the same
 * minimum over alpha_j > 0 with alpha = A c.
 *
 * @param poly - The polyhedron.
 * @param eta - The contrast `[n]`.
 * @param y - The observed response, which must lie in the polyhedron.
 * @returns V^-, V^+ and the slack of the constraints with alpha_j = 0.
 * @concept selective-inference
 * @example
 * ```ts
 * import { polyhedralBounds } from 'mdml-engine/learn'
 * // The event y_1 >= 1 (that is, -y_1 <= -1) in two dimensions, contrast y_1: z is truncated below at 1.
 * polyhedralBounds({ a: Float64Array.from([-1, 0]), b: Float64Array.from([-1]), rows: 1, n: 2 }, [1, 0], [3, 7]).vMinus // => 1
 * ```
 */
export function polyhedralBounds(poly: Polyhedron, eta: ArrayLike<number>, y: ArrayLike<number>): PolyhedralBounds {
  const { a, b, rows, n } = poly
  let ee = 0
  let z = 0
  for (let i = 0; i < n; i++) {
    ee += (eta[i] ?? 0) ** 2
    z += (eta[i] ?? 0) * (y[i] ?? 0)
  }
  let vMinus = Number.NEGATIVE_INFINITY
  let vPlus = Number.POSITIVE_INFINITY
  let vZero = Number.POSITIVE_INFINITY
  for (let j = 0; j < rows; j++) {
    let alpha = 0
    let ay = 0
    for (let i = 0; i < n; i++) {
      const aji = a[j * n + i] ?? 0
      alpha += (aji * (eta[i] ?? 0)) / ee
      ay += aji * (y[i] ?? 0)
    }
    const slack = (b[j] ?? 0) - ay
    if (Math.abs(alpha) < 1e-12) vZero = Math.min(vZero, slack)
    else if (alpha < 0) vMinus = Math.max(vMinus, slack / alpha + z)
    else vPlus = Math.min(vPlus, slack / alpha + z)
  }
  return { vMinus, vPlus, vZero }
}

/** Options of {@link selectiveInference}. */
export interface SelectiveInferenceOptions {
  /** The noise standard deviation sigma. */
  readonly sigma: number
  /** The confidence level of the interval (default 0.95). */
  readonly level?: number
  /** The null value of eta'mu for the p-value (default 0). */
  readonly null?: number
}

/** A conditional (selective) test and interval for a contrast. */
export interface SelectiveResult {
  /** The estimate eta'y. */
  readonly estimate: number
  /** The two-sided selective p-value for eta'mu = null. */
  readonly pValue: number
  /** The selective confidence interval for eta'mu. */
  readonly interval: readonly [number, number]
  readonly vMinus: number
  readonly vPlus: number
  /** The standard deviation sigma |eta| of the estimate. */
  readonly sd: number
}

/**
 * Selective inference for a contrast eta'mu after the selection event {A y <= b}: the pivot F(theta) = P_theta(eta'Y <= eta'y |
 * the event) is uniform under the truth, giving the two-sided p-value 2 min(F(null), 1 - F(null)) and the interval
 * {theta : alpha/2 <= F(theta) <= 1 - alpha/2}, found by root finding because F decreases in theta. The noise level
 * sigma is taken as known.
 *
 * @param poly - The selection polyhedron.
 * @param eta - The contrast.
 * @param y - The observed response.
 * @param options - sigma, the level and the null value.
 * @returns The estimate, p-value, interval and truncation limits.
 * @throws MdmlError with code `value` when y is not in the polyhedron.
 * @concept selective-inference
 * @example
 * ```ts
 * import { selectiveInference } from 'mdml-engine/learn'
 * // After seeing y_1 >= 1 (observed 1.5): the selective p-value for mu_1 = 0 is far larger than the naive one.
 * const r = selectiveInference({ a: Float64Array.from([-1]), b: Float64Array.from([-1]), rows: 1, n: 1 }, [1], [1.5], { sigma: 1 })
 * Math.round(r.pValue * 1000) / 1000 // => 0.842
 * ```
 */
export function selectiveInference(
  poly: Polyhedron,
  eta: ArrayLike<number>,
  y: ArrayLike<number>,
  options: SelectiveInferenceOptions,
): SelectiveResult {
  if (!(options.sigma > 0)) throw new MdmlError('value', 'selectiveInference: sigma must be positive')
  const level = options.level ?? 0.95
  let ee = 0
  let z = 0
  for (let i = 0; i < poly.n; i++) {
    ee += (eta[i] ?? 0) ** 2
    z += (eta[i] ?? 0) * (y[i] ?? 0)
  }
  const sd = options.sigma * Math.sqrt(ee)
  const { vMinus, vPlus, vZero } = polyhedralBounds(poly, eta, y)
  const tol = 1e-6 * (1 + sd)
  if (vZero < -tol || z < vMinus - tol || z > vPlus + tol)
    throw new MdmlError('value', 'selectiveInference: y is not in the selection polyhedron')
  const cdf = (theta: number): number => truncatedNormalCdf(z, theta, sd, vMinus, vPlus)
  const nullValue = options.null ?? 0
  const f0 = cdf(nullValue)
  const pValue = Math.min(1, 2 * Math.min(f0, 1 - f0))
  // F decreases in theta: F(theta_L) = 1 - alpha/2, F(theta_U) = alpha/2.
  const solve = (target: number): number => {
    let lo = z - 10 * sd
    let hi = z + 10 * sd
    for (let k = 0; k < 40 && cdf(lo) < target; k++) lo -= hi - lo
    for (let k = 0; k < 40 && cdf(hi) > target; k++) hi += hi - lo
    return brentq((t) => cdf(t) - target, lo, hi, { xtol: 1e-9 * (1 + sd) }).root
  }
  const alpha = 1 - level
  return { estimate: z, pValue, interval: [solve(1 - alpha / 2), solve(alpha / 2)], vMinus, vPlus, sd }
}

/** Options of {@link lassoSelectiveInference} and {@link forwardStepwiseInference}. */
export interface RegressionInferenceOptions {
  /** The noise standard deviation (default: the residual standard deviation of the full least-squares fit, which needs n > p + 1). */
  readonly sigma?: number
  readonly level?: number
}

/** Selective inference for the coefficients of a selected linear model. */
export interface RegressionInference {
  /** The selected variables, in the order selected (stepwise) or by index (lasso). */
  readonly variables: readonly number[]
  /** Their signs at selection. */
  readonly signs: readonly number[]
  /** Least-squares coefficients of the selected model. */
  readonly estimates: readonly number[]
  /** Selective p-values for each coefficient equalling zero. */
  readonly pValues: readonly number[]
  /** Selective confidence intervals. */
  readonly intervals: readonly (readonly [number, number])[]
  /** The naive z-test p-values and intervals that ignore the selection (for comparison). */
  readonly naivePValues: readonly number[]
  readonly naiveIntervals: readonly (readonly [number, number])[]
  readonly sigma: number
  readonly polyhedron: Polyhedron
}

function noiseLevel(x: Float64Array, y: Float64Array, n: number, p: number, given: number | undefined): number {
  if (given !== undefined) return given
  if (n <= p + 1) throw new MdmlError('value', 'selective inference: pass sigma when n <= p + 1')
  // Residual standard deviation of the full least-squares fit by the normal equations.
  const xtx = dgemm(x, x, p, n, p, { transposeA: true })
  const xty = dgemm(x, y, p, n, 1, { transposeA: true })
  const beta = dgemm(flat(inv(fromStorage(xtx, [p, p], 'float64'))), xty, p, p, 1)
  const fit = dgemm(x, beta, n, p, 1)
  let rss = 0
  for (let i = 0; i < n; i++) rss += ((y[i] ?? 0) - (fit[i] ?? 0)) ** 2
  return Math.sqrt(rss / (n - p))
}

/** Selective inference for the partial regression coefficients of the model on the columns `cols`, given the polyhedron. */
function inferModel(
  x: Float64Array,
  y: Float64Array,
  n: number,
  p: number,
  cols: readonly number[],
  signs: readonly number[],
  poly: Polyhedron,
  sigma: number,
  level: number,
): RegressionInference {
  const m = cols.length
  const xe = new Float64Array(n * m)
  cols.forEach((c, k) => {
    for (let i = 0; i < n; i++) xe[i * m + k] = x[i * p + c] ?? 0
  })
  const gi = flat(inv(fromStorage(dgemm(xe, xe, m, n, m, { transposeA: true }), [m, m], 'float64')))
  const h = dgemm(xe, gi, n, m, m)
  const estimates: number[] = []
  const pValues: number[] = []
  const intervals: [number, number][] = []
  const naiveP: number[] = []
  const naiveI: [number, number][] = []
  const zc = 1.959963984540054
  for (let k = 0; k < m; k++) {
    const eta = Float64Array.from({ length: n }, (_, i) => h[i * m + k] ?? 0)
    const r = selectiveInference(poly, eta, y, { sigma, level })
    estimates.push(r.estimate)
    pValues.push(r.pValue)
    intervals.push([r.interval[0], r.interval[1]])
    naiveP.push(2 * (1 - Phi(Math.abs(r.estimate) / r.sd)))
    naiveI.push([r.estimate - zc * r.sd, r.estimate + zc * r.sd])
  }
  return {
    variables: cols,
    signs,
    estimates,
    pValues,
    intervals,
    naivePValues: naiveP,
    naiveIntervals: naiveI,
    sigma,
    polyhedron: poly,
  }
}

/**
 * The selection polyhedron of the lasso at a fixed penalty (Lee et al. 2016, section 4.1) for the objective
 * (1/2)|y - X b|^2 + lambda |b|_1: with active set E and signs s, the event {E-hat = E, s-hat = s} is A y <= b with the
 * inactive-variable rows (1/lambda) (+- X_-E'(I - P_E)) y <= 1 -+ X_-E' X_E (X_E'X_E)^-1 s and the active-sign rows
 * -diag(s) (X_E'X_E)^-1 X_E' y <= -lambda diag(s) (X_E'X_E)^-1 s.
 *
 * @param x - The design `[n, p]`.
 * @param active - The active set E (column indices).
 * @param signs - The signs of the active coefficients.
 * @param lambda - The penalty.
 * @returns The polyhedron.
 * @concept selective-inference
 * @example
 * ```ts
 * import { lassoPolyhedron } from 'mdml-engine/learn'
 * lassoPolyhedron([[1, 0], [0, 1], [1, 1]], [0], [1], 0.5).rows // => 3
 * ```
 */
export function lassoPolyhedron(
  x: TensorLike,
  active: readonly number[],
  signs: readonly number[],
  lambda: number,
): Polyhedron {
  const m0 = matrixView(x, 'lassoPolyhedron')
  const { n, d: p } = m0
  const e = active.length
  if (e === 0) throw new MdmlError('value', 'lassoPolyhedron: the active set is empty')
  const inactive = Array.from({ length: p }, (_, j) => j).filter((j) => !active.includes(j))
  const xe = new Float64Array(n * e)
  active.forEach((c, k) => {
    for (let i = 0; i < n; i++) xe[i * e + k] = m0.data[i * p + c] ?? 0
  })
  const xo = new Float64Array(n * inactive.length)
  inactive.forEach((c, k) => {
    for (let i = 0; i < n; i++) xo[i * inactive.length + k] = m0.data[i * p + c] ?? 0
  })
  const gi = flat(inv(fromStorage(dgemm(xe, xe, e, n, e, { transposeA: true }), [e, e], 'float64')))
  const h = dgemm(xe, gi, n, e, e)
  const q = inactive.length
  // B = X_-E' (I - P_E) = X_-E' - (X_-E' H) X_E'  (q x n).
  const xoh = dgemm(xo, h, q, n, e, { transposeA: true })
  const proj = dgemm(xoh, xe, q, e, n, { transposeB: true })
  const bMat = Float64Array.from(
    { length: q * n },
    (_, k) => (xo[(k % n) * q + Math.floor(k / n)] ?? 0) - (proj[k] ?? 0),
  )
  const sv = Float64Array.from(signs)
  const shift = dgemm(xoh, sv, q, e, 1)
  const rows = 2 * q + e
  const a = new Float64Array(rows * n)
  const b = new Float64Array(rows)
  for (let r = 0; r < q; r++) {
    for (let i = 0; i < n; i++) {
      a[r * n + i] = (bMat[r * n + i] ?? 0) / lambda
      a[(q + r) * n + i] = -(bMat[r * n + i] ?? 0) / lambda
    }
    b[r] = 1 - (shift[r] ?? 0)
    b[q + r] = 1 + (shift[r] ?? 0)
  }
  // Active signs: -diag(s) Gi X_E' y <= -lambda diag(s) Gi s.
  const giXe = dgemm(gi, xe, e, e, n, { transposeB: true })
  const gis = dgemm(gi, sv, e, e, 1)
  for (let k = 0; k < e; k++) {
    for (let i = 0; i < n; i++) a[(2 * q + k) * n + i] = -(signs[k] ?? 0) * (giXe[k * n + i] ?? 0)
    b[2 * q + k] = -lambda * (signs[k] ?? 0) * (gis[k] ?? 0)
  }
  return { a, b, rows, n }
}

/**
 * Selective inference for the lasso at a fixed penalty lambda (objective (1/2)|y - X b|^2 + lambda |b|_1, the scale of
 * `lassoCoordinateDescent`): fit the lasso, condition on its active set and signs, and give for each selected variable the
 * selective p-value and confidence interval for its coefficient in the least-squares model on the selected variables
 * (Lee et al. 2016). lambda must be fixed in advance, not chosen from y (that needs the extension of Tibshirani et al.).
 *
 * @param x - The design `[n, p]`.
 * @param y - The response `[n]`.
 * @param lambda - The penalty.
 * @param options - sigma and the level.
 * @returns The selected model with selective and naive p-values and intervals.
 * @throws MdmlError with code `value` when the lasso selects nothing.
 * @concept selective-inference
 * @example
 * ```ts
 * import { lassoSelectiveInference } from 'mdml-engine/learn'
 * const x = [[1, 0], [0, 1], [1, 1], [-1, 0.5], [0.5, -1], [2, 0.2], [-0.3, 1.5], [1.2, -0.7]]
 * const y = [2.1, 0.1, 2.4, -2.2, 1.1, 4.2, -0.4, 2.9]
 * lassoSelectiveInference(x, y, 2, { sigma: 0.5 }).variables // => [0]
 * ```
 */
export function lassoSelectiveInference(
  x: TensorLike,
  y: TensorLike,
  lambda: number,
  options: RegressionInferenceOptions = {},
): RegressionInference {
  const m = matrixView(x, 'lassoSelectiveInference')
  const yv = targetVector(y, 'lassoSelectiveInference', m.n)
  const fit = lassoCoordinateDescent(fromStorage(m.data, [m.n, m.d], 'float64'), yv, lambda, {
    tol: 1e-14,
    maxIterations: 20000,
  })
  const coef = flat(fit.coef)
  const active = Array.from(coef.keys()).filter((j) => coef[j] !== 0)
  if (active.length === 0)
    throw new MdmlError('value', 'lassoSelectiveInference: the lasso selected no variables at this penalty')
  const signs = active.map((j) => Math.sign(coef[j] ?? 0))
  const poly = lassoPolyhedron(x, active, signs, lambda)
  const sigma = noiseLevel(m.data, yv, m.n, m.d, options.sigma)
  return inferModel(m.data, yv, m.n, m.d, active, signs, poly, sigma, options.level ?? 0.95)
}

/**
 * The selection polyhedron of forward stepwise regression (Loftus and Taylor 2014; Tibshirani et al. 2016): at each
 * step the variable maximising |u_j'y| is chosen, where u_j is the column j projected off the variables already chosen and
 * normalised. The path j_1, s_1, ..., j_k, s_k is the event of the 2 (p - l) constraints per step l that
 * s_l u_{j_l}'y >= +- u_j'y for every candidate j, plus s_l u_{j_l}'y >= 0.
 *
 * @param x - The design `[n, p]`.
 * @param y - The response `[n]`.
 * @param steps - The number of steps k.
 * @returns The chosen variables, their signs and the polyhedron.
 * @concept selective-inference
 * @example
 * ```ts
 * import { forwardStepwisePolyhedron } from 'mdml-engine/learn'
 * forwardStepwisePolyhedron([[1, 0, 0.1], [0, 1, 0.2], [1, 1, 0.1], [0.5, -1, 0.3], [2, 0.2, -0.2]], [2, 0.1, 2.2, -0.6, 4.1], 1).variables // => [0]
 * ```
 */
export function forwardStepwisePolyhedron(
  x: TensorLike,
  y: TensorLike,
  steps: number,
): { readonly variables: number[]; readonly signs: number[]; readonly polyhedron: Polyhedron } {
  const m = matrixView(x, 'forwardStepwisePolyhedron')
  const yv = targetVector(y, 'forwardStepwisePolyhedron', m.n)
  const { n, d: p } = m
  if (!(Number.isInteger(steps) && steps >= 1 && steps <= Math.min(p, n - 1)))
    throw new MdmlError('value', 'forwardStepwisePolyhedron: steps must be between 1 and min(p, n - 1)')
  // Columns residualised against the chosen variables, kept as an orthonormal basis of the chosen span.
  const cols: Float64Array[] = Array.from({ length: p }, (_, j) =>
    Float64Array.from({ length: n }, (_, i) => m.data[i * p + j] ?? 0),
  )
  const basis: Float64Array[] = []
  const variables: number[] = []
  const signs: number[] = []
  const rowsA: Float64Array[] = []
  const rowsB: number[] = []
  const residualise = (v: Float64Array): Float64Array => {
    const out = Float64Array.from(v)
    for (const q of basis) {
      let dot = 0
      for (let i = 0; i < n; i++) dot += (q[i] ?? 0) * (out[i] ?? 0)
      for (let i = 0; i < n; i++) out[i] = (out[i] ?? 0) - dot * (q[i] ?? 0)
    }
    return out
  }
  for (let step = 0; step < steps; step++) {
    const u = new Map<number, Float64Array>()
    for (let j = 0; j < p; j++) {
      if (variables.includes(j)) continue
      const r = residualise(cols[j] as Float64Array)
      const nrm = Math.sqrt(r.reduce((a, v) => a + v * v, 0))
      if (nrm > 1e-12)
        u.set(
          j,
          r.map((v) => v / nrm),
        )
    }
    let best = -1
    let bestVal = -1
    let bestSign = 1
    for (const [j, uj] of u) {
      let dot = 0
      for (let i = 0; i < n; i++) dot += (uj[i] ?? 0) * (yv[i] ?? 0)
      if (Math.abs(dot) > bestVal) {
        bestVal = Math.abs(dot)
        best = j
        bestSign = dot >= 0 ? 1 : -1
      }
    }
    const ub = u.get(best) as Float64Array
    // Constraints -(s u_b -+ u_j)'y <= 0 for the other candidates, and -s u_b'y <= 0.
    for (const [j, uj] of u) {
      if (j === best) continue
      rowsA.push(Float64Array.from(ub, (v, i) => -(bestSign * v - (uj[i] ?? 0))))
      rowsB.push(0)
      rowsA.push(Float64Array.from(ub, (v, i) => -(bestSign * v + (uj[i] ?? 0))))
      rowsB.push(0)
    }
    rowsA.push(Float64Array.from(ub, (v) => -bestSign * v))
    rowsB.push(0)
    variables.push(best)
    signs.push(bestSign)
    basis.push(ub)
  }
  const a = new Float64Array(rowsA.length * n)
  rowsA.forEach((r, k) => {
    a.set(r, k * n)
  })
  return { variables, signs, polyhedron: { a, b: Float64Array.from(rowsB), rows: rowsA.length, n } }
}

/**
 * Selective inference for forward stepwise regression after k steps (Tibshirani et al. 2016): for each selected variable the
 * selective p-value and interval for its coefficient in the least-squares model on all k selected variables, conditional on
 * the whole selection path.
 *
 * @param x - The design `[n, p]`.
 * @param y - The response `[n]`.
 * @param steps - The number of steps.
 * @param options - sigma and the level.
 * @returns The selected model with selective and naive p-values and intervals.
 * @concept selective-inference
 * @example
 * ```ts
 * import { forwardStepwiseInference } from 'mdml-engine/learn'
 * const x = [[1, 0, 0.1], [0, 1, 0.2], [1, 1, 0.1], [0.5, -1, 0.3], [2, 0.2, -0.2], [-1, 0.4, 0.5]]
 * const r = forwardStepwiseInference(x, [2, 0.1, 2.2, -0.6, 4.1, -2.2], 1, { sigma: 0.5 })
 * r.variables // => [0]
 * ```
 */
export function forwardStepwiseInference(
  x: TensorLike,
  y: TensorLike,
  steps: number,
  options: RegressionInferenceOptions = {},
): RegressionInference {
  const m = matrixView(x, 'forwardStepwiseInference')
  const yv = targetVector(y, 'forwardStepwiseInference', m.n)
  const path = forwardStepwisePolyhedron(x, y, steps)
  const sigma = noiseLevel(m.data, yv, m.n, m.d, options.sigma)
  return inferModel(m.data, yv, m.n, m.d, path.variables, path.signs, path.polyhedron, sigma, options.level ?? 0.95)
}
