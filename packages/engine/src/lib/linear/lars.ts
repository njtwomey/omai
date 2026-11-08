/**
 * Least angle regression (Efron, Hastie, Johnstone and Tibshirani): the whole piecewise-linear regularisation path of
 * the lasso, or of plain LARS, one knot per step. At each knot a variable joins the active set (or, in lasso mode,
 * leaves it when its coefficient would cross zero) and the coefficients move along the equiangular direction until
 * the next knot. `larsAlgorithm` is that as a traced `Algorithm` whose state carries the path so far; `larsPath` runs it
 * to the end; `lars` is the estimator on the iterative base, whose session shows the path growing knot by knot and
 * whose `fittedAt(t)` is the model at knot `t`.
 */

import { type IterativeEstimator, iterative, type LearnData, targetVector } from '../learn/index.ts'
import { ddot, dgemm, dgemv, solve } from '../numerics/linalg/index.ts'
import { dict, real } from '../space/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { centre, type Design, designOf, interceptFor, linearPredictor, vectorTensor } from './design.ts'

/** What a step of the path did to the active set. */
export type LarsEvent = { readonly kind: 'enter' | 'leave'; readonly feature: number }

/** The state of the LARS algorithm: where the path is, and the path so far. */
export type LarsState = {
  /** The coefficients at the current knot. */
  readonly coef: number[]
  /** The residual `y - X coef` (centred, weighted). */
  readonly residual: number[]
  /** The active variables, in the order they entered. */
  readonly active: number[]
  /** The sign each active variable's correlation had when it entered. */
  readonly signs: number[]
  /** Variables left out for being collinear with the active set. */
  readonly ignored: number[]
  /** Whether the last step ended with a variable leaving (so the next step adds none). */
  readonly justLeft: boolean
  /** The penalty at every knot so far, the first being the penalty above which all coefficients are zero. */
  readonly alphas: number[]
  /** The coefficients at every knot so far. */
  readonly coefs: number[][]
  /** The event that produced each knot after the first. */
  readonly events: LarsEvent[]
  /** The events of the last step (a variable entering, and one leaving). */
  readonly lastEvents: LarsEvent[]
  /** Whether the path has reached its end. */
  readonly finished: boolean
}

/** Options of {@link larsAlgorithm}. */
export interface LarsOptions {
  /** `lar` (the plain path) or `lasso` (variables leave when they cross zero: the lasso path). */
  readonly method?: 'lar' | 'lasso'
  /** Stop once the penalty falls to this (the path is then interpolated to it by {@link coefficientsAt}). */
  readonly alphaMin?: number
  /** Stop once this many variables are active. */
  readonly maxFeatures?: number
  /** Whether to fit an intercept (by centring). */
  readonly fitIntercept?: boolean
}

/** A LARS path: the knots, and the order in which variables entered. */
export type LarsPath = {
  /** The penalty at each knot, decreasing (scikit-learn's `alphas_`: the largest correlation over the sample count). */
  readonly alphas: number[]
  /** The coefficients at each knot `[knots][d]`. */
  readonly coefs: number[][]
  /** The variables in the order they first entered. */
  readonly order: number[]
  /** What happened at each knot after the first. */
  readonly events: LarsEvent[]
}

const EPS = Number.EPSILON
const TINY = 1e-30

/**
 * The LARS path as a traced algorithm, one knot per step. It logs the metrics `alpha` (the penalty at the knot),
 * `active` (variables in the active set) and `correlation` (the largest absolute correlation with the residual), the
 * tensor `coefficients` (`[d]`, the model at the knot) and the message `event` (a variable entering or leaving).
 *
 * @param design - Features, targets and weights (weights scale rows by their square roots).
 * @param options - The method, a penalty or size to stop at, and the intercept.
 * @returns The algorithm.
 * @reference B. Efron, T. Hastie, I. Johnstone and R. Tibshirani, Least angle regression, Annals of Statistics 32
 * (2004).
 * @example
 * ```ts
 * import { designOf, larsAlgorithm } from 'omai-engine/linear'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const design = designOf({ x: [[1, 0], [0, 1], [1, 1], [2, 1]], y: [1, 2, 3, 4] }, 'demo')
 * const end = runAlgorithm(larsAlgorithm(design, { method: 'lasso' }), { maxSteps: 20 }).state
 * end.events.map((e) => e.feature) // => [0, 1]
 * ```
 */
export function larsAlgorithm(design: Design, options: LarsOptions = {}): Algorithm<LarsState, never> {
  const method = options.method ?? 'lasso'
  const fitIntercept = options.fitIntercept ?? true
  const alphaMin = options.alphaMin ?? 0
  const { n, d } = design.x
  const c0 = centre(design, fitIntercept)
  const x = c0.x
  const y = c0.y
  const scale = design.w.reduce((a, b) => a + b, 0)
  const maxFeatures = Math.min(options.maxFeatures ?? d, d)
  const gram = dgemm(x, x, d, n, d, { transposeA: true })
  const correlationsOf = (r: ArrayLike<number>): Float64Array => dgemv(x, n, d, r, true)
  const largest = (c: ArrayLike<number>): number => {
    let m = 0
    for (let j = 0; j < d; j++) m = Math.max(m, Math.abs(c[j] ?? 0))
    return m
  }
  const zeroCoef = new Array<number>(d).fill(0)
  return {
    name: `lars-${method}`,
    config: { method, alphaMin, maxFeatures, fitIntercept },
    schema: channels({
      metrics: {
        alpha: {
          description: 'the penalty at the knot (largest absolute correlation over the sample count)',
          direction: 'lower',
        },
        active: { description: 'variables in the active set' },
      },
      tensors: { coefficients: { role: 'param', description: 'the coefficients at the knot', shape: [d] } },
      messages: {
        event: {
          description: 'a variable entering or leaving the active set',
          payload: { kind: 'enter or leave', feature: 'the variable' },
        },
      },
    }),
    init: () => {
      const residual = Array.from(y)
      const alpha = largest(correlationsOf(residual)) / scale
      return {
        coef: zeroCoef,
        residual,
        active: [],
        signs: [],
        ignored: [],
        justLeft: false,
        alphas: [alpha],
        coefs: [zeroCoef],
        events: [],
        lastEvents: [],
        finished: alpha <= Math.max(alphaMin, EPS) || d === 0 || maxFeatures === 0,
      }
    },
    step: (s) => {
      const c = correlationsOf(s.residual)
      let active = s.active
      let signs = s.signs
      let ignored = s.ignored
      const events: LarsEvent[] = []
      if (!s.justLeft) {
        // Bring in the inactive variable most correlated with the residual that is not collinear with the active set.
        for (;;) {
          let best = -1
          for (let j = 0; j < d; j++) {
            if (active.includes(j) || ignored.includes(j)) continue
            if (best < 0 || Math.abs(c[j] ?? 0) > Math.abs(c[best] ?? 0)) best = j
          }
          if (best < 0) return { ...s, ignored, finished: true }
          if (independent(gram, d, active, best)) {
            active = [...active, best]
            signs = [...signs, (c[best] ?? 0) >= 0 ? 1 : -1]
            events.push({ kind: 'enter', feature: best })
            break
          }
          ignored = [...ignored, best]
        }
      }
      // The equiangular direction: w = A (G_A^-1 s) with A = (s' G_A^-1 s)^-1/2, u = X_A w, and a = X'u.
      const m = active.length
      const sub = new Float64Array(m * m)
      for (let p = 0; p < m; p++)
        for (let q = 0; q < m; q++) sub[p * m + q] = gram[(active[p] ?? 0) * d + (active[q] ?? 0)] ?? 0
      const sA = Float64Array.from(signs)
      const z = toFlat(primalOf(solve(fromStorage(sub, [m, m], 'float64'), fromStorage(sA, [m], 'float64'))))
      const aa = 1 / Math.sqrt(ddot(sA, z))
      const w = Float64Array.from(z, (v) => v * aa)
      const u = new Float64Array(n)
      for (let p = 0; p < m; p++) {
        const j = active[p] ?? 0
        for (let i = 0; i < n; i++) u[i] = (u[i] ?? 0) + (x[i * d + j] ?? 0) * (w[p] ?? 0)
      }
      const cors = correlationsOf(u)
      const top = active.reduce((mx, j) => Math.max(mx, Math.abs(c[j] ?? 0)), 0)
      // How far along the direction before an inactive variable ties with the active ones, or the least squares point.
      let gamma = top / aa
      for (let j = 0; j < d; j++) {
        if (active.includes(j) || ignored.includes(j)) continue
        for (const g of [(top - (c[j] ?? 0)) / (aa - (cors[j] ?? 0)), (top + (c[j] ?? 0)) / (aa + (cors[j] ?? 0))]) {
          if (g > TINY && g < gamma) gamma = g
        }
      }
      // Lasso: a coefficient reaching zero first leaves the active set.
      let leaving = -1
      if (method === 'lasso') {
        let crossing = Number.POSITIVE_INFINITY
        active.forEach((j, p) => {
          const g = -(s.coef[j] ?? 0) / ((w[p] ?? 0) + TINY)
          if (g > TINY && g < crossing && g < gamma) {
            crossing = g
            leaving = j
          }
        })
        if (leaving >= 0) gamma = crossing
      }
      const coef = s.coef.slice()
      active.forEach((j, p) => {
        coef[j] = (coef[j] ?? 0) + gamma * (w[p] ?? 0)
      })
      const residual = s.residual.map((r, i) => r - gamma * (u[i] ?? 0))
      if (leaving >= 0) {
        const at = active.indexOf(leaving)
        coef[leaving] = 0
        active = active.filter((_, i) => i !== at)
        signs = signs.filter((_, i) => i !== at)
        events.push({ kind: 'leave', feature: leaving })
      }
      const alpha = largest(correlationsOf(residual)) / scale
      return {
        coef,
        residual,
        active,
        signs,
        ignored,
        justLeft: leaving >= 0,
        alphas: [...s.alphas, alpha],
        coefs: [...s.coefs, coef],
        events: [...s.events, ...events],
        lastEvents: events,
        finished: alpha <= Math.max(alphaMin, EPS) || active.length >= maxFeatures,
      }
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { alpha: s.alphas[s.alphas.length - 1] ?? 0, active: s.active.length },
      tensors: { coefficients: { role: 'param', value: s.coef, shape: [d] } },
      messages: s.lastEvents.map((e) => ({ kind: 'event', payload: { kind: e.kind, feature: e.feature } })),
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as LarsState,
  }
}

/** Whether variable `j` is linearly independent of the active ones, by the size of its residual after projecting on them. */
function independent(gram: Float64Array, d: number, active: readonly number[], j: number): boolean {
  const m = active.length
  const self = gram[j * d + j] ?? 0
  if (self <= 0) return false
  if (m === 0) return true
  const sub = new Float64Array(m * m)
  for (let p = 0; p < m; p++)
    for (let q = 0; q < m; q++) sub[p * m + q] = gram[(active[p] ?? 0) * d + (active[q] ?? 0)] ?? 0
  const b = Float64Array.from(active, (i) => gram[i * d + j] ?? 0)
  let projected: Float64Array
  try {
    projected = Float64Array.from(
      toFlat(primalOf(solve(fromStorage(sub, [m, m], 'float64'), fromStorage(b, [m], 'float64')))),
    )
  } catch {
    return false
  }
  return self - ddot(b, projected) > Math.sqrt(EPS) * self
}

/**
 * The coefficients at a penalty, by linear interpolation between the knots of a path (the lasso path is piecewise
 * linear in the penalty; for the plain LARS path this is the scikit-learn convention too).
 *
 * @param path - The path.
 * @param alpha - The penalty (at or above the first knot gives zeros; below the last the last knot).
 * @returns The coefficients.
 * @example
 * ```ts
 * import { coefficientsAt } from 'omai-engine/linear'
 * const path = { alphas: [2, 1, 0], coefs: [[0, 0], [1, 0], [2, 1]], order: [0, 1], events: [] }
 * coefficientsAt(path, 1.5) // => [0.5, 0]
 * coefficientsAt(path, 0.5) // => [1.5, 0.5]
 * ```
 */
export function coefficientsAt(path: LarsPath, alpha: number): number[] {
  const { alphas, coefs } = path
  if (alphas.length === 0) return []
  if (alpha >= (alphas[0] ?? 0)) return [...(coefs[0] ?? [])]
  for (let k = 1; k < alphas.length; k++) {
    const hi = alphas[k - 1] ?? 0
    const lo = alphas[k] ?? 0
    if (alpha >= lo) {
      const t = hi === lo ? 1 : (hi - alpha) / (hi - lo)
      return (coefs[k] ?? []).map((v, j) => (coefs[k - 1]?.[j] ?? 0) + t * (v - (coefs[k - 1]?.[j] ?? 0)))
    }
  }
  return [...(coefs[coefs.length - 1] ?? [])]
}

/** A path of {@link larsPath}: the knots, with the intercept at each and the features' entry order. */
export interface LarsPathResult extends LarsPath {
  /** The intercept at each knot (zero without one). */
  readonly intercepts: number[]
}

/**
 * The LARS path of a dataset, run to its end (scikit-learn's `lars_path` on centred data, the penalties its `alphas`).
 *
 * @param data - Features, targets and weights.
 * @param options - The method, where to stop and the intercept.
 * @returns The knots, the intercepts and the entry order.
 * @example
 * ```ts
 * import { larsPath } from 'omai-engine/linear'
 * const p = larsPath({ x: [[1, 0], [0, 1], [1, 1], [2, 1]], y: [1, 2, 3, 4] }, { method: 'lar' })
 * p.order // => [0, 1]
 * p.alphas.length // => 3
 * ```
 */
export function larsPath(data: LearnData, options: LarsOptions = {}): LarsPathResult {
  const design = designOf(data, 'larsPath')
  const end = runAlgorithm(larsAlgorithm(design, options), { maxSteps: 8 * Math.max(1, design.x.d) + 8 }).state
  return pathOf(end, design, options.fitIntercept ?? true)
}

function pathOf(s: LarsState, design: Design, fitIntercept: boolean): LarsPathResult {
  const c = centre(design, fitIntercept)
  const order: number[] = []
  for (const e of s.events) if (e.kind === 'enter' && !order.includes(e.feature)) order.push(e.feature)
  return {
    alphas: s.alphas,
    coefs: s.coefs,
    order,
    events: s.events,
    intercepts: s.coefs.map((coef) => (fitIntercept ? interceptFor(c, coef) : 0)),
  }
}

/** The hyperparameters of {@link lars}. */
export interface LarsHyper {
  readonly method: 'lar' | 'lasso'
  /** The penalty to stop at: the model is the path interpolated to it (0 runs to the end of the path). */
  readonly alpha: number
  /** Stop when this many variables are active. */
  readonly nNonzeroCoefs?: number
  readonly fitIntercept: boolean
}

/** The fitted parameters of {@link lars}: the model at the stopping point, and the path. */
export type LarsParams = {
  readonly coef: number[]
  readonly intercept: number
  /** The penalty of the model (the stopping penalty, or where the path ended). */
  readonly alpha: number
  readonly path: LarsPath
  /** The (weighted) feature means and target mean the intercept is formed from. */
  readonly xMean: number[]
  readonly yMean: number
}

/** What a fitted LARS model offers. */
export interface LarsVerbs {
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  /** The coefficients and intercept at another penalty on the path. */
  readonly modelAt: (alpha: number) => { readonly coef: number[]; readonly intercept: number }
  readonly score: (data: LearnData) => number
}

const LARS_DEFAULTS: LarsHyper = { method: 'lasso', alpha: 0, fitIntercept: true }

/**
 * Least angle regression and the lasso by LARS (`sklearn.linear_model.Lars` and `LassoLars`): the model is the point of
 * the path at penalty `alpha` (or where `nNonzeroCoefs` variables are active, or the end of the path), and the whole
 * path is part of the fit (`params.path`). A session adds one knot per step, so `fittedAt(t)` is the model at knot `t`,
 * with `solver/alpha`, `solver/active` and `solver/coefficients` logged.
 *
 * @param hyper - `method` (`lasso` or `lar`), `alpha`, `nNonzeroCoefs`, `fitIntercept`.
 * @returns The estimator.
 * @index estimator Least angle regression: the lasso path one knot per step.
 * @example
 * ```ts
 * import { lars } from 'omai-engine/linear'
 * import { toFlat } from 'omai-engine/tensor'
 * const f = lars({ method: 'lar', nNonzeroCoefs: 1 }).fit({ x: [[1, 0.1], [2, -0.1], [3, 0.2], [4, 0]], y: [1, 2, 3, 4.1] })
 * f.params.path.order[0] // => 0
 * toFlat(f.predict([[5, 0]]))[0] > 4 // => true
 * ```
 */
export function lars(
  hyper: Partial<LarsHyper> = {},
): IterativeEstimator<LarsHyper, LarsState, LarsParams, LarsVerbs, never> {
  return iterative<LarsHyper, LarsState, LarsParams, LarsVerbs, never>(
    {
      name: 'lars',
      kind: 'regressor',
      defaults: LARS_DEFAULTS,
      hyperSpace: (h) =>
        dict({
          alpha: real(0, Math.max(10, h.alpha), { default: h.alpha }),
        }),
      algorithm: (data, h) =>
        larsAlgorithm(designOf(data, 'lars'), {
          method: h.method,
          alphaMin: h.alpha,
          fitIntercept: h.fitIntercept,
          ...(h.nNonzeroCoefs === undefined ? {} : { maxFeatures: h.nNonzeroCoefs }),
        }),
      maxSteps: (_h, data) => 8 * Math.max(1, designOf(data, 'lars').x.d) + 8,
      params: (s, data, h) => {
        const design = designOf(data, 'lars')
        const path = pathOf(s, design, h.fitIntercept)
        const last = path.alphas[path.alphas.length - 1] ?? 0
        const alpha = h.alpha > 0 && last < h.alpha ? h.alpha : last
        const coef = alpha === last ? (path.coefs[path.coefs.length - 1] ?? []) : coefficientsAt(path, alpha)
        const c = centre(design, h.fitIntercept)
        return {
          coef: [...coef],
          intercept: h.fitIntercept ? interceptFor(c, coef) : 0,
          alpha,
          path: { alphas: path.alphas, coefs: path.coefs, order: path.order, events: path.events },
          xMean: Array.from(c.xMean),
          yMean: c.yMean,
        }
      },
      verbs: (p, h) => {
        const centreOf = (a: number): { readonly coef: number[]; readonly intercept: number } => {
          const coef = coefficientsAt(p.path, a)
          return { coef, intercept: h.fitIntercept ? p.yMean - ddot(p.xMean, coef) : 0 }
        }
        const predict = (x: TensorLike): Tensor => vectorTensor(linearPredictor(x, p.coef, p.intercept, 'lars'))
        return {
          nFeaturesIn: p.coef.length,
          predict,
          featureImportances: p.coef.map(Math.abs),
          modelAt: centreOf,
          score: (data) => {
            const pred = toFlat(predict(data.x))
            const truth = targetVector(data.y ?? [], 'lars', pred.length)
            const mean = truth.reduce((a, b) => a + b, 0) / truth.length
            let res = 0
            let tot = 0
            for (let i = 0; i < pred.length; i++) {
              res += ((truth[i] ?? 0) - (pred[i] ?? 0)) ** 2
              tot += ((truth[i] ?? 0) - mean) ** 2
            }
            return tot === 0 ? (res === 0 ? 1 : 0) : 1 - res / tot
          },
        }
      },
      logs: { 'solver/alpha': Log(), 'solver/active': Log(), 'solver/coefficients': Log({ detail: 'full' }) },
    },
    { ...LARS_DEFAULTS, ...hyper },
  )
}
