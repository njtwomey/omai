/**
 * Seemingly unrelated regressions (Zellner): several regression equations, each with its own response and its own
 * subset of the features, whose errors are correlated across equations. Fitting them jointly by feasible generalised
 * least squares, with the cross-equation error covariance estimated from the residuals, is more efficient than
 * equation-by-equation least squares unless the equations share all their regressors or the errors are uncorrelated.
 * One step is Zellner's two-step estimator; iterating to convergence reaches the Gaussian maximum likelihood.
 */

import { type Json, MdmlError, maxOf } from '../base/index.ts'
import type { CoefficientTable, Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView, requireTargets } from '../learn/index.ts'
import { dict, int } from '../space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { inverseOf, nested } from './_cov.ts'
import { waldTable } from './inference.ts'

/** The hyperparameters of {@link sur}. */
export interface SurHyper {
  /**
   * The feature columns of each equation (one list per output), or `'all'` for the same, all features, in every
   * equation (default `'all'`, in which case the fit equals equation-by-equation least squares).
   */
  readonly equations: 'all' | readonly (readonly number[])[]
  readonly fitIntercept: boolean
  /** The number of feasible-GLS steps (default 1, Zellner's two-step); more iterate toward maximum likelihood. */
  readonly maxIter: number
  /** Stop when the coefficients change by less than this (default 1e-10). */
  readonly tol: number
}

/** The system: per-equation design blocks and the stacked targets. */
interface System {
  readonly n: number
  readonly k: number
  /** Per equation: the feature columns used. */
  readonly columns: number[][]
  /** Per equation: its design (the intercept column last when fitted), row-major `[n, p_e]`. */
  readonly designs: Float64Array[]
  readonly widths: number[]
  readonly y: Float64Array[]
  readonly fitIntercept: boolean
}

function systemOf(data: LearnData, h: Pick<SurHyper, 'equations' | 'fitIntercept'>): System {
  const m = matrixView(data.x, 'sur')
  const yt = asTensor(requireTargets(data, 'sur'))
  const k = yt.shape.length === 1 ? 1 : (yt.shape[1] ?? 1)
  if ((yt.shape[0] ?? 0) !== m.n) throw new MdmlError('shape', `sur: ${yt.shape[0]} targets for ${m.n} rows`)
  if (data.weights !== undefined) throw new MdmlError('value', 'sur: sample weights are not supported')
  const columns =
    h.equations === 'all'
      ? Array.from({ length: k }, () => Array.from({ length: m.d }, (_, j) => j))
      : h.equations.map((c) => [...c])
  if (columns.length !== k) throw new MdmlError('shape', `sur: ${columns.length} equations for ${k} outputs`)
  for (const cols of columns)
    for (const j of cols)
      if (!Number.isInteger(j) || j < 0 || j >= m.d) throw new MdmlError('value', `sur: feature ${j} is not a column`)
  const flatY = toFlat(yt)
  const widths = columns.map((cols) => cols.length + (h.fitIntercept ? 1 : 0))
  const designs = columns.map((cols, e) => {
    const p = widths[e] ?? 0
    const a = new Float64Array(m.n * p)
    for (let i = 0; i < m.n; i++) {
      cols.forEach((j, c) => {
        a[i * p + c] = m.data[i * m.d + j] ?? 0
      })
      if (h.fitIntercept) a[i * p + p - 1] = 1
    }
    return a
  })
  const y = Array.from({ length: k }, (_, e) => Float64Array.from({ length: m.n }, (_, i) => flatY[i * k + e] ?? 0))
  if (m.n <= maxOf(widths)) throw new MdmlError('value', 'sur: needs more rows than coefficients per equation')
  return { n: m.n, k, columns, designs, widths, y, fitIntercept: h.fitIntercept }
}

/** `A_k^T A_l` for two equations' designs. */
function crossGram(sys: System, e: number, f: number): Float64Array {
  const pe = sys.widths[e] ?? 0
  const pf = sys.widths[f] ?? 0
  const a = sys.designs[e] ?? new Float64Array(0)
  const b = sys.designs[f] ?? new Float64Array(0)
  const out = new Float64Array(pe * pf)
  for (let i = 0; i < sys.n; i++)
    for (let r = 0; r < pe; r++) {
      const ar = a[i * pe + r] ?? 0
      for (let c = 0; c < pf; c++) out[r * pf + c] = (out[r * pf + c] ?? 0) + ar * (b[i * pf + c] ?? 0)
    }
  return out
}

/** `A_k^T y_l`. */
function crossMoment(sys: System, e: number, f: number): Float64Array {
  const pe = sys.widths[e] ?? 0
  const a = sys.designs[e] ?? new Float64Array(0)
  const y = sys.y[f] ?? new Float64Array(0)
  const out = new Float64Array(pe)
  for (let i = 0; i < sys.n; i++)
    for (let r = 0; r < pe; r++) out[r] = (out[r] ?? 0) + (a[i * pe + r] ?? 0) * (y[i] ?? 0)
  return out
}

/** The stacked GLS solution for a cross-equation covariance: coefficients and the inverse bread. */
function glsSolve(sys: System, sigma: Float64Array): { beta: number[]; cov: Float64Array; total: number } {
  const { k } = sys
  const total = sys.widths.reduce((a, b) => a + b, 0)
  const offsets = sys.widths.map((_, e) => sys.widths.slice(0, e).reduce((a, b) => a + b, 0))
  const prec = inverseOf(sigma, k)
  const a = new Float64Array(total * total)
  const rhs = new Float64Array(total)
  for (let e = 0; e < k; e++)
    for (let f = 0; f < k; f++) {
      const s = prec[e * k + f] ?? 0
      const gram = crossGram(sys, e, f)
      const pe = sys.widths[e] ?? 0
      const pf = sys.widths[f] ?? 0
      for (let r = 0; r < pe; r++)
        for (let c = 0; c < pf; c++)
          a[((offsets[e] ?? 0) + r) * total + (offsets[f] ?? 0) + c] = s * (gram[r * pf + c] ?? 0)
      const mom = crossMoment(sys, e, f)
      for (let r = 0; r < pe; r++) rhs[(offsets[e] ?? 0) + r] = (rhs[(offsets[e] ?? 0) + r] ?? 0) + s * (mom[r] ?? 0)
    }
  const cov = inverseOf(a, total)
  const beta = Array.from({ length: total }, (_, r) => {
    let v = 0
    for (let c = 0; c < total; c++) v += (cov[r * total + c] ?? 0) * (rhs[c] ?? 0)
    return v
  })
  return { beta, cov, total }
}

/** The residual cross-product matrix `E^T E / n` for stacked coefficients. */
function residualCovariance(sys: System, beta: readonly number[]): Float64Array {
  const { n, k } = sys
  const offsets = sys.widths.map((_, e) => sys.widths.slice(0, e).reduce((a, b) => a + b, 0))
  const resid = sys.y.map((y, e) => {
    const p = sys.widths[e] ?? 0
    const a = sys.designs[e] ?? new Float64Array(0)
    return Float64Array.from({ length: n }, (_, i) => {
      let v = y[i] ?? 0
      for (let c = 0; c < p; c++) v -= (a[i * p + c] ?? 0) * (beta[(offsets[e] ?? 0) + c] ?? 0)
      return v
    })
  })
  const out = new Float64Array(k * k)
  for (let e = 0; e < k; e++)
    for (let f = 0; f < k; f++) {
      let s = 0
      for (let i = 0; i < n; i++) s += (resid[e]?.[i] ?? 0) * (resid[f]?.[i] ?? 0)
      out[e * k + f] = s / n
    }
  return out
}

/** The state of {@link surAlgorithm}. */
export type SurState = {
  /** The stacked coefficients: equation by equation, each its features then its intercept. */
  readonly beta: number[]
  /** The cross-equation error covariance used for the last step, `[k][k]` (the identity before the first). */
  readonly sigma: number[][]
  readonly iteration: number
  readonly change: number
}

/**
 * The SUR estimator as a traced algorithm: it starts at equation-by-equation least squares, and each step estimates the
 * cross-equation covariance from the current residuals and re-solves the stacked generalised least squares. It logs the
 * largest coefficient change and the tensors `parameters` and `sigma`.
 *
 * @param data - Features and a `[n, k]` target matrix.
 * @param options - The equations' feature columns, the intercept, and the tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { surAlgorithm } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 20 }, (_, i) => [Math.sin(i), Math.cos(2 * i)])
 * const y = x.map((r, i) => [(r[0] ?? 0) + Math.sin(7 * i), (r[1] ?? 0) + Math.sin(7 * i) + 0.1 * Math.cos(5 * i)])
 * runAlgorithm(surAlgorithm({ x, y }, { equations: [[0], [1]] }), { maxSteps: 2 }).state.sigma.length // => 2
 * ```
 */
export function surAlgorithm(
  data: LearnData,
  options: { readonly equations?: SurHyper['equations']; readonly fitIntercept?: boolean; readonly tol?: number } = {},
): Algorithm<SurState> {
  const sys = systemOf(data, { equations: options.equations ?? 'all', fitIntercept: options.fitIntercept ?? true })
  const tol = options.tol ?? 1e-10
  const { k } = sys
  const total = sys.widths.reduce((a, b) => a + b, 0)
  const identity = Float64Array.from({ length: k * k }, (_, idx) => (Math.floor(idx / k) === idx % k ? 1 : 0))
  return {
    name: 'sur',
    config: { equations: options.equations ?? 'all', fitIntercept: sys.fitIntercept, tol } as Json,
    schema: channels({
      metrics: { change: { description: 'the largest change of the coefficients', direction: 'lower' } },
      tensors: {
        parameters: { role: 'param', description: 'the stacked coefficients', shape: [total] },
        sigma: { role: 'state', description: 'the cross-equation error covariance', shape: [k, k] },
      },
    }),
    init: () => ({
      beta: glsSolve(sys, identity).beta,
      sigma: nested(identity, k),
      iteration: 0,
      change: Number.POSITIVE_INFINITY,
    }),
    step: (s) => {
      const sigma = residualCovariance(sys, s.beta)
      const beta = glsSolve(sys, sigma).beta
      return {
        beta,
        sigma: nested(sigma, k),
        iteration: s.iteration + 1,
        change: maxOf(beta.map((v, j) => Math.abs(v - (s.beta[j] ?? 0)))),
      }
    },
    done: (s) => (s.iteration > 0 && s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { change: Number.isFinite(s.change) ? s.change : 0 },
      tensors: {
        parameters: { role: 'param', value: s.beta, shape: [total] },
        sigma: { role: 'state', value: s.sigma.flat(), shape: [k, k] },
      },
    }),
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j) as SurState,
  }
}

/** The fitted parameters of {@link sur}. */
export type SurParams = {
  /** The slopes of each equation over all features (zero for features not in the equation), `[d][k]`. */
  readonly coef: number[][]
  readonly intercept: number[]
  /** The feature columns of each equation. */
  readonly equations: number[][]
  /** The cross-equation error covariance used for the final coefficients, `[k][k]`. */
  readonly sigma: number[][]
  /** The covariance of the stacked coefficients (equation by equation, features then intercept). */
  readonly covariance: number[][]
  readonly iterations: number
  readonly converged: boolean
}

/** What a fitted SUR system offers. */
export interface SurVerbs {
  readonly nFeaturesIn: number
  /** The predicted outputs, `[n, k]`. */
  readonly predict: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  /** The uniform-average R-squared over the equations. */
  readonly score: (data: LearnData) => number
}

const DEFAULTS: SurHyper = { equations: 'all', fitIntercept: true, maxIter: 1, tol: 1e-10 }

/**
 * Seemingly unrelated regressions: `k` outputs, each regressed on its own subset of the features, estimated jointly by
 * feasible GLS with the error covariance across equations estimated from the residuals (Zellner's estimator for
 * `maxIter = 1`, iterated for more steps). The gain over separate least-squares fits comes from the correlation of the
 * errors and from the equations having different regressors; with identical regressors, or uncorrelated errors, the
 * two coincide. The exposition is the Wald table of all the coefficients, equation by equation.
 *
 * @param hyper - `equations` (feature columns per output), `fitIntercept`, `maxIter`, `tol`.
 * @returns The estimator; the targets are an `[n, k]` matrix.
 * @reference A. Zellner, An efficient method of estimating seemingly unrelated regressions and tests for aggregation
 * bias, JASA 57 (1962).
 * @example
 * ```ts
 * import { sur } from 'mdml-engine/linear'
 * const x = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i)])
 * const y = x.map((r, i) => [(r[0] ?? 0) + Math.sin(7 * i), (r[1] ?? 0) + Math.sin(7 * i)])
 * const f = sur({ equations: [[0], [1]] }).fit({ x, y })
 * f.params.coef.length // => 2
 * ```
 */
export function sur(hyper: Partial<SurHyper> = {}): IterativeEstimator<SurHyper, SurState, SurParams, SurVerbs> {
  const paramsOf = (s: SurState, data: LearnData, h: SurHyper): SurParams => {
    const sys = systemOf(data, h)
    const sigma = s.iteration === 0 ? residualCovariance(sys, s.beta) : Float64Array.from(s.sigma.flat())
    const { cov, total } = glsSolve(sys, sigma)
    const d = matrixView(data.x, 'sur').d
    const offsets = sys.widths.map((_, e) => sys.widths.slice(0, e).reduce((a, b) => a + b, 0))
    const coef = Array.from({ length: d }, () => new Array<number>(sys.k).fill(0))
    const intercept = new Array<number>(sys.k).fill(0)
    sys.columns.forEach((cols, e) => {
      cols.forEach((j, c) => {
        ;(coef[j] as number[])[e] = s.beta[(offsets[e] ?? 0) + c] ?? 0
      })
      if (h.fitIntercept) intercept[e] = s.beta[(offsets[e] ?? 0) + cols.length] ?? 0
    })
    return {
      coef,
      intercept,
      equations: sys.columns,
      sigma: nested(sigma, sys.k),
      covariance: nested(cov, total),
      iterations: s.iteration,
      converged: s.change <= h.tol,
    }
  }
  const tableOf = (par: SurParams, fitIntercept: boolean): CoefficientTable => {
    const names: string[] = []
    const estimates: number[] = []
    par.equations.forEach((cols, e) => {
      cols.forEach((j) => {
        names.push(`y${e}~x${j}`)
        estimates.push(par.coef[j]?.[e] ?? 0)
      })
      if (fitIntercept) {
        names.push(`y${e}~intercept`)
        estimates.push(par.intercept[e] ?? 0)
      }
    })
    const m = names.length
    const cov = Float64Array.from(
      { length: m * m },
      (_, idx) => par.covariance[Math.floor(idx / m)]?.[idx % m] ?? Number.NaN,
    )
    return waldTable(names, estimates, cov)
  }
  return iterative<SurHyper, SurState, SurParams, SurVerbs>(
    {
      name: 'sur',
      kind: 'regressor',
      defaults: DEFAULTS,
      hyperSpace: (h) => dict({ maxIter: int(1, 50, { default: h.maxIter }) }),
      algorithm: (data, h) => surAlgorithm(data, h),
      maxSteps: (h) => h.maxIter,
      params: paramsOf,
      verbs: (par) => {
        const d = par.coef.length
        const k = par.intercept.length
        const predictRows = (x: TensorLike): Float64Array => {
          const m = matrixView(x, 'sur', { nFeatures: d })
          const out = new Float64Array(m.n * k)
          for (let i = 0; i < m.n; i++)
            for (let e = 0; e < k; e++) {
              let v = par.intercept[e] ?? 0
              for (let j = 0; j < d; j++) v += (m.data[i * d + j] ?? 0) * (par.coef[j]?.[e] ?? 0)
              out[i * k + e] = v
            }
          return out
        }
        const base = {
          nFeaturesIn: d,
          predict: (x: TensorLike): Tensor =>
            fromStorage(predictRows(x), k === 1 ? [matrixView(x, 'sur').n] : [matrixView(x, 'sur').n, k], 'float64'),
          featureImportances: par.coef.map((row) => row.reduce((a, v) => a + Math.abs(v), 0)),
        }
        return {
          ...base,
          score: (data: LearnData): number => {
            const truth = toFlat(asTensor(requireTargets(data, 'sur score')))
            const pred = predictRows(data.x)
            const n = pred.length / k
            let total = 0
            for (let e = 0; e < k; e++) {
              let mean = 0
              for (let i = 0; i < n; i++) mean += (truth[i * k + e] ?? 0) / n
              let res = 0
              let tot = 0
              for (let i = 0; i < n; i++) {
                res += ((truth[i * k + e] ?? 0) - (pred[i * k + e] ?? 0)) ** 2
                tot += ((truth[i * k + e] ?? 0) - mean) ** 2
              }
              total += tot === 0 ? 0 : 1 - res / tot
            }
            return total / k
          },
        }
      },
      records: (s, data, h): readonly Exposition[] => [tableOf(paramsOf(s, data, h), h.fitIntercept)],
      logs: {
        'solver/change': Log(),
        'solver/parameters': Log({ detail: 'full' }),
        'solver/sigma': Log({ detail: 'full' }),
      },
    },
    { ...DEFAULTS, ...hyper },
  )
}
