import { argmaxOf, type Json } from '../../lib/base/index.ts'
import {
  coefficientTable,
  type Explained,
  type RegularisationPath,
  regularisationPath,
} from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  featureNamesOf,
  type LearnData,
} from '../../lib/learn/index.ts'
import { ddot, solve } from '../../lib/numerics/linalg/index.ts'
import { bool, dict, int, real, type Space } from '../../lib/space/index.ts'
import { primalOf, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import { centre, designOf, interceptFor, matrixTensor, vectorTensor } from './_design.ts'
import { type FittedLinearModel, type LinearModelParams, linearModelDefaults, linearVerbs } from './model.ts'

// TODO: group lasso, best subset, stepwise and stagewise selection.

/** Options of {@link larsPath}. */
export interface LarsPathOptions {
  /** Least angle regression ('lar') or its lasso modification ('lasso', dropping variables whose sign would flip). */
  readonly method?: 'lar' | 'lasso'
  /** Stop once the penalty (the largest correlation / n) falls to this (default 0). */
  readonly alphaMin?: number
  /** At most this many steps (default 500). */
  readonly maxIter?: number
  readonly fitIntercept?: boolean
}

/** The LARS path: the penalties at its knots and the coefficients there. */
export interface LarsPath {
  readonly alphas: readonly number[]
  /** coefs[i][j]: feature j at alphas[i]. */
  readonly coefs: readonly (readonly number[])[]
  /** The active set after each knot. */
  readonly active: readonly (readonly number[])[]
}

/** The state of the LARS algorithm: coefficients, active set and signs, and the path so far. */
interface LarsState {
  readonly beta: Float64Array
  readonly active: readonly number[]
  readonly alphas: readonly number[]
  readonly coefs: readonly (readonly number[])[]
  readonly sets: readonly (readonly number[])[]
  readonly done: boolean
  readonly dropped: number
}

/**
 * Least angle regression (Efron, Hastie, Johnstone and Tibshirani 2004) as a traced algorithm, one knot per step: the
 * coefficients move along the equiangular direction of the active set until another feature's correlation catches
 * up (or, for the lasso modification, an active coefficient reaches zero and leaves), as scikit-learn's `lars_path`.
 */
function larsAlgorithm(
  x: Float64Array,
  n: number,
  d: number,
  y: Float64Array,
  options: LarsPathOptions,
): Algorithm<LarsState> {
  const lasso = options.method === 'lasso'
  const alphaMin = options.alphaMin ?? 0
  const column = (j: number): Float64Array => Float64Array.from({ length: n }, (_, i) => x[i * d + j] ?? 0)
  const cols = Array.from({ length: d }, (_, j) => column(j))
  const correlations = (beta: Float64Array): Float64Array => {
    const r = Float64Array.from(y)
    for (let j = 0; j < d; j++) {
      const b = beta[j] ?? 0
      if (b === 0) continue
      const c = cols[j] as Float64Array
      for (let i = 0; i < n; i++) r[i] = (r[i] ?? 0) - b * (c[i] ?? 0)
    }
    return Float64Array.from(cols, (c) => ddot(c, r))
  }
  return {
    name: 'lars',
    config: { method: options.method ?? 'lar' } as Json,
    schema: channels({
      metrics: { alpha: { description: 'penalty at the knot' }, active: { description: 'active features' } },
    }),
    init: () => {
      const c = correlations(new Float64Array(d))
      const alpha = Math.max(...Array.from(c, Math.abs)) / n
      return {
        beta: new Float64Array(d),
        active: [],
        alphas: [alpha],
        coefs: [new Array<number>(d).fill(0)],
        sets: [[]],
        done: !(alpha > alphaMin),
        dropped: -1,
      }
    },
    step: (s) => {
      const c = correlations(s.beta)
      const big = Math.max(...Array.from(c, Math.abs))
      const active = [...s.active]
      if (s.dropped < 0) {
        const inactive = Array.from({ length: d }, (_, j) => j).filter((j) => !active.includes(j))
        if (inactive.length === 0) return { ...s, done: true }
        const j = inactive[argmaxOf(inactive.map((k) => Math.abs(c[k] ?? 0)))] ?? 0
        active.push(j)
      }
      const signs = active.map((j) => Math.sign(c[j] ?? 0) || 1)
      const k = active.length
      const g = new Float64Array(k * k)
      for (let a = 0; a < k; a++)
        for (let b = 0; b < k; b++)
          g[a * k + b] =
            (signs[a] ?? 1) *
            (signs[b] ?? 1) *
            ddot(cols[active[a] ?? 0] as Float64Array, cols[active[b] ?? 0] as Float64Array)
      const gi1 = toFlat(primalOf(solve(matrixTensor(g, k, k), vectorTensor(new Float64Array(k).fill(1)))))
      const aa = 1 / Math.sqrt(gi1.reduce((acc, v) => acc + v, 0))
      const w = gi1.map((v) => v * aa)
      // u = X_A diag(s) w, and a = X^T u.
      const u = new Float64Array(n)
      for (const [t, j] of active.entries()) {
        const col = cols[j] as Float64Array
        const f = (signs[t] ?? 1) * (w[t] ?? 0)
        for (let i = 0; i < n; i++) u[i] = (u[i] ?? 0) + f * (col[i] ?? 0)
      }
      const corrU = Float64Array.from(cols, (col) => ddot(col, u))
      let gamma = big / aa
      for (let j = 0; j < d; j++) {
        if (active.includes(j)) continue
        for (const cand of [(big - (c[j] ?? 0)) / (aa - (corrU[j] ?? 0)), (big + (c[j] ?? 0)) / (aa + (corrU[j] ?? 0))])
          if (cand > 1e-12 && cand < gamma) gamma = cand
      }
      let drop = -1
      if (lasso)
        for (const [t, j] of active.entries()) {
          const dj = (signs[t] ?? 1) * (w[t] ?? 0)
          const z = -(s.beta[j] ?? 0) / dj
          if (z > 1e-12 && z < gamma) {
            gamma = z
            drop = j
          }
        }
      // Stop exactly at alphaMin: the correlation falls linearly by gamma * aa along the step.
      const alphaNext = (big - gamma * aa) / n
      if (alphaNext < alphaMin) gamma = (big - alphaMin * n) / aa
      const beta = Float64Array.from(s.beta)
      for (const [t, j] of active.entries()) beta[j] = (beta[j] ?? 0) + gamma * (signs[t] ?? 1) * (w[t] ?? 0)
      const nextActive = drop >= 0 ? active.filter((j) => j !== drop) : active
      if (drop >= 0) beta[drop] = 0
      const alpha = Math.max((big - gamma * aa) / n, 0)
      return {
        beta,
        active: nextActive,
        alphas: [...s.alphas, alpha],
        coefs: [...s.coefs, Array.from(beta)],
        sets: [...s.sets, [...nextActive]],
        done: alpha <= alphaMin + 1e-15,
        dropped: drop,
      }
    },
    done: (s) => (s.done ? 'converged' : undefined),
    observe: (s) => ({ metrics: { alpha: s.alphas[s.alphas.length - 1] ?? 0, active: s.active.length } }),
  }
}

function runLars(data: LearnData, options: LarsPathOptions, run: FitOptions = {}) {
  const design = designOf(data, 'larsPath')
  const c = centre(design, options.fitIntercept ?? true)
  const { n, d } = design.x
  const r = runAlgorithm(larsAlgorithm(c.x, n, d, c.y, options), {
    maxSteps: options.maxIter ?? 500,
    ...(run.tracer === undefined ? {} : { tracer: run.tracer }),
    ...(run.policy === undefined ? {} : { policy: run.policy }),
  })
  return { design, c, state: r.state }
}

/**
 * The least-angle (or lasso) regression path (scikit-learn's `lars_path`): the exact piecewise-linear coefficient path
 * by LARS, with an intercept fitted by centring. The penalties are the largest absolute correlation / n at each knot.
 *
 * @param data - Features, targets and weights.
 * @param options - Method ('lar' or 'lasso'), the smallest penalty, the step limit and the intercept.
 * @returns The knots, with the regularisation-path record.
 */
export function larsPath(data: LearnData, options: LarsPathOptions = {}): Explained<LarsPath> {
  const { design, state } = runLars(data, options)
  const record: RegularisationPath = regularisationPath({
    penalties: state.alphas,
    features: featureNamesOf(data, design.x.d),
    coefficients: state.coefs,
    active: state.sets,
  })
  return { alphas: state.alphas, coefs: state.coefs, active: state.sets, exposition: [record] }
}

/** Hyperparameters of {@link lars} and {@link lassoLars}. */
export interface LarsHyper {
  /** lassoLars: the penalty (default 1). */
  readonly alpha: number
  /** lars: stop after this many features (default all). */
  readonly nNonzeroCoefs?: number
  readonly fitIntercept: boolean
}

function larsEstimator(
  name: string,
  lassoMode: boolean,
  hyper: LarsHyper,
): ExplainedEstimator<LarsHyper, FittedLinearModel> {
  return defineExplainedEstimator(
    {
      name,
      kind: 'regressor',
      hyperSpace: (h: LarsHyper): Space =>
        lassoMode
          ? dict({
              alpha: real(1e-4, 10, {
                label: 'α (L1 penalty)',
                description: 'The lasso penalty: the path stops where it reaches α.',
                log: true,
                default: h.alpha,
              }),
              fitIntercept: bool({
                label: 'fit intercept',
                description: 'Estimate an intercept (otherwise the fit passes through the origin).',
                default: h.fitIntercept,
              }),
            })
          : dict({
              nNonzeroCoefs: int(1, 50, {
                label: 'non-zero coefficients',
                description: 'Stop once this many coefficients are active.',
                default: h.nNonzeroCoefs ?? 10,
              }),
            }),
      fit: (data: LearnData, h: LarsHyper, options: FitOptions) => {
        const { design, c, state } = runLars(
          data,
          {
            method: lassoMode ? 'lasso' : 'lar',
            alphaMin: lassoMode ? h.alpha : 0,
            fitIntercept: h.fitIntercept,
            ...(lassoMode || h.nNonzeroCoefs === undefined ? {} : { maxIter: h.nNonzeroCoefs }),
          },
          options,
        )
        const coef = Array.from(state.beta)
        const intercept = h.fitIntercept ? interceptFor(c, coef) : 0
        const features = featureNamesOf(data, design.x.d)
        const params: LinearModelParams = {
          coef,
          intercept,
          scale: 1,
          solver: 'lars',
          reason: lassoMode ? 'lasso by least angle regression' : 'least angle regression',
          iterations: state.alphas.length - 1,
          converged: true,
          objective: Number.NaN,
        }
        return {
          params,
          exposition: [
            coefficientTable({
              names: [...features, ...(h.fitIntercept ? ['intercept'] : [])],
              estimates: [...coef, ...(h.fitIntercept ? [intercept] : [])],
            }),
            regularisationPath({
              penalties: state.alphas,
              features,
              coefficients: state.coefs,
              active: state.sets,
              ...(lassoMode ? { chosen: h.alpha } : {}),
            }),
          ],
        }
      },
      methods: (p: LinearModelParams) => linearVerbs(p, linearModelDefaults({}), name),
    },
    hyper,
  )
}

/**
 * Least angle regression (sklearn.linear_model.Lars), traced one knot per step. Exposition: the coefficient table and
 * the exact LARS path (coefficients and active sets at each knot).
 *
 * @param hyper - The number of features to add (default all) and the intercept.
 * @returns The estimator.
 */
export function lars(hyper: Partial<LarsHyper> = {}): ExplainedEstimator<LarsHyper, FittedLinearModel> {
  return larsEstimator('lars', false, {
    alpha: 0,
    ...(hyper.nNonzeroCoefs === undefined ? {} : { nNonzeroCoefs: hyper.nNonzeroCoefs }),
    fitIntercept: hyper.fitIntercept ?? true,
  })
}

/**
 * The lasso by least angle regression (sklearn.linear_model.LassoLars): the lasso-modified LARS path followed down to
 * the penalty alpha (scikit-learn's per-sample scaling), so the coefficients are the lasso solution there. Exposition:
 * the coefficient table and the path with the chosen alpha.
 *
 * @param hyper - The penalty (default 1) and the intercept.
 * @returns The estimator.
 */
export function lassoLars(hyper: Partial<LarsHyper> = {}): ExplainedEstimator<LarsHyper, FittedLinearModel> {
  return larsEstimator('lasso-lars', true, { alpha: hyper.alpha ?? 1, fitIntercept: hyper.fitIntercept ?? true })
}

/** {@link lassoLars} as a class. */
export class LassoLars extends ExplainedEstimatorClass<LarsHyper, FittedLinearModel> {
  /** @param hyper - As {@link lassoLars}. */
  constructor(hyper: Partial<LarsHyper> = {}) {
    super(lassoLars(hyper))
  }
}
