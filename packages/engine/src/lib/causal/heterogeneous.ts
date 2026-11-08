/**
 * Heterogeneous treatment effects by meta-learners: reductions of the conditional average treatment effect (CATE) to
 * ordinary supervised learning with any base estimator of the library. The S-learner fits one outcome model with the
 * treatment as a feature, the T-learner one per arm, the X-learner (Kunzel et al.) regresses imputed unit effects and
 * blends the two arms by the propensity, the R-learner (Nie and Wager) minimises the residual-on-residual loss as a
 * weighted regression, and the doubly robust DR-learner regresses the AIPW score. `pehe` scores an estimate against a
 * planted effect.
 */

import { OmaiError } from '../base/index.ts'
import { matrixView } from '../learn/validation.ts'
import { fromStorage, type TensorLike } from '../tensor/index.ts'
import { type Prepared, pick, pickRows, prepare, withTreatment } from './_dense.ts'
import type { CausalData } from './data.ts'
import { nuisancesFor } from './estimation.ts'
import {
  allRows,
  defaultOutcome,
  expectationOf,
  type Learner,
  type NuisanceOptions,
  propensityModel,
} from './nuisance.ts'

/** A fitted CATE model: the estimated effect of the treatment at covariate values. */
export interface CateModel {
  /** Which learner made it. */
  readonly learner: string
  /** The estimated effect at each row of `x`. */
  readonly predict: (x: TensorLike) => Float64Array
}

/** Options of the meta-learners. */
export interface MetaOptions extends NuisanceOptions {
  /** The learner of the final stage (X, R and DR learners): effects as a function of `x` (default the outcome learner). */
  readonly effect?: Learner
  /** Cross-fitting folds for the nuisance models of the R and DR learners (default 5; 0 for none). */
  readonly folds?: number
}

const matrixTensor = (v: Float64Array, d: number): TensorLike => fromStorage(v, [v.length / d, d], 'float64')

function fitMean(
  learner: Learner,
  x: Float64Array,
  d: number,
  y: ArrayLike<number>,
  key: MetaOptions['key'],
  binary: boolean,
  weights?: ArrayLike<number>,
): (q: TensorLike) => Float64Array {
  const fitted = learner.fit(
    {
      x: matrixTensor(x, d),
      y: Float64Array.from(y),
      ...(weights === undefined ? {} : { weights: Float64Array.from(weights) }),
    },
    { ...(key === undefined ? {} : { key }), explain: false },
  )
  return (q) => expectationOf(fitted, q, binary)
}

const readX = (x: TensorLike, d: number, who: string): Float64Array => {
  const m = matrixView(x, who)
  if (m.d !== d) throw new OmaiError('shape', `${who}: expected ${d} covariates, got ${m.d}`)
  return m.data
}

/**
 * The S-learner: one outcome model of the covariates and the treatment, the effect its difference between treatment 1
 * and 0. Simple and data-efficient, but a regularised base learner can shrink the treatment's contribution to zero.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The base learner (`outcome`), `binaryOutcome` and key.
 * @returns The CATE model.
 * @example
 * ```ts
 * import { sLearner } from 'omai-engine/causal'
 * const x = [[0], [1], [2], [3], [4], [5]]
 * const m = sLearner({ x, treatment: [0, 1, 0, 1, 0, 1], outcome: [0, 3, 2, 5, 4, 7] })
 * m.predict([[1], [3]])[0] // => 2
 * ```
 */
export function sLearner(data: CausalData, options: MetaOptions = {}): CateModel {
  const p = prepare(data.x, data.treatment, data.outcome, 'sLearner')
  const predictMean = fitMean(
    options.outcome ?? defaultOutcome(),
    withTreatment(p),
    p.d + 1,
    p.y,
    options.key,
    options.binaryOutcome ?? false,
  )
  return {
    learner: 's-learner',
    predict: (x) => {
      const q = readX(x, p.d, 'sLearner')
      const n = q.length / p.d
      const w = p.d + 1
      const one = new Float64Array(n * w)
      const zero = new Float64Array(n * w)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p.d; j++) {
          one[i * w + j] = q[i * p.d + j] ?? 0
          zero[i * w + j] = q[i * p.d + j] ?? 0
        }
        one[i * w + p.d] = 1
      }
      const a = predictMean(matrixTensor(one, w))
      const b = predictMean(matrixTensor(zero, w))
      return Float64Array.from(a, (v, i) => v - (b[i] ?? 0))
    },
  }
}

/** Fit the learner on the rows of an arm. */
function armModel(
  learner: Learner,
  p: Prepared,
  arm: number,
  key: MetaOptions['key'],
  binary: boolean,
): (q: TensorLike) => Float64Array {
  const rows = Array.from(allRows(p.n)).filter((i) => p.t[i] === arm)
  if (rows.length < 2)
    throw new OmaiError('value', `the ${arm === 1 ? 'treated' : 'control'} arm has fewer than two units`)
  return fitMean(learner, pickRows(p.x, p.d, rows), p.d, pick(p.y, rows), key, binary)
}

/**
 * The T-learner: an outcome model per arm, the effect their difference. It can express any heterogeneity but fits each
 * arm on its own units only, so it is noisy when one arm is small.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The base learner (`outcome`), `binaryOutcome` and key.
 * @returns The CATE model.
 * @example
 * ```ts
 * import { tLearner } from 'omai-engine/causal'
 * const x = [[0], [1], [2], [3], [4], [5]]
 * tLearner({ x, treatment: [0, 1, 0, 1, 0, 1], outcome: [0, 3, 2, 5, 4, 7] }).predict([[2]])[0] // => 2
 * ```
 */
export function tLearner(data: CausalData, options: MetaOptions = {}): CateModel {
  const p = prepare(data.x, data.treatment, data.outcome, 'tLearner')
  const learner = options.outcome ?? defaultOutcome()
  const binary = options.binaryOutcome ?? false
  const m1 = armModel(learner, p, 1, options.key, binary)
  const m0 = armModel(learner, p, 0, options.key, binary)
  return {
    learner: 't-learner',
    predict: (x) => {
      readX(x, p.d, 'tLearner')
      const a = m1(x)
      const b = m0(x)
      return Float64Array.from(a, (v, i) => v - (b[i] ?? 0))
    },
  }
}

/**
 * The X-learner: fit a model per arm, impute each unit's missing potential outcome with the other arm's model, regress
 * the imputed effects on the covariates within each arm, and combine the two effect models weighted by the propensity
 * (the effect model of the larger arm gets less weight). It shines when one arm is much smaller than the other.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - The outcome learner, the final-stage `effect` learner, the propensity learner and key.
 * @returns The CATE model.
 * @example
 * ```ts
 * import { xLearner } from 'omai-engine/causal'
 * const x = [[0], [1], [2], [3], [4], [5], [6], [7]]
 * xLearner({ x, treatment: [0, 1, 0, 1, 0, 1, 0, 1], outcome: [0, 3, 2, 5, 4, 7, 6, 9] }).learner // => 'x-learner'
 * ```
 */
export function xLearner(data: CausalData, options: MetaOptions = {}): CateModel {
  const p = prepare(data.x, data.treatment, data.outcome, 'xLearner')
  const learner = options.outcome ?? defaultOutcome()
  const effect = options.effect ?? learner
  const binary = options.binaryOutcome ?? false
  const m1 = armModel(learner, p, 1, options.key, binary)
  const m0 = armModel(learner, p, 0, options.key, binary)
  const rows1 = Array.from(allRows(p.n)).filter((i) => p.t[i] === 1)
  const rows0 = Array.from(allRows(p.n)).filter((i) => p.t[i] === 0)
  const x1 = pickRows(p.x, p.d, rows1)
  const x0 = pickRows(p.x, p.d, rows0)
  const mu0AtTreated = m0(matrixTensor(x1, p.d))
  const mu1AtControls = m1(matrixTensor(x0, p.d))
  const d1 = Float64Array.from(rows1, (r, i) => (p.y[r] ?? 0) - (mu0AtTreated[i] ?? 0))
  const d0 = Float64Array.from(rows0, (r, i) => (mu1AtControls[i] ?? 0) - (p.y[r] ?? 0))
  const tau1 = fitMean(effect, x1, p.d, d1, options.key, false)
  const tau0 = fitMean(effect, x0, p.d, d0, options.key, false)
  const g = propensityModel(p, allRows(p.n), options)
  return {
    learner: 'x-learner',
    predict: (x) => {
      readX(x, p.d, 'xLearner')
      const w = g(x)
      const a = tau0(x)
      const b = tau1(x)
      return Float64Array.from(w, (wi, i) => wi * (a[i] ?? 0) + (1 - wi) * (b[i] ?? 0))
    },
  }
}

/**
 * The R-learner (Nie and Wager): cross-fit the outcome mean `m(x)` and the propensity `e(x)`, then choose the effect
 * function minimising `sum ((Y - m) - tau(X) (T - e))^2`, which is a regression of `(Y - m) / (T - e)` on `X` weighted by
 * `(T - e)^2`. The loss is insensitive to errors in the nuisance models to first order (Neyman orthogonality).
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - Nuisance learners, the `effect` learner (must accept sample weights), `folds` and key.
 * @returns The CATE model.
 * @example
 * ```ts
 * import { rLearner } from 'omai-engine/causal'
 * const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(2 * i)])
 * const t = x.map((_, i) => (Math.sin(3 * i) > 0 ? 1 : 0))
 * const y = x.map((r, i) => (r[0] ?? 0) + 2 * (t[i] ?? 0))
 * rLearner({ x, treatment: t, outcome: y }, { folds: 2 }).learner // => 'r-learner'
 * ```
 */
export function rLearner(data: CausalData, options: MetaOptions = {}): CateModel {
  const p = prepare(data.x, data.treatment, data.outcome, 'rLearner')
  const nu = nuisancesFor(p, { ...options, folds: options.folds ?? 5, pooled: true })
  const m = nu.m ?? new Float64Array(p.n)
  const resT = Float64Array.from(p.t, (t, i) => t - (nu.e[i] ?? 0))
  const resY = Float64Array.from(p.y, (y, i) => y - (m[i] ?? 0))
  const pseudo = Float64Array.from(resY, (v, i) => v / (resT[i] ?? 1))
  const weights = Float64Array.from(resT, (v) => v * v)
  const predictMean = fitMean(options.effect ?? defaultOutcome(), p.x, p.d, pseudo, options.key, false, weights)
  return {
    learner: 'r-learner',
    predict: (x) => {
      readX(x, p.d, 'rLearner')
      return predictMean(x)
    },
  }
}

/**
 * The doubly robust DR-learner (Kennedy): cross-fit the outcome models and the propensity, form each unit's AIPW score
 * (an unbiased signal of its effect when either nuisance is right), and regress the scores on the covariates.
 *
 * @param data - Covariates, treatment and outcome.
 * @param options - Nuisance learners, the `effect` learner, `folds` and key.
 * @returns The CATE model.
 * @example
 * ```ts
 * import { drLearner } from 'omai-engine/causal'
 * const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(2 * i)])
 * const t = x.map((_, i) => (Math.sin(3 * i) > 0 ? 1 : 0))
 * const y = x.map((r, i) => (r[0] ?? 0) + 2 * (t[i] ?? 0))
 * drLearner({ x, treatment: t, outcome: y }, { folds: 2 }).learner // => 'dr-learner'
 * ```
 */
export function drLearner(data: CausalData, options: MetaOptions = {}): CateModel {
  const p = prepare(data.x, data.treatment, data.outcome, 'drLearner')
  const nu = nuisancesFor(p, { ...options, folds: options.folds ?? 5 })
  const score = Float64Array.from({ length: p.n }, (_, i) => {
    const e = nu.e[i] ?? 0.5
    const m1 = nu.mu1[i] ?? 0
    const m0 = nu.mu0[i] ?? 0
    return m1 - m0 + ((p.t[i] ?? 0) * ((p.y[i] ?? 0) - m1)) / e - ((1 - (p.t[i] ?? 0)) * ((p.y[i] ?? 0) - m0)) / (1 - e)
  })
  const predictMean = fitMean(options.effect ?? defaultOutcome(), p.x, p.d, score, options.key, false)
  return {
    learner: 'dr-learner',
    predict: (x) => {
      readX(x, p.d, 'drLearner')
      return predictMean(x)
    },
  }
}

/**
 * The precision in estimation of heterogeneous effects: the root mean squared error of estimated against true individual
 * (or conditional) effects, `sqrt(mean((tau_hat - tau)^2))`.
 *
 * @param estimated - Estimated effects.
 * @param truth - The planted effects.
 * @returns The PEHE.
 * @example
 * ```ts
 * import { pehe } from 'omai-engine/causal'
 * pehe([1, 2, 3], [1, 2, 5]) // => 1.1547005383792515
 * ```
 */
export function pehe(estimated: ArrayLike<number>, truth: ArrayLike<number>): number {
  if (estimated.length !== truth.length)
    throw new OmaiError('shape', 'pehe: estimated and true effects must have the same length')
  let s = 0
  for (let i = 0; i < estimated.length; i++) s += ((estimated[i] ?? 0) - (truth[i] ?? 0)) ** 2
  return Math.sqrt(s / estimated.length)
}
