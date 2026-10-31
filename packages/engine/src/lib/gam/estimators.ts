/**
 * Generalised additive models as estimators on the iterative base: `gam` for a numeric response of any exponential
 * family, `gamClassifier` for binary labels (binomial, logit). The fit is {@link gamAlgorithm}, one penalised IRLS
 * iteration per step, so a session shows the smoothing parameters and effective degrees of freedom settling and
 * `fittedAt(t)` is the model at any iteration. A fitted model predicts the mean, the linear predictor and each term's
 * partial effect, gives pointwise standard errors from the Bayesian posterior covariance, and reports the effective
 * degrees of freedom per term, the scale, the deviance and the AIC.
 */

import { MdmlError } from '../base/index.ts'
import { binomial } from '../dists/families/index.ts'
import type { Distribution } from '../dists/index.ts'
import { encodeLabels, type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { accuracyScore, r2Score } from '../metrics/index.ts'
import { dgemm } from '../numerics/linalg/index.ts'
import { glmSolvers } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { choice, dict, real } from '../space/index.ts'
import { fromStorage, item, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { Log } from '../trace/index.ts'
import { posteriorCovariance } from './penalised.ts'
import {
  GAM_DEFAULTS,
  type GamHyper,
  type GamProblem,
  type GamState,
  gamAlgorithm,
  gamProblem,
  gamSolution,
} from './pirls.ts'
import { buildDesign, evaluateDesign, type GamDesign } from './terms.ts'

/** The fitted parameters of a generalised additive model: plain JSON. */
export type GamParams = {
  readonly coef: number[]
  /** The log smoothing parameters, one per penalty. */
  readonly logLambda: number[]
  readonly scale: number
  readonly deviance: number
  /** The total effective degrees of freedom. */
  readonly edf: number
  /** The effective degrees of freedom of each term, in the order of the terms. */
  readonly edfTerms: number[]
  readonly labels: string[]
  /** The Bayesian posterior covariance of the coefficients `[p][p]` (scaled). */
  readonly covariance: number[][]
  /** The training covariates, response, prior weights: the design is rebuilt from them. */
  readonly xFit: number[][]
  readonly yFit: number[]
  readonly weights: number[]
  /** The class labels of a classifier (empty for a regressor). */
  readonly classes: number[]
}

/** What a fitted generalised additive model offers. */
export interface GamVerbs {
  /** The predicted mean `[m]`; for a classifier the label. */
  readonly predict: (x: TensorLike) => Tensor
  /** The linear predictor `[m]`. */
  readonly linearPredictor: (x: TensorLike) => Tensor
  /** The partial effect of every term at new rows `[m, terms]` (the intercept excluded). */
  readonly predictTerms: (x: TensorLike) => Tensor
  /** The standard error of the linear predictor `[m]`, from the Bayesian posterior covariance. */
  readonly standardErrors: (x: TensorLike) => Tensor
  /** The response distribution at the predicted means and the scale. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  /** Class probabilities `[m, 2]` (classifier only). */
  readonly predictProba?: (x: TensorLike) => Tensor
  /** The linear predictor of the positive class (classifier only). */
  readonly decisionFunction?: (x: TensorLike) => Tensor
  readonly classes?: readonly number[]
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
  readonly coef: readonly number[]
  readonly edf: number
  readonly edfTerms: Readonly<Record<string, number>>
  /** The smoothing parameters `lambda`, one per penalty. */
  readonly smoothingParameters: readonly number[]
  readonly scale: number
  readonly deviance: number
  /** `-2 loglik + 2 (edf + 1 if the scale is estimated)`. */
  readonly aic: number
}

const WHO = 'gam'

interface Prepared {
  readonly problem: GamProblem
  readonly classes: number[]
}

function prepare(data: LearnData, h: GamHyper, kind: 'regressor' | 'classifier'): Prepared {
  if (kind === 'regressor') return { problem: gamProblem(data.x, data.y ?? [], h, data.weights), classes: [] }
  const { classes, codes } = encodeLabels(data.y ?? [], WHO)
  if (classes.length !== 2)
    throw new MdmlError('value', `${WHO}: the classifier is binary, got ${classes.length} classes`)
  return { problem: gamProblem(data.x, Array.from(codes), h, data.weights), classes: [...classes] }
}

function verbsOf(p: GamParams, h: GamHyper, kind: 'regressor' | 'classifier'): GamVerbs {
  const design: GamDesign = buildDesign(p.xFit, h.terms)
  const pCols = design.p
  const link = h.link ?? h.family.defaultLink
  const coef = Float64Array.from(p.coef)
  const V = Float64Array.from(p.covariance.flat())
  const rowsOf = (x: TensorLike): { X: Float64Array; n: number } => evaluateDesign(design, x)
  const linearPredictor = (x: TensorLike): Tensor => {
    const { X, n } = rowsOf(x)
    return fromStorage(dgemm(X, coef, n, pCols, 1), [n], 'float64')
  }
  const mean = (x: TensorLike): Tensor => primalOf(link.inverse(linearPredictor(x))) as Tensor
  const labelsOf = (mu: Tensor): Tensor =>
    fromStorage(
      Float64Array.from(toFlat(mu), (v) => p.classes[v > 0.5 ? 1 : 0] ?? Number.NaN),
      [...mu.shape],
      'float64',
    )
  const predict = (x: TensorLike): Tensor => (kind === 'classifier' ? labelsOf(mean(x)) : mean(x))
  const fitted = fromStorage(Float64Array.from(p.yFit), [p.yFit.length], 'float64')
  const trainMu = (): Tensor => mean(p.xFit)
  const logLik = item(
    h.family.logLikelihood(
      fitted,
      trainMu(),
      p.scale,
      fromStorage(Float64Array.from(p.weights), [p.weights.length], 'float64'),
    ),
  )
  const scaleEstimated = h.scale === undefined && !['binomial', 'poisson', 'negativeBinomial'].includes(h.family.name)
  const base: GamVerbs = {
    predict,
    linearPredictor,
    predictTerms: (x) => {
      const { X, n } = rowsOf(x)
      const k = design.terms.length
      const out = new Float64Array(n * k)
      design.terms.forEach((t, j) => {
        for (let i = 0; i < n; i++) {
          let s = 0
          for (let c = 0; c < t.width; c++) s += (X[i * pCols + t.offset + c] ?? 0) * (coef[t.offset + c] ?? 0)
          out[i * k + j] = s
        }
      })
      return fromStorage(out, [n, k], 'float64')
    },
    standardErrors: (x) => {
      const { X, n } = rowsOf(x)
      const vx = dgemm(X, V, n, pCols, pCols)
      return fromStorage(
        Float64Array.from({ length: n }, (_, i) => {
          let s = 0
          for (let j = 0; j < pCols; j++) s += (vx[i * pCols + j] ?? 0) * (X[i * pCols + j] ?? 0)
          return Math.sqrt(Math.max(s, 0))
        }),
        [n],
        'float64',
      )
    },
    predictDistribution: (x) => h.family.distribution(mean(x), p.scale),
    score: (data) =>
      kind === 'classifier'
        ? accuracyScore(data.y ?? [], predict(data.x)).estimate.value
        : r2Score(data.y ?? [], predict(data.x)).estimate.value,
    nFeaturesIn: design.nFeatures,
    coef: p.coef,
    edf: p.edf,
    edfTerms: Object.fromEntries(p.labels.map((label, j) => [label, p.edfTerms[j] ?? 0])),
    smoothingParameters: p.logLambda.map(Math.exp),
    scale: p.scale,
    deviance: p.deviance,
    aic: -2 * logLik + 2 * (p.edf + (scaleEstimated ? 1 : 0)),
  }
  if (kind === 'regressor') return base
  return {
    ...base,
    classes: p.classes,
    predictProba: (x) => {
      const mu = toFlat(mean(x))
      return fromStorage(Float64Array.from(mu.flatMap((v) => [1 - v, v])), [mu.length, 2], 'float64')
    },
    decisionFunction: linearPredictor,
  }
}

function particular(kind: 'regressor' | 'classifier') {
  const defaults: GamHyper = kind === 'classifier' ? { ...GAM_DEFAULTS, family: binomial() } : GAM_DEFAULTS
  return {
    name: kind === 'classifier' ? 'gam/classifier' : 'gam/regressor',
    kind,
    defaults,
    hyperSpace: (h: GamHyper) =>
      dict({
        method: choice(['reml', 'gcv', 'ubre', 'fixed'], { default: h.method, label: 'smoothing selection' }),
        gamma: real(Math.min(0.5, h.gamma), Math.max(3, h.gamma), {
          default: h.gamma,
          label: 'degrees of freedom inflation',
        }),
        solver: glmSolvers.space,
      }),
    algorithm: (data: LearnData, h: GamHyper, key: Key) => gamAlgorithm(prepare(data, h, kind).problem, h, key),
    maxSteps: (h: GamHyper) => h.maxIter,
    params: (s: GamState, data: LearnData, h: GamHyper): GamParams => {
      const { problem, classes } = prepare(data, h, kind)
      const { solution, scale } = gamSolution(problem, s)
      const cov = posteriorCovariance(solution, problem.design.p, scale)
      const m = matrixView(data.x, WHO)
      const p = problem.design.p
      return {
        coef: s.beta,
        logLambda: s.logLambda,
        scale,
        deviance: s.deviance,
        edf: s.edf,
        edfTerms: s.edfTerms,
        labels: problem.design.terms.map((t) => t.label),
        covariance: Array.from({ length: p }, (_, i) => Array.from(cov.subarray(i * p, (i + 1) * p))),
        xFit: Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))),
        yFit: Array.from(problem.y),
        weights: Array.from(problem.prior),
        classes,
      }
    },
    verbs: (p: GamParams, h: GamHyper): GamVerbs => verbsOf(p, h, kind),
    logs: { 'solver/deviance': Log(), 'solver/edf': Log() },
  }
}

/**
 * A generalised additive model for a numeric response (mgcv's `gam`, pyGAM's `GAM`): an exponential-family response
 * whose link is a sum of smooth terms (`s`, `te`, `ti`), parametric terms (`linear`, `factor`) and an intercept, fitted
 * by penalised IRLS with the smoothing parameters chosen by REML (default), GCV or UBRE. Terms default to one cubic
 * regression spline per column.
 *
 * @concept generalized-additive-models
 * @reference S. N. Wood, Generalized Additive Models: An Introduction with R (2nd ed., 2017).
 * @example
 * ```ts
 * import { gam, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const y = x.map((r) => Math.sin(6 * (r[0] ?? 0)))
 * const fitted = gam.with({ terms: [s(0, { k: 8 })] }).fit({ x, y })
 * fitted.score({ x, y }) > 0.99 // => true
 * ```
 */
export const gam: IterativeEstimator<GamHyper, GamState, GamParams, GamVerbs> = iterative<
  GamHyper,
  GamState,
  GamParams,
  GamVerbs
>(particular('regressor'))

/**
 * A generalised additive model for two classes (the binomial family with the logit link): smooth effects of the
 * covariates on the log odds. The labels are the two distinct values of `y`.
 *
 * @concept generalized-additive-models
 * @example
 * ```ts
 * import { gamClassifier, s } from 'mdml-engine/gam'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 39])
 * const y = x.map((r) => ((r[0] ?? 0) > 0.5 ? 1 : 0))
 * gamClassifier.with({ terms: [s(0, { k: 5 })], method: 'gcv' }).fit({ x, y }).classes // => [0, 1]
 * ```
 */
export const gamClassifier: IterativeEstimator<GamHyper, GamState, GamParams, GamVerbs> = iterative<
  GamHyper,
  GamState,
  GamParams,
  GamVerbs
>(particular('classifier'))
