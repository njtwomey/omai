/**
 * Linear classifiers: logistic regression (binary by the GLM core's IRLS, with L1 by coordinate descent, multinomial
 * for several classes), probit regression, the ridge classifier (ridge on targets coded -1 and +1) and Rosenblatt's
 * perceptron, traced epoch by epoch. Fit functions and classes mirroring scikit-learn and statsmodels, with coefficient
 * tables and decision surfaces as exposition.
 */

import { argmaxOf, MdmlError } from '../../lib/base/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { type CoefficientTable, coefficientTable, type Exposition } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  geometricSteps,
  type LearnData,
  matrixView,
  onGeometricSchedule,
  type SurfaceStepsInput,
  sampleWeightsOf,
} from '../../lib/learn/index.ts'
import { ddot } from '../../lib/numerics/linalg/index.ts'
import { ndtr } from '../../lib/numerics/special/index.ts'
import { lbfgs, owlqn } from '../../lib/optim/index.ts'
import { permutation } from '../../lib/random/index.ts'
import { bool, choice, dict, int, real, type Space } from '../../lib/space/index.ts'
import {
  add,
  logSoftmax,
  matmul,
  mul,
  reshape,
  square,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
} from '../../lib/tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../lib/trace/index.ts'
import { type Design, matrixTensor, vectorTensor } from './_design.ts'
import { fitLinear, linearModelDefaults } from './model.ts'
import { leastSquaresSolve } from './solvers.ts'

/** Parameters of a fitted linear classifier: one score row per column of `coef` (one row for binary problems). */
export type LinearClassifierParams = {
  readonly classes: number[]
  /** coef[r][j]: score r, feature j. */
  readonly coef: number[][]
  readonly intercept: number[]
  /** How scores become probabilities: 'binary' (sigmoid of one score), 'softmax' or 'ovr' (normalised sigmoids). */
  readonly link: 'binary' | 'softmax' | 'ovr' | 'none'
  readonly iterations: number
}

/** Verbs of a fitted linear classifier. */
export interface LinearClassifierVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  readonly classes: number[]
  readonly nFeaturesIn: number
  readonly featureImportances: number[]
}

/** A fitted linear classifier. */
export type FittedLinearClassifier = Fitted<LinearClassifierParams> & LinearClassifierVerbs

const sigmoid = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)))

/** Scores x W^T + b, row-major `[m, rows]`. */
function scores(
  p: LinearClassifierParams,
  x: TensorLike,
  who: string,
): { readonly m: number; readonly s: Float64Array } {
  const d = p.coef[0]?.length ?? 0
  const v = matrixView(x, who, { nFeatures: d })
  const r = p.coef.length
  const s = new Float64Array(v.n * r)
  for (let i = 0; i < v.n; i++) {
    const row = v.data.subarray(i * d, (i + 1) * d)
    for (let k = 0; k < r; k++) s[i * r + k] = (p.intercept[k] ?? 0) + ddot(row, p.coef[k] ?? [])
  }
  return { m: v.n, s }
}

function probabilities(p: LinearClassifierParams, x: TensorLike, who: string): Tensor {
  const { m, s } = scores(p, x, who)
  const k = p.classes.length
  const r = p.coef.length
  const out = new Float64Array(m * k)
  for (let i = 0; i < m; i++) {
    if (p.link === 'binary') {
      const q = sigmoid(s[i] ?? 0)
      out[i * k] = 1 - q
      out[i * k + 1] = q
    } else if (p.link === 'softmax') {
      let max = Number.NEGATIVE_INFINITY
      for (let c = 0; c < r; c++) max = Math.max(max, s[i * r + c] ?? 0)
      let z = 0
      for (let c = 0; c < r; c++) z += Math.exp((s[i * r + c] ?? 0) - max)
      for (let c = 0; c < r; c++) out[i * k + c] = Math.exp((s[i * r + c] ?? 0) - max) / z
    } else {
      let z = 0
      for (let c = 0; c < r; c++) z += sigmoid(s[i * r + c] ?? 0)
      for (let c = 0; c < r; c++) out[i * k + c] = sigmoid(s[i * r + c] ?? 0) / z
    }
  }
  return matrixTensor(out, m, k)
}

/** The verbs of a fitted linear classifier, from its parameters. */
export function classifierVerbs(p: LinearClassifierParams, who: string, probabilistic: boolean): LinearClassifierVerbs {
  const d = p.coef[0]?.length ?? 0
  const decisionFunction = (x: TensorLike): Tensor => {
    const { m, s } = scores(p, x, who)
    return p.coef.length === 1 ? vectorTensor(s) : matrixTensor(s, m, p.coef.length)
  }
  const predict = (x: TensorLike): Tensor => {
    const { m, s } = scores(p, x, who)
    const r = p.coef.length
    return vectorTensor(
      Array.from({ length: m }, (_, i) =>
        r === 1
          ? (p.classes[(s[i] ?? 0) > 0 ? 1 : 0] ?? 0)
          : (p.classes[argmaxOf(s.subarray(i * r, (i + 1) * r))] ?? 0),
      ),
    )
  }
  const importances = Array.from({ length: d }, (_, j) => p.coef.reduce((a, row) => a + Math.abs(row[j] ?? 0), 0))
  return {
    predict,
    decisionFunction,
    ...(probabilistic
      ? {
          predictProba: (x: TensorLike) => probabilities(p, x, who),
          predictDistribution: (x: TensorLike) => categorical({ probs: probabilities(p, x, who) }),
        }
      : {}),
    classes: p.classes,
    nFeaturesIn: d,
    featureImportances: importances,
  }
}

/** The coefficient table of a linear classifier (rows named `class:feature` when there are several scores). */
export function classifierTable(
  p: LinearClassifierParams,
  features: readonly string[],
  classNames: readonly string[],
): CoefficientTable {
  const names: string[] = []
  const estimates: number[] = []
  const rows = p.coef.length
  for (const [r, row] of p.coef.entries()) {
    const prefix = rows === 1 ? '' : `${classNames[r] ?? r}:`
    for (const [j, v] of row.entries()) {
      names.push(`${prefix}${features[j] ?? `x${j}`}`)
      estimates.push(v)
    }
    names.push(`${prefix}intercept`)
    estimates.push(p.intercept[r] ?? 0)
  }
  return coefficientTable({ names, estimates })
}

/** How a linear classifier trained: its iterates (the start first), for the decision surface's steps. */
interface LinearTraining {
  /** What an iterate is ('IRLS iteration', 'update'). */
  readonly stepName: string
  readonly iterates: readonly {
    readonly step: number
    readonly coef: readonly (readonly number[])[]
    readonly intercept: readonly number[]
    readonly objective?: number
    /** The rows that mattered at this iterate (the row just seen); default the rows then misclassified. */
    readonly highlight?: readonly number[]
  }[]
  /** What the iterates' highlighted rows are (default the misclassified rows). */
  readonly highlightName?: string
  /** The objective trace, finer than the iterates (default the iterates' objectives). */
  readonly objective?: { readonly name: string; readonly iterations: number[]; readonly values: number[] }
  readonly objectiveName?: string
}

/** The models at up to 40 of a linear classifier's iterates (geometrically spaced), the last the fitted model. */
function linearSteps(
  fitted: FittedLinearClassifier,
  training: LinearTraining,
  revive: (p: LinearClassifierParams) => FittedLinearClassifier,
): SurfaceStepsInput | undefined {
  const its = training.iterates
  if (its.length < 2) return undefined
  const last = its[its.length - 1]?.step ?? 0
  const chosen = new Set<number>()
  for (const target of geometricSteps(last, 40)) {
    let best = 0
    for (let k = 1; k < its.length; k++)
      if (Math.abs((its[k]?.step ?? 0) - target) < Math.abs((its[best]?.step ?? 0) - target)) best = k
    chosen.add(best)
  }
  const kept = [...chosen].sort((a, b) => a - b).map((k) => its[k] as NonNullable<(typeof its)[number]>)
  const models = kept.map((it, t) =>
    t === kept.length - 1
      ? fitted
      : revive({ ...fitted.params, coef: it.coef.map((r) => [...r]), intercept: [...it.intercept] }),
  )
  const objective = kept.every((it) => it.objective !== undefined) ? kept.map((it) => it.objective ?? 0) : undefined
  const highlight = kept.every((it) => it.highlight !== undefined) ? kept.map((it) => it.highlight ?? []) : undefined
  return {
    models,
    iterations: kept.map((it) => it.step),
    ...(objective === undefined ? {} : { objective }),
    highlight: highlight ?? 'misclassified',
    ...(highlight === undefined || training.highlightName === undefined
      ? {}
      : { highlightName: training.highlightName }),
  }
}

/**
 * The exposition of a linear classifier: its coefficients and, for two features, its decision surface (with its steps
 * through training, when given how it trained).
 */
function classifierExposition(
  fitted: FittedLinearClassifier,
  data: LearnData,
  table: CoefficientTable,
  training?: LinearTraining,
  revive?: (p: LinearClassifierParams) => FittedLinearClassifier,
): Exposition[] {
  const d = fitted.nFeaturesIn
  if (d !== 2) return [table]
  const featureNames = featureNamesOf(data, d)
  const classNames = classNamesOf(data, fitted.classes)
  const steps = training === undefined || revive === undefined ? undefined : linearSteps(fitted, training, revive)
  // The objective at every iterate where it is finite (IRLS starts from an infinite one).
  const finite = training?.iterates.filter((it) => it.objective !== undefined && Number.isFinite(it.objective)) ?? []
  const trace =
    training?.objective ??
    (training?.objectiveName === undefined || finite.length < 2
      ? undefined
      : {
          name: training.objectiveName,
          iterations: finite.map((it) => it.step),
          values: finite.map((it) => it.objective ?? 0),
        })
  return [
    table,
    decisionSurfaceOf(fitted, data.x, {
      featureNames,
      classNames,
      ...(data.y === undefined ? {} : { y: data.y }),
      ...(steps === undefined || training === undefined ? {} : { steps, stepName: training.stepName }),
      ...(steps === undefined || trace === undefined ? {} : { objective: trace }),
    }),
  ]
}

/** The training of a binary GLM fit (IRLS iterates), when recorded. */
function glmTraining(
  f: ReturnType<typeof fitLinear>,
  options: FitOptions,
  objectiveName: string,
): LinearTraining | undefined {
  const path = f.solution.path
  if (options.explain === false || path === undefined) return undefined
  return {
    stepName: 'IRLS iteration',
    objectiveName,
    iterates: path.map((it, step) => ({
      step,
      coef: [[...it.coef]],
      intercept: [it.intercept],
      objective: it.objective,
    })),
  }
}

/** Hyperparameters of {@link logisticRegression}. */
export interface LogisticRegressionHyper {
  /** Inverse regularisation strength (scikit-learn's C). */
  readonly C: number
  readonly penalty: 'l2' | 'l1' | 'elasticnet' | 'none'
  /** The L1 share for the elastic net. */
  readonly l1Ratio: number
  readonly fitIntercept: boolean
  /** Multiclass handling: multinomial (softmax) unless one-versus-rest is asked for. */
  readonly multiClass: 'auto' | 'multinomial' | 'ovr'
  readonly maxIter: number
  readonly tol: number
}

function multinomialFit(
  design: Design,
  codes: Int32Array,
  k: number,
  h: LogisticRegressionHyper,
  options: FitOptions,
): { coef: number[][]; intercept: number[]; iterations: number } {
  const { n, d } = design.x
  const cols = d + (h.fitIntercept ? 1 : 0)
  const aug = new Float64Array(n * cols)
  for (let i = 0; i < n; i++) {
    aug.set(design.x.data.subarray(i * d, (i + 1) * d), i * cols)
    if (h.fitIntercept) aug[i * cols + d] = 1
  }
  const a = matrixTensor(aug, n, cols)
  const onehot = new Float64Array(n * k)
  for (let i = 0; i < n; i++) onehot[i * k + (codes[i] ?? 0)] = (design.w[i] ?? 0) * h.C
  const yw = matrixTensor(onehot, n, k)
  const mask = new Float64Array(k * cols)
  for (let c = 0; c < k; c++) for (let j = 0; j < d; j++) mask[c * cols + j] = 1
  const maskT = matrixTensor(mask, k, cols)
  const l1 = h.penalty === 'l1' ? 1 : h.penalty === 'elasticnet' ? h.l1Ratio : 0
  const l2 = h.penalty === 'none' ? 0 : 1 - l1
  // C sum_i w_i (-log softmax) + l2 |W|^2 / 2 (+ l1 |W|_1 by OWL-QN), over the flat [k, cols] parameters.
  const f = (flat: Value): Value => {
    const p = reshape(flat, [k, cols])
    const nll = mul(sum(mul(yw, logSoftmax(matmul(a, transpose(p)), -1))), -1)
    return l2 > 0 ? add(nll, mul(sum(square(mul(p, maskT))), l2 / 2)) : nll
  }
  const run = {
    maxIterations: h.maxIter,
    gradTol: h.tol,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  }
  const x0 = new Float64Array(k * cols)
  const r =
    l1 > 0 ? owlqn(f, x0, { ...run, l1: Array.from(mask, (m) => m * l1) }) : lbfgs(f, x0, { ...run, ftol: 1e-15 })
  const p = toFlat(r.x as Tensor)
  return {
    coef: Array.from({ length: k }, (_, c) => p.slice(c * cols, c * cols + d)),
    intercept: Array.from({ length: k }, (_, c) => (h.fitIntercept ? (p[c * cols + d] ?? 0) : 0)),
    iterations: r.steps,
  }
}

/**
 * Logistic regression (sklearn.linear_model.LogisticRegression): binary problems by the GLM core's binomial family
 * (IRLS, with coordinate descent inside for L1), several classes by the multinomial (softmax) likelihood with L-BFGS
 * (OWL-QN for L1) or one-versus-rest. The objective is C sum_i w_i log loss_i plus the penalty, as scikit-learn's.
 * Exposition: the coefficient table (with Wald errors for an unpenalised binary fit) and, for two features, the
 * decision surface with class probabilities.
 *
 * @param hyper - C (default 1), penalty (default l2), L1 share, intercept, multiclass rule and solver limits.
 * @returns The estimator.
 */
export function logisticRegression(
  hyper: Partial<LogisticRegressionHyper> = {},
): ExplainedEstimator<LogisticRegressionHyper, FittedLinearClassifier> {
  const est: ExplainedEstimator<LogisticRegressionHyper, FittedLinearClassifier> = defineExplainedEstimator(
    {
      name: 'logistic-regression',
      kind: 'classifier',
      hyperSpace: (h: LogisticRegressionHyper): Space =>
        dict({
          C: real(1e-3, 1e3, { log: true, default: h.C, label: 'inverse penalty' }),
          penalty: choice(['l2', 'l1', 'elasticnet', 'none'], {
            label: 'penalty',
            description: 'The regularisation on the coefficients: L2, L1, elastic net or none.',
            default: h.penalty,
          }),
          l1Ratio: real(0, 1, {
            label: 'L1 ratio',
            description: 'The elastic-net mix: 0 is a pure L2 penalty, 1 pure L1.',
            default: h.l1Ratio,
            when: { key: 'penalty', in: ['elasticnet'] },
          }),
          fitIntercept: bool({
            label: 'fit intercept',
            description: 'Estimate an intercept (otherwise the fit passes through the origin).',
            default: h.fitIntercept,
          }),
        }),
      fit: (data: LearnData, h: LogisticRegressionHyper, options: FitOptions) => {
        const x = matrixView(data.x, 'logisticRegression')
        const { classes: labels, codes } = encodeLabels(data.y ?? [], 'logisticRegression', x.n)
        const classes = [...labels]
        if (classes.length < 2) throw new MdmlError('value', 'logisticRegression: needs at least two classes')
        const w = sampleWeightsOf(data.weights, x.n, 'logisticRegression')
        const s = w.reduce((a, b) => a + b, 0)
        const alpha = h.penalty === 'none' ? 0 : 1 / (h.C * s)
        const l1Ratio = h.penalty === 'l1' ? 1 : h.penalty === 'elasticnet' ? h.l1Ratio : 0
        const binary = (positive: number): ReturnType<typeof fitLinear> =>
          fitLinear(
            { x: data.x, y: Array.from(codes, (c) => (c === positive ? 1 : 0)), weights: w },
            linearModelDefaults({
              family: 'binomial',
              alpha,
              l1Ratio,
              fitIntercept: h.fitIntercept,
              maxIter: h.maxIter,
              tol: 1e-12,
            }),
            options,
            'logisticRegression',
          )
        let params: LinearClassifierParams
        let table: CoefficientTable | undefined
        let training: LinearTraining | undefined
        if (classes.length === 2) {
          const f = binary(1)
          training = x.d === 2 ? glmTraining(f, options, 'penalised deviance') : undefined
          params = {
            classes,
            coef: [f.params.coef],
            intercept: [f.params.intercept],
            link: 'binary',
            iterations: f.params.iterations,
          }
          if (h.penalty === 'none') table = f.table
        } else if (h.multiClass === 'ovr') {
          const fits = classes.map((_, c) => binary(c))
          params = {
            classes,
            coef: fits.map((f) => f.params.coef),
            intercept: fits.map((f) => f.params.intercept),
            link: 'ovr',
            iterations: Math.max(...fits.map((f) => f.params.iterations)),
          }
        } else {
          const m = multinomialFit({ x, y: new Float64Array(x.n), w }, codes, classes.length, h, options)
          params = { classes, ...m, link: 'softmax' }
        }
        const fitted = est.revive(params)
        const names = featureNamesOf(data, x.d)
        return {
          params,
          exposition: classifierExposition(
            fitted,
            data,
            table ?? classifierTable(params, names, classNamesOf(data, classes)),
            training,
            (p) => est.revive(p),
          ),
        }
      },
      methods: (p: LinearClassifierParams) => classifierVerbs(p, 'logisticRegression', true),
    },
    {
      C: hyper.C ?? 1,
      penalty: hyper.penalty ?? 'l2',
      l1Ratio: hyper.l1Ratio ?? 0.5,
      fitIntercept: hyper.fitIntercept ?? true,
      multiClass: hyper.multiClass ?? 'auto',
      maxIter: hyper.maxIter ?? 1000,
      tol: hyper.tol ?? 1e-8,
    },
  )
  return est
}

/** Hyperparameters of {@link probitRegression}. */
export interface ProbitRegressionHyper {
  /** An L2 penalty on the per-sample deviance (default 0). */
  readonly alpha: number
  readonly fitIntercept: boolean
}

/**
 * Probit regression for two classes: the binomial GLM with the probit link, by IRLS (statsmodels' Probit).
 * Exposition: the coefficient table with Wald errors when unpenalised, and the decision surface for two features.
 *
 * @param hyper - L2 penalty (default 0) and intercept.
 * @returns The estimator.
 */
export function probitRegression(
  hyper: Partial<ProbitRegressionHyper> = {},
): ExplainedEstimator<ProbitRegressionHyper, FittedLinearClassifier> {
  const est: ExplainedEstimator<ProbitRegressionHyper, FittedLinearClassifier> = defineExplainedEstimator(
    {
      name: 'probit-regression',
      kind: 'classifier',
      hyperSpace: (h: ProbitRegressionHyper): Space =>
        dict({
          alpha: real(0, 1, {
            label: 'α (L2 penalty)',
            description: 'The L2 penalty on the coefficients.',
            default: h.alpha,
          }),
          fitIntercept: bool({
            label: 'fit intercept',
            description: 'Estimate an intercept (otherwise the fit passes through the origin).',
            default: h.fitIntercept,
          }),
        }),
      fit: (data: LearnData, h: ProbitRegressionHyper, options: FitOptions) => {
        const x = matrixView(data.x, 'probitRegression')
        const { classes: labels, codes } = encodeLabels(data.y ?? [], 'probitRegression', x.n)
        const classes = [...labels]
        if (classes.length !== 2) throw new MdmlError('value', 'probitRegression: needs exactly two classes')
        const f = fitLinear(
          { x: data.x, y: Array.from(codes), ...(data.weights === undefined ? {} : { weights: data.weights }) },
          linearModelDefaults({
            family: 'binomial',
            link: 'probit',
            alpha: h.alpha,
            fitIntercept: h.fitIntercept,
            tol: 1e-12,
          }),
          options,
          'probitRegression',
        )
        const params: LinearClassifierParams = {
          classes,
          coef: [f.params.coef],
          intercept: [f.params.intercept],
          link: 'none',
          iterations: f.params.iterations,
        }
        const training = x.d === 2 ? glmTraining(f, options, 'deviance') : undefined
        return {
          params,
          exposition: classifierExposition(est.revive(params), data, f.table, training, (p) => est.revive(p)),
        }
      },
      methods: (p: LinearClassifierParams) => {
        const base = classifierVerbs(p, 'probitRegression', false)
        const proba = (x: TensorLike): Tensor => {
          const q = toFlat(ndtr(base.decisionFunction(x)))
          const out = new Float64Array(q.length * 2)
          for (const [i, v] of q.entries()) {
            out[2 * i] = 1 - v
            out[2 * i + 1] = v
          }
          const s = q
          return matrixTensor(out, s.length, 2)
        }
        return {
          ...base,
          predictProba: proba,
          predictDistribution: (x: TensorLike) => categorical({ probs: proba(x) }),
        }
      },
    },
    { alpha: hyper.alpha ?? 0, fitIntercept: hyper.fitIntercept ?? true },
  )
  return est
}

/** Hyperparameters of {@link ridgeClassifier}. */
export interface RidgeClassifierHyper {
  readonly alpha: number
  readonly fitIntercept: boolean
}

/**
 * The ridge classifier (sklearn.linear_model.RidgeClassifier): ridge regression on targets coded -1/+1 (one column per
 * class beyond two), predicting the largest score. Exposition: the coefficient table and, for two features, the
 * decision surface.
 *
 * @param hyper - Penalty (default 1) and intercept.
 * @returns The estimator.
 */
export function ridgeClassifier(
  hyper: Partial<RidgeClassifierHyper> = {},
): ExplainedEstimator<RidgeClassifierHyper, FittedLinearClassifier> {
  const est: ExplainedEstimator<RidgeClassifierHyper, FittedLinearClassifier> = defineExplainedEstimator(
    {
      name: 'ridge-classifier',
      kind: 'classifier',
      hyperSpace: (h: RidgeClassifierHyper): Space =>
        dict({
          alpha: real(1e-3, 1e3, {
            label: 'α (L2 penalty)',
            description: 'The L2 penalty on the coefficients (ridge).',
            log: true,
            default: h.alpha,
          }),
          fitIntercept: bool({
            label: 'fit intercept',
            description: 'Estimate an intercept (otherwise the fit passes through the origin).',
            default: h.fitIntercept,
          }),
        }),
      fit: (data: LearnData, h: RidgeClassifierHyper) => {
        const x = matrixView(data.x, 'ridgeClassifier')
        const { classes: labels, codes } = encodeLabels(data.y ?? [], 'ridgeClassifier', x.n)
        const classes = [...labels]
        const w = sampleWeightsOf(data.weights, x.n, 'ridgeClassifier')
        const columns = classes.length === 2 ? [1] : classes.map((_, c) => c)
        const fits = columns.map((c) =>
          leastSquaresSolve(
            { x, y: Float64Array.from(codes, (k) => (k === c ? 1 : -1)), w },
            h.alpha,
            h.fitIntercept,
            'cholesky',
          ),
        )
        const params: LinearClassifierParams = {
          classes,
          coef: fits.map((f) => Array.from(f.coef)),
          intercept: fits.map((f) => f.intercept),
          link: 'none',
          iterations: 1,
        }
        const table = classifierTable(params, featureNamesOf(data, x.d), classNamesOf(data, classes))
        return { params, exposition: classifierExposition(est.revive(params), data, table) }
      },
      methods: (p: LinearClassifierParams) => classifierVerbs(p, 'ridgeClassifier', false),
    },
    { alpha: hyper.alpha ?? 1, fitIntercept: hyper.fitIntercept ?? true },
  )
  return est
}

/** Hyperparameters of {@link perceptron}. */
export interface PerceptronHyper {
  /** Epochs (default 1000). */
  readonly maxIter: number
  /** Stop when an epoch's loss improves by less than this (default 1e-3, as scikit-learn). */
  readonly tol: number
  /** The learning rate (default 1). */
  readonly eta0: number
  readonly fitIntercept: boolean
  /** Shuffle rows each epoch (default true). */
  readonly shuffle: boolean
}

/** The state of the perceptron's training algorithm. */
export interface PerceptronState {
  readonly coef: Float64Array
  readonly intercept: Float64Array
  readonly mistakes: number
  readonly loss: number
  readonly best: number
  readonly stall: number
}

/**
 * Rosenblatt's perceptron (sklearn.linear_model.Perceptron, one-versus-rest for several classes), trained by a traced
 * algorithm with one epoch per step: mistakes and the perceptron loss per epoch, stopping when the loss stops
 * improving by `tol` for five epochs. Exposition: the coefficient table and, for two features, the decision surface.
 *
 * @param hyper - Epochs, tolerance, learning rate, intercept and shuffling.
 * @returns The estimator.
 */
export function perceptron(
  hyper: Partial<PerceptronHyper> = {},
): ExplainedEstimator<PerceptronHyper, FittedLinearClassifier> {
  const est: ExplainedEstimator<PerceptronHyper, FittedLinearClassifier> = defineExplainedEstimator(
    {
      name: 'perceptron',
      kind: 'classifier',
      hyperSpace: (h: PerceptronHyper): Space =>
        dict({
          maxIter: int(1, 1000, { label: 'epochs', description: 'Passes over the data.', default: h.maxIter }),
          eta0: real(1e-3, 10, {
            label: 'learning rate η₀',
            description: 'The step size of each update.',
            log: true,
            default: h.eta0,
          }),
        }),
      fit: (data: LearnData, h: PerceptronHyper, options: FitOptions) => {
        const x = matrixView(data.x, 'perceptron')
        const { classes: labels, codes } = encodeLabels(data.y ?? [], 'perceptron', x.n)
        const classes = [...labels]
        const rows = classes.length === 2 ? 1 : classes.length
        const target = (i: number, r: number): number => {
          const c = codes[i] ?? 0
          return rows === 1 ? (c === 1 ? 1 : -1) : c === r ? 1 : -1
        }
        const { n, d } = x
        // Two features: snapshot the weights after updates on a geometric schedule, with the row that caused each.
        const record = options.explain !== false && d === 2
        let updates = 0
        const iterates: LinearTraining['iterates'][number][] = []
        const epochs: { iterations: number[]; values: number[] } = { iterations: [], values: [] }
        const snapshot = (coef: Float64Array, intercept: Float64Array, row?: number): void => {
          iterates.push({
            step: updates,
            coef: Array.from({ length: rows }, (_, k) => Array.from(coef.subarray(k * d, (k + 1) * d))),
            intercept: Array.from(intercept),
            ...(row === undefined ? {} : { highlight: [row] }),
          })
        }
        if (record) snapshot(new Float64Array(rows * d), new Float64Array(rows), undefined)
        const algorithm: Algorithm<PerceptronState> = {
          name: 'perceptron',
          config: { eta0: h.eta0, classes: classes.length },
          schema: channels({
            metrics: {
              mistakes: { description: 'misclassified rows in the epoch', direction: 'lower' },
              loss: { description: 'perceptron loss over the epoch', direction: 'lower' },
            },
          }),
          init: () => ({
            coef: new Float64Array(rows * d),
            intercept: new Float64Array(rows),
            mistakes: n,
            loss: Infinity,
            best: Infinity,
            stall: 0,
          }),
          step: (s, ctx) => {
            const coef = Float64Array.from(s.coef)
            const intercept = Float64Array.from(s.intercept)
            const order = h.shuffle
              ? Array.from(toFlat(permutation(ctx.key, n)))
              : Array.from({ length: n }, (_, i) => i)
            let mistakes = 0
            let loss = 0
            for (const i of order) {
              const row = x.data.subarray(i * d, (i + 1) * d)
              let wrong = false
              for (let r = 0; r < rows; r++) {
                const t = target(i, r)
                const z = (intercept[r] ?? 0) + ddot(row, coef.subarray(r * d, (r + 1) * d))
                if (t * z <= 0) {
                  wrong = true
                  loss -= t * z
                  for (let j = 0; j < d; j++) coef[r * d + j] = (coef[r * d + j] ?? 0) + h.eta0 * t * (row[j] ?? 0)
                  if (h.fitIntercept) intercept[r] = (intercept[r] ?? 0) + h.eta0 * t
                }
              }
              if (wrong) {
                mistakes++
                updates++
                if (record && onGeometricSchedule(updates)) snapshot(coef, intercept, i)
              }
            }
            if (record) {
              epochs.iterations.push(updates)
              epochs.values.push(mistakes)
            }
            loss /= n
            const improved = loss < s.best - h.tol
            return { coef, intercept, mistakes, loss, best: Math.min(s.best, loss), stall: improved ? 0 : s.stall + 1 }
          },
          done: (s) => (s.mistakes === 0 ? 'converged' : s.stall >= 5 ? 'converged' : undefined),
          observe: (s) => ({ metrics: { mistakes: s.mistakes, loss: s.loss } }),
        }
        const r = runAlgorithm(algorithm, {
          maxSteps: h.maxIter,
          ...(options.key === undefined ? {} : { key: options.key }),
          ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
          ...(options.policy === undefined ? {} : { policy: options.policy }),
        })
        const params: LinearClassifierParams = {
          classes,
          coef: Array.from({ length: rows }, (_, k) => Array.from(r.state.coef.subarray(k * d, (k + 1) * d))),
          intercept: Array.from(r.state.intercept),
          link: 'none',
          iterations: r.steps,
        }
        const table = classifierTable(params, featureNamesOf(data, d), classNamesOf(data, classes))
        if (record && iterates[iterates.length - 1]?.step !== updates)
          snapshot(r.state.coef, r.state.intercept, undefined)
        // The first iterate highlights nothing; the last (the fitted model) the rows it still gets wrong.
        const training: LinearTraining | undefined = record
          ? {
              stepName: 'update',
              highlightName: 'row updated on',
              iterates: iterates.map((it, k) => (k === 0 || k === iterates.length - 1 ? { ...it, highlight: [] } : it)),
              objective: { name: 'mistakes in the epoch', ...epochs },
            }
          : undefined
        return {
          params,
          exposition: classifierExposition(est.revive(params), data, table, training, (p) => est.revive(p)),
        }
      },
      methods: (p: LinearClassifierParams) => classifierVerbs(p, 'perceptron', false),
    },
    {
      maxIter: hyper.maxIter ?? 1000,
      tol: hyper.tol ?? 1e-3,
      eta0: hyper.eta0 ?? 1,
      fitIntercept: hyper.fitIntercept ?? true,
      shuffle: hyper.shuffle ?? true,
    },
  )
  return est
}

/** {@link logisticRegression} as a class. */
export class LogisticRegression extends ExplainedEstimatorClass<LogisticRegressionHyper, FittedLinearClassifier> {
  /** @param hyper - As {@link logisticRegression}. */
  constructor(hyper: Partial<LogisticRegressionHyper> = {}) {
    super(logisticRegression(hyper))
  }
}

/** {@link probitRegression} as a class. */
export class ProbitRegression extends ExplainedEstimatorClass<ProbitRegressionHyper, FittedLinearClassifier> {
  /** @param hyper - As {@link probitRegression}. */
  constructor(hyper: Partial<ProbitRegressionHyper> = {}) {
    super(probitRegression(hyper))
  }
}

/** {@link ridgeClassifier} as a class. */
export class RidgeClassifier extends ExplainedEstimatorClass<RidgeClassifierHyper, FittedLinearClassifier> {
  /** @param hyper - As {@link ridgeClassifier}. */
  constructor(hyper: Partial<RidgeClassifierHyper> = {}) {
    super(ridgeClassifier(hyper))
  }
}

/** {@link perceptron} as a class. */
export class Perceptron extends ExplainedEstimatorClass<PerceptronHyper, FittedLinearClassifier> {
  /** @param hyper - As {@link perceptron}. */
  constructor(hyper: Partial<PerceptronHyper> = {}) {
    super(perceptron(hyper))
  }
}
