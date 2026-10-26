/**
 * Regularisation paths: coefficients as the penalty varies. The elastic-net and lasso paths by warm-started coordinate
 * descent (scikit-learn's `enet_path`), the ridge path from one SVD, and cross-validated mean squared error along a
 * path with the best and one-standard-error choices of the penalty.
 */

import { argminOf, MdmlError } from '../../lib/base/index.ts'
import {
  type CrossValidationCurve,
  crossValidationCurve,
  type Explained,
  type RegularisationPath,
  regularisationPath,
} from '../../lib/exposition/index.ts'
import { featureNamesOf, type LearnData, takeRows } from '../../lib/learn/index.ts'
import { kFold, type Splitter, splitIndices } from '../../lib/learn/model-selection/index.ts'
import { ddot } from '../../lib/numerics/linalg/index.ts'
import { key as makeKey } from '../../lib/random/index.ts'
import { centre, type Design, designOf, logGrid } from './_design.ts'
import { coordinateDescentSolve, svdRidge } from './solvers.ts'

/** Options of {@link enetPath}. */
export interface EnetPathOptions {
  /** The L1 share (default 1, the lasso path). */
  readonly l1Ratio?: number
  /** The penalties, decreasing (default a log grid from alpha_max down to eps alpha_max). */
  readonly alphas?: readonly number[]
  /** Grid size (default 100, as scikit-learn). */
  readonly nAlphas?: number
  /** The grid's lower end relative to alpha_max (default 1e-3). */
  readonly eps?: number
  readonly fitIntercept?: boolean
  /** Coordinate-descent tolerance (default 1e-4). */
  readonly tol?: number
  readonly maxIter?: number
}

/** The coefficients of an elastic net along a penalty grid. */
export interface EnetPath {
  readonly alphas: readonly number[]
  /** coefs[i][j]: feature j at alphas[i]. */
  readonly coefs: readonly (readonly number[])[]
  readonly intercepts: readonly number[]
  /** The duality gap at each penalty (per sample). */
  readonly gaps: readonly number[]
  readonly iterations: readonly number[]
}

/** The smallest penalty at which every coefficient is zero (scikit-learn's alpha_max). */
export function alphaMax(design: Design, l1Ratio: number, fitIntercept: boolean): number {
  const c = centre(design, fitIntercept)
  const { n, d } = design.x
  let best = 0
  for (let j = 0; j < d; j++) {
    let s = 0
    for (let i = 0; i < n; i++) s += (c.x[i * d + j] ?? 0) * (c.y[i] ?? 0)
    best = Math.max(best, Math.abs(s))
  }
  const sw = design.w.reduce((a, b) => a + b, 0)
  return best / (sw * Math.max(l1Ratio, 1e-3))
}

/** The elastic-net path on a design, warm-started along the penalties. */
export function pathOn(design: Design, alphas: readonly number[], options: EnetPathOptions): EnetPath {
  const l1Ratio = options.l1Ratio ?? 1
  const fitIntercept = options.fitIntercept ?? true
  const coefs: number[][] = []
  const intercepts: number[] = []
  const gaps: number[] = []
  const iterations: number[] = []
  let start: { coef: ArrayLike<number>; intercept: number } | undefined
  for (const alpha of alphas) {
    const s = coordinateDescentSolve(design, alpha, l1Ratio, fitIntercept, {
      tol: options.tol ?? 1e-4,
      maxIter: options.maxIter ?? 1000,
      ...(start === undefined ? {} : { start }),
    })
    start = { coef: s.coef, intercept: s.intercept }
    coefs.push(Array.from(s.coef))
    intercepts.push(s.intercept)
    gaps.push(s.gap ?? Number.NaN)
    iterations.push(s.iterations)
  }
  return { alphas, coefs, intercepts, gaps, iterations }
}

/** The penalty grid of a path. */
export function alphaGrid(design: Design, options: EnetPathOptions): number[] {
  if (options.alphas !== undefined) return [...options.alphas]
  const max = alphaMax(design, options.l1Ratio ?? 1, options.fitIntercept ?? true)
  return logGrid(max, options.eps ?? 1e-3, options.nAlphas ?? 100)
}

/** A path's exposition record. */
export function pathRecord(path: EnetPath, features: readonly string[], chosen?: number): RegularisationPath {
  return regularisationPath({
    penalties: path.alphas,
    features,
    coefficients: path.coefs,
    active: path.coefs.map((c) => c.flatMap((v, j) => (v !== 0 ? [j] : []))),
    gaps: path.gaps,
    ...(chosen === undefined ? {} : { chosen }),
  })
}

/**
 * The elastic-net (lasso, by default) regularisation path by warm-started coordinate descent (scikit-learn's
 * `enet_path` and `lasso_path`, with an intercept fitted by centring): coefficients, active sets and duality gaps
 * along a decreasing penalty grid.
 *
 * @param data - Features, targets and weights.
 * @param options - L1 share, grid, intercept and solver tolerance.
 * @returns The path, with its regularisation-path record.
 */
export function enetPath(data: LearnData, options: EnetPathOptions = {}): Explained<EnetPath> {
  const design = designOf(data, 'enetPath')
  const path = pathOn(design, alphaGrid(design, options), options)
  return { ...path, exposition: [pathRecord(path, featureNamesOf(data, design.x.d))] }
}

/**
 * The lasso path: {@link enetPath} with l1Ratio = 1.
 *
 * @param data - Features, targets and weights.
 * @param options - Grid, intercept and solver tolerance.
 * @returns The path, with its regularisation-path record.
 */
export function lassoPath(data: LearnData, options: Omit<EnetPathOptions, 'l1Ratio'> = {}): Explained<EnetPath> {
  return enetPath(data, { ...options, l1Ratio: 1 })
}

/**
 * The ridge path from one SVD: coefficients for each penalty of min |y - x beta - b|^2 + alpha |beta|^2 (scikit-learn's
 * Ridge scaling).
 *
 * @param data - Features, targets and weights.
 * @param alphas - The penalties.
 * @param fitIntercept - Whether to fit an intercept (default true).
 * @returns The coefficients per penalty, with the regularisation-path record.
 */
export function ridgePath(
  data: LearnData,
  alphas: readonly number[],
  fitIntercept = true,
): Explained<{ readonly alphas: readonly number[]; readonly coefs: readonly (readonly number[])[] }> {
  const design = designOf(data, 'ridgePath')
  const c = centre(design, fitIntercept)
  const coefs = svdRidge(c.x, design.x.n, design.x.d, c.y, alphas).map((v) => Array.from(v))
  const record = regularisationPath({
    penalties: alphas,
    features: featureNamesOf(data, design.x.d),
    coefficients: coefs,
  })
  return { alphas, coefs, exposition: [record] }
}

/** Options of {@link cvPath}. */
export interface CvPathOptions extends EnetPathOptions {
  /** The splitter (default 5-fold without shuffling, as scikit-learn). */
  readonly cv?: Splitter
}

/** Cross-validated mean squared error along an elastic-net path. */
export interface CvPath {
  readonly alphas: readonly number[]
  /** mse[i][f]: test MSE at alphas[i] on fold f. */
  readonly mse: readonly (readonly number[])[]
  readonly bestAlpha: number
  /** The largest penalty within one standard error of the best. */
  readonly oneStandardErrorAlpha: number
  readonly curve: CrossValidationCurve
}

/**
 * Cross-validate an elastic-net path (scikit-learn's LassoCV and ElasticNetCV): the grid comes from the full data, each
 * fold's path is fitted on its training rows with warm starts and scored by mean squared error on its test rows.
 *
 * @param data - Features, targets and weights.
 * @param options - Path options and the splitter.
 * @returns The fold errors, the best and one-standard-error penalties and the cross-validation curve.
 * @throws MdmlError `value` when the splitter yields no folds.
 */
export function cvPath(data: LearnData, options: CvPathOptions = {}): CvPath {
  const design = designOf(data, 'cvPath')
  const alphas = alphaGrid(design, options)
  const folds = splitIndices(options.cv ?? kFold(5), { n: design.x.n }, makeKey(0))
  if (folds.length === 0) throw new MdmlError('value', 'cvPath: the splitter gave no folds')
  const { d } = design.x
  const take = (rows: Int32Array): Design => {
    return {
      x: takeRows(design.x, rows),
      y: Float64Array.from(rows, (r) => design.y[r] ?? 0),
      w: Float64Array.from(rows, (r) => design.w[r] ?? 0),
    }
  }
  const perFold = folds.map((fold) => {
    const train = take(fold.train)
    const test = take(fold.test)
    const path = pathOn(train, alphas, options)
    return path.coefs.map((coef, i) => {
      let se = 0
      let sw = 0
      for (let r = 0; r < test.x.n; r++) {
        const pred = (path.intercepts[i] ?? 0) + ddot(test.x.data.subarray(r * d, (r + 1) * d), coef)
        se += (test.w[r] ?? 0) * ((test.y[r] ?? 0) - pred) ** 2
        sw += test.w[r] ?? 0
      }
      return se / sw
    })
  })
  const mse = alphas.map((_, i) => perFold.map((f) => f[i] ?? Number.NaN))
  const curve = curveOf('alpha', alphas, mse, false, { scoreName: 'mean squared error', penalty: true })
  return {
    alphas,
    mse,
    bestAlpha: curve.best ?? alphas[0] ?? 0,
    oneStandardErrorAlpha: curve.oneStandardError ?? alphas[0] ?? 0,
    curve,
  }
}

/**
 * A cross-validation curve from per-fold scores: means, standard errors (standard deviation over folds / sqrt(folds)),
 * the best value and the one-standard-error choice (the largest value whose mean is within one standard error of the
 * best, values assumed ordered from most to least regularised or the reverse).
 *
 * @param parameter - The parameter's name.
 * @param values - Its values.
 * @param scores - scores[i][f]: value i on fold f.
 * @param higherIsBetter - The score's direction.
 * @param names - What the scores are, and whether the parameter is a penalty.
 * @returns The curve.
 */
export function curveOf(
  parameter: string,
  values: readonly number[],
  scores: readonly (readonly number[])[],
  higherIsBetter: boolean,
  names: { readonly scoreName?: string; readonly penalty?: boolean } = {},
): CrossValidationCurve {
  const means = scores.map((s) => s.reduce((a, b) => a + b, 0) / Math.max(s.length, 1))
  const ses = scores.map((s, i) => {
    const m = means[i] ?? 0
    const k = s.length
    if (k < 2) return 0
    const v = s.reduce((a, b) => a + (b - m) ** 2, 0) / (k - 1)
    return Math.sqrt(v / k)
  })
  const sign = higherIsBetter ? -1 : 1
  const bestIndex = argminOf(means.map((m) => sign * m))
  const bestMean = means[bestIndex] ?? 0
  const target = higherIsBetter ? bestMean - (ses[bestIndex] ?? 0) : bestMean + (ses[bestIndex] ?? 0)
  let one = bestIndex
  let largest = values[bestIndex] ?? 0
  for (const [i, m] of means.entries()) {
    const ok = higherIsBetter ? m >= target : m <= target
    if (ok && (values[i] ?? 0) > largest) {
      largest = values[i] ?? 0
      one = i
    }
  }
  return crossValidationCurve({
    parameter,
    values,
    meanScores: means,
    standardErrors: ses,
    foldScores: scores,
    higherIsBetter,
    best: values[bestIndex] ?? 0,
    oneStandardError: values[one] ?? 0,
    ...names,
  })
}
