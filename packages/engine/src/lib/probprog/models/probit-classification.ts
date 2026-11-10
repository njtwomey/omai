/**
 * Bayesian probit classification on any two-dimensional labelled point set: the model of `probitRegression` written
 * for a design matrix of any width with the scale of the prior as an argument, the polynomial feature map that makes a
 * linear model draw curved boundaries (the inputs standardised, every monomial up to a degree, each column standardised,
 * an intercept first), and what a posterior of the weights says of new points: the predictive probability, the accuracy
 * and the log predictive of held-out rows, and the predictive over a grid of the plane. A page binds the same
 * definition to the data it has, hands it to any inference method of `omai-engine/probprog/inference` and reads the
 * answers with these.
 */

import { OmaiError } from '../../base/index.ts'
import { polynomialFeatures, standardScaler } from '../../learn/preprocessing/index.ts'
import { accuracyScore, logLoss } from '../../metrics/index.ts'
import { ndtr } from '../../numerics/special/index.ts'
import { fromStorage, toFlat } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { type DataInput, defineModel, type ModelDefinition, ops } from '../model/index.ts'

/** A fitted map from points of the plane to the design rows of a probit model. */
export interface PolynomialFeatureMap {
  /** The degree of the monomials. */
  readonly degree: number
  /** The number of columns of the design (the number of weights): the intercept and every monomial of total degree 1 to `degree`. */
  readonly size: number
  /** The name of each column: `1`, `x0`, `x1`, `x0^2`, `x0 x1`, ... */
  readonly names: readonly string[]
  /**
   * The factor the design is divided by (`2 sqrt(size - 1)`): the columns are of unit scale and there are `size` of them, so
   * at the starting weights of a guide or a sampler (uniform in (-2, 2)) the linear predictor of a row is far out in the tail
   * of the normal CDF, where the log likelihood is minus infinity and its gradient not a number. Dividing the design by this
   * keeps the predictor of order one whatever the degree. The weights are then this factor times those of the unit-scale
   * columns, and a prior scale `s` on those is `s * weightScale` here.
   */
  readonly weightScale: number
  /** The design rows of points `[n][2]` (divided by `weightScale`): `[n][size]`, the intercept first and the other columns standardised on the training points. */
  readonly transform: (points: readonly (readonly number[])[]) => number[][]
}

/**
 * The polynomial feature map of a point set: the two inputs are standardised on the points, every monomial of them up to
 * the degree is formed (`polynomialFeatures`), and each monomial column is standardised on the points again (a high
 * degree of an input of unit scale is badly conditioned otherwise), with a constant column for the intercept in front.
 * With `degree` 1 the design is the intercept and the two standardised inputs.
 *
 * @param points - The training points `[n][2]` the standardisation is fitted on.
 * @param degree - The degree, 1 or more (at degree `k` there are `(k + 1)(k + 2) / 2` columns).
 * @returns The map.
 * @throws OmaiError with code `value` for a degree below 1 or an empty point set.
 * @concept feature-engineering
 * @index function The standardised polynomial design of two inputs with an intercept: `polynomialFeatureMap(points, 3).transform(rows)`.
 * @example
 * ```ts
 * import { polynomialFeatureMap } from 'omai-engine/probprog/models'
 * const map = polynomialFeatureMap([[0, 0], [1, 2], [2, 4], [3, 6]], 2)
 * [map.size, map.names, map.weightScale.toFixed(3)] // => [6, ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2'], '4.472']
 * ```
 */
export function polynomialFeatureMap(points: readonly (readonly number[])[], degree: number): PolynomialFeatureMap {
  if (!(Number.isInteger(degree) && degree >= 1))
    throw new OmaiError('value', 'polynomialFeatureMap: degree is a whole number from 1')
  if (points.length === 0) throw new OmaiError('value', 'polynomialFeatureMap: no points to fit on')
  const inputs = standardScaler().fit({ x: points as number[][] })
  const monomials = polynomialFeatures({ degree, includeBias: false }).fit({
    x: inputs.transform(points as number[][]),
  })
  const columns = standardScaler().fit({ x: monomials.transform(inputs.transform(points as number[][])) })
  const names = ['1', ...monomials.featureNamesOut()]
  const unscaled = (rows: readonly (readonly number[])[]): number[][] => {
    if (rows.length === 0) return []
    const flat = toFlat(columns.transform(monomials.transform(inputs.transform(rows as number[][]))))
    const width = names.length - 1
    return Array.from({ length: rows.length }, (_, i) => [
      1,
      ...Array.from({ length: width }, (_, j) => flat[i * width + j] ?? 0),
    ])
  }
  const weightScale = 2 * Math.sqrt(Math.max(1, names.length - 1))
  return {
    degree,
    size: names.length,
    names,
    weightScale,
    transform: (rows) => unscaled(rows).map((r) => r.map((v) => v / weightScale)),
  }
}

/**
 * The probit classifier as a model definition: `y_i ~ Bernoulli(Phi(x_i . w))` with an independent `N(0, priorScale^2)`
 * prior on every weight. The same model as `probitRegression` (whose design has three columns and a prior scale of 2) for
 * a design of any width: bind it to the rows of a design `X` `[n][p]` and the labels `y` `[n]` (`probitData`).
 *
 * @param priorScale - The standard deviation of the prior of each weight (default 2).
 * @returns The definition.
 * @concept model-spec
 * @index function The probit classifier of any design width with a chosen prior scale; bind it with `probitData`.
 * @example
 * ```ts
 * import { probitClassifier, probitData } from 'omai-engine/probprog/models'
 * probitClassifier(1).bind(probitData([[1, 0.5], [1, -0.5], [1, 2]], [1, 0, 1])).sizes // => { row: 3, feature: 2 }
 * ```
 */
export function probitClassifier(priorScale = 2): ModelDefinition {
  return defineModel('probit-classification', (m) => {
    const n = m.range('row')
    const f = m.range('feature')
    const X = m.data('X', { dims: [n, f], role: 'covariate' })
    const y = m.data('y', { dims: [n], support: 'binary', role: 'response' })
    const w = m.latent('w', d.normal(0, priorScale), { dims: [f], doc: 'the weight of each column of the design' })
    m.observe('y', d.bernoulli({ probs: ops.probit(X.matmul(w)) }), y)
  })
}

/**
 * The data a {@link probitClassifier} is bound to, from a design and its labels, as the nested numbers a worker receives.
 *
 * @param design - The design rows `[n][p]`.
 * @param labels - The labels `[n]`, 0 or 1.
 * @returns The data input.
 * @example
 * ```ts
 * import { probitData } from 'omai-engine/probprog/models'
 * probitData([[1, 2], [1, 3]], [1, 0]) // => { X: [[1, 2], [1, 3]], y: [1, 0] }
 * ```
 */
export function probitData(design: readonly (readonly number[])[], labels: readonly number[]): DataInput {
  return { X: design.map((r) => [...r]), y: [...labels] }
}

/**
 * The posterior predictive probability of the positive class at each design row: the mean over draws of the weights of
 * `Phi(x . w)`.
 *
 * @param draws - Draws of the weights `[n][p]`.
 * @param design - Design rows `[m][p]`.
 * @returns The probabilities `[m]`.
 * @throws OmaiError with code `shape` when a draw and a row differ in width, `value` for no draws.
 * @concept posterior-predictive
 * @index function Predictive probability of a probit model at design rows, from weight draws.
 * @example
 * ```ts
 * import { probitPredictive } from 'omai-engine/probprog/models'
 * probitPredictive([[0, 0], [0, 0]], [[1, 5]])[0] // => 0.5
 * ```
 */
export function probitPredictive(
  draws: readonly (readonly number[])[],
  design: readonly (readonly number[])[],
): number[] {
  if (draws.length === 0) throw new OmaiError('value', 'probitPredictive: no draws')
  const p = draws[0]?.length ?? 0
  const out = new Float64Array(design.length)
  for (const row of design)
    if (row.length !== p)
      throw new OmaiError('shape', `probitPredictive: a design row has ${row.length} columns, the draws ${p}`)
  for (const w of draws) {
    if (w.length !== p) throw new OmaiError('shape', 'probitPredictive: draws of different widths')
    const eta = new Float64Array(design.length)
    for (const [i, row] of design.entries()) {
      let s = 0
      for (let j = 0; j < p; j++) s += (row[j] ?? 0) * (w[j] ?? 0)
      eta[i] = s
    }
    const phi = toFlat(ndtr(fromStorage(eta, [eta.length], 'float64')))
    for (let i = 0; i < out.length; i++) out[i] = (out[i] ?? 0) + (phi[i] ?? 0) / draws.length
  }
  return Array.from(out)
}

/** What held-out rows say of a classifier. */
export interface HeldOutScores {
  /** The share of rows whose more probable class is the label. */
  readonly accuracy: number
  /** The mean log predictive probability of the labels, `log p(y_i | x_i, data)` averaged over the rows (higher is better). */
  readonly logPredictive: number
}

/**
 * Accuracy and mean log predictive of held-out rows, from the predictive probabilities of the positive class.
 *
 * @param probabilities - `P(y = 1)` for each row.
 * @param labels - The labels, 0 or 1.
 * @returns The scores.
 * @concept classification-metrics
 * @example
 * ```ts
 * import { heldOutScores } from 'omai-engine/probprog/models'
 * heldOutScores([0.9, 0.2], [1, 0]).accuracy // => 1
 * ```
 */
export function heldOutScores(probabilities: readonly number[], labels: readonly number[]): HeldOutScores {
  const accuracy = accuracyScore(
    labels as number[],
    probabilities.map((p) => (p >= 0.5 ? 1 : 0)),
  ).estimate.value
  const loss = logLoss(labels as number[], probabilities as number[], { labels: [0, 1] }).estimate.value
  return { accuracy, logPredictive: -loss }
}

/** The predictive probability over a rectangle of the plane on a regular grid. */
export interface PredictiveGrid {
  /** The `size` abscissae and the `size` ordinates. */
  readonly xs: readonly number[]
  readonly ys: readonly number[]
  /** `P(y = 1)` at `(xs[i], ys[j])`, row-major in `j` (`probabilities[j * size + i]`). */
  readonly probabilities: readonly number[]
}

/**
 * The posterior predictive probability of the positive class over a `size` by `size` grid covering a rectangle of the
 * plane: the points of the grid go through the feature map, then {@link probitPredictive}.
 *
 * @param draws - Draws of the weights `[n][p]`.
 * @param map - The feature map the weights are for.
 * @param extent - The rectangle, `[low, high]` along each input.
 * @param size - The points of the grid along each side (at least 2).
 * @returns The grid and the probabilities.
 * @concept posterior-predictive
 * @index function Posterior predictive probability of a probit model over a 2-D grid, for a decision boundary.
 * @example
 * ```ts
 * import { polynomialFeatureMap, probitPredictiveGrid } from 'omai-engine/probprog/models'
 * const map = polynomialFeatureMap([[-1, -1], [1, 1], [-1, 1], [1, -1]], 1)
 * probitPredictiveGrid([[0, 1, 0]], map, { x: [-1, 1], y: [-1, 1] }, 2).probabilities.length // => 4
 * ```
 */
export function probitPredictiveGrid(
  draws: readonly (readonly number[])[],
  map: PolynomialFeatureMap,
  extent: { readonly x: readonly [number, number]; readonly y: readonly [number, number] },
  size: number,
): PredictiveGrid {
  if (!(Number.isInteger(size) && size >= 2))
    throw new OmaiError('value', 'probitPredictiveGrid: size is a whole number from 2')
  const axis = (lo: number, hi: number): number[] =>
    Array.from({ length: size }, (_, i) => lo + ((hi - lo) * i) / (size - 1))
  const xs = axis(extent.x[0], extent.x[1])
  const ys = axis(extent.y[0], extent.y[1])
  const rows = ys.flatMap((y) => xs.map((x) => [x, y]))
  return { xs, ys, probabilities: probitPredictive(draws, map.transform(rows)) }
}
