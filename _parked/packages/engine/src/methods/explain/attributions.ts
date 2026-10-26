/**
 * Attributions of fitted models as exposition records: `explainPrediction` for one row and `explainDataset` for many,
 * by Tree SHAP, exact or Kernel SHAP, sampling SHAP (with its convergence) or LIME, over the output a model is read by.
 */

import { MdmlError, meanOf, varianceOf } from '../../lib/base/index.ts'
import * as explain from '../../lib/explain/index.ts'
import { curveSet, type Explained, type Exposition, featureAttributions } from '../../lib/exposition/index.ts'
import { type Fitted, featureNamesOf, type LearnData, matrixView } from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../lib/random/index.ts'
import { pearson } from '../../lib/stats/index.ts'
import { fromStorage, toFlat } from '../../lib/tensor/index.ts'
import { modelOutput, type OutputKind, treeEnsembleOf } from './_models.ts'

/**
 * The explainers {@link explainPrediction} and {@link explainDataset} offer: Tree SHAP, Kernel SHAP, exact and sampled
 * interventional Shapley values, LIME, linear SHAP (for models linear in their inputs), partition SHAP (Owen values
 * over a correlation clustering of the features) and conditional Shapley values (missing features drawn from their
 * Gaussian conditional given the present ones).
 */
export type AttributionMethod =
  | 'tree'
  | 'kernel'
  | 'exact'
  | 'sampling'
  | 'lime'
  | 'linear'
  | 'partition'
  | 'conditional'

/** Options of {@link explainPrediction} and {@link explainDataset}. */
export interface ExplainOptions {
  /** The explainer (default `tree` for trees and tree ensembles, else `kernel`). */
  readonly method?: AttributionMethod | 'auto'
  /** The output explained (default: a regressor's prediction, a classifier's probability, a booster's log odds). */
  readonly output?: OutputKind
  /** The class explained, for classifiers (default the positive class of a binary problem, else the last). */
  readonly classIndex?: number
  /** Background rows drawn from the data for the model-agnostic explainers (default 50). */
  readonly background?: number
  /** Kernel SHAP's coalitions, or LIME's perturbations (default theirs). */
  readonly nSamples?: number
  /** Sampling SHAP's permutations (default 200). */
  readonly nPermutations?: number
  /** Also compute SHAP interaction values (tree explainer only; exponential in the features). */
  readonly interactions?: boolean
  readonly key?: Key
}

/** The attributions of one row. */
export interface PredictionExplanation {
  readonly method: AttributionMethod
  /** One value per feature. */
  readonly values: readonly number[]
  readonly baseValue: number
  readonly prediction: number
}

/** The attributions of many rows. */
export interface DatasetExplanation {
  readonly method: AttributionMethod
  /** values[row][feature]. */
  readonly values: readonly (readonly number[])[]
  readonly baseValue: number
  readonly predictions: readonly number[]
  /** The rows explained. */
  readonly rows: readonly number[]
}

interface Setup {
  readonly method: AttributionMethod
  readonly x: ReturnType<typeof matrixView>
  readonly features: string[]
  readonly output: ReturnType<typeof modelOutput>
  readonly key: Key
}

function setup(fitted: Fitted, data: LearnData, options: ExplainOptions, who: string): Setup {
  const x = matrixView(data.x, who)
  const output = modelOutput(fitted, data, options.output ?? 'auto', options.classIndex)
  const trees = treeEnsembleOf(fitted, output)
  const wanted = options.method ?? 'auto'
  const method: AttributionMethod = wanted === 'auto' ? (trees === undefined ? 'kernel' : 'tree') : wanted
  if (method === 'tree' && trees === undefined)
    throw new MdmlError(
      'value',
      `${who}: the tree explainer needs a tree, a forest or a boosted model explaining its ${output.kind === 'raw' ? 'raw score' : 'prediction or probability'}`,
    )
  if ((method === 'conditional' || method === 'partition') && x.d > 12)
    throw new MdmlError('value', `${who}: the ${method} explainer takes at most 12 features`)
  if (method === 'exact' && x.d > explain.MAX_EXACT_PLAYERS)
    throw new MdmlError('value', `${who}: the exact explainer takes at most ${explain.MAX_EXACT_PLAYERS} features`)
  return { method, x, features: featureNamesOf(data, x.d), output, key: options.key ?? makeKey(0) }
}

function rowOf(x: Setup['x'], i: number): number[] {
  return Array.from(x.data.subarray(i * x.d, (i + 1) * x.d))
}

function backgroundOf(s: Setup, size: number): number[][] {
  const n = Math.min(size, s.x.n)
  const order = toFlat(permutation(foldIn(s.key, 0), s.x.n))
  return Array.from({ length: n }, (_, r) => rowOf(s.x, order[r] ?? r))
}

function predictRows(s: Setup, rows: readonly (readonly number[])[]): number[] {
  if (rows.length === 0) return []
  return toFlat(s.output.predict(fromStorage(Float64Array.from(rows.flat()), [rows.length, s.x.d], 'float64')))
}

/** Running sampling SHAP estimates in chunks of permutations, combined exactly from each chunk's mean and error. */
function samplingWithConvergence(
  s: Setup,
  row: readonly number[],
  background: readonly (readonly number[])[],
  total: number,
): {
  readonly values: number[]
  readonly standardErrors: number[]
  readonly samples: number[]
  readonly estimates: number[][]
  readonly errors: number[][]
} {
  const d = s.x.d
  const chunk = Math.max(2, Math.ceil(total / 40))
  const sum = new Float64Array(d)
  const sumSq = new Float64Array(d)
  let count = 0
  const samples: number[] = []
  const estimates: number[][] = []
  const errors: number[][] = []
  for (let c = 0; count < total; c++) {
    const b = Math.min(chunk, total - count)
    const part = explain.samplingShapley(s.output.predict, [...row], background as number[][], {
      nPermutations: b,
      key: foldIn(s.key, 1000 + c),
    })
    for (let j = 0; j < d; j++) {
      const m = part.values[j] ?? 0
      const se = part.standardErrors[j] ?? 0
      sum[j] = (sum[j] ?? 0) + b * m
      // Recover the chunk's sum of squares from its mean and standard error (zero spread for a one-sample chunk).
      sumSq[j] = (sumSq[j] ?? 0) + b * ((Number.isFinite(se) ? se * se * (b - 1) : 0) + m * m)
    }
    count += b
    const mean = Array.from(sum, (v) => v / count)
    samples.push(count)
    estimates.push(mean)
    errors.push(
      Array.from(sumSq, (v, j) =>
        count < 2 ? 0 : Math.sqrt(Math.max(0, v / count - (mean[j] ?? 0) ** 2) / (count - 1)),
      ),
    )
  }
  return {
    values: estimates[estimates.length - 1] ?? new Array<number>(d).fill(0),
    standardErrors: errors[errors.length - 1] ?? new Array<number>(d).fill(0),
    samples,
    estimates,
    errors,
  }
}

interface RowsResult {
  readonly values: number[][]
  readonly baseValue: number
  readonly predictions: number[]
  readonly standardErrors?: number[][]
  readonly interactions?: number[][][]
  readonly localFit?: number[]
  readonly baseValues?: number[]
  readonly convergence?: { samples: number[]; estimates: number[][]; standardErrors: number[][] }
}

function explainRows(
  fitted: Fitted,
  s: Setup,
  rows: readonly number[],
  options: ExplainOptions,
  convergence: boolean,
): RowsResult {
  const xs = rows.map((i) => rowOf(s.x, i))
  const predictions = predictRows(s, xs)
  if (s.method === 'tree') {
    const trees = treeEnsembleOf(fitted, s.output)
    if (trees === undefined) throw new MdmlError('value', 'explain: the model has no trees')
    const r = explain.ensembleTreeShap(trees.trees, xs, { combine: trees.combine, scale: trees.scale })
    const factor = (trees.combine === 'mean' ? 1 / trees.trees.length : 1) * trees.scale
    const interactions =
      options.interactions === true
        ? xs.map((x) => {
            const acc = Array.from({ length: s.x.d }, () => new Array<number>(s.x.d).fill(0))
            for (const t of trees.trees) {
              const m = explain.treeShapInteractions(t, [x])[0] ?? []
              m.forEach((row, i) => {
                row.forEach((v, j) => {
                  const a = acc[i] as number[]
                  a[j] = (a[j] ?? 0) + factor * v
                })
              })
            }
            return acc
          })
        : undefined
    return {
      values: r.values.map((v) => [...v]),
      baseValue: r.baseValue + trees.offset,
      predictions,
      ...(interactions === undefined ? {} : { interactions }),
    }
  }
  const background = backgroundOf(s, options.background ?? 50)
  const baseValue = predictRows(s, background).reduce((a, b) => a + b, 0) / Math.max(1, background.length)
  if (s.method === 'kernel') {
    const r = explain.kernelShap(s.output.predict, xs, background, {
      ...(options.nSamples === undefined ? {} : { nSamples: options.nSamples }),
      key: foldIn(s.key, 1),
    })
    return { values: r.values.map((v) => [...v]), baseValue: r.baseValue, predictions }
  }
  if (s.method === 'exact') {
    const values = xs.map((x) =>
      explain.shapleyValues(explain.interventionalValue(s.output.predict, x, background), s.x.d),
    )
    return { values, baseValue, predictions }
  }
  if (s.method === 'linear') {
    const { coef, intercept } = linearCoefficients(s)
    const r = explain.linearShap(coef, intercept, xs, background)
    return { values: r.values.map((v) => [...v]), baseValue: r.baseValue, predictions }
  }
  if (s.method === 'partition') {
    const all = Array.from({ length: Math.min(s.x.n, 500) }, (_, i) => rowOf(s.x, i))
    const r = explain.partitionShap(s.output.predict, xs, background, explain.correlationHierarchy(all))
    return { values: r.values.map((v) => [...v]), baseValue, predictions }
  }
  if (s.method === 'conditional') {
    const values = xs.map((x, r) =>
      explain.shapleyValues(
        explain.conditionalValue(s.output.predict, x, background, {
          nSamples: options.nSamples ?? 50,
          key: foldIn(s.key, 4 + r),
        }),
        s.x.d,
      ),
    )
    return { values, baseValue, predictions }
  }
  if (s.method === 'sampling') {
    const total = options.nPermutations ?? 200
    if (!(Number.isInteger(total) && total >= 2)) throw new MdmlError('value', 'explain: nPermutations must be >= 2')
    const parts = xs.map((x, r) =>
      r === 0 && convergence
        ? samplingWithConvergence(s, x, background, total)
        : (() => {
            const p = explain.samplingShapley(s.output.predict, x, background, {
              nPermutations: total,
              key: foldIn(s.key, 2 + r),
            })
            return { values: [...p.values], standardErrors: [...p.standardErrors] }
          })(),
    )
    const first = parts[0] as Partial<ReturnType<typeof samplingWithConvergence>> | undefined
    return {
      values: parts.map((p) => p.values),
      baseValue,
      predictions,
      standardErrors: parts.map((p) => p.standardErrors),
      ...(first?.samples === undefined
        ? {}
        : {
            convergence: {
              samples: first.samples,
              estimates: first.estimates ?? [],
              standardErrors: first.errors ?? [],
            },
          }),
    }
  }
  // LIME: the local linear model's weights times each feature's standardised offset, from its intercept.
  const training = Array.from({ length: s.x.n }, (_, i) => rowOf(s.x, i))
  const mean = Array.from({ length: s.x.d }, (_, j) => training.reduce((a, r) => a + (r[j] ?? 0), 0) / s.x.n)
  const scale = mean.map((m, j) => {
    const v = Math.sqrt(training.reduce((a, r) => a + ((r[j] ?? 0) - m) ** 2, 0) / s.x.n)
    return v === 0 ? 1 : v
  })
  const fits = xs.map((x, r) =>
    explain.lime(s.output.predict, x, training, {
      featureSelection: 'none',
      nFeatures: s.x.d,
      ...(options.nSamples === undefined ? { nSamples: 2000 } : { nSamples: options.nSamples }),
      key: foldIn(s.key, 3 + r),
    }),
  )
  return {
    values: fits.map((f, r) => {
      const out = new Array<number>(s.x.d).fill(0)
      for (const [j, w] of f.weights) out[j] = w * (((xs[r]?.[j] ?? 0) - (mean[j] ?? 0)) / (scale[j] ?? 1))
      return out
    }),
    // Each row's local model starts from its own intercept.
    baseValue: fits.reduce((a, f) => a + f.intercept, 0) / Math.max(1, fits.length),
    baseValues: fits.map((f) => f.intercept),
    predictions,
    localFit: fits.map((f) => f.score),
  }
}

/**
 * The weights and intercept of a model linear in its inputs, read off d + 1 evaluations (at 0 and each unit vector)
 * and checked at a data row, so linear SHAP applies to any fitted linear model without knowing its parameters.
 */
function linearCoefficients(s: Setup): { readonly coef: number[]; readonly intercept: number } {
  const d = s.x.d
  const probe = [
    new Array<number>(d).fill(0),
    ...Array.from({ length: d }, (_, j) => Array.from({ length: d }, (_, k) => (k === j ? 1 : 0))),
  ]
  const check = rowOf(s.x, 0)
  const out = predictRows(s, [...probe, check])
  const intercept = out[0] ?? 0
  const coef = Array.from({ length: d }, (_, j) => (out[j + 1] ?? 0) - intercept)
  const want = intercept + coef.reduce((a, w, j) => a + w * (check[j] ?? 0), 0)
  const got = out[d + 1] ?? 0
  if (Math.abs(want - got) > 1e-8 * (1 + Math.abs(got)))
    throw new MdmlError('value', 'explain: the linear explainer needs a model linear in its inputs')
  return { coef, intercept }
}

/**
 * For each feature, the feature that most interacts with it: the largest interaction values when there are any, else
 * shap's approximate interactions (sort by the feature, and add up over windows of rows the |correlation| between its
 * attributions and each other feature's values). -1 when nothing varies.
 */
function interactionPartners(
  values: readonly (readonly number[])[],
  data: readonly (readonly number[])[],
  interactions?: readonly (readonly (readonly number[])[])[],
): number[] {
  const n = values.length
  const d = values[0]?.length ?? 0
  if (interactions !== undefined)
    return Array.from({ length: d }, (_, f) => {
      let best = -1
      let top = 0
      for (let g = 0; g < d; g++) {
        if (g === f) continue
        const s = meanOf(interactions.map((m) => Math.abs(m[f]?.[g] ?? 0)))
        if (s > top) {
          top = s
          best = g
        }
      }
      return best
    })
  const inc = Math.max(Math.min(Math.floor(n / 10), 50), 1)
  return Array.from({ length: d }, (_, f) => {
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => (data[a]?.[f] ?? 0) - (data[b]?.[f] ?? 0))
    const ref = order.map((i) => values[i]?.[f] ?? 0)
    let best = -1
    let top = 0
    for (let g = 0; g < d; g++) {
      if (g === f) continue
      const other = order.map((i) => data[i]?.[g] ?? 0)
      let v = 0
      for (let j = 0; j + 1 < n; j += inc) {
        const a = ref.slice(j, j + inc)
        const b = other.slice(j, j + inc)
        if (a.length > 2 && varianceOf(a) > 0 && varianceOf(b) > 0) v += Math.abs(pearson(a, b).statistic)
      }
      if (v > top) {
        top = v
        best = g
      }
    }
    return best
  })
}

/**
 * Explain one prediction of a fitted model: how much each feature moved the output from its expected value (SHAP:
 * Tree SHAP for trees and their ensembles, Kernel SHAP, exact or permutation-sampling Shapley values for any model)
 * or a local linear surrogate's view of it (LIME). The sampling explainer records how its estimate settled.
 *
 * @param fitted - A fitted estimator.
 * @param data - The data it was fitted on (the background for the model-agnostic explainers, and feature names).
 * @param row - The index of the row of `data` to explain.
 * @param options - method, output, classIndex, background, nSamples, nPermutations, interactions and key.
 * @returns The attributions with a `featureAttributions` record.
 */
export function explainPrediction(
  fitted: Fitted,
  data: LearnData,
  row: number,
  options: ExplainOptions = {},
): Explained<PredictionExplanation> {
  const s = setup(fitted, data, options, 'explainPrediction')
  if (!(Number.isInteger(row) && row >= 0 && row < s.x.n))
    throw new MdmlError('value', `explainPrediction: row ${row} is not in the data`)
  const r = explainRows(fitted, s, [row], options, true)
  const values = r.values[0] ?? []
  const record = featureAttributions({
    method: s.method,
    features: s.features,
    output: s.output.name,
    baseValue: r.baseValue,
    values: [values],
    data: [rowOf(s.x, row)],
    predictions: r.predictions,
    indices: [row],
    ...(r.standardErrors === undefined ? {} : { standardErrors: r.standardErrors }),
    ...(r.interactions === undefined ? {} : { interactions: r.interactions }),
    ...(r.localFit === undefined ? {} : { localFit: r.localFit }),
    ...(r.baseValues === undefined ? {} : { baseValues: r.baseValues }),
    ...(r.convergence === undefined ? {} : { convergence: { row: 0, ...r.convergence } }),
  })
  const exposition: Exposition[] = [record]
  const c = r.convergence
  if (c !== undefined) {
    // The running estimates of the features that matter most, settling as permutations accrue.
    const top = values
      .map((v, j) => ({ v: Math.abs(v), j }))
      .sort((a, b) => b.v - a.v)
      .slice(0, 6)
      .map((e) => e.j)
    exposition.push(
      curveSet({
        x: c.samples,
        xLabel: 'permutations',
        yLabel: `attribution to ${s.output.name}`,
        series: top.map((j) => ({
          name: s.features[j] ?? `x${j}`,
          y: c.estimates.map((e) => e[j] ?? 0),
          spread: c.standardErrors.map((e) => 1.96 * (e[j] ?? 0)),
        })),
      }),
    )
  }
  return {
    method: s.method,
    values,
    baseValue: r.baseValues?.[0] ?? r.baseValue,
    prediction: r.predictions[0] ?? 0,
    exposition,
  }
}

/** Options of {@link explainDataset}. */
export interface ExplainDatasetOptions extends ExplainOptions {
  /** The rows to explain (default a keyed sample of at most `maxRows`). */
  readonly rows?: readonly number[]
  /** At most this many rows when `rows` is not given (default 200). */
  readonly maxRows?: number
}

/**
 * Explain many predictions of a fitted model at once, for the global pictures built from local attributions: the
 * beeswarm summary, the mean |SHAP| bar and dependence scatters.
 *
 * @param fitted - A fitted estimator.
 * @param data - The data (the rows explained, the background and feature names).
 * @param options - rows or maxRows, and the {@link explainPrediction} options.
 * @returns The attributions with a `featureAttributions` record over the rows.
 */
export function explainDataset(
  fitted: Fitted,
  data: LearnData,
  options: ExplainDatasetOptions = {},
): Explained<DatasetExplanation> {
  const s = setup(fitted, data, options, 'explainDataset')
  const rows =
    options.rows ??
    (() => {
      const m = Math.min(options.maxRows ?? 200, s.x.n)
      if (m === s.x.n) return Array.from({ length: m }, (_, i) => i)
      return [...toFlat(permutation(foldIn(s.key, 7), s.x.n)).slice(0, m)].sort((a, b) => a - b)
    })()
  for (const i of rows)
    if (!(Number.isInteger(i) && i >= 0 && i < s.x.n))
      throw new MdmlError('value', `explainDataset: row ${i} is not in the data`)
  const r = explainRows(fitted, s, rows, options, false)
  const record = featureAttributions({
    method: s.method,
    features: s.features,
    output: s.output.name,
    baseValue: r.baseValue,
    values: r.values,
    data: rows.map((i) => rowOf(s.x, i)),
    predictions: r.predictions,
    indices: rows,
    partners: interactionPartners(
      r.values,
      rows.map((i) => rowOf(s.x, i)),
      r.interactions,
    ),
    ...(r.standardErrors === undefined ? {} : { standardErrors: r.standardErrors }),
    ...(r.interactions === undefined ? {} : { interactions: r.interactions }),
    ...(r.localFit === undefined ? {} : { localFit: r.localFit }),
    ...(r.baseValues === undefined ? {} : { baseValues: r.baseValues }),
  })
  return {
    method: s.method,
    values: r.values,
    baseValue: r.baseValue,
    predictions: r.predictions,
    rows,
    exposition: [record],
  }
}
