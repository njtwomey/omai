/**
 * Evaluating explanations of fitted models: `evaluateExplanations` scores several attribution methods on the same
 * rows by faithfulness (deletion and insertion areas, faithfulness correlation), robustness (sensitivity), complexity
 * (sparseness, entropy) and completeness, with their agreement (`explanationQuality`) and their mean deletion and
 * insertion curves (`curves`); `sanityChecks` randomises a LeNet's layers from the top down, and its labels, and
 * follows how much a gradient map changes (`curves`, and the maps as `valueGrids`).
 */

import { argmaxOf, WhiteboxError } from '../../lib/base/index.ts'
import * as explain from '../../lib/explain/index.ts'
import {
  curveSet,
  type Explained,
  type ExplanationMetric,
  explanationQuality,
  valueGrids,
} from '../../lib/exposition/index.ts'
import { type Fitted, type LearnData, matrixView, targetVector } from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey, permutation, uniform } from '../../lib/random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../lib/tensor/index.ts'
import { type LeNetHyper, lenet } from '../neural/vision/index.ts'
import { modelOutput } from './_models.ts'
import { compose, networkOf } from './_networks.ts'
import { type AttributionMethod, explainDataset } from './attributions.ts'

/** Options of {@link evaluateExplanations}. */
export interface EvaluateExplanationsOptions {
  /** The explainers compared (default tree when the model has trees, kernel, sampling and LIME). */
  readonly explainers?: readonly AttributionMethod[]
  /** Rows explained (default 8, spread over the data). */
  readonly rows?: number
  /** Background rows of the model-agnostic explainers (default 30). */
  readonly background?: number
  /** Perturbed copies per row for the sensitivity (default 3; 0 skips it). */
  readonly sensitivitySamples?: number
  readonly key?: Key
}

function mean(v: readonly number[]): number {
  const f = v.filter(Number.isFinite)
  return f.length === 0 ? Number.NaN : f.reduce((a, b) => a + b, 0) / f.length
}

function spread(v: readonly number[]): number {
  const m = mean(v)
  const f = v.filter(Number.isFinite)
  return f.length < 2 ? 0 : Math.sqrt(f.reduce((a, b) => a + (b - m) ** 2, 0) / (f.length - 1))
}

/**
 * Compare attribution methods on the same rows of a fitted model, by the metrics of Quantus's categories: deletion
 * and insertion areas (features replaced by their means, most relevant first), faithfulness correlation, maximum
 * sensitivity to small perturbations of the row, sparseness and entropy, and the completeness gap against the base
 * value; with the explainers' pairwise rank agreement. Each metric is a mean over the rows with its spread.
 *
 * @param fitted - A fitted model.
 * @param data - The data (rows explained, background).
 * @param options - explainers, rows, background, sensitivitySamples and key.
 * @returns The attributions per explainer with `explanationQuality` and `curves` records.
 */
export function evaluateExplanations(
  fitted: Fitted,
  data: LearnData,
  options: EvaluateExplanationsOptions = {},
): Explained<{
  readonly explainers: readonly AttributionMethod[]
  readonly attributions: readonly (readonly (readonly number[])[])[]
}> {
  const x = matrixView(data.x, 'evaluateExplanations')
  const d = x.d
  const key = options.key ?? makeKey(0)
  const nRows = Math.min(x.n, options.rows ?? 8)
  const rows = Array.from({ length: nRows }, (_, i) => Math.floor((i * x.n) / nRows))
  const output = modelOutput(fitted, data, 'auto')
  const trees = (() => {
    try {
      explainDataset(fitted, data, { method: 'tree', rows: [0] })
      return true
    } catch {
      return false
    }
  })()
  const explainers =
    options.explainers ?? ([...(trees ? ['tree'] : []), 'kernel', 'sampling', 'lime'] as AttributionMethod[])
  const background = options.background ?? 30
  const explainOf = (method: AttributionMethod, which: readonly number[], source: LearnData = data) =>
    explainDataset(fitted, source, { method, rows: [...which], background, nPermutations: 60, nSamples: 300, key })
  const attribute = (method: AttributionMethod, which: readonly number[], source: LearnData = data): number[][] =>
    explainOf(method, which, source).values.map((v) => [...v])
  const explained = explainers.map((m) => explainOf(m, rows))
  const attributions = explained.map((r) => r.values.map((v) => [...v]))
  const xs = rows.map((i) => Array.from(x.data.subarray(i * d, (i + 1) * d)))
  const means = Array.from({ length: d }, (_, j) => {
    let s = 0
    for (let i = 0; i < x.n; i++) s += x.data[i * d + j] ?? 0
    return s / x.n
  })
  const sds = Array.from({ length: d }, (_, j) => {
    let s = 0
    for (let i = 0; i < x.n; i++) s += ((x.data[i * d + j] ?? 0) - (means[j] ?? 0)) ** 2
    return Math.sqrt(s / x.n) || 1
  })
  const predict = output.predict
  const fx = toFlat(predict(fromStorage(Float64Array.from(xs.flat()), [nRows, d], 'float64')))
  const fBase = toFlat(predict(fromStorage(Float64Array.from(means), [1, d], 'float64')))[0] ?? 0
  const deletions = attributions.map((a) => explain.deletionCurve(predict, xs, a, { baseline: means }))
  const insertions = attributions.map((a) =>
    explain.deletionCurve(predict, xs, a, { baseline: means, mode: 'insertion' }),
  )
  const faith = attributions.map((a) =>
    explain.faithfulnessCorrelation(predict, xs, a, { baseline: means, nRuns: 30, key: foldIn(key, 1) }),
  )
  const complexities = attributions.map((a) => a.map((r) => explain.complexity(r)))
  // Each explainer's attributions start from its own base value (E f over its background, a surrogate's intercept).
  const completeness = attributions.map((a, e) => explain.completenessGap(a, fx, [explained[e]?.baseValue ?? fBase]))
  const ns = options.sensitivitySamples ?? 3
  const sensitivities =
    ns <= 0
      ? undefined
      : explainers.map((m, e) => {
          // Explain perturbed copies of the rows (appended to the data, so the background still comes from it).
          const u = toFlat(uniform(foldIn(key, 2), [nRows * ns, d]))
          const z = Array.from({ length: nRows * ns }, (_, t) =>
            Array.from(
              { length: d },
              (_, j) => (xs[Math.floor(t / ns)]?.[j] ?? 0) + (2 * (u[t * d + j] ?? 0) - 1) * 0.05 * (sds[j] ?? 1),
            ),
          )
          const all = {
            ...data,
            x: fromStorage(Float64Array.from([...x.data, ...z.flat()]), [x.n + z.length, d], 'float64'),
          }
          const az = attribute(
            m,
            Array.from({ length: z.length }, (_, t) => x.n + t),
            all,
          )
          const base = attributions[e] ?? []
          return Array.from({ length: nRows }, (_, i) => {
            const e0 = base[i] ?? []
            const n0 = Math.sqrt(e0.reduce((s, v) => s + v * v, 0)) || 1
            return Math.max(
              ...Array.from({ length: ns }, (_, s) => {
                const ez = az[i * ns + s] ?? []
                return Math.sqrt(ez.reduce((acc, v, j) => acc + (v - (e0[j] ?? 0)) ** 2, 0)) / n0
              }),
            )
          })
        })
  const metric = (
    name: string,
    category: ExplanationMetric['category'],
    better: 'higher' | 'lower',
    perExplainer: readonly (readonly number[])[],
  ): ExplanationMetric => ({ name, category, better, values: perExplainer.map(mean), spread: perExplainer.map(spread) })
  const metrics: ExplanationMetric[] = [
    metric(
      'deletion AUC',
      'faithfulness',
      'lower',
      deletions.map((c) => c.auc),
    ),
    metric(
      'insertion AUC',
      'faithfulness',
      'higher',
      insertions.map((c) => c.auc),
    ),
    metric('faithfulness correlation', 'faithfulness', 'higher', faith),
    ...(sensitivities === undefined ? [] : [metric('max sensitivity', 'robustness', 'lower', sensitivities)]),
    metric(
      'sparseness',
      'complexity',
      'higher',
      complexities.map((c) => c.map((v) => v.sparseness)),
    ),
    metric('completeness gap', 'axiomatic', 'lower', completeness),
  ]
  const fractions = deletions[0]?.fractions ?? []
  return {
    explainers,
    attributions,
    exposition: [
      explanationQuality({
        explainers: [...explainers],
        metrics,
        agreement: { metric: 'Spearman rank of |attribution|', matrix: explain.explainerAgreement(attributions) },
        nRows,
      }),
      curveSet({
        x: [...fractions],
        series: [
          ...explainers.map((m, e) => ({ name: `${m} (deletion)`, y: [...(deletions[e]?.mean ?? [])], group: m })),
          ...explainers.map((m, e) => ({ name: `${m} (insertion)`, y: [...(insertions[e]?.mean ?? [])], group: m })),
        ],
        xLabel: 'share of features removed (deletion) or restored (insertion), most relevant first',
        yLabel: output.name,
      }),
    ],
  }
}

/** The gradient maps {@link sanityChecks} randomises. */
export type SanityMethod = 'saliency' | 'integratedGradients' | 'guidedBackprop' | 'gradCam'

/** Options of {@link sanityChecks}. */
export interface SanityCheckOptions {
  readonly method?: SanityMethod
  /** Retrain on shuffled labels too (default true), with these LeNet settings. */
  readonly labels?: boolean
  readonly hyper?: Partial<LeNetHyper>
  readonly key?: Key
}

/**
 * Sanity checks for saliency maps (Adebayo et al. 2018): randomise a fitted LeNet's parametrised layers one at a time
 * from the output down (cascading) and, separately, retrain it on shuffled labels; a map that explains the model
 * should change, measured by the Spearman correlation of its absolute values with the original map. The `curves`
 * record follows the similarity as layers are randomised; `valueGrids` show the maps.
 *
 * @param fitted - A fitted LeNet.
 * @param data - The images (and labels, for the label test).
 * @param row - The image explained.
 * @param options - method, labels, hyper and key.
 * @returns The similarities with `curves` and `valueGrids` records.
 */
export function sanityChecks(
  fitted: Fitted,
  data: LearnData,
  row: number,
  options: SanityCheckOptions = {},
): Explained<{ readonly cascade: readonly number[]; readonly labelSimilarity?: number }> {
  const x = matrixView(data.x, 'sanityChecks')
  if (!(Number.isInteger(row) && row >= 0 && row < x.n)) throw new WhiteboxError('value', `sanityChecks: no row ${row}`)
  const key = options.key ?? makeKey(0)
  const method = options.method ?? 'saliency'
  const image = fromStorage(Float64Array.from(x.data.subarray(row * x.d, (row + 1) * x.d)), [1, x.d], 'float64')
  const base = networkOf(fitted)
  const [, h, w] = base.shape
  const target = argmaxOf(toFlat(compose(base.layers)(image) as Tensor))
  const mapOf = (net: ReturnType<typeof networkOf>): number[] => {
    const model = compose(net.layers)
    if (method === 'saliency') return toFlat(explain.saliency(model, image, { target }))
    if (method === 'integratedGradients')
      return toFlat(explain.integratedGradients(model, image, { target, nSteps: 16 }).attributions)
    if (method === 'guidedBackprop') return toFlat(explain.guidedBackprop(net.layers, image, { target }))
    const head = compose(net.layers.slice(0, net.camLayer + 1))
    const tail = compose(net.layers.slice(net.camLayer + 1))
    return explain.interpolateMap(explain.gradCam(head, tail, image, { target })[0] ?? [], [h, w]).flat()
  }
  const original = mapOf(base)
  // Parametrised layers: two convolutions and two dense layers in a LeNet.
  const stages = [1, 2, 3, 4]
  const maps = stages.map((k) => mapOf(networkOf(fitted, { layers: k, key: foldIn(key, k) })))
  const cascade = maps.map((m) => explain.explanationSimilarity(original, m))
  let labelSimilarity: number | undefined
  let labelMap: number[] | undefined
  if ((options.labels ?? true) && data.y !== undefined) {
    const y = Array.from(targetVector(data.y, 'sanityChecks', x.n))
    const perm = toFlat(permutation(foldIn(key, 99), x.n))
    const shuffled = { x: data.x, y: perm.map((i) => y[i] ?? 0) }
    const p = fitted.params as { readonly channels?: readonly number[]; readonly hidden?: number }
    const refit = lenet({
      epochs: 3,
      snapshots: 2,
      ...(p.channels === undefined ? {} : { channels: [p.channels[0] ?? 6, p.channels[1] ?? 12] as [number, number] }),
      ...(p.hidden === undefined ? {} : { hidden: p.hidden }),
      ...options.hyper,
    }).fit(shuffled, { key: foldIn(key, 100), explain: false })
    labelMap = mapOf(networkOf(refit))
    labelSimilarity = explain.explanationSimilarity(original, labelMap)
  }
  const grid = (v: readonly number[]): number[][] => Array.from({ length: h }, (_, r) => v.slice(r * w, (r + 1) * w))
  const names = [
    'original',
    ...stages.map((k) => `top ${k} randomised`),
    ...(labelMap === undefined ? [] : ['random labels']),
  ]
  return {
    cascade,
    ...(labelSimilarity === undefined ? {} : { labelSimilarity }),
    exposition: [
      curveSet({
        x: [0, ...stages],
        series: [
          { name: 'cascading randomisation', y: [1, ...cascade] },
          ...(labelSimilarity === undefined
            ? []
            : [{ name: 'random labels', y: [1, ...stages.map(() => labelSimilarity)] }]),
        ],
        xLabel: 'layers randomised, from the output down',
        yLabel: `rank correlation with the original ${method} map`,
      }),
      valueGrids({
        grids: [original, ...maps, ...(labelMap === undefined ? [] : [labelMap])].map((m, i) => ({
          name: names[i] ?? '',
          values: grid(m),
        })),
        scale: 'each',
        diverging: method !== 'saliency' && method !== 'gradCam',
        valueName: method,
      }),
    ],
  }
}
