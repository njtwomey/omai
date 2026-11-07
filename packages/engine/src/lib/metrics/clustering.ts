/**
 * External clustering metrics, which compare a clustering with reference labels, scikit-learn style: contingency and
 * pair confusion matrices, the Rand and adjusted Rand indices, Fowlkes-Mallows, mutual information with its normalised
 * and chance-adjusted forms, homogeneity, completeness and V-measure, purity and B-cubed precision, recall and F1.
 */

import { MdmlError, type MetricResult, type MetricSpec, toRows } from '../base/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { arange, fromStorage, type Tensor, toFlat } from '../tensor/index.ts'
import { type Label, labelValues, sortLabels } from './_labels.ts'
import { type Metric, type MetricInput, type MetricOptions, metricResult, sampleWeights } from './metric.ts'
import { specOf } from './_spec.ts'

const clusteringSpec = specOf('clustering', 'labels', 'clustering-evaluation', { range: [0, 1] })
const spec = (name: string, range?: readonly [number, number]): MetricSpec => clusteringSpec(name, 'higher', range)

/**
 * A contingency table: counts of samples by true class (rows) and cluster (columns). Holds a tensor (migration,
 * D-009: `matrix` was `readonly (readonly number[])[]`).
 */
export interface ContingencyMatrix {
  readonly classes: readonly Label[]
  readonly clusters: readonly Label[]
  /** The counts, `[classes, clusters]`. */
  readonly matrix: Tensor<'float64'>
}

/**
 * The contingency table of two labellings (sklearn.metrics.cluster.contingency_matrix): C[i][j] is the (weighted)
 * number of samples of class i in cluster j.
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights (counts per sample).
 * @returns The table with its row and column labels; the matrix is a `[classes, clusters]` tensor (was `number[][]`).
 * @concept clustering-evaluation
 * @example
 * ```ts
 * import { contingencyMatrix } from 'mdml-engine/metrics'
 * contingencyMatrix([0, 0, 1, 1], [1, 1, 1, 0]).matrix // => [[0, 2], [1, 1]]
 * ```
 */
export function contingencyMatrix(
  labelsTrue: MetricInput,
  labelsPred: MetricInput,
  options: MetricOptions = {},
): ContingencyMatrix {
  const t = labelValues(labelsTrue)
  const p = labelValues(labelsPred)
  if (t.length !== p.length) throw new MdmlError('shape', 'contingencyMatrix: labellings differ in length')
  const classes = sortLabels(t)
  const clusters = sortLabels(p)
  const w = sampleWeights('contingencyMatrix', t.length, options.sampleWeight)
  const ci = new Map(classes.map((c, i) => [c, i]))
  const cj = new Map(clusters.map((c, j) => [c, j]))
  const m = classes.map(() => new Array<number>(clusters.length).fill(0))
  for (const [k, a] of t.entries()) {
    const row = m[ci.get(a) ?? 0]
    const j = cj.get(p[k] ?? 0) ?? 0
    if (row !== undefined) row[j] = (row[j] ?? 0) + (w[k] ?? 0)
  }
  return {
    classes,
    clusters,
    matrix: fromStorage(Float64Array.from(m.flat()), [classes.length, clusters.length], 'float64'),
  }
}

/** Row sums, column sums and the total of a table. */
function margins(m: readonly (readonly number[])[]): { rows: number[]; cols: number[]; n: number } {
  const rows = m.map((r) => r.reduce((a, b) => a + b, 0))
  const cols = Array.from({ length: m[0]?.length ?? 0 }, (_, j) => m.reduce((a, r) => a + (r[j] ?? 0), 0))
  return { rows, cols, n: rows.reduce((a, b) => a + b, 0) }
}

const table = (t: MetricInput, p: MetricInput, o: MetricOptions): number[][] => {
  const c = contingencyMatrix(t, p, o)
  return toRows(c.matrix.data, c.classes.length, c.clusters.length)
}

/**
 * The pair confusion matrix (sklearn.metrics.cluster.pair_confusion_matrix): over ordered pairs of distinct samples,
 * [[pairs apart in both, apart in reference but together in clusters], [together in reference but apart in clusters,
 * together in both]]. Returns a tensor (migration, D-009: was `number[][]`).
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns A float64 tensor of shape `[2, 2]`.
 * @example
 * ```ts
 * import { pairConfusionMatrix } from 'mdml-engine/metrics'
 * pairConfusionMatrix([0, 0, 1, 1], [0, 0, 1, 1]) // => [[8, 0], [0, 4]]
 * ```
 */
export function pairConfusionMatrix(
  labelsTrue: MetricInput,
  labelsPred: MetricInput,
  options: MetricOptions = {},
): Tensor<'float64'> {
  const [[tn, predOnly], [trueOnly, both]] = pairCounts(labelsTrue, labelsPred, options)
  return fromStorage(Float64Array.of(tn, predOnly, trueOnly, both), [2, 2], 'float64')
}

/** The counts of {@link pairConfusionMatrix} as rows, the plain-number kernel of the pair-counting indices. */
function pairCounts(
  labelsTrue: MetricInput,
  labelsPred: MetricInput,
  options: MetricOptions,
): [[number, number], [number, number]] {
  const m = table(labelsTrue, labelsPred, options)
  const { rows, cols, n } = margins(m)
  let sumSq = 0
  for (const r of m) for (const v of r) sumSq += v * v
  const sumRows = rows.reduce((a, v) => a + v * v, 0)
  const sumCols = cols.reduce((a, v) => a + v * v, 0)
  const both = sumSq - n
  const predOnly = sumCols - sumSq
  const trueOnly = sumRows - sumSq
  return [
    [n * n - sumCols - sumRows + sumSq, predOnly],
    [trueOnly, both],
  ]
}

const _pairs = (x: number): number => (x * (x - 1)) / 2

/**
 * The Rand index (sklearn.metrics.rand_score): the fraction of sample pairs on which the two labellings agree
 * (together in both or apart in both).
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { randScore } from 'mdml-engine/metrics'
 * randScore([0, 0, 1, 2], [0, 0, 1, 1]).estimate.value // => 0.8333333333333334
 * ```
 */
export const randScore: Metric<MetricOptions, MetricInput> = Object.assign(
  (labelsTrue: MetricInput, labelsPred: MetricInput, options: MetricOptions = {}): MetricResult => {
    const c = pairCounts(labelsTrue, labelsPred, options)
    const agree = (c[0]?.[0] ?? 0) + (c[1]?.[1] ?? 0)
    const total = agree + (c[0]?.[1] ?? 0) + (c[1]?.[0] ?? 0)
    return metricResult('randScore', total === 0 ? 1 : agree / total, labelValues(labelsTrue).length)
  },
  { spec: spec('randScore') },
)

/**
 * The adjusted Rand index (Hubert and Arabie, 1985; sklearn.metrics.adjusted_rand_score): the Rand index corrected
 * for chance, 0 for random labellings and 1 for identical ones.
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The metric result, in [-0.5, 1].
 * @example
 * ```ts
 * import { adjustedRandScore } from 'mdml-engine/metrics'
 * adjustedRandScore([0, 0, 1, 1], [1, 1, 0, 0]).estimate.value // => 1
 * ```
 */
export const adjustedRandScore: Metric<MetricOptions, MetricInput> = Object.assign(
  (labelsTrue: MetricInput, labelsPred: MetricInput, options: MetricOptions = {}): MetricResult => {
    const c = pairCounts(labelsTrue, labelsPred, options)
    const tn = c[0]?.[0] ?? 0
    const fp = c[0]?.[1] ?? 0
    const fn = c[1]?.[0] ?? 0
    const tp = c[1]?.[1] ?? 0
    // sklearn's form: 2 (tp tn - fn fp) / ((tp + fn)(fn + tn) + (tp + fp)(fp + tn)).
    const value = fn === 0 && fp === 0 ? 1 : (2 * (tp * tn - fn * fp)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return metricResult('adjustedRandScore', value, labelValues(labelsTrue).length)
  },
  { spec: spec('adjustedRandScore', [-0.5, 1]) },
)

/**
 * The Fowlkes-Mallows index (sklearn.metrics.fowlkes_mallows_score): the geometric mean of pairwise precision and
 * recall, TP / sqrt((TP + FP)(TP + FN)) over sample pairs.
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { fowlkesMallowsScore } from 'mdml-engine/metrics'
 * fowlkesMallowsScore([0, 0, 1, 1], [0, 0, 1, 1]).estimate.value // => 1
 * ```
 */
export const fowlkesMallowsScore: Metric<MetricOptions, MetricInput> = Object.assign(
  (labelsTrue: MetricInput, labelsPred: MetricInput, options: MetricOptions = {}): MetricResult => {
    const m = table(labelsTrue, labelsPred, options)
    const { rows, cols, n } = margins(m)
    let tk = 0
    for (const r of m) for (const v of r) tk += v * v
    tk -= n
    const pk = cols.reduce((a, v) => a + v * v, 0) - n
    const qk = rows.reduce((a, v) => a + v * v, 0) - n
    const value = tk === 0 ? 0 : Math.sqrt(tk / pk) * Math.sqrt(tk / qk)
    return metricResult('fowlkesMallowsScore', value, labelValues(labelsTrue).length)
  },
  { spec: spec('fowlkesMallowsScore') },
)

/** The entropy (nats) of a vector of counts. */
function entropy(counts: readonly number[]): number {
  const n = counts.reduce((a, b) => a + b, 0)
  let h = 0
  for (const c of counts) if (c > 0) h -= (c / n) * Math.log(c / n)
  return h
}

/** The mutual information (nats) of a contingency table. */
function mutualInformation(m: readonly (readonly number[])[]): number {
  const { rows, cols, n } = margins(m)
  if (rows.length === 1 || cols.length === 1) return 0
  let mi = 0
  for (const [i, r] of m.entries()) {
    for (const [j, v] of r.entries()) {
      if (v > 0) mi += (v / n) * (Math.log(v * n) - Math.log((rows[i] ?? 0) * (cols[j] ?? 0)))
    }
  }
  return Math.max(mi, 0)
}

/**
 * The mutual information (in nats) between two labellings (sklearn.metrics.mutual_info_score).
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { mutualInfoScore } from 'mdml-engine/metrics'
 * Math.abs(mutualInfoScore([0, 0, 1, 1], [1, 1, 0, 0]).estimate.value - Math.LN2) < 1e-15 // => true
 * ```
 */
export const mutualInfoScore: Metric<MetricOptions, MetricInput> = Object.assign(
  (labelsTrue: MetricInput, labelsPred: MetricInput, options: MetricOptions = {}): MetricResult =>
    metricResult(
      'mutualInfoScore',
      mutualInformation(table(labelsTrue, labelsPred, options)),
      labelValues(labelsTrue).length,
    ),
  { spec: spec('mutualInfoScore', [0, Infinity]) },
)

/** How the normalised mutual informations combine the two entropies. */
export type AverageMethod = 'arithmetic' | 'geometric' | 'min' | 'max'

const generalisedMean = (a: number, b: number, method: AverageMethod): number => {
  if (method === 'min') return Math.min(a, b)
  if (method === 'max') return Math.max(a, b)
  if (method === 'geometric') return Math.sqrt(a * b)
  return (a + b) / 2
}

/**
 * Normalised mutual information (sklearn.metrics.normalized_mutual_info_score): MI / mean(H(true), H(pred)) with the
 * arithmetic (default, equal to the V-measure), geometric, min or max mean.
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - averageMethod and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { normalizedMutualInfoScore } from 'mdml-engine/metrics'
 * Number(normalizedMutualInfoScore([0, 0, 1, 1], [1, 1, 0, 0]).estimate.value.toFixed(12)) // => 1
 * ```
 */
export const normalizedMutualInfoScore: Metric<
  MetricOptions & { readonly averageMethod?: AverageMethod },
  MetricInput
> = Object.assign(
  (
    labelsTrue: MetricInput,
    labelsPred: MetricInput,
    options: MetricOptions & { readonly averageMethod?: AverageMethod } = {},
  ): MetricResult => {
    const m = table(labelsTrue, labelsPred, options)
    const n = labelValues(labelsTrue).length
    const { rows, cols } = margins(m)
    // sklearn: a single class and a single cluster (or none) are a perfect match.
    if ((rows.length === 1 && cols.length === 1) || (rows.length === 0 && cols.length === 0))
      return metricResult('normalizedMutualInfoScore', 1, n)
    const mi = mutualInformation(m)
    if (mi === 0) return metricResult('normalizedMutualInfoScore', 0, n)
    const norm = generalisedMean(entropy(rows), entropy(cols), options.averageMethod ?? 'arithmetic')
    return metricResult('normalizedMutualInfoScore', mi / norm, n)
  },
  { spec: spec('normalizedMutualInfoScore') },
)

/** The expected mutual information of a contingency table under the permutation model (Vinh et al., 2010). */
function expectedMutualInformation(rows: readonly number[], cols: readonly number[], n: number): number {
  const lg = toFlat(lgamma(arange(1, n + 2)))
  const gl = (k: number): number => lg[k] ?? 0 // lgamma(k + 1) = log(k!)
  let emi = 0
  for (const a of rows) {
    for (const b of cols) {
      const start = Math.max(1, a - n + b)
      const end = Math.min(a, b)
      for (let nij = start; nij <= end; nij++) {
        const term2 = Math.log(n) + Math.log(nij) - Math.log(a) - Math.log(b)
        const logP =
          gl(a) + gl(b) + gl(n - a) + gl(n - b) - gl(n) - gl(nij) - gl(a - nij) - gl(b - nij) - gl(n - a - b + nij)
        emi += (nij / n) * term2 * Math.exp(logP)
      }
    }
  }
  return emi
}

/**
 * Adjusted mutual information (Vinh, Epps and Bailey, 2010; sklearn.metrics.adjusted_mutual_info_score):
 * (MI - E[MI]) / (mean(H(true), H(pred)) - E[MI]), corrected for chance under random labellings with the same
 * cluster sizes. Weights must be integer counts.
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - averageMethod (default 'arithmetic') and integer sample weights.
 * @returns The metric result.
 * @throws MdmlError with code `value` for non-integer sample weights.
 * @example
 * ```ts
 * import { adjustedMutualInfoScore } from 'mdml-engine/metrics'
 * Number(adjustedMutualInfoScore([0, 0, 1, 1], [1, 1, 0, 0]).estimate.value.toFixed(12)) // => 1
 * ```
 */
export const adjustedMutualInfoScore: Metric<MetricOptions & { readonly averageMethod?: AverageMethod }, MetricInput> =
  Object.assign(
    (
      labelsTrue: MetricInput,
      labelsPred: MetricInput,
      options: MetricOptions & { readonly averageMethod?: AverageMethod } = {},
    ): MetricResult => {
      const m = table(labelsTrue, labelsPred, options)
      const { rows, cols, n } = margins(m)
      if (!Number.isInteger(n) || m.some((r) => r.some((v) => !Number.isInteger(v))))
        throw new MdmlError('value', 'adjustedMutualInfoScore: sample weights must be integer counts')
      const count = labelValues(labelsTrue).length
      if ((rows.length === 1 && cols.length === 1) || (rows.length === 0 && cols.length === 0))
        return metricResult('adjustedMutualInfoScore', 1, count)
      const mi = mutualInformation(m)
      const emi = expectedMutualInformation(rows, cols, n)
      const normaliser = generalisedMean(entropy(rows), entropy(cols), options.averageMethod ?? 'arithmetic')
      let denominator = normaliser - emi
      // sklearn keeps the sign and guards against a zero denominator with machine epsilon.
      const eps = Number.EPSILON
      denominator = denominator < 0 ? Math.min(denominator, -eps) : Math.max(denominator, eps)
      return metricResult('adjustedMutualInfoScore', (mi - emi) / denominator, count)
    },
    { spec: spec('adjustedMutualInfoScore', [-Infinity, 1]) },
  )

/** Homogeneity, completeness and V-measure. */
export interface HomogeneityCompleteness {
  readonly homogeneity: number
  readonly completeness: number
  readonly vMeasure: number
}

/**
 * Homogeneity (every cluster holds one class), completeness (every class lies in one cluster) and their weighted
 * harmonic mean, the V-measure (Rosenberg and Hirschberg, 2007; sklearn.metrics.homogeneity_completeness_v_measure).
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - beta (weight of completeness, default 1) and sample weights.
 * @returns The three scores.
 * @example
 * ```ts
 * import { homogeneityCompletenessVMeasure } from 'mdml-engine/metrics'
 * homogeneityCompletenessVMeasure([0, 0, 1, 1], [0, 1, 2, 3]).homogeneity // => 1
 * ```
 */
export function homogeneityCompletenessVMeasure(
  labelsTrue: MetricInput,
  labelsPred: MetricInput,
  options: MetricOptions & { readonly beta?: number } = {},
): HomogeneityCompleteness {
  const m = table(labelsTrue, labelsPred, options)
  const { rows, cols } = margins(m)
  if (rows.length === 0) return { homogeneity: 1, completeness: 1, vMeasure: 1 }
  const hTrue = entropy(rows)
  const hPred = entropy(cols)
  const mi = mutualInformation(m)
  const homogeneity = hTrue === 0 ? 1 : mi / hTrue
  const completeness = hPred === 0 ? 1 : mi / hPred
  const beta = options.beta ?? 1
  const vMeasure =
    homogeneity + completeness === 0
      ? 0
      : ((1 + beta) * homogeneity * completeness) / (beta * homogeneity + completeness)
  return { homogeneity, completeness, vMeasure }
}

const hcvMetric = (
  name: string,
  pick: (r: HomogeneityCompleteness) => number,
): Metric<MetricOptions & { readonly beta?: number }, MetricInput> =>
  Object.assign(
    (
      labelsTrue: MetricInput,
      labelsPred: MetricInput,
      options: MetricOptions & { readonly beta?: number } = {},
    ): MetricResult =>
      metricResult(
        name,
        pick(homogeneityCompletenessVMeasure(labelsTrue, labelsPred, options)),
        labelValues(labelsTrue).length,
      ),
    { spec: spec(name) },
  )

/**
 * Homogeneity (sklearn.metrics.homogeneity_score): 1 - H(class | cluster) / H(class).
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { homogeneityScore } from 'mdml-engine/metrics'
 * homogeneityScore([0, 0, 1, 1], [0, 0, 0, 0]).estimate.value // => 0
 * ```
 */
export const homogeneityScore: Metric<MetricOptions, MetricInput> = hcvMetric('homogeneityScore', (r) => r.homogeneity)

/**
 * Completeness (sklearn.metrics.completeness_score): 1 - H(cluster | class) / H(cluster).
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { completenessScore } from 'mdml-engine/metrics'
 * completenessScore([0, 0, 1, 1], [0, 0, 0, 0]).estimate.value // => 1
 * ```
 */
export const completenessScore: Metric<MetricOptions, MetricInput> = hcvMetric(
  'completenessScore',
  (r) => r.completeness,
)

/**
 * V-measure (sklearn.metrics.v_measure_score): the weighted harmonic mean of homogeneity and completeness.
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - beta and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { vMeasureScore } from 'mdml-engine/metrics'
 * Number(vMeasureScore([0, 0, 1, 1], [1, 1, 0, 0]).estimate.value.toFixed(12)) // => 1
 * ```
 */
export const vMeasureScore: Metric<MetricOptions & { readonly beta?: number }, MetricInput> = hcvMetric(
  'vMeasureScore',
  (r) => r.vMeasure,
)

/**
 * Purity: the fraction of samples in the majority class of their cluster, sum_j max_i C[i][j] / N.
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { purityScore } from 'mdml-engine/metrics'
 * purityScore([0, 0, 1, 1, 1], [0, 0, 0, 1, 1]).estimate.value // => 0.8
 * ```
 */
export const purityScore: Metric<MetricOptions, MetricInput> = Object.assign(
  (labelsTrue: MetricInput, labelsPred: MetricInput, options: MetricOptions = {}): MetricResult => {
    const m = table(labelsTrue, labelsPred, options)
    const { cols, n } = margins(m)
    let hit = 0
    for (let j = 0; j < cols.length; j++) hit += m.reduce((best, r) => Math.max(best, r[j] ?? 0), 0)
    return metricResult('purityScore', hit / n, labelValues(labelsTrue).length)
  },
  { spec: spec('purityScore') },
)

/** B-cubed precision, recall and F-measure. */
export interface BCubed {
  readonly precision: number
  readonly recall: number
  readonly f1: number
}

/**
 * B-cubed precision, recall and F1 (Bagga and Baldwin, 1998; Amigo et al., 2009): per sample, the fraction of its
 * cluster that shares its class (precision) and of its class that shares its cluster (recall), averaged over samples.
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The three scores.
 * @example
 * ```ts
 * import { bCubed } from 'mdml-engine/metrics'
 * bCubed([0, 0, 1, 1], [0, 0, 1, 1]).f1 // => 1
 * ```
 */
export function bCubed(labelsTrue: MetricInput, labelsPred: MetricInput, options: MetricOptions = {}): BCubed {
  const m = table(labelsTrue, labelsPred, options)
  const { rows, cols, n } = margins(m)
  // Sample in cell (i, j): precision C_ij / col_j, recall C_ij / row_i; C_ij such samples.
  let precision = 0
  let recall = 0
  for (const [i, r] of m.entries()) {
    for (const [j, v] of r.entries()) {
      if (v === 0) continue
      precision += (v * v) / (cols[j] ?? 1)
      recall += (v * v) / (rows[i] ?? 1)
    }
  }
  precision /= n
  recall /= n
  return { precision, recall, f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall) }
}

/**
 * The B-cubed F1 of a clustering against reference labels, as a metric (see {@link bCubed}).
 *
 * @param labelsTrue - Reference labels `[N]`.
 * @param labelsPred - Cluster labels `[N]`.
 * @param options - Sample weights.
 * @returns The metric result, with precision and recall in the breakdown.
 * @example
 * ```ts
 * import { bCubedF1 } from 'mdml-engine/metrics'
 * bCubedF1([0, 0, 1, 1], [0, 0, 0, 0]).breakdown?.recall // => 1
 * ```
 */
export const bCubedF1: Metric<MetricOptions, MetricInput> = Object.assign(
  (labelsTrue: MetricInput, labelsPred: MetricInput, options: MetricOptions = {}): MetricResult => {
    const b = bCubed(labelsTrue, labelsPred, options)
    return metricResult('bCubedF1', b.f1, labelValues(labelsTrue).length, { precision: b.precision, recall: b.recall })
  },
  { spec: spec('bCubedF1') },
)
