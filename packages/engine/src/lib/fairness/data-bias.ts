/**
 * Bias in data, before or around a model: how each group is represented (against a population when known), base-rate
 * gaps in the labels (AIF360's dataset statistical parity difference and disparate impact), label bias against clean
 * labels, proxies (features that reveal the protected attribute, by dependence and predictability), and bias
 * amplification by a model (Zhao et al. 2017; Wang and Russakovsky 2021).
 */

import { WhiteboxError } from '../base/index.ts'
import { labelValues, sortLabels } from '../metrics/_labels.ts'
import { normalizedMutualInfoScore } from '../metrics/clustering.ts'
import { rocAucScore } from '../metrics/curves.ts'
import type { MetricInput } from '../metrics/metric.ts'
import { lstsq } from '../numerics/linalg/index.ts'
import { toFlat } from '../tensor/index.ts'

/** Each group's count and share of the rows, with the population's shares when given. */
export interface Representation {
  readonly groups: readonly string[]
  readonly counts: readonly number[]
  readonly shares: readonly number[]
  /** The population's share of each group (when given). */
  readonly populationShares?: readonly number[]
  /** share / population share: below 1 for an under-represented group (when the population is given). */
  readonly ratios?: readonly number[]
}

/**
 * How each group is represented in the data: its count and share, and, given the population's shares, the ratio of
 * the two (sampling bias: below 1 for an under-represented group).
 *
 * @param groups - Sensitive feature `[N]`.
 * @param options - The population's share of each group, by name.
 * @returns The groups (sorted), their counts and shares, and the ratios to the population.
 * @throws WhiteboxError `value` when a group has no population share.
 * @example
 * ```ts
 * import { representation } from 'whitebox-engine/fairness'
 * const r = representation(['a', 'a', 'a', 'b'], { populationShares: { a: 0.5, b: 0.5 } })
 * r.shares // => [0.75, 0.25]
 * r.ratios // => [1.5, 0.5]
 * ```
 */
export function representation(
  groups: MetricInput,
  options: { readonly populationShares?: Readonly<Record<string, number>> } = {},
): Representation {
  const g = labelValues(groups)
  const names = sortLabels(g).map(String)
  const counts = names.map((name) => g.filter((v) => String(v) === name).length)
  const shares = counts.map((c) => c / g.length)
  const pop = options.populationShares
  if (pop === undefined) return { groups: names, counts, shares }
  const populationShares = names.map((name) => {
    const v = pop[name]
    if (v === undefined) throw new WhiteboxError('value', `representation: no population share for group ${name}`)
    return v
  })
  return {
    groups: names,
    counts,
    shares,
    populationShares,
    ratios: shares.map((s, i) => s / (populationShares[i] ?? 1)),
  }
}

/** Options of {@link baseRates}. */
export interface BaseRateOptions {
  /** The positive class (default 1). */
  readonly posLabel?: number | string
  /** The reference (privileged) group (default the largest). */
  readonly reference?: number | string
}

/** Each group's base rate and the gaps in the labels. */
export interface BaseRates {
  readonly groups: readonly string[]
  readonly rates: readonly number[]
  readonly overall: number
  readonly reference: number
  /** The base rate furthest from the reference's, minus the reference's (AIF360's statistical parity difference). */
  readonly statisticalParityDifference: number
  /** The smallest base rate over the reference's (AIF360's disparate impact of the labels). */
  readonly disparateImpact: number
}

/**
 * Each group's base rate P(y = 1 | group) in the labels and the gaps between them (AIF360's BinaryLabelDatasetMetric:
 * statistical parity difference and disparate impact of the dataset): bias already in the data, before any model.
 *
 * @param yTrue - Labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - posLabel and the reference group.
 * @returns The base rates and the gaps.
 * @example
 * ```ts
 * import { baseRates } from 'whitebox-engine/fairness'
 * const r = baseRates([1, 1, 0, 0, 1, 0], ['a', 'a', 'a', 'a', 'b', 'b'], { reference: 'a' })
 * r.rates // => [0.5, 0.5]
 * r.statisticalParityDifference // => 0
 * ```
 */
export function baseRates(yTrue: MetricInput, groups: MetricInput, options: BaseRateOptions = {}): BaseRates {
  const pos = options.posLabel ?? 1
  const t = labelValues(yTrue).map((v) => (v === pos ? 1 : 0))
  const g = labelValues(groups).map(String)
  if (t.length !== g.length) throw new WhiteboxError('shape', 'baseRates: inputs differ in length')
  const names = sortLabels(labelValues(groups)).map(String)
  const counts = names.map(() => 0)
  const positives = names.map(() => 0)
  for (const [i, v] of g.entries()) {
    const k = names.indexOf(v)
    counts[k] = (counts[k] ?? 0) + 1
    positives[k] = (positives[k] ?? 0) + (t[i] ?? 0)
  }
  const rates = counts.map((c, k) => (c === 0 ? 0 : (positives[k] ?? 0) / c))
  let reference = 0
  if (options.reference !== undefined) {
    reference = names.indexOf(String(options.reference))
    if (reference < 0) throw new WhiteboxError('value', `baseRates: no group ${String(options.reference)}`)
  } else for (const [k, c] of counts.entries()) if (c > (counts[reference] ?? 0)) reference = k
  const ref = rates[reference] ?? 0
  let spd = 0
  for (const [k, r] of rates.entries()) if (k !== reference && Math.abs(r - ref) > Math.abs(spd)) spd = r - ref
  const others = rates.filter((_, k) => k !== reference)
  const di = ref === 0 || others.length === 0 ? Number.NaN : Math.min(...others) / ref
  return {
    groups: names,
    rates,
    overall: t.reduce((s: number, v) => s + v, 0) / Math.max(1, t.length),
    reference,
    statisticalParityDifference: spd,
    disparateImpact: di,
  }
}

/** Each group's label flips against clean labels. */
export interface LabelBias {
  readonly groups: readonly string[]
  /** P(recorded negative | clean positive, group). */
  readonly positiveToNegative: readonly number[]
  /** P(recorded positive | clean negative, group). */
  readonly negativeToPositive: readonly number[]
}

/**
 * Label bias: how often each group's recorded labels disagree with clean (true, or audited) labels, in each direction —
 * a group whose good outcomes were recorded as bad more often than another's carries historical bias in its labels.
 *
 * @param recorded - The labels as recorded `[N]` (1 positive).
 * @param clean - The clean labels `[N]` (1 positive).
 * @param groups - Sensitive feature `[N]`.
 * @returns Each group's flip rates (0 where the group has no clean labels of that class).
 * @example
 * ```ts
 * import { labelBias } from 'whitebox-engine/fairness'
 * labelBias([1, 0, 1, 1], [1, 1, 1, 1], ['a', 'a', 'b', 'b']).positiveToNegative // => [0.5, 0]
 * ```
 */
export function labelBias(recorded: MetricInput, clean: MetricInput, groups: MetricInput): LabelBias {
  const r = labelValues(recorded).map((v) => (v === 1 ? 1 : 0))
  const c = labelValues(clean).map((v) => (v === 1 ? 1 : 0))
  const g = labelValues(groups).map(String)
  if (r.length !== c.length || r.length !== g.length)
    throw new WhiteboxError('shape', 'labelBias: inputs differ in length')
  const names = sortLabels(labelValues(groups)).map(String)
  const rate = (name: string, cls: 0 | 1): number => {
    let n = 0
    let flipped = 0
    for (const [i, v] of g.entries())
      if (v === name && c[i] === cls) {
        n++
        if (r[i] !== cls) flipped++
      }
    return n === 0 ? 0 : flipped / n
  }
  return {
    groups: names,
    positiveToNegative: names.map((n) => rate(n, 1)),
    negativeToPositive: names.map((n) => rate(n, 0)),
  }
}

/** Options of {@link proxyScores}. */
export interface ProxyOptions {
  /** The features' names (default `x0`, `x1`, ...). */
  readonly featureNames?: readonly string[]
  /** Quantile bins for the mutual information (default 10). */
  readonly bins?: number
  /** Folds for the combined model's held-out AUC (default 5; 0 to skip it). */
  readonly folds?: number
  /** The group whose membership is predicted (default the smallest group). */
  readonly target?: number | string
}

/** How much each feature reveals the protected attribute. */
export interface ProxyScores {
  readonly features: readonly string[]
  /** Each feature's AUC for telling the target group from the rest (at least 0.5: direction does not matter). */
  readonly auc: readonly number[]
  /** Each feature's normalised mutual information with the group, after quantile binning. */
  readonly dependence: readonly number[]
  /** The held-out AUC of a linear model of every feature predicting the target group. */
  readonly combinedAuc?: number
  readonly target: string
}

/** A column's quantile bin per row (numpy's 'lower' quantiles as edges; its distinct values when there are few). */
function binned(column: readonly number[], bins: number): number[] {
  const distinct = [...new Set(column)].sort((a, b) => a - b)
  if (distinct.length <= bins) return column.map((v) => distinct.indexOf(v))
  const sorted = [...column].sort((a, b) => a - b)
  const edges = Array.from(
    { length: bins - 1 },
    (_, k) => sorted[Math.floor(((k + 1) * (sorted.length - 1)) / bins)] ?? 0,
  )
  return column.map((v) => edges.filter((e) => v >= e).length)
}

/**
 * Proxy detection: how much each feature reveals the protected attribute (a model can learn the group through it even
 * when the group itself is withheld — fairness through unawareness fails). For each feature, its AUC for telling the
 * target group from the rest and its normalised mutual information with the group; and the held-out AUC of a linear
 * model of all the features together.
 *
 * @param x - Features `[N, d]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - Feature names, bins, folds and the target group.
 * @returns Each feature's scores and the combined AUC.
 * @example
 * ```ts
 * import { proxyScores } from 'whitebox-engine/fairness'
 * const x = [[0, 5], [1, 3], [2, 4], [10, 3], [11, 5], [12, 4]]
 * const r = proxyScores(x, ['a', 'a', 'a', 'b', 'b', 'b'], { featureNames: ['postcode', 'noise'], folds: 0 })
 * r.auc // => [1, 0.5]
 * ```
 */
export function proxyScores(
  x: readonly (readonly number[])[],
  groups: MetricInput,
  options: ProxyOptions = {},
): ProxyScores {
  const g = labelValues(groups).map(String)
  if (x.length !== g.length) throw new WhiteboxError('shape', 'proxyScores: inputs differ in length')
  const names = sortLabels(labelValues(groups)).map(String)
  if (names.length < 2) throw new WhiteboxError('value', 'proxyScores: needs at least two groups')
  const counts = names.map((n) => g.filter((v) => v === n).length)
  let target = names[counts.indexOf(Math.min(...counts))] ?? ''
  if (options.target !== undefined) target = String(options.target)
  const y = g.map((v) => (v === target ? 1 : 0))
  const d = x[0]?.length ?? 0
  const features = options.featureNames ?? Array.from({ length: d }, (_, j) => `x${j}`)
  const auc: number[] = []
  const dependence: number[] = []
  for (let j = 0; j < d; j++) {
    const col = x.map((row) => row[j] ?? 0)
    const a = rocAucScore(y, col).estimate.value
    auc.push(Math.max(a, 1 - a))
    dependence.push(normalizedMutualInfoScore(g, binned(col, options.bins ?? 10)).estimate.value)
  }
  const folds = options.folds ?? 5
  let combinedAuc: number | undefined
  if (folds >= 2 && x.length >= 2 * folds) {
    // Standardise, then fit least squares of the indicator on each training fold and score the held-out fold.
    const mean = Array.from({ length: d }, (_, j) => x.reduce((s, r) => s + (r[j] ?? 0), 0) / x.length)
    const sd = Array.from({ length: d }, (_, j) => {
      const v = x.reduce((s, r) => s + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / x.length
      return v > 0 ? Math.sqrt(v) : 1
    })
    const z = x.map((r) => [1, ...r.map((v, j) => (v - (mean[j] ?? 0)) / (sd[j] ?? 1))])
    const scores = new Array<number>(x.length).fill(0)
    for (let f = 0; f < folds; f++) {
      const train = z.flatMap((_, i) => (i % folds === f ? [] : [i]))
      const beta = toFlat(
        lstsq(
          train.map((i) => z[i] ?? []),
          train.map((i) => [y[i] ?? 0]),
        ) as never,
      ) as number[]
      for (let i = f; i < x.length; i += folds) scores[i] = (z[i] ?? []).reduce((s, v, j) => s + v * (beta[j] ?? 0), 0)
    }
    combinedAuc = rocAucScore(y, scores).estimate.value
  }
  return { features, auc, dependence, target, ...(combinedAuc === undefined ? {} : { combinedAuc }) }
}

/** The result of {@link biasAmplification}. */
export interface BiasAmplification {
  /** BiasAmp: positive when the model exaggerates the label's association with the groups. */
  readonly value: number
  readonly groups: readonly string[]
  /** Each group's positive rate in the labels. */
  readonly baseRates: readonly number[]
  /** Each group's positive rate in the predictions. */
  readonly selectionRates: readonly number[]
}

/**
 * Bias amplification (Wang and Russakovsky 2021, directional BiasAmp from attribute to task; after Zhao et al. 2017):
 * for each group and outcome that are positively associated in the labels, how much more often the model predicts
 * the outcome for the group than the labels have it (and the reverse for negatively associated pairs), averaged. Zero
 * when the model reproduces the labels' rates; positive when it amplifies their association with the groups.
 *
 * @param yTrue - Labels `[N]` (1 positive).
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @returns The amplification, with each group's base and selection rates.
 * @example
 * ```ts
 * import { biasAmplification } from 'whitebox-engine/fairness'
 * // The labels favour a (0.75 against 0.25); the model favours it more (1 against 0).
 * const y = [1, 1, 1, 0, 1, 0, 0, 0]
 * const p = [1, 1, 1, 1, 0, 0, 0, 0]
 * biasAmplification(y, p, ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']).value // => 0.25
 * ```
 */
export function biasAmplification(yTrue: MetricInput, yPred: MetricInput, groups: MetricInput): BiasAmplification {
  const t = labelValues(yTrue).map((v) => (v === 1 ? 1 : 0))
  const p = labelValues(yPred).map((v) => (v === 1 ? 1 : 0))
  const g = labelValues(groups).map(String)
  if (t.length !== p.length || t.length !== g.length)
    throw new WhiteboxError('shape', 'biasAmplification: inputs differ in length')
  const names = sortLabels(labelValues(groups)).map(String)
  const n = t.length
  const pT = t.reduce((s: number, v) => s + v, 0) / n
  const base: number[] = []
  const selection: number[] = []
  let sum = 0
  for (const name of names) {
    const rows = g.flatMap((v, i) => (v === name ? [i] : []))
    const pG = rows.length / n
    const b = rows.reduce((s, i) => s + (t[i] ?? 0), 0) / rows.length
    const sel = rows.reduce((s, i) => s + (p[i] ?? 0), 0) / rows.length
    base.push(b)
    selection.push(sel)
    // The outcome 1 and the outcome 0, each against the group.
    for (const [joint, marginal, delta] of [
      [pG * b, pG * pT, sel - b],
      [pG * (1 - b), pG * (1 - pT), b - sel],
    ] as const)
      sum += joint > marginal ? delta : -delta
  }
  return { value: sum / (2 * names.length), groups: names, baseRates: base, selectionRates: selection }
}
