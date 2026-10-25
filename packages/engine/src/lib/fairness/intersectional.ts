/**
 * Intersectional and subgroup fairness: rates over every combination of several protected attributes with the worst
 * subgroup and the largest gaps (Buolamwini and Gebru's intersectional audit), rich-subgroup auditing over linear
 * functions of the attributes (GerryFair's auditor, Kearns et al. 2018), and the generalised entropy and Theil indices
 * of individual benefits with their between- and within-group parts (Speicher et al. 2018; AIF360).
 */

import { MdmlError } from '../base/index.ts'
import { type GroupRateName, type GroupRateValues, tableRates } from '../exposition/index.ts'
import { labelValues } from '../metrics/_labels.ts'
import type { MetricInput } from '../metrics/metric.ts'
import { lstsq } from '../numerics/linalg/index.ts'
import { toFlat } from '../tensor/index.ts'

/** The subgroups of several attributes: each row's combination and every combination present. */
export interface IntersectionalGroups {
  /** Each row's subgroup, an index into `subgroups`. */
  readonly index: readonly number[]
  /** Each subgroup's value of every attribute, in sorted order (first attribute slowest). */
  readonly subgroups: readonly (readonly string[])[]
  /** Each subgroup's name, its values joined by ` × `. */
  readonly names: readonly string[]
  readonly counts: readonly number[]
  /** Each attribute's values, sorted. */
  readonly values: readonly (readonly string[])[]
}

const byText = (a: string, b: string): number => {
  const x = Number(a)
  const y = Number(b)
  if (a !== '' && b !== '' && Number.isFinite(x) && Number.isFinite(y)) return x - y
  return a < b ? -1 : a > b ? 1 : 0
}

/**
 * The intersections of several protected attributes: every combination of their values that occurs, and each row's.
 *
 * @param attributes - One or more sensitive features, each `[N]`.
 * @returns The subgroups, their names and sizes, and each row's subgroup.
 * @throws MdmlError `shape` when the attributes differ in length or there are none.
 * @example
 * ```ts
 * import { intersectionalGroups } from 'mdml-engine/fairness'
 * const g = intersectionalGroups([['f', 'm', 'f', 'm'], ['old', 'old', 'young', 'old']])
 * g.names // => ['f × old', 'f × young', 'm × old']
 * g.counts // => [1, 1, 2]
 * g.index // => [0, 2, 1, 2]
 * ```
 */
export function intersectionalGroups(attributes: readonly MetricInput[]): IntersectionalGroups {
  if (attributes.length === 0) throw new MdmlError('shape', 'intersectionalGroups: give at least one attribute')
  const columns = attributes.map((a) => labelValues(a).map(String))
  const n = columns[0]?.length ?? 0
  if (columns.some((c) => c.length !== n))
    throw new MdmlError('shape', 'intersectionalGroups: attributes differ in length')
  const values = columns.map((c) => [...new Set(c)].sort(byText))
  const keys = Array.from({ length: n }, (_, i) => columns.map((c) => c[i] ?? ''))
  const seen = new Map<string, readonly string[]>()
  for (const k of keys) seen.set(JSON.stringify(k), k)
  const order = (a: readonly string[], b: readonly string[]): number => {
    for (let j = 0; j < a.length; j++) {
      const d = byText(a[j] ?? '', b[j] ?? '')
      if (d !== 0) return d
    }
    return 0
  }
  const subgroups = [...seen.values()].sort(order)
  const at = new Map(subgroups.map((s, i) => [JSON.stringify(s), i]))
  const index = keys.map((k) => at.get(JSON.stringify(k)) ?? 0)
  const counts = subgroups.map(() => 0)
  for (const i of index) counts[i] = (counts[i] ?? 0) + 1
  return { index, subgroups, names: subgroups.map((s) => s.join(' × ')), counts, values }
}

/** Options of {@link subgroupRates}. */
export interface SubgroupRatesOptions {
  /** Subgroups with fewer rows are left out of the gaps and the worst subgroup (default 1). */
  readonly minSize?: number
  /** The positive class (default 1). */
  readonly posLabel?: number | string
}

/** The rates of every subgroup, with the worst subgroup and the largest gap in each rate. */
export interface SubgroupRates {
  readonly groups: IntersectionalGroups
  /** tables[s][actual][predicted]. */
  readonly tables: readonly (readonly (readonly number[])[])[]
  readonly rates: readonly GroupRateValues[]
  readonly overall: GroupRateValues
  /** Per rate: the largest minus the smallest over the subgroups large enough. */
  readonly gaps: Readonly<Record<GroupRateName, number>>
  /** Per rate: the subgroup furthest from the overall rate (an index into the subgroups) and its distance. */
  readonly worst: Readonly<Record<GroupRateName, { readonly subgroup: number; readonly gap: number }>>
  readonly minSize: number
}

const RATES: readonly GroupRateName[] = [
  'selectionRate',
  'truePositiveRate',
  'falsePositiveRate',
  'precision',
  'accuracy',
  'baseRate',
]

/**
 * A classifier's rates in every intersection of several protected attributes (fairlearn's MetricFrame over several
 * sensitive features; Buolamwini and Gebru's intersectional audit): each subgroup's 2 by 2 table and rates, the largest
 * gap in each rate between subgroups, and the subgroup furthest from the overall rate. Subgroups smaller than
 * `minSize` keep their rates but are left out of the gaps.
 *
 * @param yTrue - True binary labels `[N]`.
 * @param yPred - Predicted binary labels `[N]`.
 * @param attributes - Sensitive features, each `[N]`.
 * @param options - minSize and posLabel.
 * @returns The subgroups' tables and rates, the gaps and the worst subgroups.
 * @example
 * ```ts
 * import { subgroupRates } from 'mdml-engine/fairness'
 * const r = subgroupRates([1, 0, 1, 0], [1, 0, 0, 0], [['f', 'f', 'm', 'm'], ['a', 'b', 'a', 'b']])
 * r.groups.names // => ['f × a', 'f × b', 'm × a', 'm × b']
 * r.gaps.truePositiveRate // => 1
 * ```
 */
export function subgroupRates(
  yTrue: MetricInput,
  yPred: MetricInput,
  attributes: readonly MetricInput[],
  options: SubgroupRatesOptions = {},
): SubgroupRates {
  const groups = intersectionalGroups(attributes)
  const pos = options.posLabel ?? 1
  const t = labelValues(yTrue).map((v) => (v === pos ? 1 : 0))
  const p = labelValues(yPred).map((v) => (v === pos ? 1 : 0))
  if (t.length !== p.length || t.length !== groups.index.length)
    throw new MdmlError('shape', 'subgroupRates: inputs differ in length')
  const tables = groups.subgroups.map(() => [
    [0, 0],
    [0, 0],
  ])
  for (const [i, s] of groups.index.entries()) {
    const row = tables[s]?.[t[i] ?? 0]
    if (row !== undefined) row[p[i] ?? 0] = (row[p[i] ?? 0] ?? 0) + 1
  }
  const rates = tables.map(tableRates)
  const total = [0, 1].map((a) => [0, 1].map((b) => tables.reduce((sum, tb) => sum + (tb[a]?.[b] ?? 0), 0)))
  const overall = tableRates(total)
  const minSize = options.minSize ?? 1
  const large = groups.counts.flatMap((c, s) => (c >= minSize ? [s] : []))
  const gaps = {} as Record<GroupRateName, number>
  const worst = {} as Record<GroupRateName, { subgroup: number; gap: number }>
  for (const name of RATES) {
    const vs = large.map((s) => rates[s]?.[name] ?? 0)
    gaps[name] = vs.length === 0 ? 0 : Math.max(...vs) - Math.min(...vs)
    let best = { subgroup: large[0] ?? 0, gap: -1 }
    for (const s of large) {
      const gap = Math.abs((rates[s]?.[name] ?? 0) - overall[name])
      if (gap > best.gap) best = { subgroup: s, gap }
    }
    worst[name] = { subgroup: best.subgroup, gap: Math.max(0, best.gap) }
  }
  return { groups, tables, rates, overall, gaps, worst, minSize }
}

/** The result of {@link richSubgroupAudit}. */
export interface RichSubgroupAudit {
  /** Which error rate was audited: false positives (among negatives) or false negatives (among positives). */
  readonly rate: 'falsePositive' | 'falseNegative'
  /** The subgroup found: 1 for each row inside it. */
  readonly members: readonly number[]
  /** The linear function of the protected features defining it: inside where intercept + w·x has `sign`. */
  readonly weights: readonly number[]
  readonly intercept: number
  /** The rate over the audited class (negatives for false positives) in all rows. */
  readonly baseline: number
  /** The rate over the audited class inside the subgroup. */
  readonly subgroupRate: number
  /** The subgroup's share of all rows that are of the audited class. */
  readonly size: number
  /** |subgroupRate - baseline|. */
  readonly disparity: number
  /** size × disparity: GerryFair's gamma-unfairness of the subgroup. */
  readonly weightedDisparity: number
  /** +1 when the subgroup's rate is above the baseline, -1 below. */
  readonly sign: 1 | -1
}

/**
 * Rich-subgroup auditing (GerryFair; Kearns, Neel, Roth and Wu 2018): search the subgroups defined by linear threshold
 * functions of the protected features for the one whose false-positive (or false-negative) rate differs most from the
 * overall rate, weighted by its size. As GerryFair's auditor, it regresses each audited row's excess rate on the
 * protected features (one linear regression per direction) and takes the rows the regression places on the costly
 * side; the larger of the two weighted disparities wins.
 *
 * @param protectedFeatures - The protected features `[N, d]` (numbers; encode categories as indicators).
 * @param yTrue - True binary labels `[N]`.
 * @param yPred - Predictions in [0, 1] `[N]` (hard decisions or probabilities of a positive decision).
 * @param options - The rate audited (default false positives).
 * @returns The subgroup, its defining function, and its rate, size and disparity.
 * @throws MdmlError `shape` when the inputs differ in length; `value` without rows of the audited class.
 * @example
 * ```ts
 * import { richSubgroupAudit } from 'mdml-engine/fairness'
 * // Only the negative with a = 1 is flagged: the rest (a = 0) are flagged less than the average.
 * const r = richSubgroupAudit([[0], [0], [0], [1]], [0, 0, 0, 0], [0, 0, 0, 1])
 * r.members // => [1, 1, 1, 0]
 * r.weightedDisparity // => 0.1875
 * r.sign // => -1
 * ```
 */
export function richSubgroupAudit(
  protectedFeatures: readonly (readonly number[])[],
  yTrue: MetricInput,
  yPred: readonly number[],
  options: { readonly rate?: 'falsePositive' | 'falseNegative' } = {},
): RichSubgroupAudit {
  const rate = options.rate ?? 'falsePositive'
  const fn = rate === 'falseNegative'
  // Auditing false negatives is auditing false positives with the labels and predictions flipped.
  const y = labelValues(yTrue)
    .map((v) => (v === 1 ? 1 : 0))
    .map((v) => (fn ? 1 - v : v))
  const pred = yPred.map((v) => (fn ? 1 - v : v))
  const n = y.length
  if (protectedFeatures.length !== n || pred.length !== n)
    throw new MdmlError('shape', 'richSubgroupAudit: inputs differ in length')
  const subset = y.flatMap((v, i) => (v === 0 ? [i] : []))
  if (subset.length === 0) throw new MdmlError('value', 'richSubgroupAudit: no rows of the audited class')
  const baseline = subset.reduce((s, i) => s + (pred[i] ?? 0), 0) / subset.length
  const d = protectedFeatures[0]?.length ?? 0
  const design = subset.map((i) => [1, ...(protectedFeatures[i] ?? [])])
  // Regress each audited row's excess decision on the features; the subgroup is where the cost of including it is
  // negative (the regression of the cost of labelling 0 is identically 0).
  const fit = (target: number[]): number[] =>
    toFlat(
      lstsq(
        design,
        target.map((v) => [v]),
      ) as never,
    ) as number[]
  const candidate = (direction: 1 | -1) => {
    const beta = fit(subset.map((i) => (-direction / n) * (baseline - (pred[i] ?? 0))))
    const score = (x: readonly number[]): number =>
      (beta[0] ?? 0) + x.reduce((s, v, j) => s + v * (beta[j + 1] ?? 0), 0)
    const members = protectedFeatures.map((x) => (score(x) < 0 ? 1 : 0))
    const inside = subset.filter((i) => members[i] === 1)
    const groupRate = inside.length === 0 ? 0 : inside.reduce((s, i) => s + (pred[i] ?? 0), 0) / inside.length
    const disparity = Math.abs(groupRate - baseline)
    const size = inside.length / n
    return { beta, members, groupRate, disparity, size, weighted: disparity * size }
  }
  const up = candidate(1)
  const down = candidate(-1)
  const best = down.weighted > up.weighted ? down : up
  return {
    rate,
    members: best.members,
    weights: Array.from({ length: d }, (_, j) => best.beta[j + 1] ?? 0),
    intercept: best.beta[0] ?? 0,
    baseline,
    subgroupRate: best.groupRate,
    size: best.size,
    disparity: best.disparity,
    weightedDisparity: best.weighted,
    sign: best.groupRate >= baseline ? 1 : -1,
  }
}

/** Options of the entropy indices. */
export interface EntropyIndexOptions {
  /**
   * The index's order: 2 (default, half the squared coefficient of variation), 1 for Theil, 0 for the mean log
   * deviation (infinite when anyone's benefit is 0, a false negative).
   */
  readonly alpha?: number
  /** The positive class (default 1). */
  readonly posLabel?: number | string
}

/** The generalised entropy of benefits. */
function entropyOf(b: readonly number[], alpha: number): number {
  const n = b.length
  if (n === 0) return 0
  const mu = b.reduce((s, v) => s + v, 0) / n
  if (mu === 0) return 0
  let sum = 0
  for (const v of b) {
    const r = v / mu
    if (alpha === 1) sum += r > 0 ? r * Math.log(r) : 0
    else if (alpha === 0) sum += -Math.log(r)
    else sum += r ** alpha - 1
  }
  if (alpha === 1 || alpha === 0) return sum / n
  return sum / (n * alpha * (alpha - 1))
}

/** Each row's benefit, as Speicher et al.: yhat - y + 1 (2 for a false positive, 0 for a false negative). */
function benefits(yTrue: MetricInput, yPred: MetricInput, pos: number | string): number[] {
  const t = labelValues(yTrue)
  const p = labelValues(yPred)
  if (t.length !== p.length) throw new MdmlError('shape', 'entropy index: inputs differ in length')
  return t.map((v, i) => (p[i] === pos ? 1 : 0) - (v === pos ? 1 : 0) + 1)
}

/**
 * The generalised entropy index of individual benefits (Speicher et al. 2018; AIF360's generalized_entropy_index):
 * each person's benefit is yhat - y + 1, and the index measures how unequally benefits are spread (0 when everyone is
 * treated alike). Alpha 1 gives the Theil index, 2 half the squared coefficient of variation.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param options - alpha and posLabel.
 * @returns The index.
 * @example
 * ```ts
 * import { generalisedEntropyIndex } from 'mdml-engine/fairness'
 * generalisedEntropyIndex([1, 1, 0, 0], [1, 1, 0, 0]) // => 0
 * generalisedEntropyIndex([1, 0, 0, 0], [1, 1, 0, 0]).toFixed(4) // => '0.0600'
 * ```
 */
export function generalisedEntropyIndex(
  yTrue: MetricInput,
  yPred: MetricInput,
  options: EntropyIndexOptions = {},
): number {
  return entropyOf(benefits(yTrue, yPred, options.posLabel ?? 1), options.alpha ?? 2)
}

/**
 * The Theil index of individual benefits (the generalised entropy index of order 1; AIF360's theil_index).
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param options - posLabel.
 * @returns The index.
 * @example
 * ```ts
 * import { theilIndex } from 'mdml-engine/fairness'
 * theilIndex([1, 0, 0, 0], [1, 1, 0, 0]).toFixed(4) // => '0.0541'
 * ```
 */
export function theilIndex(
  yTrue: MetricInput,
  yPred: MetricInput,
  options: { readonly posLabel?: number | string } = {},
): number {
  return generalisedEntropyIndex(yTrue, yPred, { ...options, alpha: 1 })
}

/** The between- and within-group parts of a generalised entropy index. */
export interface EntropyDecomposition {
  readonly total: number
  /** The index with everyone given their group's mean benefit: unfairness between groups. */
  readonly between: number
  /** total - between: unfairness within groups. */
  readonly within: number
  readonly alpha: number
}

/**
 * The generalised entropy index split into between- and within-group parts (Speicher et al. 2018; AIF360's
 * between_group_generalized_entropy_index): the between part replaces each benefit by its group's mean, so it counts
 * only the inequality between groups; the rest is inequality within them.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels `[N]`.
 * @param groups - Sensitive feature `[N]` (or several, by intersection).
 * @param options - alpha and posLabel.
 * @returns The total, between and within parts.
 * @example
 * ```ts
 * import { entropyDecomposition } from 'mdml-engine/fairness'
 * // Group a gets every benefit, group b none: all of the inequality is between groups.
 * const r = entropyDecomposition([0, 0, 1, 1], [1, 1, 0, 0], ['a', 'a', 'b', 'b'])
 * r.within // => 0
 * ```
 */
export function entropyDecomposition(
  yTrue: MetricInput,
  yPred: MetricInput,
  groups: MetricInput | readonly MetricInput[],
  options: EntropyIndexOptions = {},
): EntropyDecomposition {
  const alpha = options.alpha ?? 2
  const b = benefits(yTrue, yPred, options.posLabel ?? 1)
  const isList = Array.isArray(groups) && groups.length > 0 && typeof groups[0] === 'object' && groups[0] !== null
  const g = intersectionalGroups(isList ? (groups as readonly MetricInput[]) : [groups as MetricInput])
  if (g.index.length !== b.length) throw new MdmlError('shape', 'entropyDecomposition: inputs differ in length')
  const sums = g.counts.map(() => 0)
  for (const [i, s] of g.index.entries()) sums[s] = (sums[s] ?? 0) + (b[i] ?? 0)
  const means = sums.map((s, k) => s / (g.counts[k] ?? 1))
  const total = entropyOf(b, alpha)
  const between = entropyOf(
    g.index.map((s) => means[s] ?? 0),
    alpha,
  )
  return { total, between, within: total - between, alpha }
}
