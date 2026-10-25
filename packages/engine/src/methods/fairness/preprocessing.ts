/**
 * Fairness before training on a dataset with a protected attribute: reweighing, which up-weights the outcomes that are
 * rare in a group so label and group become independent, a classifier fitted with and without those weights and audited
 * both ways, and disparate-impact removal, which repairs the other features towards a distribution shared by the
 * groups.
 */

import { coefficientTable, curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import {
  type ReweighingCell as CoreCell,
  type FairRepresentation,
  type FairRepresentationOptions,
  fairRepresentationMemberships,
  type LabelMapping,
  type LabelMappingOptions,
  learnFairRepresentation,
  massageLabels,
  optimisedLabelMapping,
  preferentialSample,
  repairFeatures,
  reweighingWeights,
} from '../../lib/fairness/index.ts'
import { type Estimator, type Fitted, featureNamesOf, matrixView } from '../../lib/learn/index.ts'
import { foldIn, type Key, key as makeKey, uniform } from '../../lib/random/index.ts'
import { fromStorage, toFlat } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { baselineScores, type MitigationOptions, mitigationData, reportOf } from './_mitigation.ts'
import { fairnessAudit } from './audit.ts'
import {
  binaryLabels,
  type FairnessData,
  positiveScores,
  type SensitiveSpec,
  sensitiveGroups,
  withoutColumns,
} from './groups.ts'

/** One cell of {@link reweighing}: as the core's, with the group named. */
export type ReweighingCell = Omit<CoreCell, 'group'> & { readonly group: string }

/**
 * Reweighing (`reweighingWeights` of `mdml-engine/fairness`) on a dataset with a protected attribute: groups
 * whose good outcomes are rare get those outcomes weighted up. Exposition: a `coefficientTable` of the weights per group and label.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute (default the data's first).
 * @returns A weight per row and the cells, with the record.
 */
export function reweighing(
  data: FairnessData,
  options: { readonly sensitive?: SensitiveSpec } = {},
): Explained<{ readonly weights: readonly number[]; readonly cells: readonly ReweighingCell[] }> {
  const groups = sensitiveGroups(data, options.sensitive)
  const labels = binaryLabels(data, 'reweighing')
  const core = reweighingWeights(labels, groups.index)
  const cells = core.cells.map((c) => ({ ...c, group: groups.names[c.group] ?? String(c.group) }))
  const weights = core.weights
  const classes = data.classes ?? ['0', '1']
  return {
    weights,
    cells,
    exposition: [
      coefficientTable({
        names: cells.map((c) => `${c.group}, ${classes[c.label] ?? c.label}`),
        estimates: cells.map((c) => c.weight),
      }),
    ],
  }
}

/** Options of {@link reweighedFit}. */
export interface ReweighedFitOptions {
  readonly sensitive?: SensitiveSpec
  /** The classifier (default logistic regression); it must accept sample weights. */
  readonly estimator?: Estimator
  /** Fit without the sensitive column (default false). */
  readonly dropSensitive?: boolean
  readonly key?: Key
}

/**
 * A classifier fitted with and without reweighing, audited on the same rows: the weights' table and the group reports
 * before and after.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, estimator, whether to drop the attribute, and key.
 * @returns Both fitted models and the weights, with the records.
 */
export function reweighedFit(
  data: FairnessData,
  options: ReweighedFitOptions = {},
): Explained<{ readonly plain: Fitted; readonly reweighed: Fitted; readonly weights: readonly number[] }> {
  const groups = sensitiveGroups(data, options.sensitive)
  const r = reweighing(data, { sensitive: groups.index.map((g) => groups.names[g] ?? '') })
  const estimator = options.estimator ?? logisticRegression()
  const drop = options.dropSensitive === true && groups.column !== undefined
  const train = drop ? withoutColumns(data, [groups.column as number]) : data
  const key = options.key ?? makeKey(0)
  const plain = estimator.fit(train, { key, explain: false })
  const reweighed = estimator.fit({ ...train, weights: r.weights }, { key, explain: false })
  const sensitive = options.sensitive ?? groups.attribute
  const audit = (m: Fitted, label: string): Exposition[] =>
    fairnessAudit(m, data, { sensitive, dropSensitive: drop, label }).exposition as Exposition[]
  return {
    plain,
    reweighed,
    weights: r.weights,
    exposition: [...r.exposition, ...audit(plain, 'unweighted'), ...audit(reweighed, 'reweighed')],
  }
}

/** Options of {@link disparateImpactRemover}. */
export interface DisparateImpactRemoverOptions {
  readonly sensitive?: SensitiveSpec
  /** How far to move each value towards the groups' median distribution, from 0 (none) to 1 (all the way). */
  readonly repairLevel?: number
}

/**
 * Disparate-impact removal (`repairFeatures` of `mdml-engine/fairness`, Feldman et al.) on a dataset with a
 * protected attribute: every feature but the protected one moves towards a distribution shared by the groups, keeping
 * ranks within groups, by `repairLevel`. A classifier fitted on fully repaired features cannot tell the groups apart from any single feature. Exposition: a
 * `curves` record of each group's distribution (empirical CDF) of the most group-dependent feature before and after.
 *
 * @param data - Data holding the protected attribute as a column, or with labels given in `sensitive`.
 * @param options - The attribute and the repair level (default 1).
 * @returns The repaired data, with the record.
 * @throws MdmlError `value` for a repair level outside [0, 1].
 */
export function disparateImpactRemover(
  data: FairnessData,
  options: DisparateImpactRemoverOptions = {},
): Explained<{ readonly data: FairnessData }> {
  const level = options.repairLevel ?? 1
  const groups = sensitiveGroups(data, options.sensitive)
  const m = matrixView(data.x, 'disparateImpactRemover')
  const names = featureNamesOf(data, m.d)
  const k = groups.names.length
  const members = groups.names.map((): number[] => [])
  for (const [i, g] of groups.index.entries()) members[g]?.push(i)
  const out = repairFeatures(m.data, m.d, groups.index, {
    level,
    ...(groups.column === undefined ? {} : { skip: [groups.column] }),
  })
  // The feature whose groups' medians differ most, to draw.
  let worst = -1
  let worstGap = -1
  for (let j = 0; j < m.d; j++) {
    if (j === groups.column) continue
    const medians = members.map((rows) => {
      const v = rows.map((i) => m.data[i * m.d + j] ?? 0).sort((a, b) => a - b)
      return v[Math.floor(v.length / 2)] ?? 0
    })
    const gap = Math.max(...medians) - Math.min(...medians)
    if (gap > worstGap) {
      worstGap = gap
      worst = j
    }
  }
  const repaired: FairnessData = { ...data, x: fromStorage(out, [m.n, m.d], 'float64') }
  const exposition: Exposition[] = []
  if (worst >= 0 && k > 0) {
    const lo = Math.min(...Array.from({ length: m.n }, (_, i) => m.data[i * m.d + worst] ?? 0))
    const hi = Math.max(...Array.from({ length: m.n }, (_, i) => m.data[i * m.d + worst] ?? 0))
    const grid = Array.from({ length: 101 }, (_, t) => lo + ((hi - lo) * t) / 100)
    const ecdf = (values: readonly number[]): number[] =>
      grid.map((x) => values.filter((v) => v <= x).length / Math.max(1, values.length))
    exposition.push(
      curveSet({
        x: grid,
        xLabel: names[worst] ?? `x${worst}`,
        yLabel: 'cumulative share of the group',
        series: groups.names.flatMap((name, g) => {
          const rows = members[g] ?? []
          return [
            { name: `${name}, before`, y: ecdf(rows.map((i) => m.data[i * m.d + worst] ?? 0)), group: name },
            { name: `${name}, repaired`, y: ecdf(rows.map((i) => out[i * m.d + worst] ?? 0)), group: name },
          ]
        }),
      }),
    )
  }
  return { data: repaired, exposition }
}

/** The result of the label and sample mitigations: the classifier fitted on the mitigated data, and its scores. */
export interface MitigatedFit {
  readonly fitted: Fitted
  /** The mitigated classifier's positive-class scores on the original rows. */
  readonly scores: readonly number[]
}

/**
 * Massaging (`massageLabels` of `mdml-engine/fairness`, Kamiran and Calders) on a dataset: a logistic regression
 * ranks the rows, the fewest labels that equalise the groups' label rates are swapped (the unprivileged negatives
 * ranked highest promoted, the privileged positives ranked lowest demoted), and a classifier is fitted to the massaged
 * labels. Exposition: the group reports of the classifier fitted to the original labels and to the massaged ones.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, privileged group and key.
 * @returns The fitted classifier, its scores and the rows relabelled, with the records.
 */
export function massaging(
  data: FairnessData,
  options: MitigationOptions = {},
): Explained<MitigatedFit & { readonly promoted: readonly number[]; readonly demoted: readonly number[] }> {
  const d = mitigationData(data, options, 'massaging')
  const before = baselineScores(d)
  const m = massageLabels(d.labels, before, d.groups.index, { privileged: d.privileged })
  const fitted = logisticRegression().fit({ ...d.unaware, y: m.labels }, { key: d.key, explain: false })
  const scores = positiveScores(fitted, d.unaware.x).scores
  return {
    fitted,
    scores,
    promoted: m.promoted,
    demoted: m.demoted,
    exposition: [reportOf(d, 'original labels', before), reportOf(d, `massaged (${m.promoted.length} swaps)`, scores)],
  }
}

/**
 * Preferential sampling (`preferentialSample` of `mdml-engine/fairness`, Kamiran and Calders) on a dataset: every
 * group-and-label cell is resampled to its size under independence, removing or duplicating the rows a logistic
 * regression ranks nearest the boundary, and a classifier is fitted to the resample. Exposition: the group reports
 * before and after, and a `coefficientTable` of each cell's size before and after.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, privileged group and key.
 * @returns The fitted classifier, its scores and the resampled rows, with the records.
 */
export function preferentialSampling(
  data: FairnessData,
  options: MitigationOptions = {},
): Explained<MitigatedFit & { readonly rows: readonly number[] }> {
  const d = mitigationData(data, options, 'preferentialSampling')
  const before = baselineScores(d)
  const s = preferentialSample(d.labels, before, d.groups.index)
  const x = s.rows.map((i) => d.rows[i] ?? [])
  const fitted = logisticRegression().fit({ x, y: s.rows.map((i) => d.labels[i] ?? 0) }, { key: d.key, explain: false })
  const scores = positiveScores(fitted, d.unaware.x).scores
  const classes = d.classes ?? ['0', '1']
  return {
    fitted,
    scores,
    rows: s.rows,
    exposition: [
      reportOf(d, 'original sample', before),
      reportOf(d, 'preferential sample', scores),
      coefficientTable({
        names: s.cells.map((c) => `${d.groups.names[c.group] ?? c.group}, ${classes[c.label] ?? c.label}`),
        estimates: s.cells.map((c) => c.expected - c.count),
      }),
    ],
  }
}

/** Options of {@link optimisedPreprocessing}. */
export interface OptimisedPreprocessingOptions extends MitigationOptions, LabelMappingOptions {}

/**
 * Optimised pre-processing of the labels (`optimisedLabelMapping` of `mdml-engine/fairness`, after Calmon et al.)
 * on a dataset: the randomised relabelling that changes the fewest labels while bringing every group's positive rate
 * within epsilon of a shared rate, labels drawn from it with the key, and a classifier fitted to them. Exposition: the
 * group reports before and after.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, epsilon, flip cap and key.
 * @returns The mapping, the fitted classifier and its scores, with the records.
 */
export function optimisedPreprocessing(
  data: FairnessData,
  options: OptimisedPreprocessingOptions = {},
): Explained<MitigatedFit & { readonly mapping: LabelMapping }> {
  const d = mitigationData(data, options, 'optimisedPreprocessing')
  const mapping = optimisedLabelMapping(d.labels, d.groups.index, options)
  const u = toFlat(uniform(foldIn(d.key, 7), [d.labels.length]))
  const relabelled = d.labels.map((y, i) => {
    const p = mapping.positive[d.groups.index[i] ?? 0]?.[y === 1 ? 1 : 0] ?? y
    return (u[i] ?? 1) < p ? 1 : 0
  })
  const fitted = logisticRegression().fit({ ...d.unaware, y: relabelled }, { key: d.key, explain: false })
  const scores = positiveScores(fitted, d.unaware.x).scores
  return {
    fitted,
    scores,
    mapping,
    exposition: [reportOf(d, 'original labels', baselineScores(d)), reportOf(d, 'optimised labels', scores)],
  }
}

/** Options of {@link fairRepresentation}. */
export interface FairRepresentationMethodOptions extends MitigationOptions, Omit<FairRepresentationOptions, 'key'> {}

/**
 * Learning fair representations (`learnFairRepresentation` of `mdml-engine/fairness`, Zemel et al.) on a dataset:
 * the features (standardised) are mapped to memberships of prototypes that reconstruct them, predict the label and are
 * shared alike by the groups; the prototypes' labels decide. Exposition: the group reports of a logistic regression on
 * the features and of the representation's predictions.
 *
 * @param data - Labelled data holding the protected attribute.
 * @param options - The attribute, prototypes, loss weights and key.
 * @returns The representation and its scores, with the records.
 */
export function fairRepresentation(
  data: FairnessData,
  options: FairRepresentationMethodOptions = {},
): Explained<{ readonly representation: FairRepresentation; readonly scores: readonly number[] }> {
  const d = mitigationData(data, options, 'fairRepresentation')
  const dims = d.rows[0]?.length ?? 0
  const mean = Array.from({ length: dims }, (_, j) => d.rows.reduce((s, r) => s + (r[j] ?? 0), 0) / d.rows.length)
  const sd = Array.from(
    { length: dims },
    (_, j) => Math.sqrt(d.rows.reduce((s, r) => s + ((r[j] ?? 0) - (mean[j] ?? 0)) ** 2, 0) / d.rows.length) || 1,
  )
  const z = d.rows.map((r) => r.map((v, j) => (v - (mean[j] ?? 0)) / (sd[j] ?? 1)))
  const representation = learnFairRepresentation(z, d.labels, d.binary, { ...options, key: d.key })
  const memberships = fairRepresentationMemberships(representation, z)
  const scores = memberships.map((m) => m.reduce((s, v, k) => s + v * (representation.labels[k] ?? 0), 0))
  return {
    representation,
    scores,
    exposition: [reportOf(d, 'logistic regression', baselineScores(d)), reportOf(d, 'fair representation', scores)],
  }
}
