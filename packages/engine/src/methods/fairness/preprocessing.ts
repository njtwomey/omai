import { WhiteboxError } from '../../lib/base/index.ts'
import { coefficientTable, curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import { type Estimator, type Fitted, featureNamesOf, matrixView } from '../../lib/learn/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { fromStorage } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { fairnessAudit } from './audit.ts'
import { binaryLabels, type FairnessData, type SensitiveSpec, sensitiveGroups, withoutColumns } from './groups.ts'

/** One cell of {@link reweighing}: a group and a label, how many rows it has and expects, and its weight. */
export interface ReweighingCell {
  readonly group: string
  readonly label: number
  readonly count: number
  /** The count were group and label independent: n P(group) P(label). */
  readonly expected: number
  /** expected / count. */
  readonly weight: number
}

/**
 * Reweighing (Kamiran and Calders, 2012; AIF360's `Reweighing`): weight each row by `P(group) P(label) / P(group,
 * label)`, so that in the weighted data the label is independent of the group. Groups whose good outcomes are rare get
 * those outcomes weighted up. Exposition: a `coefficientTable` of the weights per group and label.
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
  const n = labels.length
  const k = groups.names.length
  const joint = Array.from({ length: k }, () => [0, 0])
  for (const [i, g] of groups.index.entries()) {
    const row = joint[g] as number[]
    const y = labels[i] ?? 0
    row[y] = (row[y] ?? 0) + 1
  }
  const byGroup = joint.map((r) => (r[0] ?? 0) + (r[1] ?? 0))
  const byLabel = [0, 1].map((y) => joint.reduce((s, r) => s + (r[y] ?? 0), 0))
  const cells: ReweighingCell[] = joint.flatMap((r, g) =>
    [0, 1].map((y) => {
      const count = r[y] ?? 0
      const expected = ((byGroup[g] ?? 0) * (byLabel[y] ?? 0)) / n
      return {
        group: groups.names[g] ?? String(g),
        label: y,
        count,
        expected,
        weight: count === 0 ? 0 : expected / count,
      }
    }),
  )
  const weights = groups.index.map((g, i) => cells[2 * g + (labels[i] ?? 0)]?.weight ?? 1)
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

/** The empirical quantile (type 7, numpy's default) of sorted values. */
function quantileSorted(sorted: readonly number[], q: number): number {
  const h = (sorted.length - 1) * q
  const lo = Math.floor(h)
  const hi = Math.min(sorted.length - 1, lo + 1)
  return (sorted[lo] ?? 0) + (h - lo) * ((sorted[hi] ?? 0) - (sorted[lo] ?? 0))
}

/**
 * Disparate-impact removal (Feldman et al., 2015; AIF360's `DisparateImpactRemover`): repair each feature other than
 * the protected one so its distribution is the same in every group, keeping each row's rank within its group: a row
 * at quantile u of its group moves towards the median, across groups, of the groups' quantiles at u, by `repairLevel`.
 * A classifier fitted on fully repaired features cannot tell the groups apart from any single feature. Exposition: a
 * `curves` record of each group's distribution (empirical CDF) of the most group-dependent feature before and after.
 *
 * @param data - Data holding the protected attribute as a column, or with labels given in `sensitive`.
 * @param options - The attribute and the repair level (default 1).
 * @returns The repaired data, with the record.
 * @throws WhiteboxError `value` for a repair level outside [0, 1].
 */
export function disparateImpactRemover(
  data: FairnessData,
  options: DisparateImpactRemoverOptions = {},
): Explained<{ readonly data: FairnessData }> {
  const level = options.repairLevel ?? 1
  if (!(level >= 0 && level <= 1)) throw new WhiteboxError('value', 'disparateImpactRemover: repairLevel is in [0, 1]')
  const groups = sensitiveGroups(data, options.sensitive)
  const m = matrixView(data.x, 'disparateImpactRemover')
  const names = featureNamesOf(data, m.d)
  const k = groups.names.length
  const members = groups.names.map((): number[] => [])
  for (const [i, g] of groups.index.entries()) members[g]?.push(i)
  const out = Float64Array.from(m.data)
  let worst = -1
  let worstGap = -1
  for (let j = 0; j < m.d; j++) {
    if (j === groups.column) continue
    const column = (i: number): number => m.data[i * m.d + j] ?? 0
    const sorted = members.map((rows) => rows.map(column).sort((a, b) => a - b))
    const medians = sorted.map((s) => quantileSorted(s, 0.5))
    const gap = Math.max(...medians) - Math.min(...medians)
    if (gap > worstGap) {
      worstGap = gap
      worst = j
    }
    for (const [g, rows] of members.entries()) {
      const own = sorted[g] ?? []
      for (const i of rows) {
        const v = column(i)
        // Mid-rank quantile of the value within its group.
        let below = 0
        let equal = 0
        for (const w of own) {
          if (w < v) below++
          else if (w === v) equal++
        }
        const u = own.length <= 1 ? 0.5 : (below + (equal - 1) / 2) / (own.length - 1)
        const targets = sorted
          .filter((s) => s.length > 0)
          .map((s) => quantileSorted(s, u))
          .sort((a, b) => a - b)
        const target = quantileSorted(targets, 0.5)
        out[i * m.d + j] = (1 - level) * v + level * target
      }
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
