import { coefficientTable, curveSet, type Explained, type Exposition } from '../../lib/exposition/index.ts'
import { type ReweighingCell as CoreCell, repairFeatures, reweighingWeights } from '../../lib/fairness/index.ts'
import { type Estimator, type Fitted, featureNamesOf, matrixView } from '../../lib/learn/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { fromStorage } from '../../lib/tensor/index.ts'
import { logisticRegression } from '../linear/index.ts'
import { fairnessAudit } from './audit.ts'
import { binaryLabels, type FairnessData, type SensitiveSpec, sensitiveGroups, withoutColumns } from './groups.ts'

/** One cell of {@link reweighing}: as the core's, with the group named. */
export type ReweighingCell = Omit<CoreCell, 'group'> & { readonly group: string }

/**
 * Reweighing (`reweighingWeights` of `whitebox-engine/fairness`) on a dataset with a protected attribute: groups
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
 * Disparate-impact removal (`repairFeatures` of `whitebox-engine/fairness`, Feldman et al.) on a dataset with a
 * protected attribute: every feature but the protected one moves towards a distribution shared by the groups, keeping
 * ranks within groups, by `repairLevel`. A classifier fitted on fully repaired features cannot tell the groups apart from any single feature. Exposition: a
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
