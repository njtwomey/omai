/**
 * What every fairness method starts from: the protected groups of a dataset's rows (from a named feature column or a
 * label per row), the features without the protected attribute (fairness through unawareness), and a classifier's
 * positive-class scores and binary labels.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import {
  type Fitted,
  featureNamesOf,
  fromView,
  type LearnData,
  matrixView,
  requireVerb,
  takeColumns,
  targetVector,
} from '../../lib/learn/index.ts'
import { type TensorLike, toFlat } from '../../lib/tensor/index.ts'

/** Learn data that may name its features and classes and carry dataset metadata (a `ClassificationData`). */
export type FairnessData = LearnData & {
  readonly featureNames?: readonly string[]
  readonly classes?: readonly string[]
  readonly meta?: { readonly sensitive?: Readonly<Record<string, readonly string[]>> }
}

/** Where the protected attribute is: a feature column by name or index, or a group label per row. */
export type SensitiveSpec = string | number | readonly (number | string)[]

/** The protected groups of the rows. */
export interface SensitiveGroups {
  /** The attribute's name. */
  readonly attribute: string
  /** The group names, in index order. */
  readonly names: readonly string[]
  /** Each row's group, an index into `names`. */
  readonly index: readonly number[]
  /** The feature column holding the attribute, when it is one. */
  readonly column?: number
}

/**
 * The protected groups of a dataset's rows: from a feature column (by name or index; its coded values named by
 * `meta.sensitive` when the dataset names them), from a label per row, or by default from the first attribute the
 * dataset's `meta.sensitive` names.
 *
 * @param data - The data.
 * @param sensitive - The column or the labels (default the dataset's first sensitive attribute).
 * @returns The attribute, group names and each row's group.
 * @throws WhiteboxError `value` when no attribute is given or named, or a named column is missing; `shape` when the
 *   labels do not match the rows.
 */
export function sensitiveGroups(data: FairnessData, sensitive?: SensitiveSpec): SensitiveGroups {
  const m = matrixView(data.x, 'sensitiveGroups')
  const names = featureNamesOf(data, m.d)
  const known = data.meta?.sensitive ?? {}
  const spec = sensitive ?? Object.keys(known)[0]
  if (spec === undefined)
    throw new WhiteboxError('value', 'sensitiveGroups: give the protected attribute (the dataset names none)')
  if (typeof spec === 'object') {
    if (spec.length !== m.n) throw new WhiteboxError('shape', `sensitiveGroups: ${spec.length} labels for ${m.n} rows`)
    return fromValues('group', spec, undefined)
  }
  const column = typeof spec === 'number' ? spec : names.indexOf(spec)
  if (column < 0 || column >= m.d) throw new WhiteboxError('value', `sensitiveGroups: no feature ${String(spec)}`)
  const attribute = names[column] ?? `x${column}`
  const values = Array.from({ length: m.n }, (_, i) => m.data[i * m.d + column] ?? 0)
  return { ...fromValues(attribute, values, known[attribute]), column }
}

function fromValues(
  attribute: string,
  values: readonly (number | string)[],
  coded: readonly string[] | undefined,
): SensitiveGroups {
  if (coded !== undefined && values.every((v) => typeof v === 'number' && Number.isInteger(v) && v < coded.length))
    return { attribute, names: coded, index: values as number[] }
  const distinct = [...new Set(values)].sort((a, b) =>
    typeof a === 'number' && typeof b === 'number' ? a - b : String(a) < String(b) ? -1 : 1,
  )
  const at = new Map(distinct.map((v, i) => [v, i]))
  return { attribute, names: distinct.map(String), index: values.map((v) => at.get(v) ?? 0) }
}

/**
 * The data without some feature columns (fairness through unawareness: fit without the protected attribute), with
 * its feature names, classes and metadata kept.
 *
 * @param data - The data.
 * @param columns - Column indices or names to drop.
 * @returns The data without them.
 */
export function withoutColumns(data: FairnessData, columns: readonly (number | string)[]): FairnessData {
  const m = matrixView(data.x, 'withoutColumns')
  const names = featureNamesOf(data, m.d)
  const drop = new Set(columns.map((c) => (typeof c === 'number' ? c : names.indexOf(c))))
  const keep = Array.from({ length: m.d }, (_, j) => j).filter((j) => !drop.has(j))
  return { ...data, x: fromView(takeColumns(m, keep)), featureNames: keep.map((j) => names[j] ?? `x${j}`) }
}

/**
 * A fitted binary classifier's positive-class scores: the probability of its second class, or the decision function
 * when it gives no probabilities.
 *
 * @param model - The fitted classifier.
 * @param x - The rows.
 * @returns A score per row (higher means more positive) and whether they are probabilities.
 */
export function positiveScores(
  model: Fitted,
  x: TensorLike,
): { readonly scores: number[]; readonly probability: boolean } {
  if (typeof model.predictProba === 'function') {
    const p = model.predictProba(x)
    const k = p.shape[1] ?? 1
    const flat = toFlat(p)
    return { scores: Array.from({ length: flat.length / k }, (_, i) => flat[i * k + k - 1] ?? 0), probability: true }
  }
  return { scores: Array.from(toFlat(requireVerb(model, 'decisionFunction', 'positiveScores')(x))), probability: false }
}

/** Binary labels (1 for the positive class, the larger label) of the data's targets. */
export function binaryLabels(data: LearnData, who: string): number[] {
  if (data.y === undefined) throw new WhiteboxError('value', `${who}: the data needs labels`)
  const y = Array.from(targetVector(data.y, who))
  const positive = y.every((v) => v === 0 || v === 1) ? 1 : Math.max(...y)
  return y.map((v) => (v === positive ? 1 : 0))
}
