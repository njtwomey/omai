import type { Dataset, DatasetMeta } from 'mdml-engine/data'
import { asTensor, toFlat } from 'mdml-engine/tensor'

/** One column as the page shows it: numbers (NaN is missing), or labels (null is missing). */
export type FrameColumn =
  | { readonly name: string; readonly kind: 'numeric'; readonly values: readonly number[]; readonly target: boolean }
  | {
      readonly name: string
      readonly kind: 'label'
      readonly values: readonly (string | null)[]
      readonly target: boolean
    }

/** A dataset as columns of equal length, whatever its schema. */
export interface Frame {
  readonly rows: number
  readonly columns: readonly FrameColumn[]
  readonly meta: DatasetMeta
}

const numbers = (t: unknown): number[] => Array.from(toFlat(asTensor(t as never)))

/** Columns `j` of a row-major `[n, d]` tensor. */
function matrixColumns(x: unknown, names: readonly string[], target: boolean): FrameColumn[] {
  const t = asTensor(x as never)
  const [n = 0, d = 1] = t.shape
  const flat = toFlat(t)
  return Array.from({ length: d }, (_, j) => ({
    name: names[j] ?? `x${j}`,
    kind: 'numeric' as const,
    values: Array.from({ length: n }, (_, i) => flat[i * d + j] ?? Number.NaN),
    target,
  }))
}

/**
 * The columns of a dataset for display: the feature columns of a regression or classification set (and its target, the
 * classes by name), or the columns of a table. Other kinds of data have no columns to show.
 *
 * @param data - The dataset.
 * @returns The frame, or `undefined` for a dataset that is not a table of columns.
 */
export function toFrame(data: Dataset): Frame | undefined {
  switch (data.kind) {
    case 'regression': {
      const x = matrixColumns(data.x, data.featureNames, false)
      const y = matrixColumns(data.y, data.targetNames, true)
      const first = x[0]
      return first === undefined ? undefined : { rows: first.values.length, columns: [...x, ...y], meta: data.meta }
    }
    case 'classification': {
      const x = matrixColumns(data.x, data.featureNames, false)
      const first = x[0]
      if (first === undefined) return undefined
      const labels = numbers(data.y).map((c) => data.classes[c] ?? null)
      return {
        rows: first.values.length,
        columns: [...x, { name: 'class', kind: 'label', values: labels, target: true }],
        meta: data.meta,
      }
    }
    case 'table': {
      const columns: FrameColumn[] = Object.entries(data.columns).map(([name, c]) => {
        const target = name === data.target
        if (c.kind === 'numeric') return { name, kind: 'numeric', values: numbers(c.values), target }
        if (c.kind === 'categorical')
          return {
            name,
            kind: 'label',
            values: numbers(c.codes).map((k) => (k < 0 ? null : (c.levels[k] ?? null))),
            target,
          }
        if (c.kind === 'boolean')
          return { name, kind: 'label', values: numbers(c.values).map((v) => (v ? 'true' : 'false')), target }
        return { name, kind: 'label', values: c.values, target }
      })
      const first = columns[0]
      return first === undefined ? undefined : { rows: first.values.length, columns, meta: data.meta }
    }
    default:
      return undefined
  }
}

/** What a column holds, in a line: its range or its commonest labels, and how much is missing. */
export interface ColumnSummary {
  readonly missing: number
  /** Numeric: the minimum, mean and maximum of the present values. */
  readonly range?: readonly [number, number, number]
  /** Numeric: counts in equal-width bins from the minimum to the maximum. */
  readonly bins?: readonly number[]
  /** Labels: the commonest levels with their counts, most first, and how many levels there are. */
  readonly levels?: readonly (readonly [string, number])[]
  readonly levelCount?: number
}

/**
 * Summarise a column.
 *
 * @param column - The column.
 * @param binCount - Bins of a numeric column's histogram.
 * @param top - Levels kept for a label column.
 * @returns The summary.
 */
export function summarise(column: FrameColumn, binCount = 12, top = 5): ColumnSummary {
  if (column.kind === 'numeric') {
    const present = column.values.filter((v) => Number.isFinite(v))
    const missing = column.values.length - present.length
    if (present.length === 0) return { missing }
    const lo = Math.min(...present)
    const hi = Math.max(...present)
    const mean = present.reduce((a, b) => a + b, 0) / present.length
    const bins = new Array<number>(binCount).fill(0)
    for (const v of present) {
      const k = hi === lo ? 0 : Math.min(binCount - 1, Math.floor(((v - lo) / (hi - lo)) * binCount))
      bins[k] = (bins[k] ?? 0) + 1
    }
    return { missing, range: [lo, mean, hi], bins }
  }
  const counts = new Map<string, number>()
  let missing = 0
  for (const v of column.values) {
    if (v === null) missing++
    else counts.set(v, (counts.get(v) ?? 0) + 1)
  }
  const sorted = [...counts].sort((a, b) => b[1] - a[1])
  return { missing, levels: sorted.slice(0, top), levelCount: sorted.length }
}
