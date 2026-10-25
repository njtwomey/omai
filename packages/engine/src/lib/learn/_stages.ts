// What a pipeline records of each stage beyond its shapes: the stage's data (rows, classes, column statistics), its
// fitted parameters in full, and, for the final estimator, the whole pipeline's surface over the raw inputs with its
// loss (private to the domain).
import type { ColumnStats, DecisionSurface, StageData, StageLoss, StageParameter } from '../exposition/index.ts'
import { asTensor, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import type { Fitted, LearnData } from './estimator.ts'
import { classNamesOf, decisionSurfaceOf, featureNamesOf } from './explained.ts'
import { matrixView } from './validation.ts'

/** The most rows a stage's data carries by default (evenly spaced beyond it). */
export const STAGE_ROWS = 500

/** At most `limit` row indices of n, evenly spaced (all of them when n is small). */
export function spacedRows(n: number, limit: number): number[] {
  if (n <= limit) return Array.from({ length: n }, (_, i) => i)
  return Array.from({ length: Math.max(0, limit) }, (_, i) => Math.floor((i * n) / limit))
}

/** How a pipeline's rows are coloured: by class index (with the class values) or by their numeric target. */
export type RowTargets =
  | { readonly kind: 'classes'; readonly values: readonly number[]; readonly names: readonly string[] }
  | { readonly kind: 'values' }
  | undefined

/** How the training targets colour the rows: classes for a classifier (or class-named data), values for a regressor. */
export function rowTargetsOf(data: LearnData, finalKind: string): RowTargets {
  if (data.y === undefined) return undefined
  const y = toFlat(asTensor(data.y))
  if (y.length === 0) return undefined
  const named = Array.isArray((data as { readonly classes?: unknown }).classes)
  if (finalKind === 'classifier' || named) {
    const values = Array.from(new Set(Array.from(y))).sort((a, b) => a - b)
    if (values.length > 64) return { kind: 'values' }
    return { kind: 'classes', values, names: classNamesOf(data, values) }
  }
  return { kind: 'values' }
}

function columnStats(data: Float64Array, n: number, d: number): ColumnStats[] {
  return Array.from({ length: d }, (_, j) => {
    let count = 0
    let sum = 0
    let lo = Number.POSITIVE_INFINITY
    let hi = Number.NEGATIVE_INFINITY
    for (let i = 0; i < n; i++) {
      const v = data[i * d + j] ?? Number.NaN
      if (!Number.isFinite(v)) continue
      count++
      sum += v
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
    const mean = count === 0 ? Number.NaN : sum / count
    let ss = 0
    for (let i = 0; i < n; i++) {
      const v = data[i * d + j] ?? Number.NaN
      if (Number.isFinite(v)) ss += (v - mean) ** 2
    }
    return {
      mean,
      std: count === 0 ? Number.NaN : Math.sqrt(ss / count),
      min: count === 0 ? Number.NaN : lo,
      max: count === 0 ? Number.NaN : hi,
    }
  })
}

/**
 * A stage's data: up to `limit` evenly spaced rows of `x` with their classes or targets, and statistics of every
 * column over all rows.
 */
export function stageDataOf(
  x: TensorLike,
  columns: readonly string[],
  y: TensorLike | undefined,
  targets: RowTargets,
  limit: number,
): StageData {
  const m = matrixView(x, 'pipeline', { allowNaN: true, allowInfinite: true })
  const idx = spacedRows(m.n, limit)
  const rows = idx.map((i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  const ys = y === undefined ? undefined : toFlat(asTensor(y))
  const aligned = ys !== undefined && ys.length === m.n ? ys : undefined
  let labels: number[] | undefined
  let values: number[] | undefined
  if (aligned !== undefined && targets?.kind === 'classes') {
    const index = new Map(targets.values.map((c, i) => [c, i]))
    labels = idx.map((i) => index.get(aligned[i] ?? Number.NaN) ?? -1)
  } else if (aligned !== undefined && targets?.kind === 'values') values = idx.map((i) => aligned[i] ?? Number.NaN)
  return {
    columns: [...columns],
    rows,
    n: m.n,
    ...(labels === undefined ? {} : { labels }),
    ...(values === undefined ? {} : { targets: values }),
    stats: columnStats(m.data, m.n, m.d),
  }
}

const MAX_PARAMETERS = 8
const MAX_ENTRIES = 4096

function isNumbers(v: unknown): v is number[] {
  return Array.isArray(v) && v.length > 0 && v.every((u) => typeof u === 'number')
}

/**
 * A fitted step's parameters in full: numbers, short strings, number vectors and matrices (up to a few thousand
 * entries), named by the stage's columns where their lengths match (a vector per input or output column; a matrix
 * with a row per output column or a column per input column). Counts that are the same for every column (rows
 * seen) say nothing per column and are left out.
 */
export function parametersOf(
  f: Fitted,
  input: readonly string[],
  output: readonly string[],
): { readonly parameters: StageParameter[]; readonly explainedVarianceRatio?: number[] } {
  const params = f.params as unknown
  const out: StageParameter[] = []
  if (typeof params !== 'object' || params === null || Array.isArray(params)) return { parameters: out }
  let ratio: number[] | undefined
  const named = (len: number): readonly string[] | undefined =>
    len === input.length ? input : len === output.length ? output : undefined
  for (const [name, v] of Object.entries(params as Record<string, unknown>)) {
    if (name === 'explainedVarianceRatio' && isNumbers(v)) ratio = [...v]
    if (out.length >= MAX_PARAMETERS) continue
    if (typeof v === 'number' && Number.isFinite(v)) out.push({ name, value: v })
    else if (typeof v === 'string' && v.length <= 32) out.push({ name, value: v })
    else if (isNumbers(v) && v.length <= MAX_ENTRIES) {
      if (v.length > 1 && v.every((u) => Number.isInteger(u) && u === v[0])) continue
      const columns = named(v.length)
      out.push({ name, value: [...v], ...(columns === undefined ? {} : { columns }) })
    } else if (Array.isArray(v) && v.length > 0 && v.every(isNumbers)) {
      const m = v as number[][]
      const width = m[0]?.length ?? 0
      if (m.some((r) => r.length !== width) || m.length * width > MAX_ENTRIES) continue
      const columns = width === input.length ? input : undefined
      const rows = m.length === output.length && m.length !== 1 ? output : undefined
      out.push({
        name,
        value: m.map((r) => [...r]),
        ...(columns === undefined ? {} : { columns }),
        ...(rows === undefined ? {} : { rows }),
      })
    }
  }
  return { parameters: out, ...(ratio === undefined ? {} : { explainedVarianceRatio: ratio }) }
}

/** The binary or multiclass entropy of each grid cell's predicted distribution: the log loss the model expects. */
function entropyGrid(probabilities: readonly (readonly (readonly number[])[])[]): number[][] {
  const first = probabilities[0] ?? []
  return first.map((row, i) =>
    row.map((_, j) => {
      let h = 0
      for (const p of probabilities) {
        const v = p[i]?.[j] ?? 0
        if (v > 0) h -= v * Math.log(v)
      }
      return h
    }),
  )
}

/**
 * The whole fitted pipeline's predictions over its raw inputs (one or two features), on up to `limit` of the
 * training rows, with the loss on the training rows: log loss for a probabilistic classifier (per row, the rows it
 * misclassifies, and the loss it expects over the grid), squared error for a regressor.
 */
export function surfaceAndLossOf(
  fitted: Fitted,
  data: LearnData,
  targets: RowTargets,
  limit: number,
): { readonly surface?: DecisionSurface; readonly loss?: StageLoss } {
  const m = matrixView(data.x, 'pipeline', { allowNaN: true, allowInfinite: true })
  if (m.d < 1 || m.d > 2 || m.n === 0 || fitted.predict === undefined) return {}
  if (!m.data.every(Number.isFinite)) return {}
  const idx = spacedRows(m.n, limit)
  const xs = idx.map((i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
  const y = data.y === undefined ? undefined : toFlat(asTensor(data.y))
  const aligned = y !== undefined && y.length === m.n ? y : undefined
  const ys = aligned === undefined ? undefined : idx.map((i) => aligned[i] ?? 0)
  const classes = fitted.classes
  const surface = decisionSurfaceOf(fitted, xs, {
    featureNames: featureNamesOf(data, m.d),
    resolution: 40,
    rowsLimit: limit,
    ...(ys === undefined ? {} : { y: ys }),
    ...(classes === undefined ? {} : { classNames: classNamesOf(data, classes) }),
  })
  if (aligned === undefined || surface.rows === undefined) return { surface }
  if (fitted.kind === 'classifier' && classes !== undefined && fitted.predictProba !== undefined) {
    const proba = fitted.predictProba(data.x) as Tensor
    const flat = toFlat(proba)
    const k = classes.length
    const index = new Map(classes.map((c, i) => [c, i]))
    const lossAt = (i: number): number => {
      const c = index.get(aligned[i] ?? Number.NaN)
      const p = c === undefined ? 0 : (flat[i * k + c] ?? 0)
      return -Math.log(Math.max(p, 1e-15))
    }
    let total = 0
    for (let i = 0; i < m.n; i++) total += lossAt(i)
    const predicted = toFlat(fitted.predict(xs) as Tensor)
    const misclassified = idx.flatMap((i, r) => (predicted[r] === aligned[i] ? [] : [r]))
    return {
      surface,
      loss: {
        name: 'log loss',
        mean: total / m.n,
        values: idx.map(lossAt),
        misclassified,
        ...(surface.probabilities === undefined ? {} : { expected: entropyGrid(surface.probabilities) }),
      },
    }
  }
  if (fitted.kind === 'regressor' && targets?.kind !== 'classes') {
    const predicted = toFlat(fitted.predict(data.x) as Tensor)
    if (predicted.length !== m.n) return { surface }
    const err = (i: number): number => ((predicted[i] ?? 0) - (aligned[i] ?? 0)) ** 2
    let total = 0
    for (let i = 0; i < m.n; i++) total += err(i)
    return { surface, loss: { name: 'squared error', mean: total / m.n, values: idx.map(err) } }
  }
  return { surface }
}
