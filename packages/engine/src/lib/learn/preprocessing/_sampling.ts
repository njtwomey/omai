// Private helpers shared by the resamplers: the classes of the data, assembly of a resampled set, nearest neighbours
// within a matrix, interpolation between neighbours (SMOTE's step) and the neighbour tests of the cleaning rules.

import { foldIn, type Key, randint, uniform } from '../../random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../tensor/index.ts'
import type { LearnData } from '../estimator.ts'
import { encodeLabels, type MatrixView, matrixView, takeRows } from '../validation.ts'
import type { Resampled } from './resampling.ts'

export interface Classes {
  readonly m: MatrixView
  readonly labels: Float64Array
  readonly classes: readonly number[]
  readonly members: number[][]
}

export function classesOf(data: LearnData, who: string): Classes {
  const m = matrixView(data.x, who)
  const { classes, codes } = encodeLabels(data.y ?? [], who, m.n)
  const members = classes.map(() => [] as number[])
  codes.forEach((c, i) => {
    members[c]?.push(i)
  })
  return { m, labels: Float64Array.from(codes, (c) => classes[c] ?? 0), classes, members }
}

export function assemble(
  c: Classes,
  rows: readonly number[],
  synthetic: readonly number[][],
  syntheticLabels: readonly number[],
): Resampled {
  const base = takeRows(c.m, rows)
  const n = rows.length + synthetic.length
  const out = new Float64Array(n * c.m.d)
  out.set(base.data)
  synthetic.forEach((r, s) => {
    out.set(r, (rows.length + s) * c.m.d)
  })
  return {
    x: fromStorage(out, [n, c.m.d], 'float64'),
    y: fromStorage(Float64Array.from([...rows.map((i) => c.labels[i] ?? 0), ...syntheticLabels]), [n], 'float64'),
    indices: Int32Array.from([...rows, ...synthetic.map(() => -1)]),
  }
}

/** Squared Euclidean distances from row `i` of `a` to every row of `b`. */
export function distancesFrom(a: MatrixView, i: number, b: MatrixView): Float64Array {
  const out = new Float64Array(b.n)
  for (let r = 0; r < b.n; r++) {
    let s = 0
    for (let j = 0; j < a.d; j++) s += ((a.data[i * a.d + j] ?? 0) - (b.data[r * b.d + j] ?? 0)) ** 2
    out[r] = s
  }
  return out
}

/** The indices of the k nearest rows of `b` to row i of `a` (ties by index), optionally skipping one index. */
export function nearest(a: MatrixView, i: number, b: MatrixView, k: number, skip = -1): number[] {
  const d = distancesFrom(a, i, b)
  return Array.from(d.keys())
    .filter((r) => r !== skip)
    .sort((p, q) => (d[p] ?? 0) - (d[q] ?? 0) || p - q)
    .slice(0, k)
}

/** The draws behind SMOTE: for each of `count` points, a row of `rows` sample, one of its `k` neighbours, and a step in [0, 1). */
export function drawPairs(
  key: Key,
  count: number,
  nRows: number,
  k: number,
): { readonly row: Int32Array; readonly col: Int32Array; readonly step: Float64Array } {
  const picks = toFlat(randint(foldIn(key, 0), [count], 0, nRows * k))
  return {
    row: Int32Array.from(picks, (p) => Math.floor(p / k)),
    col: Int32Array.from(picks, (p) => p % k),
    step: Float64Array.from(toFlat(uniform(foldIn(key, 1), [count]))),
  }
}

/** Interpolate `count` points between rows and their neighbours (imblearn's `_make_samples`). */
export function interpolate(
  key: Key,
  base: MatrixView,
  rows: readonly number[],
  neighbours: readonly (readonly number[])[],
  donors: MatrixView,
  count: number,
  stepScale: number,
): number[][] {
  const k = neighbours[0]?.length ?? 0
  if (count <= 0 || rows.length === 0 || k === 0) return []
  const { row, col, step } = drawPairs(key, count, rows.length, k)
  return Array.from(row, (r, s) => {
    const i = rows[r] ?? 0
    const nb = neighbours[r]?.[col[s] ?? 0] ?? 0
    const t = stepScale * (step[s] ?? 0)
    return Array.from({ length: base.d }, (_, j) => {
      const x = base.data[i * base.d + j] ?? 0
      return x + t * ((donors.data[nb * donors.d + j] ?? 0) - x)
    })
  })
}

/** A matrix view as a float tensor. */
export function fromRows(m: MatrixView): Tensor {
  return fromStorage(m.data, [m.n, m.d], 'float64')
}

/** The k nearest rows of `b` for every row of `a`, each list skipping the row's own index when `self` (a and b alike). */
export function neighbourLists(a: MatrixView, b: MatrixView, k: number, self = false): number[][] {
  return Array.from({ length: a.n }, (_, i) => nearest(a, i, b, k, self ? i : -1))
}

/** The most frequent value, the smallest among ties (scipy's `mode`, and the vote of a k-NN classifier). */
export function modeOf(values: readonly number[]): number {
  const counts = new Map<number, number>()
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1)
  let best = Number.NaN
  let bestCount = 0
  for (const [v, c] of counts) if (c > bestCount || (c === bestCount && v < best)) [best, bestCount] = [v, c]
  return best
}

/**
 * Mutual nearest neighbours of different classes (Tomek links) among some rows: the rows to keep (in the order given),
 * dropping a member of a link when its class is one of the classes `clean` accepts.
 */
export function tomekKeep(
  m: MatrixView,
  rows: readonly number[],
  labels: ArrayLike<number>,
  clean: (label: number) => boolean,
): number[] {
  const sub = takeRows(m, rows)
  const nn = Array.from({ length: sub.n }, (_, i) => nearest(sub, i, sub, 1, i)[0] ?? i)
  return rows.filter((row, i) => {
    const label = labels[row] ?? 0
    if (!clean(label)) return true
    const j = nn[i] ?? i
    return !(labels[rows[j] ?? 0] !== label && nn[j] === i)
  })
}
