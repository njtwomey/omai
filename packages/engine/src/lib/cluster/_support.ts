/**
 * What the clusterers of this folder share and nobody else needs: rows picked out of a flat matrix, labels renumbered by
 * first appearance, and the prediction verb of a clusterer defined by labelled points (a new row takes the label of the
 * nearest labelled point within a radius, or -1).
 */

import { type DistanceAlias, type DistanceName, pairwiseDistances } from '../distances/index.ts'
import { matrixView } from '../learn/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'

/** Rows of a matrix as plain arrays. */
export type Matrix = readonly (readonly number[])[]

/** Rows `idx` of a flat row-major matrix with `d` columns. */
export function rowsAt(data: ArrayLike<number>, d: number, idx: readonly number[]): number[][] {
  return idx.map((i) => Array.from({ length: d }, (_, j) => data[i * d + j] ?? 0))
}

/** A flat row-major matrix as rows. */
export function toRows(flat: ArrayLike<number>, n: number, d: number): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => flat[i * d + j] ?? 0))
}

/** Labels renumbered 0, 1, ... in order of first appearance; negative labels (noise) become -1. */
export function canonicalLabels(labels: readonly number[]): number[] {
  const seen = new Map<number, number>()
  return labels.map((l) => {
    if (l < 0) return -1
    if (!seen.has(l)) seen.set(l, seen.size)
    return seen.get(l) ?? 0
  })
}

/** The k-th smallest of a row of a flat `n x n` matrix (k from 1), the row's own entry included. */
export function kthInRow(dist: ArrayLike<number>, n: number, i: number, k: number): number {
  const row = Array.from({ length: n }, (_, j) => dist[i * n + j] ?? 0).sort((a, b) => a - b)
  return row[Math.min(Math.max(k, 1), n) - 1] ?? 0
}

/**
 * The `predict` verb of a clusterer defined by labelled points: a new row takes the label of the nearest of the points
 * within `radius` (-1 when none is).
 */
export function labelledPredict(
  points: Matrix,
  labels: readonly number[],
  radius: number,
  metric: DistanceName | DistanceAlias,
  d: number,
): (x: TensorLike) => Tensor {
  return (x) => {
    const m = matrixView(x, 'predict', points.length === 0 ? {} : { nFeatures: d })
    const out = new Int32Array(m.n).fill(-1)
    if (points.length > 0) {
      const dist = toFlat(pairwiseDistances(fromStorage(m.data, [m.n, d], 'float64'), points as number[][], metric))
      for (let i = 0; i < m.n; i++) {
        let best = radius
        for (let j = 0; j < points.length; j++) {
          const v = dist[i * points.length + j] ?? 0
          if (v <= best) {
            best = v
            out[i] = labels[j] ?? -1
          }
        }
      }
    }
    return fromStorage(out, [m.n], 'int32')
  }
}
