/**
 * Graphs from precision matrices and how well one graph recovers another: the adjacency of a precision matrix (the
 * conditional-independence graph of a Gaussian model: an edge where the entry is non-zero), its edge list, and the
 * precision, recall and F1 of an estimated edge set against a true one, with the structural Hamming distance.
 */

import { OmaiError } from '../base/index.ts'

/**
 * The conditional-independence graph of a precision matrix: `graph[a][b]` is true where `|precision[a][b]|` exceeds
 * `tol` (the diagonal is false).
 *
 * @param precision - The precision matrix `[p][p]`.
 * @param tol - The magnitude below which an entry counts as zero (default 1e-10).
 * @returns The symmetric adjacency matrix (an edge needs both entries above `tol`).
 * @example
 * ```ts
 * import { precisionGraph } from 'omai-engine/covariance'
 * precisionGraph([[2, -0.5, 0], [-0.5, 2, 0], [0, 0, 1]]) // => [[false, true, false], [true, false, false], [false, false, false]]
 * ```
 */
export function precisionGraph(precision: readonly (readonly number[])[], tol = 1e-10): boolean[][] {
  const p = precision.length
  return Array.from({ length: p }, (_, a) =>
    Array.from(
      { length: p },
      (__, b) => a !== b && Math.abs(precision[a]?.[b] ?? 0) > tol && Math.abs(precision[b]?.[a] ?? 0) > tol,
    ),
  )
}

/**
 * The edges `[a, b]`, `a < b`, of an adjacency matrix.
 *
 * @param graph - The adjacency matrix `[p][p]`.
 * @returns The edges in row-major order.
 * @example
 * ```ts
 * import { graphEdges } from 'omai-engine/covariance'
 * graphEdges([[false, true, false], [true, false, true], [false, true, false]]) // => [[0, 1], [1, 2]]
 * ```
 */
export function graphEdges(graph: readonly (readonly boolean[])[]): [number, number][] {
  const edges: [number, number][] = []
  for (let a = 0; a < graph.length; a++)
    for (let b = a + 1; b < graph.length; b++) if (graph[a]?.[b] === true) edges.push([a, b])
  return edges
}

/** How well an estimated graph recovers a true one, over the node pairs (each undirected edge counted once). */
export interface EdgeRecovery {
  /** Edges in both graphs. */
  readonly truePositives: number
  /** Edges only in the estimate. */
  readonly falsePositives: number
  /** Edges only in the truth. */
  readonly falseNegatives: number
  /** `truePositives / (truePositives + falsePositives)` (1 when the estimate has no edges). */
  readonly precision: number
  /** `truePositives / (truePositives + falseNegatives)` (1 when the truth has no edges). */
  readonly recall: number
  /** The harmonic mean of the two (0 when both are 0). */
  readonly f1: number
  /** The structural Hamming distance: `falsePositives + falseNegatives`. */
  readonly hamming: number
}

/**
 * Support recovery of a graph estimate: counts and the precision, recall and F1 of its edges against the true edges.
 *
 * @param estimated - The estimated adjacency matrix `[p][p]` (the upper triangle is read).
 * @param truth - The true adjacency matrix `[p][p]`.
 * @returns The counts, rates and the Hamming distance.
 * @throws OmaiError with code `shape` when the graphs differ in size.
 * @index function Edge precision, recall, F1 and Hamming distance of an estimated graph against the true one.
 * @example
 * ```ts
 * import { edgeRecovery } from 'omai-engine/covariance'
 * const truth = [[false, true, false], [true, false, true], [false, true, false]]
 * const found = [[false, true, true], [true, false, false], [true, false, false]]
 * edgeRecovery(found, truth).f1 // => 0.5
 * ```
 */
export function edgeRecovery(
  estimated: readonly (readonly boolean[])[],
  truth: readonly (readonly boolean[])[],
): EdgeRecovery {
  if (estimated.length !== truth.length) throw new OmaiError('shape', 'edgeRecovery: the graphs differ in size')
  let tp = 0
  let fp = 0
  let fn = 0
  for (let a = 0; a < truth.length; a++)
    for (let b = a + 1; b < truth.length; b++) {
      const e = estimated[a]?.[b] === true
      const t = truth[a]?.[b] === true
      if (e && t) tp++
      else if (e) fp++
      else if (t) fn++
    }
  const precision = tp + fp === 0 ? 1 : tp / (tp + fp)
  const recall = tp + fn === 0 ? 1 : tp / (tp + fn)
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall)
  return { truePositives: tp, falsePositives: fp, falseNegatives: fn, precision, recall, f1, hamming: fp + fn }
}
