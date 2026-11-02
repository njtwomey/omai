/**
 * The self-organising map (Kohonen 1982): a lattice of prototype vectors trained online so that neighbouring nodes come to
 * represent similar rows, giving a topology-preserving map of the data. One epoch per step with shrinking learning rate and
 * neighbourhood, the U-matrix of neighbour distances, and an estimator whose transform is the grid position of a row's
 * best-matching unit.
 */

import { MdmlError } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { geometry } from '../numerics/index.ts'
import { permutation, subsample } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

/** The state of a self-organising map: the node weights in row-major grid order, and the quantisation error. */
export interface SomState {
  readonly weights: number[][]
  /** The mean distance of the rows to their best-matching units. */
  readonly error: number
}

/** The best-matching unit (nearest node) of each row, and the mean distance to it (the quantisation error). */
function bestUnits(x: TensorLike, weights: readonly (readonly number[])[]): { units: number[]; error: number } {
  const d = toFlat(geometry.cdist(x, weights as number[][]))
  const k = weights.length
  const n = d.length / k
  const units: number[] = []
  let error = 0
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let j = 1; j < k; j++) if ((d[i * k + j] ?? 0) < (d[i * k + best] ?? 0)) best = j
    units.push(best)
    error += d[i * k + best] ?? 0
  }
  return { units, error: error / n }
}

/**
 * A self-organising map (Kohonen 1982) trained online as a traced {@link Algorithm}: each step (an epoch) visits the rows
 * in a random order, moving every node towards the row by lr * exp(-g^2 / (2 sigma^2)) for grid distance g to the
 * best-matching unit, with lr and sigma decaying as 1 / (1 + t / (T / 2)) (MiniSom's default). It observes the
 * quantisation error, the node `weights` and the `embedding` (each row's grid position).
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial node weights `[rows * cols, d]`.
 * @param grid - Rows and columns of the lattice.
 * @param options - Epochs, the initial learning rate (0.5) and neighbourhood width (max(rows, cols) / 2).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { somAlgorithm } from 'mdml-engine/manifold'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [1], [2], [3]]
 * const alg = somAlgorithm(x, [[0], [1], [2]], [1, 3], { epochs: 10 })
 * runAlgorithm(alg, { maxSteps: 10, key: key(0) }).state.weights.length // => 3
 * ```
 */
export function somAlgorithm(
  x: TensorLike,
  init: readonly (readonly number[])[],
  grid: readonly [number, number],
  options: { readonly epochs: number; readonly learningRate?: number; readonly sigma?: number },
): Algorithm<SomState> {
  const m = matrixView(x, 'som')
  const [rows, cols] = grid
  if (rows * cols !== init.length)
    throw new MdmlError('shape', `som: ${init.length} nodes for a ${rows} x ${cols} grid`)
  const lr0 = options.learningRate ?? 0.5
  const s0 = options.sigma ?? Math.max(rows, cols) / 2
  const coords = Array.from({ length: rows * cols }, (_, u) => [Math.floor(u / cols), u % cols] as const)
  return {
    name: 'som',
    config: { rows, cols, learningRate: lr0, sigma: s0 },
    schema: channels({
      metrics: {
        quantisationError: {
          description: 'the mean distance of the rows to their best-matching units',
          direction: 'lower',
        },
      },
      tensors: {
        weights: { role: 'state', description: 'the prototype vectors in grid order', shape: ['units', 'd'] },
        embedding: {
          role: 'state',
          description: 'the grid position of each row’s best-matching unit',
          shape: ['n', 2],
        },
      },
    }),
    init: () => {
      const weights = init.map((r) => [...r])
      return { weights, error: bestUnits(x, weights).error }
    },
    step: (s, ctx) => {
      const decay = 1 / (1 + ctx.t / (options.epochs / 2))
      const lr = lr0 * decay
      const sigma = Math.max(s0 * decay, 1e-3)
      const weights = s.weights.map((r) => [...r])
      for (const i of permutation(ctx.key, m.n).data) {
        const row = m.data.subarray(i * m.d, (i + 1) * m.d)
        let best = 0
        let bestD = Number.POSITIVE_INFINITY
        for (const [u, w] of weights.entries()) {
          let dd = 0
          for (let q = 0; q < m.d; q++) dd += ((row[q] ?? 0) - (w[q] ?? 0)) ** 2
          if (dd < bestD) {
            bestD = dd
            best = u
          }
        }
        const [br, bc] = coords[best] as readonly [number, number]
        for (const [u, w] of weights.entries()) {
          const [r, c] = coords[u] as readonly [number, number]
          const h = Math.exp(-((r - br) ** 2 + (c - bc) ** 2) / (2 * sigma * sigma))
          if (h < 1e-6) continue
          for (let q = 0; q < m.d; q++) w[q] = (w[q] ?? 0) + lr * h * ((row[q] ?? 0) - (w[q] ?? 0))
        }
      }
      return { weights, error: bestUnits(x, weights).error }
    },
    done: (_s, t) => (t >= options.epochs ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { quantisationError: s.error },
      tensors: {
        weights: { role: 'state', value: s.weights.flat(), shape: [s.weights.length, m.d] },
        embedding: {
          role: 'state',
          value: bestUnits(x, s.weights).units.flatMap((u) => [Math.floor(u / cols), u % cols]),
          shape: [m.n, 2],
        },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SomState,
  }
}

/**
 * The U-matrix of a map: each node's mean distance to its 4-connected grid neighbours.
 *
 * @param weights - Node weights in row-major grid order.
 * @param grid - Rows and columns.
 * @returns The distances `[rows, cols]`.
 * @example
 * ```ts
 * import { uMatrix } from 'mdml-engine/manifold'
 * uMatrix([[0], [1], [3]], [1, 3]) // => [[1, 1.5, 2]]
 * ```
 */
export function uMatrix(weights: readonly (readonly number[])[], grid: readonly [number, number]): number[][] {
  const [rows, cols] = grid
  const at = (r: number, c: number): readonly number[] => weights[r * cols + c] ?? []
  const dist = (a: readonly number[], b: readonly number[]): number =>
    Math.sqrt(a.reduce((s, v, q) => s + (v - (b[q] ?? 0)) ** 2, 0))
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => {
      const near = [
        [r - 1, c],
        [r + 1, c],
        [r, c - 1],
        [r, c + 1],
      ].filter(([a = 0, b = 0]) => a >= 0 && a < rows && b >= 0 && b < cols)
      return near.reduce((s, [a = 0, b = 0]) => s + dist(at(r, c), at(a, b)), 0) / Math.max(1, near.length)
    }),
  )
}

/** Hyperparameters of {@link selfOrganisingMap}. */
export interface SomHyper {
  readonly rows: number
  readonly cols: number
  readonly epochs: number
  readonly learningRate: number
  /** The neighbourhood width (`null`: max(rows, cols) / 2). */
  readonly sigma: number | null
}

/** The fitted parameters of a map. */
export type SomParams = {
  readonly weights: number[][]
  readonly grid: [number, number]
  readonly quantisationError: number
  readonly uMatrix: number[][]
}

/** The verbs of a fitted map: the grid position of each row's best-matching unit. */
export interface SomVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly predict: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * A self-organising map on the iterative base: a rows x cols lattice of prototypes trained online, one epoch per step.
 * `transform` gives the grid position of a row's best-matching unit and `predict` its unit index.
 *
 * @index estimator Self-organising map.
 * @example
 * ```ts
 * import { selfOrganisingMap } from 'mdml-engine/manifold'
 * const x = Array.from({ length: 20 }, (_, i) => [i / 20, (i % 5) / 5])
 * selfOrganisingMap.with({ rows: 3, cols: 3, epochs: 5 }).fit({ x }).params.weights.length // => 9
 * ```
 */
export const selfOrganisingMap: IterativeEstimator<SomHyper, SomState, SomParams, SomVerbs> = iterative<
  SomHyper,
  SomState,
  SomParams,
  SomVerbs
>({
  name: 'manifold/som',
  kind: 'transformer',
  defaults: { rows: 8, cols: 8, epochs: 20, learningRate: 0.5, sigma: null },
  hyperSpace: (h) =>
    dict({
      rows: int(2, 30, { default: h.rows, label: 'rows' }),
      cols: int(2, 30, { default: h.cols, label: 'columns' }),
      epochs: int(1, 200, { default: h.epochs, label: 'epochs' }),
      learningRate: real(0.01, 1, { default: h.learningRate, label: 'learning rate' }),
    }),
  algorithm: (data, h, key) => {
    const m = matrixView(data.x, 'som')
    const units = h.rows * h.cols
    const picks = Array.from(subsample(key, m.n, Math.min(units, m.n)).data)
    const init = Array.from({ length: units }, (_, u) => {
      const i = picks[u % picks.length] ?? 0
      return Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
    })
    return somAlgorithm(data.x, init, [h.rows, h.cols], {
      epochs: h.epochs,
      learningRate: h.learningRate,
      ...(h.sigma === null ? {} : { sigma: h.sigma }),
    })
  },
  maxSteps: (h) => h.epochs,
  params: (s, _data, h) => ({
    weights: s.weights,
    grid: [h.rows, h.cols],
    quantisationError: s.error,
    uMatrix: uMatrix(s.weights, [h.rows, h.cols]),
  }),
  verbs: (p) => {
    const d = p.weights[0]?.length ?? 0
    const unitsOf = (x: TensorLike): number[] => {
      matrixView(x, 'som', { nFeatures: d })
      return bestUnits(x, p.weights).units
    }
    return {
      transform: (x) => {
        const u = unitsOf(x)
        return fromStorage(
          Float64Array.from(u.flatMap((v) => [Math.floor(v / p.grid[1]), v % p.grid[1]])),
          [u.length, 2],
          'float64',
        )
      },
      predict: (x) => {
        const u = unitsOf(x)
        return fromStorage(Int32Array.from(u), [u.length], 'int32')
      },
      nFeaturesIn: d,
    }
  },
})
