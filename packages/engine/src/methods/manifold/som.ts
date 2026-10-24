import { WhiteboxError } from '../../lib/base/index.ts'
import { embeddingSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { geometry } from '../../lib/numerics/index.ts'
import { foldIn, type Key, permutation, subsample } from '../../lib/random/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import type { Algorithm } from '../../lib/trace/index.ts'
import { recordedRun } from './_record.ts'

/** The state of a self-organising map: the node weights in row-major grid order. */
export interface SomState {
  readonly weights: number[][]
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
 * A self-organising map (Kohonen 1982) trained online as a traced algorithm: each step (an epoch) visits the rows in
 * a random order, moving every node towards the row by lr * exp(-g^2 / (2 sigma^2)) for grid distance g to the
 * best-matching unit, with lr and sigma decaying as 1 / (1 + t / (T / 2)) (MiniSom's default).
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial node weights `[rows * cols][d]`.
 * @param grid - Rows and columns of the lattice.
 * @param options - Epochs, the initial learning rate (0.5) and neighbourhood width (max(rows, cols) / 2).
 * @returns The algorithm.
 */
export function somAlgorithm(
  x: TensorLike,
  init: readonly (readonly number[])[],
  grid: readonly [number, number],
  options: { readonly epochs: number; readonly learningRate?: number; readonly sigma?: number },
): Algorithm<SomState> {
  const m = matrixView(x, 'som')
  const [rows, cols] = grid
  const lr0 = options.learningRate ?? 0.5
  const s0 = options.sigma ?? Math.max(rows, cols) / 2
  const coords = Array.from({ length: rows * cols }, (_, u) => [Math.floor(u / cols), u % cols] as const)
  return {
    name: 'som',
    config: { rows, cols, learningRate: lr0, sigma: s0 },
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
    observe: (s) => ({ metrics: { quantisationError: s.error } }),
  }
}

/**
 * The U-matrix of a map: each node's mean distance to its 4-connected grid neighbours.
 *
 * @param weights - Node weights in row-major grid order.
 * @param grid - Rows and columns.
 * @returns The distances `[rows][cols]`.
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

/** Hyperparameters of the self-organising map. */
export type SomHyper = {
  readonly rows?: number
  readonly cols?: number
  readonly epochs?: number
  readonly learningRate?: number
  readonly sigma?: number
  /** Record the map every this many epochs (default 1). */
  readonly recordEvery?: number
}

/** The parameters of a fitted map. */
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
 * A self-organising map: a rows x cols lattice of prototypes trained online, traced epoch by epoch. Its exposition is
 * an `embeddingSteps` record whose positions are each row's best-matching unit on the grid per recorded epoch, with
 * the quantisation error.
 *
 * @param hyper - Lattice size, epochs, learning rate, neighbourhood width and recording.
 * @returns The estimator.
 */
export function selfOrganisingMap(hyper: SomHyper = {}): Estimator<SomHyper, ExplainedFitted<SomParams, SomVerbs>> {
  return defineExplainedEstimator<SomHyper, SomParams, SomVerbs>(
    {
      name: 'som',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          rows: int(2, 30, { default: h.rows ?? 8, label: 'rows' }),
          cols: int(2, 30, { default: h.cols ?? 8, label: 'columns' }),
          epochs: int(1, 200, { default: h.epochs ?? 20, label: 'epochs' }),
          learningRate: real(0.01, 1, { default: h.learningRate ?? 0.5 }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'som')
        const grid: [number, number] = [h.rows ?? 8, h.cols ?? 8]
        const units = grid[0] * grid[1]
        if (units < 1) throw new WhiteboxError('value', 'som: the grid must have nodes')
        const run = fitRunOptions(options, 'som')
        const key = run.key as Key
        const picks = Array.from(subsample(foldIn(key, 0), m.n, Math.min(units, m.n)).data)
        const init = Array.from({ length: units }, (_, u) => {
          const i = picks[u % picks.length] ?? 0
          return Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
        })
        const toGrid = (w: number[][]): number[][] =>
          bestUnits(data.x, w).units.map((u) => [Math.floor(u / grid[1]), u % grid[1]])
        const epochs = h.epochs ?? 20
        const out = recordedRun(
          somAlgorithm(data.x, init, grid, {
            epochs,
            ...(h.learningRate === undefined ? {} : { learningRate: h.learningRate }),
            ...(h.sigma === undefined ? {} : { sigma: h.sigma }),
          }),
          { ...run, key: foldIn(key, 1), maxSteps: epochs },
          h.recordEvery ?? 1,
          (s) => toGrid(s.weights),
          (s) => s.error,
        )
        const weights = out.state.weights
        return {
          params: { weights, grid, quantisationError: out.state.error, uMatrix: uMatrix(weights, grid) },
          exposition: [
            embeddingSteps({
              method: 'som',
              positions: out.positions,
              iterations: out.iterations,
              objective: out.objective,
            }),
          ],
        }
      },
      methods: (p) => {
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
    },
    hyper,
  )
}

/** The self-organising map as a class. */
export class SelfOrganisingMap extends EstimatorClass<SomHyper, ExplainedFitted<SomParams, SomVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link selfOrganisingMap}.
   */
  constructor(hyper: SomHyper = {}) {
    super(selfOrganisingMap(hyper))
  }
}
