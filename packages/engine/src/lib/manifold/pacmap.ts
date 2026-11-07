/**
 * PaCMAP (Wang, Huang, Rudin and Shaposhnik 2021; the `pacmap` package): a layout from three kinds of pairs, each with a
 * bounded loss, so that local and global structure are kept together. Near pairs are each row's nearest neighbours by a
 * locally scaled distance, mid-near pairs are the second closest of six random rows, further pairs are random rows that
 * are not neighbours. The weights of the three losses follow a three-phase schedule (global structure first, then a
 * balance, then local detail), and the layout is Adam on the weighted pair loss, one iteration per step. The loss at
 * fixed weights is a smooth problem of the layout (`pacmapProblem`), so any solver of the smooth slot can minimise it.
 */

import { MdmlError } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { analyticObjective, type OptimiserShape, optimisers, type SmoothProblem } from '../optim/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { choice, dict, int, real, type SpaceValue } from '../space/index.ts'
import { asTensor, primalOf, type TensorLike, toArray, toFlat, type Value } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  drawIndices,
  type EmbeddingParams,
  type EmbeddingVerbs,
  embeddingVerbs,
  pcaStart,
  scaledNeighbours,
  trainingRows,
} from './_support.ts'

/** A pair of row indices. */
export type PacmapPair = readonly [i: number, j: number]

/** The three kinds of pair of PaCMAP. */
export interface PacmapPairs {
  /** Each row with its nearest neighbours by the locally scaled distance. */
  readonly near: PacmapPair[]
  /** Each row with the second closest of six random rows, repeated. */
  readonly mid: PacmapPair[]
  /** Each row with random rows that are neither it nor its neighbours. */
  readonly far: PacmapPair[]
}

/** Options of {@link pacmapPairs}. */
export interface PacmapPairOptions {
  /** Near pairs per row (default 10). */
  readonly nNeighbors?: number
  /** Mid-near pairs per row as a fraction of `nNeighbors` (default 0.5). */
  readonly mnRatio?: number
  /** Further pairs per row as a multiple of `nNeighbors` (default 2). */
  readonly fpRatio?: number
}

/**
 * The three kinds of pairs of PaCMAP. Near pairs: for each row the `nNeighbors` rows of smallest scaled squared distance
 * d_ij^2 / (sigma_i sigma_j) among its `min(max(nNeighbors + 50, 60), n - 1)` nearest (sigma the mean distance to the 4th
 * to 6th nearest). Mid-near pairs: `round(nNeighbors mnRatio)` per row, each the second closest to the row (by Euclidean
 * distance) of six distinct random other rows. Further pairs: `round(nNeighbors fpRatio)` distinct random rows per row
 * that are not it and not one of its near pairs. The pairs are a function of the data and the key.
 *
 * @param x - The rows `[n, d]`.
 * @param key - The key for the random pairs.
 * @param options - Neighbours per row and the mid-near and further ratios.
 * @returns The three lists of pairs.
 * @throws MdmlError with code `value` for fewer than 8 rows, or too few rows for the pairs asked for.
 * @example
 * ```ts
 * import { pacmapPairs } from 'mdml-engine/manifold'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 20 }, (_, i) => [i, (i * 7) % 5])
 * const pairs = pacmapPairs(x, key(0), { nNeighbors: 4 })
 * [pairs.near.length, pairs.mid.length, pairs.far.length] // => [80, 40, 160]
 * ```
 */
export function pacmapPairs(x: TensorLike, key: Key, options: PacmapPairOptions = {}): PacmapPairs {
  const n = matrixView(x, 'pacmapPairs', { minSamples: 8 }).n
  const kNear = options.nNeighbors ?? 10
  const kMid = Math.round(kNear * (options.mnRatio ?? 0.5))
  const kFar = Math.round(kNear * (options.fpRatio ?? 2))
  if (!(Number.isInteger(kNear) && kNear >= 1 && kNear <= n - 1))
    throw new MdmlError('value', `pacmap: nNeighbors must be in 1..${n - 1}`)
  if (n - 1 - kNear < kFar)
    throw new MdmlError('value', `pacmap: need at least ${kNear + kFar + 1} rows for these pairs`)
  const extra = Math.min(Math.max(kNear + 50, 60), n - 1)
  const nb = scaledNeighbours(x, extra)
  const near: PacmapPair[] = []
  const mid: PacmapPair[] = []
  const far: PacmapPair[] = []
  const nearOf: Set<number>[] = []
  for (let i = 0; i < n; i++) {
    const ranked = (nb.nbrs[i] ?? [])
      .map((j, a) => ({ j, s: nb.scaled[i]?.[a] ?? 0, a }))
      .sort((p, q) => p.s - q.s || p.a - q.a)
      .slice(0, kNear)
    nearOf.push(new Set(ranked.map((r) => r.j)))
    for (const r of ranked) near.push([i, r.j])
  }
  const midKey = foldIn(key, 1)
  const farKey = foldIn(key, 2)
  for (let i = 0; i < n; i++) {
    const stream = drawIndices(foldIn(midKey, i), n)
    for (let m = 0; m < kMid; m++) {
      const six: number[] = []
      while (six.length < 6) {
        const c = stream.next().value
        if (c !== i && !six.includes(c)) six.push(c)
      }
      const order = six.map((c, a) => ({ c, d: nb.all[i * n + c] ?? 0, a })).sort((p, q) => p.d - q.d || p.a - q.a)
      mid.push([i, order[1]?.c ?? six[0] ?? 0])
    }
    const fstream = drawIndices(foldIn(farKey, i), n)
    const taken = new Set<number>([i, ...(nearOf[i] ?? [])])
    while (taken.size < 1 + (nearOf[i]?.size ?? 0) + kFar) {
      const c = fstream.next().value
      if (!taken.has(c)) {
        taken.add(c)
        far.push([i, c])
      }
    }
  }
  return { near, mid, far }
}

/** The weights of the three pair losses at one iteration. */
export interface PacmapWeights {
  readonly near: number
  readonly mid: number
  readonly far: number
}

/**
 * PaCMAP's weight schedule. Phase one (the first `phases[0]` iterations) starts with a large mid-near weight
 * `midInit` that falls linearly to 3, near weight 2 and further weight 1: the mid-near pairs lay out the global
 * structure. Phase two (the next `phases[1]`) holds the mid-near weight at 3 and raises the near weight to 3 to
 * balance local and global. Phase three drops the mid-near pairs and the near weight to 1 to refine local structure.
 *
 * @param t - The iteration, from 0.
 * @param phases - The lengths of the three phases (default 100, 100, 250).
 * @param midInit - The initial mid-near weight (default 1000).
 * @returns The three weights.
 * @example
 * ```ts
 * import { pacmapWeights } from 'mdml-engine/manifold'
 * [pacmapWeights(0).mid, pacmapWeights(50).mid, pacmapWeights(150).mid, pacmapWeights(300).mid] // => [1000, 501.5, 3, 0]
 * ```
 */
export function pacmapWeights(
  t: number,
  phases: readonly [number, number, number] = [100, 100, 250],
  midInit = 1000,
): PacmapWeights {
  const [a, b] = phases
  if (t < a) {
    const f = t / a
    return { near: 2, mid: (1 - f) * midInit + f * 3, far: 1 }
  }
  if (t < a + b) return { near: 3, mid: 3, far: 1 }
  return { near: 1, mid: 0, far: 1 }
}

/**
 * PaCMAP's loss and its gradient at a layout. With d_ij = 1 + |y_i - y_j|^2 a near pair costs
 * w_near d / (10 + d), a mid-near pair w_mid d / (10000 + d) and a further pair w_far / (1 + d): attraction that
 * saturates, so no pair dominates, and a repulsion that vanishes for far pairs.
 *
 * @param y - The layout, flat row-major `[n * k]`.
 * @param k - The embedding dimension.
 * @param pairs - The three lists of pairs.
 * @param w - The three weights.
 * @returns The loss (a sum over pairs) and its gradient.
 * @example
 * ```ts
 * import { pacmapLoss } from 'mdml-engine/manifold'
 * // one near pair at squared distance 3 (d = 4): loss 4 / 14
 * pacmapLoss([0, 0, 1, Math.sqrt(2)], 2, { near: [[0, 1]], mid: [], far: [] }, { near: 1, mid: 0, far: 0 }).loss // => 0.2857142857142857
 * ```
 */
export function pacmapLoss(
  y: ArrayLike<number>,
  k: number,
  pairs: PacmapPairs,
  w: PacmapWeights,
): { readonly loss: number; readonly grad: Float64Array } {
  const grad = new Float64Array(y.length)
  let loss = 0
  const run = (list: readonly PacmapPair[], weight: number, kind: 'near' | 'mid' | 'far'): void => {
    if (weight === 0) return
    for (const [i, j] of list) {
      let d = 1
      for (let q = 0; q < k; q++) d += ((y[i * k + q] ?? 0) - (y[j * k + q] ?? 0)) ** 2
      let coef: number
      if (kind === 'near') {
        loss += (weight * d) / (10 + d)
        coef = (weight * 20) / (10 + d) ** 2
      } else if (kind === 'mid') {
        loss += (weight * d) / (10000 + d)
        coef = (weight * 20000) / (10000 + d) ** 2
      } else {
        loss += weight / (1 + d)
        coef = (-weight * 2) / (1 + d) ** 2
      }
      for (let q = 0; q < k; q++) {
        const g = coef * ((y[i * k + q] ?? 0) - (y[j * k + q] ?? 0))
        grad[i * k + q] = (grad[i * k + q] ?? 0) + g
        grad[j * k + q] = (grad[j * k + q] ?? 0) - g
      }
    }
  }
  run(pairs.near, w.near, 'near')
  run(pairs.mid, w.mid, 'mid')
  run(pairs.far, w.far, 'far')
  return { loss, grad }
}

/**
 * The PaCMAP loss at fixed weights as a smooth problem of the flattened layout, for the solver slot: any solver of
 * `smoothSolvers` (L-BFGS, conjugate gradient, the stochastic `gradient` mode with Adam) can minimise it. The three-phase
 * schedule changes the weights between phases; this is the objective of one of them.
 *
 * @param pairs - The pairs.
 * @param w - The weights (one phase).
 * @param y0 - The start layout `[n, k]`.
 * @returns The problem.
 * @example
 * ```ts
 * import { pacmapProblem } from 'mdml-engine/manifold'
 * const p = pacmapProblem({ near: [[0, 1]], mid: [], far: [[0, 2]] }, { near: 1, mid: 0, far: 1 }, [[0], [1], [5]])
 * p.x0 // => [0, 1, 5]
 * ```
 */
export function pacmapProblem(pairs: PacmapPairs, w: PacmapWeights, y0: readonly (readonly number[])[]): SmoothProblem {
  const k = y0[0]?.length ?? 0
  return {
    objective: analyticObjective({
      value: (y) => pacmapLoss(y, k, pairs, w).loss,
      gradient: (y) => pacmapLoss(y, k, pairs, w).grad,
    }),
    x0: y0.flat(),
  }
}

/** The state of a PaCMAP run: the layout, the optimiser's state, and the loss and gradient at the layout. */
export interface PacmapState {
  readonly y: number[][]
  readonly opt: unknown
  /** The loss at the layout with the weights of the step it is the start of (a new phase changes the weights). */
  readonly loss: number
  /** The gradient at the layout, flat. */
  readonly grad: number[]
  /** Its norm. */
  readonly gradNorm: number
}

/** Options of {@link pacmapAlgorithm}. */
export interface PacmapOptions {
  /** The optimiser as a shape over the learning rate (default Adam with eps 1e-7 as the `pacmap` package). */
  readonly optimiser?: OptimiserShape
  /** The learning rate (default 1). */
  readonly learningRate?: number
  /** The lengths of the three phases (default 100, 100, 250). */
  readonly phases?: readonly [number, number, number]
  /** The initial mid-near weight (default 1000). */
  readonly midInit?: number
}

/** The optimiser the `pacmap` package uses: Adam with beta 0.9 and 0.999, eps 1e-7. */
export const pacmapAdam: SpaceValue = { kind: 'adam', b1: 0.9, b2: 0.999, eps: 1e-7, weightDecay: 0 }

/**
 * PaCMAP's layout as a traced {@link Algorithm}, one iteration per step: the loss and gradient at the current weights
 * of the schedule, then the optimiser's update at the learning rate (Adam by default). It observes the loss, the
 * gradient norm, the phase (1 to 3) and the `embedding`.
 *
 * @param pairs - The pairs.
 * @param init - The initial layout `[n, k]`.
 * @param options - The optimiser, learning rate, phase lengths and initial mid-near weight.
 * @returns The algorithm; it completes after the three phases.
 * @example
 * ```ts
 * import { pacmapAlgorithm, pacmapPairs } from 'mdml-engine/manifold'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 16 }, (_, i) => [i % 4, Math.floor(i / 4)])
 * const init = x.map((r) => [r[0] * 0.01, r[1] * 0.01])
 * const alg = pacmapAlgorithm(pacmapPairs(x, key(0), { nNeighbors: 3 }), init, { phases: [5, 5, 10] })
 * runAlgorithm(alg, { maxSteps: 100, key: key(0) }).steps // => 20
 * ```
 */
export function pacmapAlgorithm(
  pairs: PacmapPairs,
  init: readonly (readonly number[])[],
  options: PacmapOptions = {},
): Algorithm<PacmapState> {
  const n = init.length
  const k = init[0]?.length ?? 0
  const lr = options.learningRate ?? 1
  const phases = options.phases ?? [100, 100, 250]
  const midInit = options.midInit ?? 1000
  const total = phases[0] + phases[1] + phases[2]
  const optimiser = (options.optimiser ?? optimisers.make(pacmapAdam))(lr)
  const phaseOf = (t: number): number => (t < phases[0] ? 1 : t < phases[0] + phases[1] ? 2 : 3)
  return {
    name: 'pacmap',
    config: {
      learningRate: lr,
      phases: [...phases],
      midInit,
      optimiser: optimiser.name,
      near: pairs.near.length,
      mid: pairs.mid.length,
      far: pairs.far.length,
    },
    schema: channels({
      metrics: {
        loss: {
          description: 'the weighted pair loss at the iterate (the weights change with the phase)',
          direction: 'lower',
        },
        gradNorm: { description: 'the norm of the gradient at the iterate', direction: 'lower' },
        phase: { description: 'the phase of the weight schedule, 1 to 3' },
      },
      tensors: { embedding: { role: 'state', description: 'the embedding', shape: ['n', 'k'] } },
    }),
    init: () => {
      const flat = Float64Array.from(init.flat())
      const { loss, grad } = pacmapLoss(flat, k, pairs, pacmapWeights(0, phases, midInit))
      return {
        y: init.map((r) => [...r]),
        opt: optimiser.init(asTensor(flat)),
        loss,
        grad: Array.from(grad),
        gradNorm: dnrm2(grad),
      }
    },
    step: (s, ctx) => {
      const flat = Float64Array.from(s.y.flat())
      const { updates, state } = optimiser.update(asTensor(Float64Array.from(s.grad)), s.opt, asTensor(flat))
      const update = toFlat(primalOf(updates as Value))
      const y = s.y.map((r, i) => r.map((v, q) => v + (update[i * k + q] ?? 0)))
      const next = pacmapLoss(y.flat(), k, pairs, pacmapWeights(ctx.t + 1, phases, midInit))
      return { y, opt: state, loss: next.loss, grad: Array.from(next.grad), gradNorm: dnrm2(next.grad) }
    },
    done: (_s, t) => (t >= total ? 'completed' : undefined),
    observe: (s, t) => ({
      metrics: { loss: s.loss, gradNorm: s.gradNorm, phase: phaseOf(t) },
      tensors: { embedding: { role: 'state', value: s.y.flat(), shape: [n, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PacmapState,
  }
}

/** Hyperparameters of {@link pacmap}. */
export interface PacmapHyper {
  readonly nComponents: number
  readonly nNeighbors: number
  readonly mnRatio: number
  readonly fpRatio: number
  /** The lengths of the three phases of the weight schedule. */
  readonly phases: readonly [number, number, number]
  readonly midInit: number
  readonly learningRate: number
  /** The optimiser, a value of `optimisers.space` (default Adam with eps 1e-7). */
  readonly optimiser: SpaceValue
  /** The PCA start scaled to standard deviation 0.01 (default), or Gaussian noise of 1e-2 scale. */
  readonly init: 'pca' | 'random'
}

/**
 * PaCMAP (the `pacmap` package) on the iterative base: the near, mid-near and further pairs, a PCA start, then Adam on
 * the weighted pair loss through the three phases of the schedule, one iteration per step. The embedding is on the
 * tape at every step; new rows are placed by their nearest training rows.
 *
 * @example
 * ```ts
 * import { pacmap } from 'mdml-engine/manifold'
 * const x = Array.from({ length: 24 }, (_, i) => [i < 12 ? 0 : 6, (i % 12) * 0.3, (i % 5) * 0.1])
 * pacmap.with({ nNeighbors: 4, phases: [10, 10, 20] }).fit({ x }).params.embedding.length // => 24
 * ```
 */
export const pacmap: IterativeEstimator<PacmapHyper, PacmapState, EmbeddingParams, EmbeddingVerbs> = iterative<
  PacmapHyper,
  PacmapState,
  EmbeddingParams,
  EmbeddingVerbs
>({
  name: 'manifold/pacmap',
  kind: 'transformer',
  defaults: {
    nComponents: 2,
    nNeighbors: 10,
    mnRatio: 0.5,
    fpRatio: 2,
    phases: [100, 100, 250],
    midInit: 1000,
    learningRate: 1,
    optimiser: pacmapAdam,
    init: 'pca',
  },
  hyperSpace: (h) =>
    dict({
      nNeighbors: int(2, 100, { default: h.nNeighbors, label: 'neighbours' }),
      mnRatio: real(0, 2, { default: h.mnRatio, label: 'mid-near ratio' }),
      fpRatio: real(0, 5, { default: h.fpRatio, label: 'further ratio' }),
      learningRate: real(1e-3, 10, { log: true, default: h.learningRate, label: 'learning rate' }),
      optimiser: optimisers.space,
      init: choice(['pca', 'random'] as const, { default: h.init, label: 'initialisation' }),
    }),
  algorithm: (data, h, key) => {
    const m = matrixView(data.x, 'pacmap', { minSamples: 8 })
    const start =
      h.init === 'pca'
        ? pcaStart(data.x, h.nComponents, 0.01)
        : (toArray(normal(foldIn(key, 9), [m.n, h.nComponents])) as number[][]).map((r) => r.map((v) => v * 0.01))
    return pacmapAlgorithm(
      pacmapPairs(data.x, key, { nNeighbors: h.nNeighbors, mnRatio: h.mnRatio, fpRatio: h.fpRatio }),
      start,
      { optimiser: optimisers.make(h.optimiser), learningRate: h.learningRate, phases: h.phases, midInit: h.midInit },
    )
  },
  maxSteps: (h) => h.phases[0] + h.phases[1] + h.phases[2],
  params: (s, data) => ({ embedding: s.y, data: trainingRows(data.x, 'pacmap'), objective: s.loss }),
  verbs: (p) => embeddingVerbs(p, 'pacmap'),
})
