/**
 * TriMap (Amid and Warmuth 2019; the `trimap` package): a layout from triplets (i, j, k) saying that row i is closer to
 * row j than to row k. The triplets are each row's nearest neighbours as j with random outliers as k, plus a few random
 * triplets ordered by the data, and each carries a weight that grows with how much closer j is than k. The loss is the
 * weighted sum of d_ij / (d_ij + d_ik) with d = 1 + the squared embedded distance, minimised by gradient descent with
 * momentum and delta-bar-delta gains. The loss at fixed triplets is a smooth problem of the layout (`trimapProblem`), so
 * the optimiser is a slot: the delta-bar-delta rule of the reference, or any optimiser of the `optim` table.
 */

import { MdmlError } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { analyticObjective, type OptimiserShape, optimisers, type SmoothProblem } from '../optim/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
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

/** A triplet (i, j, k): row i should be nearer to row j than to row k. */
export type Triplet = readonly [i: number, j: number, k: number]

/** Triplets with their weights. */
export interface TrimapTriplets {
  readonly triplets: Triplet[]
  readonly weights: number[]
}

/** Options of {@link trimapTriplets}. */
export interface TrimapTripletOptions {
  /** Nearest neighbours per row used as j (default 10). */
  readonly nInliers?: number
  /** Outliers k per inlier (default 5). */
  readonly nOutliers?: number
  /** Random triplets per row (default 3). */
  readonly nRandom?: number
  /** The scale of the weights before the tempered logarithm (default 500). */
  readonly weightAdj?: number
  /** The temperature of the logarithm: 1 is the logarithm, below 1 a power (default 0.5). */
  readonly weightTemp?: number
}

/**
 * The tempered logarithm of TriMap's weights: log x at temperature 1, otherwise (x^(1 - t) - 1) / (1 - t), which is
 * concave and increasing on x >= 1 and flattens the weights less than the logarithm for t < 1.
 *
 * @param x - The argument (at least 1 here).
 * @param t - The temperature.
 * @returns The tempered logarithm.
 * @example
 * ```ts
 * import { temperedLog } from 'mdml-engine/manifold'
 * [temperedLog(Math.E, 1), temperedLog(9, 0.5)] // => [1, 4]
 * ```
 */
export function temperedLog(x: number, t: number): number {
  return Math.abs(t - 1) < 1e-2 ? Math.log(x) : (x ** (1 - t) - 1) / (1 - t)
}

/**
 * TriMap's triplets and weights. For each row i, each of its `nInliers` nearest rows j (by the locally scaled distance
 * s_ij = d_ij^2 / (sigma_i sigma_j) of PaCMAP) is paired with `nOutliers` random rows k that are neither i nor among its
 * nearest `min(nInliers + 50, n - 1)` (when fewer than `nOutliers` rows are left, any row that is not i or an inlier);
 * `nRandom` more triplets per row take two random rows ordered so that j is the nearer by s. The weight of a triplet is
 * the similarity ratio p_ij / p_ik = exp(s_ik - s_ij) with p = exp(-s), divided by the largest ratio and then
 * `temperedLog(1 + weightAdj * ratio, weightTemp)`: a triplet where j is much nearer than k pulls harder.
 *
 * @param x - The rows `[n, d]`.
 * @param key - The key for the outliers and the random triplets.
 * @param options - Counts of inliers, outliers and random triplets, and the weight scale and temperature.
 * @returns The triplets and their weights.
 * @throws MdmlError with code `value` for fewer than 8 rows or invalid counts.
 * @example
 * ```ts
 * import { trimapTriplets } from 'mdml-engine/manifold'
 * import { key } from 'mdml-engine/random'
 * const x = Array.from({ length: 20 }, (_, i) => [i, (i * 7) % 5])
 * trimapTriplets(x, key(0), { nInliers: 3, nOutliers: 2, nRandom: 1 }).triplets.length // => 140
 * ```
 */
export function trimapTriplets(x: TensorLike, key: Key, options: TrimapTripletOptions = {}): TrimapTriplets {
  const n = matrixView(x, 'trimapTriplets', { minSamples: 8 }).n
  const nIn = options.nInliers ?? 10
  const nOut = options.nOutliers ?? 5
  const nRand = options.nRandom ?? 3
  const adj = options.weightAdj ?? 500
  const temp = options.weightTemp ?? 0.5
  if (!(Number.isInteger(nIn) && nIn >= 1 && nIn <= n - 2))
    throw new MdmlError('value', `trimap: nInliers must be in 1..${n - 2}`)
  if (!(Number.isInteger(nOut) && nOut >= 0 && Number.isInteger(nRand) && nRand >= 0))
    throw new MdmlError('value', 'trimap: nOutliers and nRandom must be non-negative integers')
  const extra = Math.min(nIn + 50, n - 1)
  const nb = scaledNeighbours(x, extra)
  const s = (i: number, j: number): number => (nb.all[i * n + j] ?? 0) ** 2 / ((nb.sigma[i] ?? 1) * (nb.sigma[j] ?? 1))
  const triplets: Triplet[] = []
  const logRatio: number[] = []
  const outKey = foldIn(key, 1)
  const randKey = foldIn(key, 2)
  for (let i = 0; i < n; i++) {
    const ranked = (nb.nbrs[i] ?? [])
      .map((j, a) => ({ j, s: nb.scaled[i]?.[a] ?? 0, a }))
      .sort((p, q) => p.s - q.s || p.a - q.a)
    const inliers = ranked.slice(0, nIn).map((r) => r.j)
    const wide = n - 1 - extra >= nOut ? new Set(ranked.map((r) => r.j)) : new Set(inliers)
    wide.add(i)
    const stream = drawIndices(foldIn(outKey, i), n)
    for (const j of inliers) {
      const chosen = new Set<number>()
      while (chosen.size < nOut) {
        const k = stream.next().value
        if (!wide.has(k) && !chosen.has(k)) {
          chosen.add(k)
          triplets.push([i, j, k])
          logRatio.push(s(i, k) - s(i, j))
        }
      }
    }
    const rand = drawIndices(foldIn(randKey, i), n)
    for (let r = 0; r < nRand; r++) {
      let a = i
      let b = i
      while (a === i) a = rand.next().value
      while (b === i || b === a) b = rand.next().value
      const [j, k] = s(i, a) <= s(i, b) ? [a, b] : [b, a]
      triplets.push([i, j, k])
      logRatio.push(s(i, k) - s(i, j))
    }
  }
  const top = Math.max(...logRatio)
  const weights = logRatio.map((l) => temperedLog(1 + adj * Math.exp(l - top), temp))
  return { triplets, weights }
}

/**
 * TriMap's loss and gradient at a layout. With d_ij = 1 + |y_i - y_j|^2 a triplet costs w d_ij / (d_ij + d_ik), which
 * is below one half exactly when j is embedded nearer to i than k is. Also counts the violated triplets (d_ij > d_ik).
 *
 * @param y - The layout, flat row-major `[n * k]`.
 * @param k - The embedding dimension.
 * @param triplets - The triplets.
 * @param weights - Their weights.
 * @returns The loss, its gradient and the number of violated triplets.
 * @example
 * ```ts
 * import { trimapLoss } from 'mdml-engine/manifold'
 * // y_i = 0, y_j = 1, y_k = 3: d_ij = 2, d_ik = 10, loss 2 / 12
 * trimapLoss([0, 1, 3], 1, [[0, 1, 2]], [1]).loss // => 0.16666666666666666
 * ```
 */
export function trimapLoss(
  y: ArrayLike<number>,
  k: number,
  triplets: readonly Triplet[],
  weights: readonly number[],
): { readonly loss: number; readonly grad: Float64Array; readonly violations: number } {
  const grad = new Float64Array(y.length)
  let loss = 0
  let violations = 0
  for (const [t, [i, j, o]] of triplets.entries()) {
    let dij = 1
    let dik = 1
    for (let q = 0; q < k; q++) {
      dij += ((y[i * k + q] ?? 0) - (y[j * k + q] ?? 0)) ** 2
      dik += ((y[i * k + q] ?? 0) - (y[o * k + q] ?? 0)) ** 2
    }
    const w = weights[t] ?? 0
    if (dij > dik) violations++
    loss += (w * dij) / (dij + dik)
    const c = (2 * w) / (dij + dik) ** 2
    for (let q = 0; q < k; q++) {
      const yij = (y[i * k + q] ?? 0) - (y[j * k + q] ?? 0)
      const yik = (y[i * k + q] ?? 0) - (y[o * k + q] ?? 0)
      grad[i * k + q] = (grad[i * k + q] ?? 0) + c * (dik * yij - dij * yik)
      grad[j * k + q] = (grad[j * k + q] ?? 0) - c * dik * yij
      grad[o * k + q] = (grad[o * k + q] ?? 0) + c * dij * yik
    }
  }
  return { loss, grad, violations }
}

/**
 * The TriMap loss at fixed triplets as a smooth problem of the flattened layout, for the solver slot: any solver of
 * `smoothSolvers` can minimise it from a start.
 *
 * @param triplets - The triplets.
 * @param weights - Their weights.
 * @param y0 - The start layout `[n, k]`.
 * @returns The problem.
 * @example
 * ```ts
 * import { trimapProblem } from 'mdml-engine/manifold'
 * trimapProblem([[0, 1, 2]], [1], [[0], [1], [3]]).x0 // => [0, 1, 3]
 * ```
 */
export function trimapProblem(
  triplets: readonly Triplet[],
  weights: readonly number[],
  y0: readonly (readonly number[])[],
): SmoothProblem {
  const k = y0[0]?.length ?? 0
  return {
    objective: analyticObjective({
      value: (y) => trimapLoss(y, k, triplets, weights).loss,
      gradient: (y) => trimapLoss(y, k, triplets, weights).grad,
    }),
    x0: y0.flat(),
  }
}

/** The state of a TriMap run. */
export interface TrimapState {
  readonly y: number[][]
  /** The velocity of the momentum update (delta-bar-delta), zero for a slot optimiser. */
  readonly vel: number[][]
  /** The per-coordinate gains of delta-bar-delta. */
  readonly gain: number[][]
  /** The step size, adapted by the bold-driver rule of the reference (delta-bar-delta only). */
  readonly lr: number
  /** The slot optimiser's state, when there is one. */
  readonly opt: unknown
  readonly loss: number
  readonly grad: number[]
  readonly gradNorm: number
  readonly violations: number
}

/** Options of {@link trimapAlgorithm}. */
export interface TrimapOptions {
  /** The step size. For the delta-bar-delta rule it is scaled by n / (the number of triplets) as the reference does. */
  readonly learningRate?: number
  /** An optimiser of the `optim` table in place of the delta-bar-delta rule. */
  readonly optimiser?: OptimiserShape
  /** The iterations (default 400). */
  readonly nIters?: number
  /** The iteration at which the momentum rises from 0.5 to 0.8 (default 150). */
  readonly switchIter?: number
}

/**
 * TriMap's descent as a traced {@link Algorithm}, one iteration per step. The default is the reference's: momentum 0.5
 * (0.8 after `switchIter`) with Jacobs' delta-bar-delta gains (a coordinate's gain grows by 0.2 while its gradient and
 * its velocity point opposite ways, that is the descent continues, and shrinks by a factor 0.8 when they agree, floored at
 * 0.01), and a step size that shrinks by 0.9 when the loss rose and grows by 1.01 otherwise. With an `optimiser` of the
 * `optim` table that optimiser updates the layout instead. It observes the loss, the number of violated triplets, the
 * gradient norm, the step size and the `embedding`.
 *
 * @param triplets - The triplets.
 * @param weights - Their weights.
 * @param init - The initial layout `[n, k]`.
 * @param options - Step size, optimiser, iterations and the momentum switch.
 * @returns The algorithm; it completes after `nIters` steps.
 * @example
 * ```ts
 * import { trimapAlgorithm, trimapTriplets } from 'mdml-engine/manifold'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = Array.from({ length: 16 }, (_, i) => [i % 4, Math.floor(i / 4)])
 * const t = trimapTriplets(x, key(0), { nInliers: 3, nOutliers: 2, nRandom: 1 })
 * const init = x.map((r) => [(r[0] ?? 0) * 0.01, (r[1] ?? 0) * 0.01])
 * runAlgorithm(trimapAlgorithm(t.triplets, t.weights, init, { nIters: 10 }), { maxSteps: 100, key: key(0) }).steps // => 10
 * ```
 */
export function trimapAlgorithm(
  triplets: readonly Triplet[],
  weights: readonly number[],
  init: readonly (readonly number[])[],
  options: TrimapOptions = {},
): Algorithm<TrimapState> {
  const n = init.length
  const k = init[0]?.length ?? 0
  const nIters = options.nIters ?? 400
  const switchIter = options.switchIter ?? 150
  const slot = options.optimiser
  const lr0 =
    (options.learningRate ?? (slot === undefined ? 1000 : 0.1)) *
    (slot === undefined ? n / Math.max(1, triplets.length) : 1)
  const optimiser = slot?.(lr0)
  const zeros = (): number[][] => init.map((r) => r.map(() => 0))
  const evaluate = (y: number[][]): { loss: number; grad: number[]; gradNorm: number; violations: number } => {
    const r = trimapLoss(y.flat(), k, triplets, weights)
    return { loss: r.loss, grad: Array.from(r.grad), gradNorm: Math.hypot(...r.grad), violations: r.violations }
  }
  return {
    name: 'trimap',
    config: {
      learningRate: lr0,
      optimiser: optimiser?.name ?? 'delta-bar-delta',
      nIters,
      switchIter,
      triplets: triplets.length,
    },
    schema: channels({
      metrics: {
        loss: { description: 'the weighted triplet loss', direction: 'lower' },
        violations: { description: 'triplets whose embedded order is wrong', direction: 'lower' },
        gradNorm: { description: 'the norm of the gradient', direction: 'lower' },
        lr: { description: 'the step size' },
      },
      tensors: { embedding: { role: 'state', description: 'the embedding', shape: ['n', 'k'] } },
    }),
    init: () => {
      const y = init.map((r) => [...r])
      return {
        y,
        vel: zeros(),
        gain: init.map((r) => r.map(() => 1)),
        lr: lr0,
        opt: optimiser?.init(asTensor(Float64Array.from(init.flat()))),
        ...evaluate(y),
      }
    },
    step: (s, ctx) => {
      if (optimiser !== undefined) {
        const { updates, state } = optimiser.update(
          asTensor(Float64Array.from(s.grad)),
          s.opt,
          asTensor(Float64Array.from(s.y.flat())),
        )
        const update = toFlat(primalOf(updates as Value))
        const y = s.y.map((r, i) => r.map((v, q) => v + (update[i * k + q] ?? 0)))
        return { ...s, y, opt: state, ...evaluate(y) }
      }
      const momentum = ctx.t < switchIter ? 0.5 : 0.8
      const gain = s.gain.map((r, i) =>
        r.map((g, q) => {
          const opposite = (s.vel[i]?.[q] ?? 0) > 0 !== (s.grad[i * k + q] ?? 0) > 0
          return Math.max(opposite ? g + 0.2 : g * 0.8, 0.01)
        }),
      )
      const vel = s.vel.map((r, i) =>
        r.map((v, q) => momentum * v - s.lr * (gain[i]?.[q] ?? 1) * (s.grad[i * k + q] ?? 0)),
      )
      const y = s.y.map((r, i) => r.map((v, q) => v + (vel[i]?.[q] ?? 0)))
      const next = evaluate(y)
      return { ...s, y, vel, gain, lr: next.loss > s.loss + 1e-7 ? s.lr * 0.9 : s.lr * 1.01, ...next }
    },
    done: (_s, t) => (t >= nIters ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { loss: s.loss, violations: s.violations, gradNorm: s.gradNorm, lr: s.lr },
      tensors: { embedding: { role: 'state', value: s.y.flat(), shape: [n, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TrimapState,
  }
}

/** Hyperparameters of {@link trimap}. */
export interface TrimapHyper {
  readonly nComponents: number
  readonly nInliers: number
  readonly nOutliers: number
  readonly nRandom: number
  readonly weightAdj: number
  readonly weightTemp: number
  /** The step size (`null`: 1000 for delta-bar-delta, 0.1 for the others). */
  readonly learningRate: number | null
  readonly nIters: number
  /** `'dbd'`, the reference's delta-bar-delta rule, or an optimiser of `optimisers` by name (with its defaults). */
  readonly optimiser: string
  readonly init: 'pca' | 'random'
}

/**
 * TriMap (the `trimap` package) on the iterative base: the triplets with their weights, a PCA start, then
 * delta-bar-delta gradient descent on the weighted triplet loss, one iteration per step; the optimiser is a slot
 * (`optimiser`: `'dbd'` or any of the `optimisers` table). The embedding is on the tape at every step; new rows are
 * placed by their nearest training rows.
 *
 * @example
 * ```ts
 * import { trimap } from 'mdml-engine/manifold'
 * const x = Array.from({ length: 24 }, (_, i) => [i < 12 ? 0 : 6, (i % 12) * 0.3, (i % 5) * 0.1])
 * trimap.with({ nInliers: 3, nOutliers: 2, nIters: 20 }).fit({ x }).params.embedding.length // => 24
 * ```
 */
export const trimap: IterativeEstimator<TrimapHyper, TrimapState, EmbeddingParams, EmbeddingVerbs> = iterative<
  TrimapHyper,
  TrimapState,
  EmbeddingParams,
  EmbeddingVerbs
>({
  name: 'manifold/trimap',
  kind: 'transformer',
  defaults: {
    nComponents: 2,
    nInliers: 10,
    nOutliers: 5,
    nRandom: 3,
    weightAdj: 500,
    weightTemp: 0.5,
    learningRate: null,
    nIters: 400,
    optimiser: 'dbd',
    init: 'pca',
  },
  hyperSpace: (h) =>
    dict({
      nInliers: int(2, 50, { default: h.nInliers, label: 'inliers' }),
      nOutliers: int(0, 20, { default: h.nOutliers, label: 'outliers per inlier' }),
      nRandom: int(0, 20, { default: h.nRandom, label: 'random triplets' }),
      weightAdj: real(1, 5000, { log: true, default: h.weightAdj, label: 'weight scale' }),
      weightTemp: real(0, 1, { default: h.weightTemp, label: 'weight temperature' }),
      nIters: int(10, 2000, { default: h.nIters, label: 'iterations' }),
      optimiser: choice(['dbd', ...optimisers.names] as const, { default: h.optimiser, label: 'optimiser' }),
      init: choice(['pca', 'random'] as const, { default: h.init, label: 'initialisation' }),
    }),
  algorithm: (data, h, key) => {
    const m = matrixView(data.x, 'trimap', { minSamples: 8 })
    const start =
      h.init === 'pca'
        ? pcaStart(data.x, h.nComponents, 0.01)
        : (toArray(normal(foldIn(key, 9), [m.n, h.nComponents])) as number[][]).map((r) => r.map((v) => v * 0.01))
    const { triplets, weights } = trimapTriplets(data.x, key, {
      nInliers: h.nInliers,
      nOutliers: h.nOutliers,
      nRandom: h.nRandom,
      weightAdj: h.weightAdj,
      weightTemp: h.weightTemp,
    })
    return trimapAlgorithm(triplets, weights, start, {
      ...(h.learningRate === null ? {} : { learningRate: h.learningRate }),
      ...(h.optimiser === 'dbd' ? {} : { optimiser: optimisers.make({ kind: h.optimiser }) }),
      nIters: h.nIters,
    })
  },
  maxSteps: (h) => h.nIters,
  params: (s, data) => ({ embedding: s.y, data: trainingRows(data.x, 'trimap'), objective: s.loss }),
  verbs: (p) => embeddingVerbs(p, 'trimap'),
})
