/**
 * UMAP (McInnes, Healy and Melville 2018; as in umap-learn): a fuzzy graph of each row's nearest neighbours is laid out
 * in low dimensions by stochastic gradient descent on a fuzzy cross-entropy, pulling neighbours together and pushing
 * negative samples apart, from a spectral start. The curve parameters a and b come from a least-squares fit, the layout is
 * one epoch per step.
 */

import { MdmlError } from '../base/index.ts'
import { spectralEmbedding as graphSpectralEmbedding } from '../graph/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { curveFit } from '../optim/index.ts'
import { foldIn, type Key, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import {
  add,
  div,
  fromStorage,
  mul,
  pow,
  sliceAxis,
  type TensorLike,
  toArray,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { type EmbeddingParams, type EmbeddingVerbs, embeddingVerbs, trainingRows } from './_support.ts'
import { neighbourEdges } from './isomap.ts'

/**
 * UMAP's curve parameters a and b (umap-learn's `find_ab_params`): the least-squares fit of 1 / (1 + a d^(2b)) to 1
 * below `minDist` and exp(-(d - minDist) / spread) beyond, by `optim`'s curve fit.
 *
 * @param spread - The scale of the embedded points (default 1).
 * @param minDist - The closest embedded distance (default 0.1).
 * @returns a and b.
 * @example
 * ```ts
 * import { findAbParams } from 'mdml-engine/manifold'
 * Math.round(findAbParams(1, 0.1).a * 1000) // => 1577
 * ```
 */
export function findAbParams(spread = 1, minDist = 0.1): { readonly a: number; readonly b: number } {
  const xs = Array.from({ length: 299 }, (_, i) => ((i + 1) * 3 * spread) / 299)
  const ys = xs.map((x) => (x < minDist ? 1 : Math.exp(-(x - minDist) / spread)))
  const model = (x: Value, p: unknown): Value => {
    const a = sliceAxis(p as Value, 0, 0, 1)
    const b = sliceAxis(p as Value, 0, 1, 2)
    return div(1, add(1, mul(a, pow(x, mul(b, 2)))))
  }
  const [a = 1.577, b = 0.895] = toFlat(curveFit(model, xs, ys, [1, 1]).popt as Value)
  return { a, b }
}

/**
 * The fuzzy simplicial set of UMAP (McInnes, Healy and Melville 2018): per row, rho = the distance to its nearest
 * neighbour and sigma found by bisection so that sum_j exp(-(d_ij - rho) / sigma) = log2(k); memberships
 * exp(-(d_ij - rho) / sigma) symmetrised by the fuzzy union a + b - ab.
 *
 * @param x - Points `[n, d]`.
 * @param k - Neighbours (itself included, as umap-learn).
 * @returns The membership matrix `[n, n]` and each row's sigma and rho.
 * @example
 * ```ts
 * import { fuzzySimplicialSet } from 'mdml-engine/manifold'
 * fuzzySimplicialSet([[0], [1], [2], [10]], 2).weights[0]?.[1] // => 1
 * ```
 */
export function fuzzySimplicialSet(
  x: TensorLike,
  k: number,
): { readonly weights: number[][]; readonly sigmas: number[]; readonly rhos: number[] } {
  const n = matrixView(x, 'fuzzySimplicialSet').n
  if (!(Number.isInteger(k) && k >= 2 && k <= n)) throw new MdmlError('value', `umap: nNeighbors must be in 2..${n}`)
  const { edges } = neighbourEdges(x, k - 1)
  const dist: number[][] = Array.from({ length: n }, () => [])
  const nbr: number[][] = Array.from({ length: n }, () => [])
  for (const [i, j, d] of edges) {
    dist[i]?.push(d)
    nbr[i]?.push(j)
  }
  const target = Math.log2(k)
  const meanAll = edges.reduce((a, e) => a + e[2], 0) / Math.max(1, edges.length)
  const rhos: number[] = []
  const sigmas: number[] = []
  const w = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    const ds = dist[i] ?? []
    const rho = ds.find((v) => v > 0) ?? 0
    let lo = 0
    let hi = Number.POSITIVE_INFINITY
    let mid = 1
    for (let it = 0; it < 64; it++) {
      const psum = ds.reduce((a, d) => a + (d - rho > 0 ? Math.exp(-(d - rho) / mid) : 1), 0)
      if (Math.abs(psum - target) < 1e-5) break
      if (psum > target) {
        hi = mid
        mid = (lo + hi) / 2
      } else {
        lo = mid
        mid = hi === Number.POSITIVE_INFINITY ? mid * 2 : (lo + hi) / 2
      }
    }
    const meanI = ds.reduce((a, v) => a + v, 0) / Math.max(1, ds.length)
    const sigma = Math.max(mid, 1e-3 * (rho > 0 ? meanI : meanAll))
    rhos.push(rho)
    sigmas.push(sigma)
    for (const [a, j] of (nbr[i] ?? []).entries()) {
      const d = ds[a] ?? 0
      ;(w[i] as number[])[j] = d - rho <= 0 || sigma === 0 ? 1 : Math.exp(-(d - rho) / sigma)
    }
  }
  return {
    weights: w.map((r, i) => r.map((v, j) => v + (w[j]?.[i] ?? 0) - v * (w[j]?.[i] ?? 0))),
    sigmas,
    rhos,
  }
}

/** The state of UMAP's layout optimisation: the embedding and the schedules of the edge and negative samples. */
export interface UmapState {
  readonly y: number[][]
  readonly nextSample: number[]
  readonly nextNegative: number[]
}

/**
 * UMAP's fuzzy cross-entropy between the graph memberships and the embedding's 1 / (1 + a d^(2b)).
 *
 * @param weights - Memberships `[n, n]`.
 * @param y - The embedding.
 * @param a - Curve parameter a.
 * @param b - Curve parameter b.
 * @returns The cross-entropy over all pairs.
 * @example
 * ```ts
 * import { fuzzyCrossEntropy } from 'mdml-engine/manifold'
 * fuzzyCrossEntropy([[0, 1], [1, 0]], [[0], [0.01]], 1.9, 0.8) < 0.01 // => true
 * ```
 */
export function fuzzyCrossEntropy(
  weights: readonly (readonly number[])[],
  y: readonly (readonly number[])[],
  a: number,
  b: number,
): number {
  let ce = 0
  const eps = 1e-12
  for (let i = 0; i < y.length; i++)
    for (let j = i + 1; j < y.length; j++) {
      const d2 = (y[i] ?? []).reduce((s, v, q) => s + (v - (y[j]?.[q] ?? 0)) ** 2, 0)
      const q = 1 / (1 + a * d2 ** b)
      const p = weights[i]?.[j] ?? 0
      ce -= p * Math.log(Math.max(q, eps)) + (1 - p) * Math.log(Math.max(1 - q, eps))
    }
  return ce
}

const clip = (v: number): number => Math.max(-4, Math.min(4, v))

/** Options of {@link umapAlgorithm}. */
export interface UmapOptions {
  readonly nEpochs: number
  readonly a: number
  readonly b: number
  /** Negative samples per positive one (default 5). */
  readonly negativeSampleRate?: number
  /** The initial learning rate, decaying linearly to zero (default 1). */
  readonly learningRate?: number
}

/**
 * UMAP's layout by stochastic gradient descent (umap-learn's `optimize_layout_euclidean`) as a traced {@link Algorithm},
 * one epoch per step: each epoch samples every edge in proportion to its membership, pulls its ends together, and pushes
 * the head away from `negativeSampleRate` random points, with a learning rate decaying linearly to zero. It observes the
 * fuzzy cross-entropy and the `embedding`.
 *
 * @param weights - Memberships `[n, n]`.
 * @param init - The initial embedding.
 * @param options - Epochs, curve parameters, negative sample rate and initial learning rate.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { umapAlgorithm } from 'mdml-engine/manifold'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const w = [[0, 1, 0], [1, 0, 0.5], [0, 0.5, 0]]
 * const out = runAlgorithm(umapAlgorithm(w, [[0, 0], [5, 5], [9, 0]], { nEpochs: 20, a: 1.9, b: 0.8 }), { maxSteps: 20, key: key(0) })
 * out.state.y.length // => 3
 * ```
 *
 * The optimiser is deliberately not a solver slot. umap-learn's SGD is not a gradient step on one objective: each epoch
 * samples edges by membership and draws fresh negative samples, with per-term clipped gradients and a linearly decaying
 * rate, so the iterate follows a random walk whose stationary layout is what the reference produces. The full
 * cross-entropy is available as `fuzzyCrossEntropy`, but a deterministic `smoothSolvers` mode would minimise a
 * different (all-pairs) objective than the one this samples.
 */
export function umapAlgorithm(
  weights: readonly (readonly number[])[],
  init: readonly (readonly number[])[],
  options: UmapOptions,
): Algorithm<UmapState> {
  const n = init.length
  const k = init[0]?.length ?? 0
  const { a, b, nEpochs } = options
  const rate = options.negativeSampleRate ?? 5
  const alpha0 = options.learningRate ?? 1
  const all: [number, number, number][] = []
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (i !== j && (weights[i]?.[j] ?? 0) > 0) all.push([i, j, weights[i]?.[j] ?? 0])
  const wmax = Math.max(...all.map((e) => e[2]), 0)
  const edges = all.filter((e) => e[2] >= wmax / nEpochs)
  const perSample = edges.map((e) => nEpochs / (nEpochs * (e[2] / wmax)))
  return {
    name: 'umap',
    config: { nEpochs, a, b, negativeSampleRate: rate },
    schema: channels({
      metrics: {
        crossEntropy: {
          description: 'the fuzzy cross-entropy between the graph and the embedding',
          direction: 'lower',
        },
      },
      tensors: { embedding: { role: 'state', description: 'the embedding', shape: ['n', 'k'] } },
    }),
    init: () => ({
      y: init.map((r) => [...r]),
      nextSample: [...perSample],
      nextNegative: perSample.map((v) => v / rate),
    }),
    step: (s, ctx) => {
      const epoch = ctx.t
      const alpha = alpha0 * (1 - epoch / nEpochs)
      const y = s.y.map((r) => [...r])
      const nextSample = [...s.nextSample]
      const nextNegative = [...s.nextNegative]
      const u = toFlat(uniform(ctx.key, [Math.max(1, edges.length * (rate + 1))]))
      let draw = 0
      for (const [e, [i, j]] of edges.entries()) {
        if ((nextSample[e] ?? 0) > epoch + 1) continue
        const cur = y[i] as number[]
        const other = y[j] as number[]
        const d2 = cur.reduce((acc, v, q) => acc + (v - (other[q] ?? 0)) ** 2, 0)
        const coef = d2 > 0 ? (-2 * a * b * d2 ** (b - 1)) / (a * d2 ** b + 1) : 0
        for (let q = 0; q < cur.length; q++) {
          const g = clip(coef * ((cur[q] ?? 0) - (other[q] ?? 0)))
          cur[q] = (cur[q] ?? 0) + g * alpha
          other[q] = (other[q] ?? 0) - g * alpha
        }
        nextSample[e] = (nextSample[e] ?? 0) + (perSample[e] ?? 0)
        const perNeg = (perSample[e] ?? 1) / rate
        const negatives = Math.floor((epoch + 1 - (nextNegative[e] ?? 0)) / perNeg)
        for (let r = 0; r < negatives; r++) {
          const c = Math.min(n - 1, Math.floor((u[draw++ % u.length] ?? 0) * n))
          if (c === i) continue
          const o = y[c] as number[]
          const dk = cur.reduce((acc, v, q) => acc + (v - (o[q] ?? 0)) ** 2, 0)
          const rep = dk > 0 ? (2 * b) / ((0.001 + dk) * (a * dk ** b + 1)) : 0
          for (let q = 0; q < cur.length; q++) {
            const g = rep > 0 ? clip(rep * ((cur[q] ?? 0) - (o[q] ?? 0))) : 4
            cur[q] = (cur[q] ?? 0) + g * alpha
          }
        }
        nextNegative[e] = (nextNegative[e] ?? 0) + negatives * perNeg
      }
      return { y, nextSample, nextNegative }
    },
    done: (_s, t) => (t >= nEpochs ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { crossEntropy: fuzzyCrossEntropy(weights, s.y, a, b) },
      tensors: { embedding: { role: 'state', value: s.y.flat(), shape: [n, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as UmapState,
  }
}

/** Hyperparameters of {@link umap}. */
export interface UmapHyper {
  readonly nNeighbors: number
  readonly nComponents: number
  readonly minDist: number
  readonly spread: number
  readonly nEpochs: number
  readonly negativeSampleRate: number
  readonly learningRate: number
  /** Spectral initialisation of the fuzzy graph, or uniform random. */
  readonly init: 'spectral' | 'random'
}

/**
 * The fuzzy graph and the start of a UMAP run: memberships, each row's local scale, the curve parameters and the initial
 * layout (spectral or random, scaled to [0, 10] with a little noise).
 *
 * @param x - The rows to embed.
 * @param h - The hyperparameters.
 * @param key - The key for random initialisation and the jitter.
 * @returns The graph, scales, curve and start.
 */
export function umapSetup(
  x: TensorLike,
  h: UmapHyper,
  key: Key,
): {
  readonly weights: number[][]
  readonly sigmas: number[]
  readonly start: number[][]
  readonly a: number
  readonly b: number
} {
  const m = matrixView(x, 'umap', { minSamples: 4 })
  const k = h.nComponents
  const { weights, sigmas } = fuzzySimplicialSet(x, Math.min(h.nNeighbors, m.n))
  let start: number[][]
  if (h.init === 'spectral')
    start = toArray(
      graphSpectralEmbedding(fromStorage(Float64Array.from(weights.flat()), [m.n, m.n], 'float64'), k, {
        dropFirst: true,
      }),
    ) as number[][]
  else start = toArray(uniform(foldIn(key, 0), [m.n, k], { low: -10, high: 10 })) as number[][]
  const noise = toFlat(uniform(foldIn(key, 2), [m.n * k], { low: -1e-4, high: 1e-4 }))
  const lo = Array.from({ length: k }, (_, c) => Math.min(...start.map((r) => r[c] ?? 0)))
  const hi = Array.from({ length: k }, (_, c) => Math.max(...start.map((r) => r[c] ?? 0)))
  start = start.map((r, i) =>
    r.map((v, c) => (10 * (v - (lo[c] ?? 0))) / Math.max((hi[c] ?? 1) - (lo[c] ?? 0), 1e-12) + (noise[i * k + c] ?? 0)),
  )
  const { a, b } = findAbParams(h.spread, h.minDist)
  return { weights, sigmas, start, a, b }
}

/**
 * UMAP (umap-learn) on the iterative base: the fuzzy simplicial set of the k-nearest-neighbour graph, spectral
 * initialisation (`graph`'s spectral embedding) and the SGD layout with negative sampling, one epoch per step. New rows
 * are placed by their nearest training rows.
 *
 * @index estimator UMAP embedding.
 * @example
 * ```ts
 * import { umap } from 'mdml-engine/manifold'
 * const x = Array.from({ length: 12 }, (_, i) => [i < 6 ? 0 : 5, (i % 6) * 0.1, i * 0.01])
 * umap.with({ nNeighbors: 4, nEpochs: 20 }).fit({ x }).params.embedding.length // => 12
 * ```
 */
export const umap: IterativeEstimator<UmapHyper, UmapState, EmbeddingParams, EmbeddingVerbs> = iterative<
  UmapHyper,
  UmapState,
  EmbeddingParams,
  EmbeddingVerbs
>({
  name: 'manifold/umap',
  kind: 'transformer',
  defaults: {
    nNeighbors: 15,
    nComponents: 2,
    minDist: 0.1,
    spread: 1,
    nEpochs: 200,
    negativeSampleRate: 5,
    learningRate: 1,
    init: 'spectral',
  },
  hyperSpace: (h) =>
    dict({
      nNeighbors: int(2, 100, { default: h.nNeighbors, label: 'neighbours' }),
      minDist: real(0, 0.99, { default: h.minDist, label: 'minimum distance' }),
      nEpochs: int(10, 1000, { default: h.nEpochs, label: 'epochs' }),
      init: choice(['spectral', 'random'] as const, { default: h.init, label: 'initialisation' }),
    }),
  algorithm: (data, h, key) => {
    const { weights, start, a, b } = umapSetup(data.x, h, foldIn(key, 7))
    return umapAlgorithm(weights, start, {
      nEpochs: h.nEpochs,
      a,
      b,
      negativeSampleRate: h.negativeSampleRate,
      learningRate: h.learningRate,
    })
  },
  maxSteps: (h) => h.nEpochs,
  params: (s, data, h) => {
    const { weights } = umapGraphCache(data.x, h)
    const { a, b } = findAbParams(h.spread, h.minDist)
    return { embedding: s.y, data: trainingRows(data.x, 'umap'), objective: fuzzyCrossEntropy(weights, s.y, a, b) }
  },
  verbs: (p) => embeddingVerbs(p, 'umap'),
})

const graphs = new WeakMap<object, { readonly key: string; readonly weights: number[][] }>()

/** The fuzzy graph of a dataset, kept per data object so that probing many steps does not rebuild it. */
function umapGraphCache(x: TensorLike, h: UmapHyper): { readonly weights: number[][] } {
  const id = `${h.nNeighbors}`
  const hit = typeof x === 'object' && x !== null ? graphs.get(x) : undefined
  if (hit?.key === id) return hit
  const weights = fuzzySimplicialSet(x, Math.min(h.nNeighbors, matrixView(x, 'umap').n)).weights
  if (typeof x === 'object' && x !== null) graphs.set(x, { key: id, weights })
  return { weights }
}
