/**
 * UMAP (McInnes, Healy and Melville, 2018; as in umap-learn): a fuzzy graph of each row's nearest neighbours is laid
 * out in low dimensions by stochastic gradient descent on a fuzzy cross-entropy, pulling neighbours together and
 * pushing negative samples apart, from a spectral start. The layout is traced epoch by epoch, and `umapSession` sets a
 * run up for live stepping.
 */

import { MdmlError } from '../../lib/base/index.ts'
import { embeddingSteps } from '../../lib/exposition/index.ts'
import { spectralEmbedding as graphSpectralEmbedding } from '../../lib/graph/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  fitRunOptions,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { curveFit } from '../../lib/optim/index.ts'
import { foldIn, type Key, uniform } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { add, div, mul, pow, sliceAxis, type TensorLike, toArray, toFlat, type Value } from '../../lib/tensor/index.ts'
import type { Algorithm } from '../../lib/trace/index.ts'
import { recordedRun } from './_record.ts'
import { neighbourEdges } from './isomap.ts'
import type { EmbeddingParams, FittedEmbedding } from './mds.ts'

/**
 * UMAP's curve parameters a and b (umap-learn's `find_ab_params`): the least-squares fit of 1 / (1 + a d^(2b)) to 1
 * below `minDist` and exp(-(d - minDist) / spread) beyond, by `optim`'s curve fit.
 *
 * @param spread - The scale of the embedded points (default 1).
 * @param minDist - The closest embedded distance (default 0.1).
 * @returns a and b.
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
 * @returns The membership matrix `[n][n]` and each row's sigma and rho.
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

/** The state of UMAP's layout optimisation. */
export interface UmapState {
  readonly y: number[][]
  readonly nextSample: number[]
  readonly nextNegative: number[]
  readonly crossEntropy: number
}

/**
 * UMAP's fuzzy cross-entropy between the graph memberships and the embedding's 1 / (1 + a d^(2b)).
 *
 * @param weights - Memberships `[n][n]`.
 * @param y - The embedding.
 * @param a - Curve parameter a.
 * @param b - Curve parameter b.
 * @returns The cross-entropy over all pairs.
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

/**
 * UMAP's layout by stochastic gradient descent (umap-learn's `optimize_layout_euclidean`) as a traced algorithm:
 * each epoch samples every edge in proportion to its membership, pulls its ends together, and pushes the head away
 * from `negativeSampleRate` random points, with a learning rate decaying linearly to zero.
 *
 * @param weights - Memberships `[n][n]`.
 * @param init - The initial embedding.
 * @param options - Epochs, curve parameters, negative sample rate (5) and initial learning rate (1).
 * @returns The algorithm.
 */
export function umapLayoutAlgorithm(
  weights: readonly (readonly number[])[],
  init: readonly (readonly number[])[],
  options: {
    readonly nEpochs: number
    readonly a: number
    readonly b: number
    readonly negativeSampleRate?: number
    readonly learningRate?: number
  },
): Algorithm<UmapState> {
  const n = init.length
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
    name: 'umap-layout',
    config: { nEpochs, a, b, negativeSampleRate: rate },
    init: () => ({
      y: init.map((r) => [...r]),
      nextSample: [...perSample],
      nextNegative: perSample.map((v) => v / rate),
      crossEntropy: fuzzyCrossEntropy(weights, init, a, b),
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
          const k = Math.min(n - 1, Math.floor((u[draw++ % u.length] ?? 0) * n))
          if (k === i) continue
          const o = y[k] as number[]
          const dk = cur.reduce((acc, v, q) => acc + (v - (o[q] ?? 0)) ** 2, 0)
          const rep = dk > 0 ? (2 * b) / ((0.001 + dk) * (a * dk ** b + 1)) : 0
          for (let q = 0; q < cur.length; q++) {
            const g = rep > 0 ? clip(rep * ((cur[q] ?? 0) - (o[q] ?? 0))) : 4
            cur[q] = (cur[q] ?? 0) + g * alpha
          }
        }
        nextNegative[e] = (nextNegative[e] ?? 0) + negatives * perNeg
      }
      return { y, nextSample, nextNegative, crossEntropy: fuzzyCrossEntropy(weights, y, a, b) }
    },
    observe: (s) => ({ metrics: { crossEntropy: s.crossEntropy } }),
  }
}

/** Hyperparameters of UMAP. */
export type UmapHyper = {
  readonly nNeighbors?: number
  readonly nComponents?: number
  readonly minDist?: number
  readonly spread?: number
  readonly nEpochs?: number
  readonly negativeSampleRate?: number
  readonly learningRate?: number
  /** Spectral initialisation of the fuzzy graph (default) or uniform random. */
  readonly init?: 'spectral' | 'random'
  /** Record the embedding every this many epochs (default 10). */
  readonly recordEvery?: number
}

/** A UMAP run set up and ready to step: what {@link umap} fits, for pages that run it live. */
export interface UmapSession {
  readonly algorithm: Algorithm<UmapState>
  /** The epoch budget (`nEpochs`). */
  readonly maxSteps: number
  /** The fuzzy graph's memberships and each row's local scale. */
  readonly weights: readonly (readonly number[])[]
  readonly sigmas: readonly number[]
}

/**
 * Set up UMAP without running it: the fuzzy simplicial set, the initial layout (spectral or random, scaled to
 * [0, 10] with a little noise) and the layout-optimisation algorithm {@link umap} runs. Step it with `runAlgorithm`,
 * or live frame by frame (render's `useLiveRun`), and read each state's positions (`state.y`) and cross-entropy
 * (`state.crossEntropy`).
 *
 * @param data - The rows to embed.
 * @param hyper - As {@link umap}.
 * @param key - The key for random initialisation and the jitter.
 * @returns The algorithm, the epoch budget, and the graph and local scales.
 */
export function umapSession(data: LearnData, hyper: UmapHyper, key: Key): UmapSession {
  const m = matrixView(data.x, 'umap', { minSamples: 4 })
  const k = hyper.nComponents ?? 2
  const nn = Math.min(hyper.nNeighbors ?? 15, m.n)
  const { weights, sigmas } = fuzzySimplicialSet(data.x, nn)
  let start: number[][]
  if ((hyper.init ?? 'spectral') === 'spectral') {
    start = toArray(graphSpectralEmbedding(weights, k, { dropFirst: true })) as number[][]
  } else start = toArray(uniform(foldIn(key, 0), [m.n, k], { low: -10, high: 10 })) as number[][]
  const noise = toFlat(uniform(foldIn(key, 2), [m.n * k], { low: -1e-4, high: 1e-4 }))
  const lo = Array.from({ length: k }, (_, c) => Math.min(...start.map((r) => r[c] ?? 0)))
  const hi = Array.from({ length: k }, (_, c) => Math.max(...start.map((r) => r[c] ?? 0)))
  start = start.map((r, i) =>
    r.map((v, c) => (10 * (v - (lo[c] ?? 0))) / Math.max((hi[c] ?? 1) - (lo[c] ?? 0), 1e-12) + (noise[i * k + c] ?? 0)),
  )
  const { a, b } = findAbParams(hyper.spread ?? 1, hyper.minDist ?? 0.1)
  const nEpochs = hyper.nEpochs ?? 200
  return {
    algorithm: umapLayoutAlgorithm(weights, start, {
      nEpochs,
      a,
      b,
      ...(hyper.negativeSampleRate === undefined ? {} : { negativeSampleRate: hyper.negativeSampleRate }),
      ...(hyper.learningRate === undefined ? {} : { learningRate: hyper.learningRate }),
    }),
    maxSteps: nEpochs,
    weights,
    sigmas,
  }
}

/**
 * UMAP (McInnes, Healy and Melville 2018; umap-learn): the fuzzy simplicial set of the k-nearest-neighbour graph,
 * spectral initialisation (`graph`'s spectral embedding) and the traced SGD layout with negative sampling. Its
 * exposition is an `embeddingSteps` record: positions every few epochs, the fuzzy cross-entropy, the membership graph
 * and each row's sigma.
 *
 * @param hyper - Neighbours, dimension, min_dist and spread, epochs, sampling, learning rate, init and recording.
 * @returns The estimator.
 */
export function umap(hyper: UmapHyper = {}): Estimator<UmapHyper, FittedEmbedding> {
  return defineExplainedEstimator<UmapHyper, EmbeddingParams, { readonly nFeaturesIn: number }>(
    {
      name: 'umap',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nNeighbors: int(2, 100, { default: h.nNeighbors ?? 15, label: 'neighbours' }),
          minDist: real(0, 0.99, { default: h.minDist ?? 0.1, label: 'minimum distance' }),
          nEpochs: int(10, 1000, { default: h.nEpochs ?? 200, label: 'epochs' }),
          init: choice(['spectral', 'random'], {
            label: 'initialisation',
            description: 'Start from the spectral embedding of the graph or from random positions.',
            default: h.init ?? 'spectral',
          }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'umap', { minSamples: 4 })
        const run = fitRunOptions(options, 'umap')
        const key = run.key as Key
        const session = umapSession(data, h, key)
        const { weights, sigmas } = session
        const out = recordedRun(
          session.algorithm,
          { ...run, key: foldIn(key, 1), maxSteps: session.maxSteps },
          h.recordEvery ?? 10,
          (s) => s.y,
          (s) => s.crossEntropy,
        )
        const graph = weights.flatMap((r, i) =>
          r.flatMap((v, j): [number, number, number][] => (j > i && v > 0 ? [[i, j, v]] : [])),
        )
        return {
          params: { embedding: out.state.y, objective: out.state.crossEntropy, nIter: out.steps, nFeaturesIn: m.d },
          exposition: [
            embeddingSteps({
              method: 'umap',
              positions: out.positions,
              iterations: out.iterations,
              objective: out.objective,
              graph,
              bandwidths: sigmas,
            }),
          ],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}

/** UMAP as a class. */
export class UMAP extends EstimatorClass<UmapHyper, FittedEmbedding> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link umap}.
   */
  constructor(hyper: UmapHyper = {}) {
    super(umap(hyper))
  }
}
