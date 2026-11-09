/**
 * Gaussian graphical models with a known sparse precision matrix, for graph-estimation work (the graphical lasso and its
 * penalty selection, neighbourhood selection, nonparanormal estimators): a band, hub or random (Erdos-Renyi) graph with
 * chosen partial correlations on its edges, a precision matrix that is exactly sparse on it and a covariance with a
 * unit diagonal, and samples from it, optionally pushed through monotone marginal transforms (a nonparanormal sample
 * has the same graph). The graph, the precision, the partial correlations and the latent Gaussian rows are the truth.
 *
 * @index module Gaussian graphical models with a known sparse precision matrix, optionally nonparanormal.
 */

import { OmaiError } from '../base/index.ts'
import { type DatasetGenerator, type RegressionData, xy } from '../data/index.ts'
import { cholesky, inv } from '../numerics/linalg/index.ts'
import { foldIn, type Key, key as makeKey, normal, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { toArray } from '../tensor/index.ts'
import { matrix, stamp } from './_util.ts'

/** The shape of the graph of a {@link makeGaussianGraph} model. */
export type GraphStructure = 'band' | 'hub' | 'random'

/** Options of {@link makeGaussianGraph}. */
export interface GaussianGraphOptions {
  /** The graph (default `'random'`). */
  readonly structure?: GraphStructure
  /** For `'band'`: nodes within this index distance are joined (default 1, a chain). */
  readonly bandwidth?: number
  /** For `'hub'`: the number of hubs, each joined to its share of the other nodes (default 3). */
  readonly hubs?: number
  /** For `'random'`: the probability of each edge (default 0.1). */
  readonly edgeProbability?: number
  /** The size of the partial correlations on edges, before the positive-definiteness rescaling (default 0.3). */
  readonly strength?: number
}

/** A Gaussian graphical model: the graph and the matrices that are exactly sparse on it. */
export interface GaussianGraph {
  /** The precision matrix `[p][p]`, zero off the graph. */
  readonly precision: number[][]
  /** The covariance matrix `[p][p]`, with a unit diagonal. */
  readonly covariance: number[][]
  /** The partial correlations `[p][p]`: zero off the graph, one on the diagonal. */
  readonly partialCorrelations: number[][]
  /** Whether each pair is joined `[p][p]` (false on the diagonal). */
  readonly adjacency: boolean[][]
  /** The edges `[i, j]` with `i < j`. */
  readonly edges: readonly (readonly [number, number])[]
}

/**
 * A Gaussian graphical model on `p` variables: edges of a band, hub or random graph carry partial correlations of
 * random sign and magnitude between 0.75 and 1 times `strength`, rescaled together so that the matrix `I - P` of
 * partial correlations is diagonally dominant (hence the precision positive definite). The precision is that matrix
 * scaled so that the covariance has a unit diagonal, so it is exactly zero off the graph.
 *
 * @param key - The random key (the graph and the signs depend on it alone).
 * @param p - The number of variables (at least 2).
 * @param options - The graph structure, its size parameter and the strength of the edges.
 * @returns The precision, covariance, partial correlations, adjacency and edges.
 * @throws OmaiError with code `value` for fewer than two variables.
 * @example
 * ```ts
 * import { makeGaussianGraph } from 'omai-engine/datasets'
 * import { key } from 'omai-engine/random'
 * const g = makeGaussianGraph(key(0), 5, { structure: 'band', bandwidth: 1 })
 * g.edges.map(([a, b]) => b - a) // => [1, 1, 1, 1]
 * ```
 */
export function makeGaussianGraph(key: Key, p: number, options: GaussianGraphOptions = {}): GaussianGraph {
  if (!Number.isInteger(p) || p < 2) throw new OmaiError('value', 'makeGaussianGraph: need at least two variables')
  const structure = options.structure ?? 'random'
  const strength = options.strength ?? 0.3
  const draws = uniform(foldIn(key, 0), [p, p]).data
  const signs = uniform(foldIn(key, 1), [p, p]).data
  const sizes = uniform(foldIn(key, 2), [p, p]).data
  const adjacency = Array.from({ length: p }, () => new Array<boolean>(p).fill(false))
  const join = (a: number, b: number): void => {
    if (a === b) return
    ;(adjacency[a] as boolean[])[b] = true
    ;(adjacency[b] as boolean[])[a] = true
  }
  if (structure === 'band') {
    const w = Math.max(1, Math.floor(options.bandwidth ?? 1))
    for (let a = 0; a < p; a++) for (let b = a + 1; b <= Math.min(p - 1, a + w); b++) join(a, b)
  } else if (structure === 'hub') {
    const hubs = Math.min(p - 1, Math.max(1, Math.floor(options.hubs ?? 3)))
    for (let a = hubs; a < p; a++) join(a % hubs, a)
  } else {
    const prob = options.edgeProbability ?? 0.1
    for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) if ((draws[a * p + b] ?? 1) < prob) join(a, b)
  }
  const partial = Array.from({ length: p }, () => new Array<number>(p).fill(0))
  for (let a = 0; a < p; a++)
    for (let b = a + 1; b < p; b++)
      if (adjacency[a]?.[b] === true) {
        const v = ((signs[a * p + b] ?? 0) < 0.5 ? -1 : 1) * strength * (0.75 + 0.25 * (sizes[a * p + b] ?? 0))
        ;(partial[a] as number[])[b] = v
        ;(partial[b] as number[])[a] = v
      }
  const rowSum = Math.max(...partial.map((r) => r.reduce((s, v) => s + Math.abs(v), 0)))
  const shrink = Math.max(1, rowSum / 0.9)
  const scaled = partial.map((r) => r.map((v) => v / shrink))
  // theta0 = I - P is the precision of the standardised model; rescale it to a unit-diagonal covariance.
  const theta0 = scaled.map((r, a) => r.map((v, b) => (a === b ? 1 : 0 - v)))
  const sigma0 = toArray(inv(theta0)) as number[][]
  const d = sigma0.map((r, a) => Math.sqrt(r[a] ?? 1))
  const covariance = sigma0.map((r, a) => r.map((v, b) => v / ((d[a] ?? 1) * (d[b] ?? 1))))
  const precision = theta0.map((r, a) => r.map((v, b) => v * (d[a] ?? 1) * (d[b] ?? 1) + 0))
  const edges: [number, number][] = []
  for (let a = 0; a < p; a++) for (let b = a + 1; b < p; b++) if (adjacency[a]?.[b] === true) edges.push([a, b])
  return {
    precision,
    covariance,
    partialCorrelations: scaled.map((r, a) => r.map((v, b) => (a === b ? 1 : v))),
    adjacency,
    edges,
  }
}

/** How the latent Gaussian variables are observed: unchanged, or through a monotone marginal transform. */
export type GraphMarginal = 'gaussian' | 'cubic' | 'exp'

/**
 * Samples from a Gaussian graphical model with a planted sparse precision matrix (see {@link makeGaussianGraph}),
 * optionally observed through a monotone marginal transform per variable (`cubic`: heavy tails; `exp`: log-normal): a
 * nonparanormal sample whose graph is still the latent Gaussian's. The graph depends on `graphSeed` alone, so samples
 * of different sizes (and keys) come from the same model. Features are the observed variables and targets the latent
 * Gaussian ones. The truth has the `precision`, `covariance`, `partialCorrelations`, `adjacency` and `edges` of the
 * latent model, the `structure` and the `marginal`.
 *
 * @example
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { gaussianGraph } from 'omai-engine/datasets'
 * const { data, truth } = generate(gaussianGraph, { knobs: { n: 30, p: 4, structure: 'band' } })
 * [data.x.shape, (truth.extra?.['edges'] as unknown[]).length] // => [[30, 4], 3]
 * ```
 */
export const gaussianGraph: DatasetGenerator<RegressionData> = {
  name: 'gaussianGraph',
  kind: 'regression',
  description:
    'Gaussian samples with a planted sparse precision matrix (band, hub or random graph), optionally nonparanormal.',
  knobs: dict({
    n: int(2, 10_000_000, { label: 'samples', description: 'The number of observations.', default: 200 }),
    p: int(2, 500, { label: 'variables', description: 'The number of variables (nodes of the graph).', default: 10 }),
    structure: choice(['band', 'hub', 'random'], {
      label: 'graph',
      description: 'The shape of the graph: a band, hubs or random (Erdos-Renyi) edges.',
      default: 'random',
    }),
    bandwidth: int(1, 50, { label: 'bandwidth', description: 'For a band: the index distance joined.', default: 1 }),
    hubs: int(1, 50, { label: 'hubs', description: 'For hubs: how many.', default: 3 }),
    edgeProbability: real(0, 1, {
      label: 'edge probability',
      description: 'For a random graph: the probability of each edge.',
      default: 0.1,
    }),
    strength: real(0.01, 0.95, {
      label: 'edge strength',
      description: 'The size of the partial correlations on the edges.',
      default: 0.3,
    }),
    marginal: choice(['gaussian', 'cubic', 'exp'], {
      label: 'marginal transform',
      description: 'A monotone transform of each variable: cubic (heavy tails) or exponential (log-normal).',
      default: 'gaussian',
    }),
    graphSeed: int(0, 1_000_000, {
      label: 'graph seed',
      description: 'Seeds the graph, so samples with different keys share it.',
      default: 0,
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const p = knobs['p'] as number
    const marginal = knobs['marginal'] as GraphMarginal
    const graph = makeGaussianGraph(makeKey(knobs['graphSeed'] as number), p, {
      structure: knobs['structure'] as GraphStructure,
      bandwidth: knobs['bandwidth'] as number,
      hubs: knobs['hubs'] as number,
      edgeProbability: knobs['edgeProbability'] as number,
      strength: knobs['strength'] as number,
    })
    const l = toArray(cholesky(graph.covariance)) as number[][]
    const z = normal(foldIn(k, 3), [n, p]).data
    const latent = new Float64Array(n * p)
    for (let i = 0; i < n; i++)
      for (let a = 0; a < p; a++) {
        let v = 0
        for (let b = 0; b <= a; b++) v += (l[a]?.[b] ?? 0) * (z[i * p + b] ?? 0)
        latent[i * p + a] = v
      }
    const observe = (v: number): number => (marginal === 'cubic' ? v * v * v : marginal === 'exp' ? Math.exp(v) : v)
    return {
      data: xy(matrix(Float64Array.from(latent, observe), n, p), matrix(latent, n, p), {
        ...stamp('gaussianGraph', gaussianGraph.description, knobs),
      }),
      truth: {
        extra: {
          precision: graph.precision,
          covariance: graph.covariance,
          partialCorrelations: graph.partialCorrelations,
          adjacency: graph.adjacency,
          edges: graph.edges,
          structure: knobs['structure'],
          marginal,
        },
      },
    }
  },
}
