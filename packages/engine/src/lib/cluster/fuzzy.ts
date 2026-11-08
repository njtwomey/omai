/**
 * Fuzzy c-means (Bezdek 1981): clustering with soft memberships, where every row belongs to each cluster to a degree
 * set by its relative distances and a fuzzifier `m`. The alternating membership and centre updates are one traced
 * algorithm, started from k-means++ centres.
 */

import { maxOf, OmaiError, toRows } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, tensor } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { initialCentres, type Rows, squaredDistancesToCentres } from './kmeans.ts'

/** The state of fuzzy c-means: the centres, the memberships `[n, c]` (flat), the objective and the last centre shift. */
export interface FuzzyState {
  readonly centres: number[][]
  readonly memberships: number[]
  readonly objective: number
  /** The largest distance a centre moved in the last step (a large number before the first). */
  readonly shift: number
}

/**
 * Fuzzy memberships u_ik = 1 / sum_j (d_ik / d_jk)^(2 / (m - 1)), with a point on a centre wholly in it.
 *
 * @param x - Points `[n, d]`.
 * @param centres - Centres `[c, d]`.
 * @param m - The fuzzifier (> 1).
 * @returns Memberships `[n, c]` flat, and the objective sum u^m d^2.
 * @example
 * ```ts
 * import { fuzzyMemberships } from 'omai-engine/cluster'
 * fuzzyMemberships([[0], [1]], [[0], [1]], 2).memberships // => [1, 0, 0, 1]
 * ```
 */
export function fuzzyMemberships(
  x: TensorLike,
  centres: Rows,
  m: number,
): { readonly memberships: number[]; readonly objective: number } {
  const c = centres.length
  const sq = squaredDistancesToCentres(x, centres)
  const n = sq.length / Math.max(c, 1)
  const memberships = new Array<number>(n * c).fill(0)
  let objective = 0
  const power = 1 / (m - 1)
  for (let i = 0; i < n; i++) {
    let zero = -1
    for (let k = 0; k < c; k++) if ((sq[i * c + k] ?? 0) === 0) zero = k
    for (let k = 0; k < c; k++) {
      const dk = sq[i * c + k] ?? 0
      let sum = 0
      if (zero < 0) for (let j = 0; j < c; j++) sum += (dk / (sq[i * c + j] ?? 1)) ** power
      const u = zero >= 0 ? (k === zero ? 1 : 0) : 1 / sum
      memberships[i * c + k] = u
      objective += u ** m * dk
    }
  }
  return { memberships, objective }
}

/** The index of the largest membership of each row of `[n, c]`. */
const hardLabels = (u: readonly number[], c: number): number[] =>
  Array.from({ length: u.length / Math.max(c, 1) }, (_, i) => {
    let best = 0
    for (let k = 1; k < c; k++) if ((u[i * c + k] ?? 0) > (u[i * c + best] ?? 0)) best = k
    return best
  })

/**
 * Fuzzy c-means as a traced {@link Algorithm}: each step updates the centres to the u^m-weighted means and then the
 * memberships, until the centres move less than `tol`. It observes the objective and the shift, the centres and the
 * memberships.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres `[c, d]`.
 * @param m - The fuzzifier (default 2).
 * @param tol - Centre-shift tolerance (default 1e-5).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { fuzzyCMeansAlgorithm } from 'omai-engine/cluster'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const out = runAlgorithm(fuzzyCMeansAlgorithm([[0], [1], [9], [10]], [[0], [10]]), { maxSteps: 100 })
 * out.state.centres.map((c) => Math.round((c[0] ?? 0) / 10)) // => [0, 1]
 * ```
 */
export function fuzzyCMeansAlgorithm(x: TensorLike, init: Rows, m = 2, tol = 1e-5): Algorithm<FuzzyState> {
  if (!(m > 1)) throw new OmaiError('value', 'fuzzyCMeansAlgorithm: the fuzzifier must exceed 1')
  const v = matrixView(x, 'fuzzyCMeansAlgorithm')
  const c = init.length
  return {
    name: 'fuzzy-c-means',
    config: { m, tol, c },
    schema: channels({
      metrics: {
        objective: { description: 'sum of u^m d^2', direction: 'lower' },
        shift: { description: 'the largest distance a centre moved (-1 before the first step)' },
      },
      tensors: {
        centres: { role: 'state', description: 'the cluster centres', shape: ['c', 'd'] },
        memberships: {
          role: 'state',
          description: 'the degree of membership of each row in each cluster',
          shape: ['n', 'c'],
        },
      },
    }),
    init: () => {
      const centres = init.map((r) => [...r])
      return { centres, ...fuzzyMemberships(x, centres, m), shift: Number.MAX_VALUE }
    },
    step: (s) => {
      const centres = s.centres.map((_, k) => {
        let w = 0
        const mean = new Array<number>(v.d).fill(0)
        for (let i = 0; i < v.n; i++) {
          const wi = (s.memberships[i * c + k] ?? 0) ** m
          w += wi
          for (let q = 0; q < v.d; q++) mean[q] = (mean[q] ?? 0) + wi * (v.data[i * v.d + q] ?? 0)
        }
        return mean.map((t) => t / (w || 1))
      })
      const shift = maxOf(centres.map((cent, k) => Math.hypot(...cent.map((t, q) => t - (s.centres[k]?.[q] ?? 0)))))
      return { centres, ...fuzzyMemberships(x, centres, m), shift }
    },
    done: (s) => (s.shift < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, shift: s.shift === Number.MAX_VALUE ? -1 : s.shift },
      tensors: {
        centres: { role: 'state', value: s.centres.flat(), shape: [c, v.d] },
        memberships: { role: 'state', value: s.memberships, shape: [v.n, c] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FuzzyState,
  }
}

/** Hyperparameters of {@link fuzzyCMeans}. */
export interface FuzzyCMeansHyper {
  readonly nClusters: number
  /** The fuzzifier (> 1; larger is softer). */
  readonly m: number
  readonly maxIter: number
  readonly tol: number
}

/** The fitted parameters of {@link fuzzyCMeans}. */
export type FuzzyCMeansParams = {
  readonly centres: number[][]
  /** Memberships of the training rows, `[n][c]`. */
  readonly memberships: number[][]
  readonly labels: number[]
  readonly m: number
}

/** The verbs of fitted fuzzy c-means: hard labels and memberships. */
export interface FuzzyVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
}

/**
 * Fuzzy c-means on the iterative base, from k-means++ centres.
 *
 * @example
 * ```ts
 * import { fuzzyCMeans } from 'omai-engine/cluster'
 * fuzzyCMeans.with({ nClusters: 2 }).fit({ x: [[0], [0.5], [10], [10.5]] }).params.labels.length // => 4
 * ```
 */
export const fuzzyCMeans: IterativeEstimator<FuzzyCMeansHyper, FuzzyState, FuzzyCMeansParams, FuzzyVerbs> = iterative<
  FuzzyCMeansHyper,
  FuzzyState,
  FuzzyCMeansParams,
  FuzzyVerbs
>({
  name: 'cluster/fuzzy-c-means',
  kind: 'clusterer',
  defaults: { nClusters: 3, m: 2, maxIter: 300, tol: 1e-5 },
  hyperSpace: (h) =>
    dict({
      nClusters: int(1, 20, { default: h.nClusters, label: 'clusters' }),
      m: real(1.1, 5, { default: h.m, label: 'fuzzifier' }),
    }),
  algorithm: (data, h, key) =>
    fuzzyCMeansAlgorithm(data.x, initialCentres(data.x, h.nClusters, 'k-means++', key), h.m, h.tol),
  maxSteps: (h) => h.maxIter,
  params: (s, _data, h) => {
    const c = s.centres.length
    const n = s.memberships.length / Math.max(c, 1)
    return {
      centres: s.centres,
      memberships: toRows(s.memberships, n, c),
      labels: hardLabels(s.memberships, c),
      m: h.m,
    }
  },
  verbs: (p) => {
    const d = p.centres[0]?.length ?? 0
    const u = (x: TensorLike): number[] => {
      matrixView(x, 'fuzzyCMeans', { nFeatures: d })
      return fuzzyMemberships(x, p.centres, p.m).memberships
    }
    return {
      predict: (x) => tensor(hardLabels(u(x), p.centres.length)),
      predictProba: (x) => {
        const r = u(x)
        return fromStorage(
          Float64Array.from(r),
          [r.length / Math.max(p.centres.length, 1), p.centres.length],
          'float64',
        )
      },
    }
  },
})
