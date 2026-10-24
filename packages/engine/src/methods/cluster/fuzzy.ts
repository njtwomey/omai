import { WhiteboxError } from '../../lib/base/index.ts'
import { clusteringSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  fitRunOptions,
  matrixView,
} from '../../lib/learn/index.ts'
import { geometry } from '../../lib/numerics/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toArray } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { initialCentres } from './kmeans.ts'

/** The state of fuzzy c-means. */
export interface FuzzyState {
  readonly centres: number[][]
  readonly memberships: number[][]
  readonly objective: number
  readonly shift: number
  readonly history: { readonly centres: number[][][]; readonly labels: number[][]; readonly objective: number[] }
}

/**
 * Fuzzy memberships u_ik = 1 / sum_j (d_ik / d_jk)^(2 / (m - 1)), with a point on a centre wholly in it.
 *
 * @param x - Points `[n, d]`.
 * @param centres - Centres `[c][d]`.
 * @param m - The fuzzifier (> 1).
 * @returns Memberships `[n][c]` and the objective sum u^m d^2.
 */
export function fuzzyMemberships(
  x: TensorLike,
  centres: readonly (readonly number[])[],
  m: number,
): { readonly memberships: number[][]; readonly objective: number } {
  const d = toArray(geometry.cdist(x, centres as number[][])) as number[][]
  let objective = 0
  const memberships = d.map((row) => {
    const zero = row.indexOf(0)
    const u =
      zero >= 0
        ? row.map((_, k) => (k === zero ? 1 : 0))
        : row.map((dk) => 1 / row.reduce((s, dj) => s + (dk / dj) ** (2 / (m - 1)), 0))
    for (const [k, v] of u.entries()) objective += v ** m * (row[k] ?? 0) ** 2
    return u
  })
  return { memberships, objective }
}

/**
 * Fuzzy c-means (Bezdek 1981) as a traced algorithm: alternate membership and centre updates (centres the
 * u^m-weighted means) until the centres move less than tol.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres.
 * @param m - The fuzzifier (default 2).
 * @param tol - Centre-shift tolerance (default 1e-5).
 * @returns The algorithm.
 */
export function fuzzyCMeansAlgorithm(
  x: TensorLike,
  init: readonly (readonly number[])[],
  m = 2,
  tol = 1e-5,
): Algorithm<FuzzyState> {
  const v = matrixView(x, 'fuzzyCMeans')
  const labelsOf = (u: number[][]): number[] => u.map((r) => r.indexOf(Math.max(...r)))
  return {
    name: 'fuzzy-c-means',
    config: { m, tol },
    init: () => {
      const centres = init.map((r) => [...r])
      const { memberships, objective } = fuzzyMemberships(x, centres, m)
      return {
        centres,
        memberships,
        objective,
        shift: Number.POSITIVE_INFINITY,
        history: { centres: [centres], labels: [labelsOf(memberships)], objective: [objective] },
      }
    },
    step: (s) => {
      const centres = s.centres.map((_, k) => {
        let w = 0
        const c = new Array<number>(v.d).fill(0)
        for (let i = 0; i < v.n; i++) {
          const wi = (s.memberships[i]?.[k] ?? 0) ** m
          w += wi
          for (let q = 0; q < v.d; q++) c[q] = (c[q] ?? 0) + wi * (v.data[i * v.d + q] ?? 0)
        }
        return c.map((t) => t / (w || 1))
      })
      const shift = Math.max(...centres.map((c, k) => Math.hypot(...c.map((t, q) => t - (s.centres[k]?.[q] ?? 0)))))
      const { memberships, objective } = fuzzyMemberships(x, centres, m)
      return {
        centres,
        memberships,
        objective,
        shift,
        history: {
          centres: [...s.history.centres, centres],
          labels: [...s.history.labels, labelsOf(memberships)],
          objective: [...s.history.objective, objective],
        },
      }
    },
    done: (s) => (s.shift < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: { objective: s.objective } }),
  }
}

/** Hyperparameters of fuzzy c-means. */
export type FuzzyCMeansHyper = {
  readonly nClusters?: number
  readonly m?: number
  readonly maxIter?: number
  readonly tol?: number
}

/** The parameters of fitted fuzzy c-means. */
export type FuzzyCMeansParams = {
  readonly centres: number[][]
  readonly memberships: number[][]
  readonly labels: number[]
  readonly m: number
}

/** The verbs of fitted fuzzy c-means: hard labels and memberships. */
export interface FuzzyVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * Fuzzy c-means: soft memberships with fuzzifier m, k-means++ initial centres, traced. Its exposition is a
 * `clusteringSteps` record (hard labels, centres and objective per iteration).
 *
 * @param hyper - Clusters, fuzzifier, iteration cap and tolerance.
 * @returns The estimator.
 */
export function fuzzyCMeans(
  hyper: FuzzyCMeansHyper = {},
): Estimator<FuzzyCMeansHyper, ExplainedFitted<FuzzyCMeansParams, FuzzyVerbs>> {
  return defineExplainedEstimator<FuzzyCMeansHyper, FuzzyCMeansParams, FuzzyVerbs>(
    {
      name: 'fuzzy-c-means',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          nClusters: int(1, 20, { default: h.nClusters ?? 3, label: 'clusters' }),
          m: real(1.1, 5, { default: h.m ?? 2, label: 'fuzzifier' }),
        }),
      fit: (data, h, options) => {
        const m = h.m ?? 2
        if (!(m > 1)) throw new WhiteboxError('value', 'fuzzyCMeans: the fuzzifier must exceed 1')
        const run = fitRunOptions(options, 'fuzzy-c-means')
        const start = initialCentres(data.x, h.nClusters ?? 3, 'k-means++', run.key as Key)
        const out = runAlgorithm(fuzzyCMeansAlgorithm(data.x, start, m, h.tol ?? 1e-5), {
          ...run,
          maxSteps: h.maxIter ?? 300,
        })
        const s = out.state
        return {
          params: { centres: s.centres, memberships: s.memberships, labels: s.history.labels.at(-1) ?? [], m },
          exposition: [
            clusteringSteps({
              method: 'fuzzyCMeans',
              assignments: s.history.labels,
              centres: s.history.centres,
              objective: s.history.objective,
            }),
          ],
        }
      },
      methods: (p) => {
        const d = p.centres[0]?.length ?? 0
        const u = (x: TensorLike): number[][] => {
          matrixView(x, 'fuzzyCMeans', { nFeatures: d })
          return fuzzyMemberships(x, p.centres, p.m).memberships
        }
        return {
          predict: (x) => {
            const labels = u(x).map((r) => r.indexOf(Math.max(...r)))
            return fromStorage(Int32Array.from(labels), [labels.length], 'int32')
          },
          predictProba: (x) => {
            const r = u(x)
            return fromStorage(Float64Array.from(r.flat()), [r.length, p.centres.length], 'float64')
          },
          nFeaturesIn: d,
        }
      },
    },
    hyper,
  )
}

/** Fuzzy c-means as a class. */
export class FuzzyCMeans extends EstimatorClass<FuzzyCMeansHyper, ExplainedFitted<FuzzyCMeansParams, FuzzyVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link fuzzyCMeans}.
   */
  constructor(hyper: FuzzyCMeansHyper = {}) {
    super(fuzzyCMeans(hyper))
  }
}
