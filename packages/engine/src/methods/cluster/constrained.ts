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
import type { Key } from '../../lib/random/index.ts'
import { dict, int } from '../../lib/space/index.ts'
import type { TensorLike } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import {
  type CentroidParams,
  type CentroidVerbs,
  centroidVerbs,
  initialCentres,
  squaredDistancesToCentres,
} from './kmeans.ts'

/** Pairwise constraints: rows that must share a cluster, and rows that must not. */
export interface PairConstraints {
  readonly mustLink?: readonly (readonly [number, number])[]
  readonly cannotLink?: readonly (readonly [number, number])[]
}

/** The state of COP-k-means. */
export interface CopKMeansState {
  readonly centres: number[][]
  readonly labels: number[]
  readonly inertia: number
  readonly changed: boolean
  readonly history: { readonly assignments: number[][]; readonly centres: number[][][]; readonly objective: number[] }
}

/**
 * Constrained assignment (Wagstaff et al. 2001): rows in order, each to its nearest centre that violates no constraint
 * with rows already assigned; must-linked rows follow their partner.
 *
 * @param x - Points `[n, d]`.
 * @param centres - The centres.
 * @param constraints - Must-link and cannot-link pairs.
 * @returns Labels and inertia.
 * @throws WhiteboxError `value` when some row has no feasible cluster.
 */
export function constrainedAssignment(
  x: TensorLike,
  centres: readonly (readonly number[])[],
  constraints: PairConstraints,
): { readonly labels: number[]; readonly inertia: number } {
  const k = centres.length
  const dist = squaredDistancesToCentres(x, centres)
  const n = dist.length / k
  const must = Array.from({ length: n }, () => [] as number[])
  const cannot = Array.from({ length: n }, () => [] as number[])
  for (const [a, b] of constraints.mustLink ?? []) {
    must[a]?.push(b)
    must[b]?.push(a)
  }
  for (const [a, b] of constraints.cannotLink ?? []) {
    cannot[a]?.push(b)
    cannot[b]?.push(a)
  }
  const labels = new Array<number>(n).fill(-1)
  let inertia = 0
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: k }, (_, c) => c).sort((p, q) => (dist[i * k + p] ?? 0) - (dist[i * k + q] ?? 0))
    const ok = order.find(
      (c) =>
        (must[i] ?? []).every((j) => (labels[j] ?? -1) < 0 || labels[j] === c) &&
        (cannot[i] ?? []).every((j) => labels[j] !== c),
    )
    if (ok === undefined)
      throw new WhiteboxError('value', `copKMeans: row ${i} has no cluster satisfying its constraints`)
    labels[i] = ok
    inertia += dist[i * k + ok] ?? 0
  }
  return { labels, inertia }
}

/**
 * COP-k-means as a traced algorithm: constrained assignment, then centre means, until the labels stop changing.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres.
 * @param constraints - The constraints.
 * @returns The algorithm.
 */
export function copKMeansAlgorithm(
  x: TensorLike,
  init: readonly (readonly number[])[],
  constraints: PairConstraints,
): Algorithm<CopKMeansState> {
  const m = matrixView(x, 'copKMeans')
  return {
    name: 'cop-kmeans',
    config: { k: init.length },
    init: () => {
      const centres = init.map((r) => [...r])
      const { labels, inertia } = constrainedAssignment(x, centres, constraints)
      return {
        centres,
        labels,
        inertia,
        changed: true,
        history: { assignments: [labels], centres: [centres], objective: [inertia] },
      }
    },
    step: (s) => {
      const centres = s.centres.map((old, c) => {
        const rows = s.labels.flatMap((l, i) => (l === c ? [i] : []))
        if (rows.length === 0) return old
        return old.map((_, q) => rows.reduce((a, i) => a + (m.data[i * m.d + q] ?? 0), 0) / rows.length)
      })
      const { labels, inertia } = constrainedAssignment(x, centres, constraints)
      return {
        centres,
        labels,
        inertia,
        changed: labels.some((l, i) => l !== s.labels[i]),
        history: {
          assignments: [...s.history.assignments, labels],
          centres: [...s.history.centres, centres],
          objective: [...s.history.objective, inertia],
        },
      }
    },
    done: (s, t) => (t > 0 && !s.changed ? 'converged' : undefined),
    observe: (s) => ({ metrics: { inertia: s.inertia } }),
  }
}

/** Hyperparameters of COP-k-means. */
export type CopKMeansHyper = PairConstraints & { readonly nClusters?: number; readonly maxIter?: number }

/**
 * COP-k-means (Wagstaff et al. 2001): k-means whose assignments respect must-link and cannot-link constraints,
 * traced. Its exposition is a `clusteringSteps` record.
 *
 * @param hyper - Clusters, iteration cap and the constraints.
 * @returns The estimator.
 */
export function copKMeans(
  hyper: CopKMeansHyper = {},
): Estimator<CopKMeansHyper, ExplainedFitted<CentroidParams, CentroidVerbs>> {
  return defineExplainedEstimator<CopKMeansHyper, CentroidParams, CentroidVerbs>(
    {
      name: 'cop-kmeans',
      kind: 'clusterer',
      hyperSpace: (h) => dict({ nClusters: int(1, 20, { default: h.nClusters ?? 3, label: 'clusters' }) }),
      fit: (data, h, options) => {
        const run = fitRunOptions(options, 'cop-kmeans')
        const start = initialCentres(data.x, h.nClusters ?? 3, 'k-means++', run.key as Key)
        const out = runAlgorithm(copKMeansAlgorithm(data.x, start, h), { ...run, maxSteps: h.maxIter ?? 300 })
        const s = out.state
        return {
          params: { centres: s.centres, labels: s.labels, inertia: s.inertia, nIter: out.steps },
          exposition: [
            clusteringSteps({
              method: 'copKMeans',
              assignments: s.history.assignments,
              centres: s.history.centres,
              objective: s.history.objective,
              ...(h.mustLink === undefined && h.cannotLink === undefined
                ? {}
                : {
                    constraints: {
                      ...(h.mustLink === undefined ? {} : { mustLink: h.mustLink }),
                      ...(h.cannotLink === undefined ? {} : { cannotLink: h.cannotLink }),
                    },
                  }),
            }),
          ],
        }
      },
      methods: centroidVerbs,
    },
    hyper,
  )
}

/** COP-k-means as a class. */
export class COPKMeans extends EstimatorClass<CopKMeansHyper, ExplainedFitted<CentroidParams, CentroidVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link copKMeans}.
   */
  constructor(hyper: CopKMeansHyper = {}) {
    super(copKMeans(hyper))
  }
}
