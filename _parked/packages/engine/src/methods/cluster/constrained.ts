/**
 * Constrained clustering: COP-k-means (Wagstaff et al., 2001), k-means whose assignments must respect must-link and
 * cannot-link pairs between rows. The constrained assignment step, a traced algorithm, the fit function and class, and
 * `brokenConstraints` for the pairs a labelling violates.
 */

import { MdmlError } from '../../lib/base/index.ts'
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
  /** Rows the last assignment had to place against a constraint. */
  readonly relaxed: number[]
  readonly history: { readonly assignments: number[][]; readonly centres: number[][][]; readonly objective: number[] }
}

/** What a constrained assignment returns: labels, inertia and the rows placed against some constraint. */
export interface ConstrainedAssignment {
  readonly labels: number[]
  readonly inertia: number
  /** Rows with no cluster satisfying their constraints, placed where they break the fewest (nearest on ties). */
  readonly relaxed: number[]
}

/**
 * Constrained assignment (Wagstaff et al. 2001): rows in order, each to its nearest centre that violates no constraint
 * with rows already assigned; must-linked rows follow their partner. A row with no such centre either fails (`strict`,
 * as in the paper) or goes to the centre breaking the fewest constraints, nearest on ties, and is reported.
 *
 * @param x - Points `[n, d]`.
 * @param centres - The centres.
 * @param constraints - Must-link and cannot-link pairs.
 * @param strict - Throw instead of relaxing (default false).
 * @returns Labels, inertia and the relaxed rows.
 * @throws MdmlError `value` when `strict` and some row has no feasible cluster.
 */
export function constrainedAssignment(
  x: TensorLike,
  centres: readonly (readonly number[])[],
  constraints: PairConstraints,
  strict = false,
): ConstrainedAssignment {
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
  const relaxed: number[] = []
  let inertia = 0
  for (let i = 0; i < n; i++) {
    // Constraints broken by putting row i in cluster c, given the rows placed so far.
    const broken = (c: number): number =>
      (must[i] ?? []).filter((j) => (labels[j] ?? -1) >= 0 && labels[j] !== c).length +
      (cannot[i] ?? []).filter((j) => labels[j] === c).length
    let best = -1
    let bestBroken = Number.POSITIVE_INFINITY
    for (let c = 0; c < k; c++) {
      const b = broken(c)
      const closer = best < 0 || (dist[i * k + c] ?? 0) < (dist[i * k + best] ?? 0)
      if (b < bestBroken || (b === bestBroken && closer)) {
        best = c
        bestBroken = b
      }
    }
    if (bestBroken > 0) {
      if (strict) throw new MdmlError('value', `copKMeans: row ${i} has no cluster satisfying its constraints`)
      relaxed.push(i)
    }
    labels[i] = best
    inertia += dist[i * k + best] ?? 0
  }
  return { labels, inertia, relaxed }
}

/**
 * The constraints a labelling breaks: indices into the must-link pairs split across clusters and into the cannot-link
 * pairs sharing one.
 *
 * @param labels - The cluster of each row.
 * @param constraints - The constraints.
 * @returns The broken must-link and cannot-link pairs, by index.
 */
export function brokenConstraints(
  labels: readonly number[],
  constraints: PairConstraints,
): { readonly mustLink: number[]; readonly cannotLink: number[] } {
  return {
    mustLink: (constraints.mustLink ?? []).flatMap(([a, b], j) => (labels[a] !== labels[b] ? [j] : [])),
    cannotLink: (constraints.cannotLink ?? []).flatMap(([a, b], j) => (labels[a] === labels[b] ? [j] : [])),
  }
}

/**
 * COP-k-means as a traced algorithm: constrained assignment, then centre means, until the labels stop changing.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres.
 * @param constraints - The constraints.
 * @param strict - Fail on a row with no feasible cluster instead of relaxing (default false).
 * @returns The algorithm.
 */
export function copKMeansAlgorithm(
  x: TensorLike,
  init: readonly (readonly number[])[],
  constraints: PairConstraints,
  strict = false,
): Algorithm<CopKMeansState> {
  const m = matrixView(x, 'copKMeans')
  return {
    name: 'cop-kmeans',
    config: { k: init.length },
    init: () => {
      const centres = init.map((r) => [...r])
      const { labels, inertia, relaxed } = constrainedAssignment(x, centres, constraints, strict)
      return {
        centres,
        labels,
        inertia,
        changed: true,
        relaxed,
        history: { assignments: [labels], centres: [centres], objective: [inertia] },
      }
    },
    step: (s) => {
      const centres = s.centres.map((old, c) => {
        const rows = s.labels.flatMap((l, i) => (l === c ? [i] : []))
        if (rows.length === 0) return old
        return old.map((_, q) => rows.reduce((a, i) => a + (m.data[i * m.d + q] ?? 0), 0) / rows.length)
      })
      const { labels, inertia, relaxed } = constrainedAssignment(x, centres, constraints, strict)
      return {
        centres,
        labels,
        inertia,
        changed: labels.some((l, i) => l !== s.labels[i]),
        relaxed,
        history: {
          assignments: [...s.history.assignments, labels],
          centres: [...s.history.centres, centres],
          objective: [...s.history.objective, inertia],
        },
      }
    },
    done: (s, t) => (t > 0 && !s.changed ? 'converged' : undefined),
    observe: (s) => ({ metrics: { inertia: s.inertia, relaxed: s.relaxed.length } }),
  }
}

/** Hyperparameters of COP-k-means. */
export type CopKMeansHyper = PairConstraints & {
  readonly nClusters?: number
  readonly maxIter?: number
  /** Fail when a row has no cluster satisfying its constraints, as the paper does (default false: relax and report). */
  readonly strict?: boolean
}

/** The parameters of a fitted COP-k-means: the centroid parameters and the constraints the result breaks. */
export type CopKMeansParams = CentroidParams & {
  /** Indices into the must-link pairs split across clusters. */
  readonly brokenMustLink: number[]
  /** Indices into the cannot-link pairs sharing a cluster. */
  readonly brokenCannotLink: number[]
}

/**
 * COP-k-means (Wagstaff et al. 2001): k-means whose assignments respect must-link and cannot-link constraints,
 * traced. A row with no feasible cluster (the greedy assignment can paint itself into a corner, or the constraints
 * may contradict each other or need more clusters) is placed where it breaks the fewest constraints rather than
 * failing, unless `strict`; the constraints the result breaks are in the parameters. Its exposition is a
 * `clusteringSteps` record with the constraints and the broken ones.
 *
 * @param hyper - Clusters, iteration cap, the constraints and strictness.
 * @returns The estimator.
 */
export function copKMeans(
  hyper: CopKMeansHyper = {},
): Estimator<CopKMeansHyper, ExplainedFitted<CopKMeansParams, CentroidVerbs>> {
  return defineExplainedEstimator<CopKMeansHyper, CopKMeansParams, CentroidVerbs>(
    {
      name: 'cop-kmeans',
      kind: 'clusterer',
      hyperSpace: (h) => dict({ nClusters: int(1, 20, { default: h.nClusters ?? 3, label: 'clusters' }) }),
      fit: (data, h, options) => {
        const run = fitRunOptions(options, 'cop-kmeans')
        const start = initialCentres(data.x, h.nClusters ?? 3, 'k-means++', run.key as Key)
        const out = runAlgorithm(copKMeansAlgorithm(data.x, start, h, h.strict ?? false), {
          ...run,
          maxSteps: h.maxIter ?? 300,
        })
        const s = out.state
        const broken = brokenConstraints(s.labels, h)
        const mustLink = (h.mustLink ?? []).map(([a, b]): [number, number] => [a, b])
        const cannotLink = (h.cannotLink ?? []).map(([a, b]): [number, number] => [a, b])
        return {
          params: {
            centres: s.centres,
            labels: s.labels,
            inertia: s.inertia,
            nIter: out.steps,
            brokenMustLink: broken.mustLink,
            brokenCannotLink: broken.cannotLink,
          },
          exposition: [
            clusteringSteps({
              method: 'copKMeans',
              assignments: s.history.assignments,
              centres: s.history.centres,
              objective: s.history.objective,
              constraints: {
                mustLink,
                cannotLink,
                brokenMustLink: broken.mustLink,
                brokenCannotLink: broken.cannotLink,
              },
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
export class COPKMeans extends EstimatorClass<CopKMeansHyper, ExplainedFitted<CopKMeansParams, CentroidVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link copKMeans}.
   */
  constructor(hyper: CopKMeansHyper = {}) {
    super(copKMeans(hyper))
  }
}
