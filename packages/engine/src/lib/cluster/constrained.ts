/**
 * Constrained clustering: COP-k-means (Wagstaff et al. 2001), k-means whose assignments must respect must-link and
 * cannot-link pairs between rows. The constrained assignment step, the traced algorithm, an estimator, and
 * `brokenConstraints` for the pairs a labelling violates.
 */

import { MdmlError } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { dict, int } from '../space/index.ts'
import { type Tensor, type TensorLike, tensor } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { initialCentres, nearestCentres, type Rows, squaredDistancesToCentres } from './kmeans.ts'

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
 * @example
 * ```ts
 * import { constrainedAssignment } from 'mdml-engine/cluster'
 * constrainedAssignment([[0], [1]], [[0], [1]], { cannotLink: [[0, 1]] }).labels // => [0, 1]
 * constrainedAssignment([[0], [0.1]], [[0], [5]], { cannotLink: [[0, 1]] }).labels // => [0, 1]
 * ```
 */
export function constrainedAssignment(
  x: TensorLike,
  centres: Rows,
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
 * @example
 * ```ts
 * import { brokenConstraints } from 'mdml-engine/cluster'
 * brokenConstraints([0, 1, 1], { mustLink: [[0, 1]], cannotLink: [[1, 2]] }) // => { mustLink: [0], cannotLink: [0] }
 * ```
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
 * COP-k-means as a traced {@link Algorithm}: each step moves the centres to the means and then makes the constrained
 * assignment, until the labels stop changing. It observes the inertia, how many rows were placed against a constraint,
 * the centres and the labels.
 *
 * @param x - Points `[n, d]`.
 * @param init - Initial centres.
 * @param constraints - The constraints.
 * @param strict - Fail on a row with no feasible cluster instead of relaxing (default false).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { copKMeansAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [1], [10], [11]]
 * runAlgorithm(copKMeansAlgorithm(x, [[0], [10]], { mustLink: [[0, 1]] }), { maxSteps: 20 }).state.labels // => [0, 0, 1, 1]
 * ```
 */
export function copKMeansAlgorithm(
  x: TensorLike,
  init: Rows,
  constraints: PairConstraints,
  strict = false,
): Algorithm<CopKMeansState> {
  const m = matrixView(x, 'copKMeans')
  const k = init.length
  return {
    name: 'cop-kmeans',
    config: { k },
    schema: channels({
      metrics: {
        inertia: { description: 'sum of squared distances of the rows to their centres', direction: 'lower' },
        relaxed: { description: 'rows placed against a constraint' },
      },
      tensors: {
        centres: { role: 'state', description: 'the cluster centres', shape: ['k', 'd'] },
        labels: { role: 'state', description: 'the cluster of each row', shape: ['n'] },
      },
    }),
    init: () => {
      const centres = init.map((r) => [...r])
      return { centres, ...constrainedAssignment(x, centres, constraints, strict), changed: true }
    },
    step: (s) => {
      const centres = s.centres.map((old, c) => {
        const rows = s.labels.flatMap((l, i) => (l === c ? [i] : []))
        if (rows.length === 0) return old
        return old.map((_, q) => rows.reduce((a, i) => a + (m.data[i * m.d + q] ?? 0), 0) / rows.length)
      })
      const next = constrainedAssignment(x, centres, constraints, strict)
      return { centres, ...next, changed: next.labels.some((l, i) => l !== s.labels[i]) }
    },
    done: (s, t) => (t > 0 && !s.changed ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { inertia: s.inertia, relaxed: s.relaxed.length },
      tensors: {
        centres: { role: 'state', value: s.centres.flat(), shape: [k, m.d] },
        labels: { role: 'state', value: s.labels, shape: [m.n] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CopKMeansState,
  }
}

/** Hyperparameters of {@link copKMeans}. */
export interface CopKMeansHyper {
  readonly nClusters: number
  readonly maxIter: number
  /** Fail when a row has no cluster satisfying its constraints, as the paper does (false: relax and report). */
  readonly strict: boolean
  readonly mustLink: [number, number][]
  readonly cannotLink: [number, number][]
}

/** The fitted parameters of {@link copKMeans}: the centroids and the constraints the result breaks. */
export type CopKMeansParams = {
  readonly centres: number[][]
  readonly labels: number[]
  readonly inertia: number
  /** Indices into the must-link pairs split across clusters. */
  readonly brokenMustLink: number[]
  /** Indices into the cannot-link pairs sharing a cluster. */
  readonly brokenCannotLink: number[]
}

/** The verbs of fitted COP-k-means: the nearest centre (unconstrained). */
export interface CopKMeansVerbs {
  readonly predict: (x: TensorLike) => Tensor
}

/**
 * COP-k-means (Wagstaff et al. 2001) on the iterative base: k-means whose assignments respect must-link and cannot-link
 * constraints. A row with no feasible cluster (the greedy assignment can paint itself into a corner, or the constraints
 * may contradict each other) is placed where it breaks the fewest constraints rather than failing, unless `strict`;
 * the constraints the result breaks are in the parameters.
 *
 * @example
 * ```ts
 * import { copKMeans } from 'mdml-engine/cluster'
 * const fit = copKMeans.with({ nClusters: 2, mustLink: [[0, 3]] }).fit({ x: [[0], [1], [10], [11]] })
 * fit.params.labels[0] === fit.params.labels[3] // => true
 * ```
 */
export const copKMeans: IterativeEstimator<CopKMeansHyper, CopKMeansState, CopKMeansParams, CopKMeansVerbs> = iterative<
  CopKMeansHyper,
  CopKMeansState,
  CopKMeansParams,
  CopKMeansVerbs
>({
  name: 'cluster/cop-kmeans',
  kind: 'clusterer',
  defaults: { nClusters: 3, maxIter: 300, strict: false, mustLink: [], cannotLink: [] },
  hyperSpace: (h) => dict({ nClusters: int(1, 20, { default: h.nClusters, label: 'clusters' }) }),
  algorithm: (data, h, key) =>
    copKMeansAlgorithm(data.x, initialCentres(data.x, h.nClusters, 'k-means++', key), h, h.strict),
  maxSteps: (h) => h.maxIter,
  params: (s, _data, h) => {
    const broken = brokenConstraints(s.labels, h)
    return {
      centres: s.centres,
      labels: s.labels,
      inertia: s.inertia,
      brokenMustLink: broken.mustLink,
      brokenCannotLink: broken.cannotLink,
    }
  },
  verbs: (p) => ({ predict: (x) => tensor(nearestCentres(x, p.centres).labels) }),
})
