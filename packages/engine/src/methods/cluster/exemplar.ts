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
import { dict, int, real } from '../../lib/space/index.ts'
import { quantile } from '../../lib/stats/index.ts'
import { toArray } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { cutTree, linkageMatrix } from './hierarchical.ts'
import { type CentroidParams, type CentroidVerbs, centroidVerbs } from './kmeans.ts'

/** The state of affinity propagation: responsibilities, availabilities and the exemplar history. */
export interface AffinityPropagationState {
  readonly r: number[][]
  readonly a: number[][]
  /** The exemplar indicator of the last `convergenceIter` iterations, per row. */
  readonly recent: boolean[][]
  readonly stable: boolean
  readonly history: number[][]
}

/** Labels from exemplars: each row to its most similar exemplar, exemplars to themselves. */
function exemplarLabels(s: readonly (readonly number[])[], exemplars: readonly number[]): number[] {
  return s.map((row, i) => {
    const own = exemplars.indexOf(i)
    if (own >= 0) return own
    let best = 0
    for (let k = 1; k < exemplars.length; k++)
      if ((row[exemplars[k] ?? 0] ?? 0) > (row[exemplars[best] ?? 0] ?? 0)) best = k
    return exemplars.length === 0 ? -1 : best
  })
}

/**
 * Affinity propagation's message passing (Frey and Dueck 2007; sklearn's loop, without its tie-breaking noise) as a
 * traced algorithm: damped responsibility and availability updates; it has converged when the exemplar set has not
 * changed for `convergenceIter` iterations.
 *
 * @param s - Similarities `[n][n]` with the preferences on the diagonal.
 * @param options - Damping (0.5) and the convergence window (15).
 * @returns The algorithm.
 */
export function affinityPropagationAlgorithm(
  s: readonly (readonly number[])[],
  options: { readonly damping?: number; readonly convergenceIter?: number } = {},
): Algorithm<AffinityPropagationState> {
  const n = s.length
  const damping = options.damping ?? 0.5
  const window = options.convergenceIter ?? 15
  const zeros = (): number[][] => Array.from({ length: n }, () => new Array<number>(n).fill(0))
  return {
    name: 'affinity-propagation',
    config: { damping, convergenceIter: window },
    init: () => ({ r: zeros(), a: zeros(), recent: [], stable: false, history: [] }),
    step: (st) => {
      const r = st.r.map((row, i) => {
        // The largest and second largest a + s in the row: max over j != k of (a + s) for every k.
        let first = Number.NEGATIVE_INFINITY
        let second = Number.NEGATIVE_INFINITY
        let at = -1
        for (let j = 0; j < n; j++) {
          const v = (st.a[i]?.[j] ?? 0) + (s[i]?.[j] ?? 0)
          if (v > first) {
            second = first
            first = v
            at = j
          } else if (v > second) second = v
        }
        return row.map((old, k) => damping * old + (1 - damping) * ((s[i]?.[k] ?? 0) - (k === at ? second : first)))
      })
      const colPos = Array.from({ length: n }, (_, k) => {
        let t = 0
        for (let i = 0; i < n; i++) if (i !== k) t += Math.max(0, r[i]?.[k] ?? 0)
        return t
      })
      const a = st.a.map((row, i) =>
        row.map((old, k) => {
          const value =
            i === k ? (colPos[k] ?? 0) : Math.min(0, (r[k]?.[k] ?? 0) + (colPos[k] ?? 0) - Math.max(0, r[i]?.[k] ?? 0))
          return damping * old + (1 - damping) * value
        }),
      )
      const exemplarFlags = Array.from({ length: n }, (_, i) => (a[i]?.[i] ?? 0) + (r[i]?.[i] ?? 0) > 0)
      const recent = [...st.recent, exemplarFlags].slice(-window)
      const stable =
        recent.length === window &&
        exemplarFlags.some((v) => v) &&
        exemplarFlags.every((_, i) => recent.every((e) => e[i] === recent[0]?.[i]))
      const exemplars = exemplarFlags.flatMap((v, i) => (v ? [i] : []))
      return { r, a, recent, stable, history: [...st.history, exemplarLabels(s, exemplars)] }
    },
    done: (st) => (st.stable ? 'converged' : undefined),
    observe: (st) => ({ metrics: { exemplars: (st.recent.at(-1) ?? []).filter((v) => v).length } }),
  }
}

/** Hyperparameters of affinity propagation. */
export type AffinityPropagationHyper = {
  readonly damping?: number
  /** The self-similarity: higher gives more clusters (default the median similarity). */
  readonly preference?: number
  readonly maxIter?: number
  readonly convergenceIter?: number
}

/** The parameters of a fitted affinity propagation. */
export type AffinityPropagationParams = CentroidParams & { readonly exemplarIndices: number[] }

/**
 * Affinity propagation (sklearn.cluster.AffinityPropagation) on negative squared Euclidean similarities: traced
 * message passing, then sklearn's refinement of each cluster's exemplar. Its exposition is a `clusteringSteps` record
 * of the assignments to the current exemplars per iteration.
 *
 * @param hyper - Damping, preference, iteration cap and convergence window.
 * @returns The estimator.
 */
export function affinityPropagation(
  hyper: AffinityPropagationHyper = {},
): Estimator<AffinityPropagationHyper, ExplainedFitted<AffinityPropagationParams, CentroidVerbs>> {
  return defineExplainedEstimator<AffinityPropagationHyper, AffinityPropagationParams, CentroidVerbs>(
    {
      name: 'affinity-propagation',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          damping: real(0.5, 0.99, { default: h.damping ?? 0.5 }),
          maxIter: int(10, 1000, { default: h.maxIter ?? 200, label: 'iterations' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'affinityPropagation', { minSamples: 2 })
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const sim = (toArray(geometry.cdist(rows, rows, 'sqeuclidean')) as number[][]).map((r) => r.map((v) => -v))
        const pref = h.preference ?? quantile(sim.flat(), 0.5)
        const s = sim.map((r, i) => r.map((v, j) => (i === j ? pref : v)))
        const out = runAlgorithm(
          affinityPropagationAlgorithm(s, {
            ...(h.damping === undefined ? {} : { damping: h.damping }),
            ...(h.convergenceIter === undefined ? {} : { convergenceIter: h.convergenceIter }),
          }),
          { ...fitRunOptions(options, 'affinity-propagation'), maxSteps: h.maxIter ?? 200 },
        )
        let exemplars = (out.state.recent.at(-1) ?? []).flatMap((v, i) => (v ? [i] : []))
        if (exemplars.length === 0) throw new WhiteboxError('convergence', 'affinityPropagation: no exemplars emerged')
        // sklearn's refinement: within each cluster, the member with the largest total similarity to the others.
        let c = exemplarLabels(s, exemplars)
        exemplars = exemplars.map((_, k) => {
          const members = c.flatMap((l, i) => (l === k ? [i] : []))
          let best = members[0] ?? 0
          let bestSum = Number.NEGATIVE_INFINITY
          for (const j of members) {
            const t = members.reduce((acc, i) => acc + (s[i]?.[j] ?? 0), 0)
            if (t > bestSum) {
              bestSum = t
              best = j
            }
          }
          return best
        })
        exemplars.sort((p, q) => p - q)
        c = exemplarLabels(s, exemplars)
        const centres = exemplars.map((i) => rows[i] ?? [])
        const inertia = c.reduce((acc, l, i) => acc - (sim[i]?.[exemplars[l] ?? 0] ?? 0), 0)
        return {
          params: { centres, labels: c, inertia, nIter: out.steps, exemplarIndices: exemplars },
          exposition: [clusteringSteps({ method: 'affinityPropagation', assignments: out.state.history })],
        }
      },
      methods: centroidVerbs,
    },
    hyper,
  )
}

/** Affinity propagation as a class. */
export class AffinityPropagation extends EstimatorClass<
  AffinityPropagationHyper,
  ExplainedFitted<AffinityPropagationParams, CentroidVerbs>
> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link affinityPropagation}.
   */
  constructor(hyper: AffinityPropagationHyper = {}) {
    super(affinityPropagation(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// BIRCH

/** A clustering feature: count, linear sum and squared sum of the points absorbed. */
export interface ClusteringFeature {
  readonly n: number
  readonly ls: number[]
  readonly ss: number
}

/**
 * The leaf level of BIRCH's CF tree (Zhang, Ramakrishnan and Livny 1996): each point joins the nearest subcluster if
 * the merged radius stays within `threshold`, else starts a new one. TODO: the branching-factor tree and its
 * rebuilding.
 *
 * @param rows - Points `[n][d]`.
 * @param threshold - The largest subcluster radius.
 * @returns The subclusters and each row's subcluster.
 */
export function birchSubclusters(
  rows: readonly (readonly number[])[],
  threshold: number,
): { readonly features: ClusteringFeature[]; readonly members: number[] } {
  const features: ClusteringFeature[] = []
  const members = rows.map((r) => {
    const sq = r.reduce((a, v) => a + v * v, 0)
    let best = -1
    let bestD = Number.POSITIVE_INFINITY
    for (const [k, f] of features.entries()) {
      const d = r.reduce((a, v, q) => a + (v - (f.ls[q] ?? 0) / f.n) ** 2, 0)
      if (d < bestD) {
        bestD = d
        best = k
      }
    }
    if (best >= 0) {
      const f = features[best] as ClusteringFeature
      const n = f.n + 1
      const ls = f.ls.map((v, q) => v + (r[q] ?? 0))
      const ss = f.ss + sq
      const radius = Math.sqrt(Math.max(0, ss / n - ls.reduce((a, v) => a + (v / n) ** 2, 0)))
      if (radius <= threshold) {
        features[best] = { n, ls, ss }
        return best
      }
    }
    features.push({ n: 1, ls: [...r], ss: sq })
    return features.length - 1
  })
  return { features, members }
}

/** Hyperparameters of BIRCH. */
export type BirchHyper = { readonly threshold?: number; readonly nClusters?: number }

/**
 * BIRCH (sklearn.cluster.Birch): subclusters of radius at most `threshold`, then Ward agglomeration of their centroids
 * into `nClusters` (the global step). Its exposition is a `clusteringSteps` record with two steps: the subclusters,
 * then the global clusters (with their centres).
 *
 * @param hyper - The subcluster radius (default 0.5) and the number of clusters (default 3).
 * @returns The estimator.
 */
export function birch(hyper: BirchHyper = {}): Estimator<BirchHyper, ExplainedFitted<CentroidParams, CentroidVerbs>> {
  return defineExplainedEstimator<BirchHyper, CentroidParams, CentroidVerbs>(
    {
      name: 'birch',
      kind: 'clusterer',
      hyperSpace: (h) =>
        dict({
          threshold: real(0.05, 5, { log: true, default: h.threshold ?? 0.5, label: 'subcluster radius' }),
          nClusters: int(1, 20, { default: h.nClusters ?? 3, label: 'clusters' }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'birch')
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const { features, members } = birchSubclusters(rows, h.threshold ?? 0.5)
        const centroids = features.map((f) => f.ls.map((v) => v / f.n))
        const k = Math.min(h.nClusters ?? 3, centroids.length)
        const global =
          centroids.length > 1 ? cutTree(linkageMatrix(centroids, 'ward'), centroids.length, { nClusters: k }) : [0]
        const labels = members.map((s) => global[s] ?? 0)
        const centres = Array.from({ length: k }, (_, c) => {
          const fs = features.filter((_, s) => global[s] === c)
          const n = fs.reduce((a, f) => a + f.n, 0)
          return Array.from({ length: m.d }, (_, q) => fs.reduce((a, f) => a + (f.ls[q] ?? 0), 0) / n)
        })
        const inertia = rows.reduce(
          (a, r, i) => a + r.reduce((s, v, q) => s + (v - (centres[labels[i] ?? 0]?.[q] ?? 0)) ** 2, 0),
          0,
        )
        return {
          params: { centres, labels, inertia, nIter: 1 },
          exposition: [
            clusteringSteps({
              method: 'birch',
              assignments: [members, labels],
              centres: [centroids, centres],
            }),
          ],
        }
      },
      methods: centroidVerbs,
    },
    hyper,
  )
}

/** BIRCH as a class. */
export class Birch extends EstimatorClass<BirchHyper, ExplainedFitted<CentroidParams, CentroidVerbs>> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link birch}.
   */
  constructor(hyper: BirchHyper = {}) {
    super(birch(hyper))
  }
}
