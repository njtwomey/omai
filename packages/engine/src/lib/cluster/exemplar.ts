/**
 * Exemplar-based clustering: affinity propagation (Frey and Dueck 2007), where rows pass responsibility and
 * availability messages until a set of exemplars emerges, one damped round of messages per step.
 */

import { MdmlError } from '../base/index.ts'
import { distanceMatrix } from '../distances/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { foldIn, type Key, normal } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { quantile } from '../stats/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { labelledPredict, rowsAt } from './_support.ts'

/** The state of affinity propagation: responsibilities and availabilities (flat `[n, n]`), recent exemplars, labels. */
export interface AffinityPropagationState {
  readonly r: number[]
  readonly a: number[]
  /** The exemplar indicator of the last `convergenceIter` iterations, per row. */
  readonly recent: boolean[][]
  readonly stable: boolean
  /** The index among the current exemplars of each row's most similar exemplar (-1 while there are none). */
  readonly labels: number[]
}

const EPS = 2.220446049250313e-16
/** 100 times the smallest normal double, sklearn's additive noise scale. */
const TINY_100 = 100 * 2.2250738585072014e-308

/** Labels from exemplars: each row to its most similar exemplar, exemplars to themselves. */
function exemplarLabels(s: ArrayLike<number>, n: number, exemplars: readonly number[]): number[] {
  return Array.from({ length: n }, (_, i) => {
    const own = exemplars.indexOf(i)
    if (own >= 0) return own
    let best = 0
    for (let k = 1; k < exemplars.length; k++)
      if ((s[i * n + (exemplars[k] ?? 0)] ?? 0) > (s[i * n + (exemplars[best] ?? 0)] ?? 0)) best = k
    return exemplars.length === 0 ? -1 : best
  })
}

/** Options of {@link affinityPropagationAlgorithm}. */
export interface AffinityPropagationOptions {
  readonly damping?: number
  /** The self-similarity: higher gives more clusters (default the median similarity). */
  readonly preference?: number
  readonly convergenceIter?: number
  /**
   * The key of sklearn's tie-breaking noise, (eps |s| + 100 tiny) N(0, 1) added to every similarity: without it,
   * symmetric similarities (two equally good exemplars) make the messages oscillate instead of converging.
   */
  readonly noiseKey?: Key
}

/**
 * The similarities affinity propagation works on: negative squared Euclidean distances with the preference on the
 * diagonal and sklearn's tie-breaking noise.
 *
 * @param x - Points `[n, d]`.
 * @param options - Preference (default the median similarity) and the noise key.
 * @returns The similarities `[n, n]`, flat, the same without the noise, and `n`.
 * @example
 * ```ts
 * import { affinitySimilarities } from 'mdml-engine/cluster'
 * affinitySimilarities([[0], [2]], { preference: -1 }).plain // => [-1, -4, -4, -1]
 * ```
 */
export function affinitySimilarities(
  x: TensorLike,
  options: Pick<AffinityPropagationOptions, 'preference' | 'noiseKey'> = {},
): { readonly n: number; readonly s: number[]; readonly plain: number[] } {
  const { n, data } = distanceMatrix(x, 'squaredEuclidean')
  const sim = Array.from(data, (v) => -v)
  const pref = options.preference ?? quantile(sim, 0.5)
  const plain = sim.map((v, p) => (Math.floor(p / n) === p % n ? pref : v))
  const noise = options.noiseKey === undefined ? undefined : normal(foldIn(options.noiseKey, 0), [n, n]).data
  const s = plain.map((v, p) => v + (noise === undefined ? 0 : (EPS * v + TINY_100) * (noise[p] ?? 0)))
  return { n, s, plain }
}

/**
 * Affinity propagation's message passing (Frey and Dueck 2007; sklearn's loop) as a traced {@link Algorithm}: each step
 * is one damped round of responsibility and availability updates, and the algorithm has converged when the exemplar
 * set has not changed for `convergenceIter` rounds. It observes the number of exemplars and each row's exemplar.
 *
 * @param s - Similarities `[n, n]` flat, with the preferences on the diagonal.
 * @param n - The number of rows.
 * @param options - Damping (0.5) and the convergence window (15).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { affinityPropagationAlgorithm } from 'mdml-engine/cluster'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const s = [-1, -1, -50, -1, -1, -50, -50, -50, -1]
 * runAlgorithm(affinityPropagationAlgorithm(s, 3), { maxSteps: 200 }).state.labels.length // => 3
 * ```
 */
export function affinityPropagationAlgorithm(
  s: ArrayLike<number>,
  n: number,
  options: Pick<AffinityPropagationOptions, 'damping' | 'convergenceIter'> = {},
): Algorithm<AffinityPropagationState> {
  const damping = options.damping ?? 0.5
  const window = options.convergenceIter ?? 15
  return {
    name: 'affinity-propagation',
    config: { damping, convergenceIter: window, n },
    schema: channels({
      metrics: { exemplars: { description: 'the number of rows that are exemplars now' } },
      tensors: { labels: { role: 'state', description: 'the exemplar each row follows', shape: ['n'] } },
    }),
    init: () => ({
      r: new Array<number>(n * n).fill(0),
      a: new Array<number>(n * n).fill(0),
      recent: [],
      stable: false,
      labels: new Array<number>(n).fill(-1),
    }),
    step: (st) => {
      const r = new Array<number>(n * n).fill(0)
      for (let i = 0; i < n; i++) {
        // The largest and second largest a + s in the row: max over j != k of (a + s) for every k.
        let first = Number.NEGATIVE_INFINITY
        let second = Number.NEGATIVE_INFINITY
        let at = -1
        for (let j = 0; j < n; j++) {
          const v = (st.a[i * n + j] ?? 0) + (s[i * n + j] ?? 0)
          if (v > first) {
            second = first
            first = v
            at = j
          } else if (v > second) second = v
        }
        for (let k = 0; k < n; k++)
          r[i * n + k] =
            damping * (st.r[i * n + k] ?? 0) + (1 - damping) * ((s[i * n + k] ?? 0) - (k === at ? second : first))
      }
      const colPos = Array.from({ length: n }, (_, k) => {
        let t = 0
        for (let i = 0; i < n; i++) if (i !== k) t += Math.max(0, r[i * n + k] ?? 0)
        return t
      })
      const a = new Array<number>(n * n).fill(0)
      for (let i = 0; i < n; i++)
        for (let k = 0; k < n; k++) {
          const value =
            i === k
              ? (colPos[k] ?? 0)
              : Math.min(0, (r[k * n + k] ?? 0) + (colPos[k] ?? 0) - Math.max(0, r[i * n + k] ?? 0))
          a[i * n + k] = damping * (st.a[i * n + k] ?? 0) + (1 - damping) * value
        }
      const flags = Array.from({ length: n }, (_, i) => (a[i * n + i] ?? 0) + (r[i * n + i] ?? 0) > 0)
      const recent = [...st.recent, flags].slice(-window)
      const stable =
        recent.length === window &&
        flags.some((v) => v) &&
        flags.every((_, i) => recent.every((e) => e[i] === recent[0]?.[i]))
      const exemplars = flags.flatMap((v, i) => (v ? [i] : []))
      return { r, a, recent, stable, labels: exemplarLabels(s, n, exemplars) }
    },
    done: (st) => (st.stable ? 'converged' : undefined),
    observe: (st) => ({
      metrics: { exemplars: (st.recent.at(-1) ?? []).filter((v) => v).length },
      tensors: { labels: { role: 'state', value: st.labels, shape: [n] } },
    }),
    serialise: (st) => stateToJson(st),
    deserialise: (j) => stateFromJson(j) as AffinityPropagationState,
  }
}

/** The exemplars of a state, with sklearn's refinement: within each cluster, the member most similar to the others. */
export function affinityExemplars(
  state: AffinityPropagationState,
  s: ArrayLike<number>,
  n: number,
): { readonly exemplars: number[]; readonly labels: number[] } {
  const raw = (state.recent.at(-1) ?? []).flatMap((v, i) => (v ? [i] : []))
  if (raw.length === 0) return { exemplars: [], labels: new Array<number>(n).fill(-1) }
  const c = exemplarLabels(s, n, raw)
  const exemplars = raw.map((_, k) => {
    const members = c.flatMap((l, i) => (l === k ? [i] : []))
    let best = members[0] ?? 0
    let bestSum = Number.NEGATIVE_INFINITY
    for (const j of members) {
      let t = 0
      for (const i of members) t += s[i * n + j] ?? 0
      if (t > bestSum) {
        bestSum = t
        best = j
      }
    }
    return best
  })
  exemplars.sort((p, q) => p - q)
  return { exemplars, labels: exemplarLabels(s, n, exemplars) }
}

/** Hyperparameters of {@link affinityPropagation}. */
export interface AffinityPropagationHyper {
  readonly damping: number
  /** The self-similarity (`null`: the median similarity). */
  readonly preference: number | null
  readonly maxIter: number
  readonly convergenceIter: number
}

/** The fitted parameters of {@link affinityPropagation}. */
export type AffinityPropagationParams = {
  readonly exemplarIndices: number[]
  readonly centres: number[][]
  readonly labels: number[]
}

/** The verbs of fitted affinity propagation: the nearest exemplar. */
export interface ExemplarVerbs {
  readonly predict: (x: TensorLike) => Tensor
}

/**
 * Affinity propagation (sklearn.cluster.AffinityPropagation) on negative squared Euclidean similarities on the
 * iterative base: one damped round of messages per step. The fitted value at a step is the current exemplars (none
 * yet at the start), refined as sklearn does.
 *
 * @example
 * ```ts
 * import { affinityPropagation } from 'mdml-engine/cluster'
 * const x = [[0], [0.1], [0.2], [9], [9.1], [9.2]]
 * affinityPropagation.with({ preference: -10 }).fit({ x }).params.labels // => [0, 0, 0, 1, 1, 1]
 * ```
 */
export const affinityPropagation: IterativeEstimator<
  AffinityPropagationHyper,
  AffinityPropagationState,
  AffinityPropagationParams,
  ExemplarVerbs
> = iterative<AffinityPropagationHyper, AffinityPropagationState, AffinityPropagationParams, ExemplarVerbs>({
  name: 'cluster/affinity-propagation',
  kind: 'clusterer',
  defaults: { damping: 0.5, preference: null, maxIter: 200, convergenceIter: 15 },
  hyperSpace: (h) =>
    dict({
      damping: real(0.5, 0.99, { default: h.damping, label: 'damping' }),
      maxIter: int(10, 1000, { default: h.maxIter, label: 'iterations' }),
    }),
  algorithm: (data, h, key) => {
    const { n, s } = affinitySimilarities(data.x, {
      noiseKey: key,
      ...(h.preference === null ? {} : { preference: h.preference }),
    })
    return affinityPropagationAlgorithm(s, n, { damping: h.damping, convergenceIter: h.convergenceIter })
  },
  maxSteps: (h) => h.maxIter,
  params: (st, data, h) => {
    const m = matrixView(data.x, 'affinityPropagation', { minSamples: 2 })
    const { s } = affinitySimilarities(data.x, h.preference === null ? {} : { preference: h.preference })
    const { exemplars, labels } = affinityExemplars(st, s, m.n)
    return { exemplarIndices: exemplars, centres: rowsAt(m.data, m.d, exemplars), labels }
  },
  verbs: (p) => ({
    predict: labelledPredict(
      p.centres,
      p.centres.map((_, j) => j),
      Number.POSITIVE_INFINITY,
      'sqeuclidean',
      p.centres[0]?.length ?? 0,
    ),
  }),
})

/**
 * Affinity propagation run to the end on given similarities, as a function: the exemplars and labels.
 *
 * @param x - Points `[n, d]`.
 * @param options - Damping, preference, window and the noise key.
 * @param maxIter - The most rounds (default 200).
 * @returns The exemplar rows, the labels and the rounds taken.
 * @throws MdmlError `convergence` when no exemplars emerged.
 * @example
 * ```ts
 * import { affinityPropagationFit } from 'mdml-engine/cluster'
 * import { key } from 'mdml-engine/random'
 * affinityPropagationFit([[0], [0.1], [9], [9.1]], { preference: -10, noiseKey: key(0) }).exemplars // => [0, 3]
 * ```
 */
export function affinityPropagationFit(
  x: TensorLike,
  options: AffinityPropagationOptions = {},
  maxIter = 200,
): { readonly exemplars: number[]; readonly labels: number[]; readonly nIter: number } {
  const { n, s } = affinitySimilarities(x, options)
  const out = runAlgorithm(affinityPropagationAlgorithm(s, n, options), { maxSteps: maxIter })
  const { exemplars, labels } = affinityExemplars(out.state, s, n)
  if (exemplars.length === 0) throw new MdmlError('convergence', 'affinityPropagationFit: no exemplars emerged')
  return { exemplars, labels, nIter: out.steps }
}
