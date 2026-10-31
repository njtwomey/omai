/**
 * Variants of hidden Markov model training beside Baum-Welch: the left-right (Bakis) topology, where states are only
 * visited in order and a state can be left only for itself or one of the next few (Baum-Welch keeps the zeros of the
 * transition matrix, so constraining the start is enough), and Viterbi training (segmental k-means), the hard-assignment
 * EM that decodes every sequence with the best path under the current parameters and re-estimates from those counts. The
 * hard version maximises the joint probability of the data and the best paths, which it never decreases, and stops when
 * the paths stop changing.
 */

import { MdmlError } from '../../base/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../../trace/index.ts'
import { viterbi } from '../inference/chain.ts'
import { hmmChain } from '../models/state-space.ts'
import { type HmmObservation, type HmmParameters, type HmmSequence, hmmFromParameters } from './baum-welch.ts'

const vector = (o: HmmObservation): readonly number[] => (typeof o === 'number' ? [o] : o)

/**
 * Constrain parameters to the left-right topology: the start is state 0, a state moves to itself or up to `maxJump`
 * states ahead, the last state absorbs. Baum-Welch and Viterbi training started from the result keep that structure.
 *
 * @param params - Starting parameters of any topology.
 * @param maxJump - The farthest forward jump (default 1: stay or advance one).
 * @returns The parameters with a banded upper-triangular transition matrix, rows renormalised.
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * const p = { kind: 'categorical', start: [0.5, 0.5], transition: [[0.5, 0.5], [0.5, 0.5]], emission: [[0.5, 0.5], [0.5, 0.5]] } as const
 * learning.leftRightParameters({ ...p, start: [0.5, 0.5], transition: [[0.5, 0.5], [0.5, 0.5]], emission: [[0.5, 0.5], [0.5, 0.5]] }).transition // => [[0.5, 0.5], [0, 1]]
 */
export function leftRightParameters(params: HmmParameters, maxJump = 1): HmmParameters {
  const K = params.start.length
  const transition = params.transition.map((row, i) => {
    const masked = row.map((v, j) => (j >= i && j <= i + maxJump ? Math.max(v, 1e-3) : 0))
    const z = masked.reduce((a, b) => a + b, 0)
    return masked.map((v) => v / z)
  })
  const start = Array.from({ length: K }, (_, i) => (i === 0 ? 1 : 0))
  return { ...params, start, transition }
}

/** Whether a transition matrix is left-right: no probability on a move backwards. */
export function isLeftRight(transition: readonly (readonly number[])[]): boolean {
  return transition.every((row, i) => row.every((v, j) => j >= i || v === 0))
}

/** The state of {@link viterbiTrainingAlgorithm}. */
export type ViterbiTrainingState = {
  params: HmmParameters
  /** The best-path assignment of every sequence under the parameters the step started from. */
  paths: number[][]
  /** The joint log probability of the data and these paths. */
  score: number | null
  /** Whether the assignment changed in the last step. */
  changed: boolean
}

/** Options of {@link viterbiTrainingAlgorithm}. */
export interface ViterbiTrainingOptions {
  /** Added to the counts of transitions that are allowed (non-zero), keeping the structure (default 1e-3). */
  readonly smoothing?: number
  /** Added to the variance of a Gaussian state, hmmlearn's prior (default 1e-2). */
  readonly covariancePrior?: number
}

/**
 * Viterbi training as an algorithm: each step decodes every sequence by the Viterbi path under the current parameters
 * and re-estimates the start, transitions (zeros stay zero) and emissions (categorical frequencies, or the mean and
 * diagonal variance of the frames in each state) from those hard counts. Gaussian emissions are diagonal.
 *
 * @param sequences - The training sequences.
 * @param start - The starting parameters.
 * @param options - Smoothing and the variance prior.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { learning } from 'mdml-engine/pgm'
 * const start = { kind: 'categorical', start: [0.5, 0.5], transition: [[0.6, 0.4], [0.4, 0.6]], emission: [[0.7, 0.3], [0.2, 0.8]] } as const
 * learning.viterbiTrainingAlgorithm([[0, 0, 1, 1]], { ...start, start: [0.5, 0.5], transition: [[0.6, 0.4], [0.4, 0.6]], emission: [[0.7, 0.3], [0.2, 0.8]] }).name // => 'viterbi-training'
 */
export function viterbiTrainingAlgorithm(
  sequences: readonly HmmSequence[],
  start: HmmParameters,
  options: ViterbiTrainingOptions = {},
): Algorithm<ViterbiTrainingState> {
  const K = start.start.length
  const smoothing = options.smoothing ?? 1e-3
  const prior = options.covariancePrior ?? 1e-2
  if (sequences.length === 0) throw new MdmlError('value', 'viterbiTraining: no sequences')
  const observations = sequences.map((s) => s.map((o) => (start.kind === 'categorical' ? o : vector(o))))
  const decode = (params: HmmParameters): { paths: number[][]; score: number } => {
    const model = hmmFromParameters(params)
    let score = 0
    const paths = observations.map((obs) => {
      const r = viterbi(hmmChain(model, obs))
      score += r.logProbability
      return r.path
    })
    return { paths, score }
  }
  const reestimate = (params: HmmParameters, paths: number[][]): HmmParameters => {
    const startCounts = new Array<number>(K).fill(0)
    const trans = Array.from({ length: K }, () => new Array<number>(K).fill(0))
    paths.forEach((path) => {
      startCounts[path[0] ?? 0] = (startCounts[path[0] ?? 0] ?? 0) + 1
      for (let t = 1; t < path.length; t++)
        (trans[path[t - 1] ?? 0] as number[])[path[t] ?? 0] = (trans[path[t - 1] ?? 0]?.[path[t] ?? 0] ?? 0) + 1
    })
    const norm = (row: number[], allowed: readonly number[]): number[] => {
      const r = row.map((v, j) => ((allowed[j] ?? 0) > 0 ? v + smoothing : 0))
      const z = r.reduce((a, b) => a + b, 0)
      return z > 0 ? r.map((v) => v / z) : allowed.map((v) => (v > 0 ? 1 : 0))
    }
    const newStart = norm(startCounts, params.start)
    const transition = trans.map((row, i) => norm(row, params.transition[i] ?? []))
    if (params.kind === 'categorical') {
      const S = params.emission[0]?.length ?? 0
      const counts = Array.from({ length: K }, () => new Array<number>(S).fill(0))
      observations.forEach((obs, n) =>
        obs.forEach((o, t) => {
          const k = paths[n]?.[t] ?? 0
          ;(counts[k] as number[])[o as number] = (counts[k]?.[o as number] ?? 0) + 1
        }),
      )
      const emission = counts.map((row, k) => {
        const z = row.reduce((a, b) => a + b, 0)
        return z > 0 ? row.map((v) => v / z) : (params.emission[k] ?? row)
      })
      return { kind: 'categorical', start: newStart, transition, emission }
    }
    const D = params.means[0]?.length ?? 1
    const means = params.means.map((m) => [...m])
    const covariances = params.covariances.map((c) => c.map((r) => [...r]))
    for (let k = 0; k < K; k++) {
      const frames: (readonly number[])[] = []
      observations.forEach((obs, n) =>
        obs.forEach((o, t) => (paths[n]?.[t] === k ? frames.push(o as readonly number[]) : null)),
      )
      if (frames.length === 0) continue
      for (let d = 0; d < D; d++) {
        const mean = frames.reduce((a, f) => a + (f[d] ?? 0), 0) / frames.length
        const variance = frames.reduce((a, f) => a + ((f[d] ?? 0) - mean) ** 2, 0)
        ;(means[k] as number[])[d] = mean
        ;((covariances[k] as number[][])[d] as number[])[d] = (variance + prior) / (frames.length + 1)
      }
    }
    return { kind: 'gaussian', start: newStart, transition, means, covariances }
  }
  return {
    name: 'viterbi-training',
    config: { states: K, emission: start.kind, smoothing },
    schema: channels({
      metrics: {
        score: { description: 'joint log probability of the data and the best paths', direction: 'higher' },
        changed: { description: 'whether any path changed in the step (1) or not (0)', direction: 'lower' },
      },
      tensors: { transition: { role: 'param', description: 'the transition matrix', shape: [K, K] } },
    }),
    init: () => ({ params: start, paths: [], score: null, changed: true }),
    step: (s) => {
      const { paths, score } = decode(s.params)
      const changed = JSON.stringify(paths) !== JSON.stringify(s.paths)
      return { params: reestimate(s.params, paths), paths, score, changed }
    },
    done: (s) => (!s.changed ? 'converged' : undefined),
    observe: (s) => ({
      metrics: s.score === null ? {} : { score: s.score, changed: s.changed ? 1 : 0 },
      tensors: { transition: { role: 'param', value: s.params.transition.flat(), shape: [K, K] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ViterbiTrainingState,
  }
}
