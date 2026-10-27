/**
 * The hidden Markov model as a particular of the iterative base: Baum-Welch from a seeded start over one sequence (rows
 * are time steps), with categorical or Gaussian emissions. Each iteration puts the log likelihood, the transition matrix
 * and the emissions on the tape, and the state posteriors and the decoded path at any step are a probe, drawn against
 * the true states when the data has them.
 */

import { stateSequence } from '../../lib/exposition/index.ts'
import { type IterativeEstimator, iterative, type LearnData } from '../../lib/learn/index.ts'
import { learning } from '../../lib/pgm/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { derived, Log } from '../../lib/trace/index.ts'

/** The hyperparameters of {@link hmm}. */
export interface HmmHyper {
  /** The number of hidden states. */
  readonly nStates: number
  /**
   * The emission family. Categorical needs symbol indices (non-negative integers, one per step); real-valued series
   * are fitted with Gaussian emissions whichever is asked for.
   */
  readonly emissions: 'categorical' | 'gaussian'
  /** The iteration budget of a fit. */
  readonly maxIter: number
  /** Converged when an iteration gains less than this in log likelihood. */
  readonly tol: number
  /** The probability of staying in a state the first iteration starts from (sticky starts converge faster). */
  readonly stay: number
}

/** The fitted parameters of {@link hmm}: the start distribution, the transition matrix and the emissions, plain JSON. */
export type HmmParams = learning.HmmParameters

/** What a fitted hidden Markov model offers. */
export interface HmmVerbs {
  /** The most probable path of states (Viterbi) through a sequence, one row per time step. */
  readonly predict: (x: TensorLike) => Tensor
  /** The posterior of each state at each step `[T, K]`. */
  readonly predictProba: (x: TensorLike) => Tensor
  /** The log likelihood of a sequence. */
  readonly score: (data: LearnData) => number
}

// The rows of a sequence: a symbol per step for categorical emissions, a vector per step otherwise.
function sequenceOf(x: TensorLike, kind: 'categorical' | 'gaussian'): learning.HmmSequence {
  const t = asTensor(x)
  const n = t.shape[0] ?? 0
  const d = t.shape.length === 1 ? 1 : (t.shape[1] ?? 1)
  const flat = toFlat(t)
  return Array.from({ length: n }, (_, i) =>
    kind === 'categorical' ? (flat[i * d] ?? 0) : Array.from(flat.slice(i * d, (i + 1) * d)),
  )
}

// Categorical emissions fit symbol data; anything else (reals, several columns) is fitted as Gaussian.
function emissionOf(x: TensorLike, asked: 'categorical' | 'gaussian'): 'categorical' | 'gaussian' {
  const t = asTensor(x)
  const symbols = t.shape.length === 1 || t.shape[1] === 1
  return asked === 'categorical' && symbols && toFlat(t).every((v) => Number.isInteger(v) && v >= 0)
    ? 'categorical'
    : 'gaussian'
}

// The display position of each model state that agrees best with the true states (states are numbered arbitrarily).
function renumbering(path: readonly number[], truth: readonly number[], K: number): number[] {
  const counts = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  for (const [t, k] of path.entries()) {
    const c = truth[t] ?? 0
    if (c < K) (counts[k] as number[])[c] = (counts[k]?.[c] ?? 0) + 1
  }
  let best: number[] = Array.from({ length: K }, (_, k) => k)
  let bestScore = -1
  const place = (k: number, taken: readonly number[]): void => {
    if (k === K) {
      const score = taken.reduce((a, c, s) => a + (counts[s]?.[c] ?? 0), 0)
      if (score > bestScore) [best, bestScore] = [[...taken], score]
      return
    }
    for (let c = 0; c < K; c++) if (!taken.includes(c)) place(k + 1, [...taken, c])
  }
  place(0, [])
  return best
}

// What the model makes of the data: the posterior of each state, the decoded path, and the truth when the data has it.
function readingOf(params: HmmParams, data: LearnData): ReturnType<typeof stateSequence> {
  const x = asTensor(data.x)
  const { posteriors, path } = learning.hmmDecode(params, sequenceOf(x, params.kind))
  const K = params.start.length
  const truthOf = data.y === undefined ? undefined : Array.from(toFlat(asTensor(data.y)))
  const truth =
    truthOf !== undefined && truthOf.length === path.length && truthOf.every((c) => c >= 0 && c < K)
      ? truthOf
      : undefined
  const display = truth === undefined ? Array.from({ length: K }, (_, k) => k) : renumbering(path, truth, K)
  const classes = (data as { readonly classes?: readonly string[] }).classes
  const names = Array.from({ length: K }, (_, c) => (truth === undefined ? undefined : classes?.[c]) ?? `state ${c}`)
  const featureName = (data as { readonly featureNames?: readonly string[] }).featureNames?.[0] ?? 'observation'
  const values = toFlat(x)
  return stateSequence({
    states: names,
    posteriors: posteriors.map((row) => {
      const out = new Array<number>(K).fill(0)
      for (const [k, p] of row.entries()) out[display[k] ?? k] = p
      return out
    }),
    decoded: path.map((k) => display[k] ?? k),
    ...(truth === undefined ? {} : { truth }),
    ...(x.shape.length === 1 || x.shape[1] === 1
      ? { observations: { name: featureName, values: Array.from(values) } }
      : {}),
  })
}

/**
 * A hidden Markov model fitted by Baum-Welch: the rows of `x` are the time steps of one sequence, with a symbol index
 * (categorical emissions) or a vector (Gaussian) per step. `predict` is the Viterbi path and `predictProba` the state
 * posteriors. On the tape: `solver/logLikelihood`, `solver/transition` and `solver/emission` (the categorical table or
 * the Gaussian means) per iteration; the probe `model/posterior` is a `stateSequence` record, which numbers the states to
 * agree best with the true ones when the data has `y`.
 *
 * @example
 * ```ts
 * import { hmm } from 'mdml-engine/methods/sequence'
 * const x = [[0], [0], [0], [1], [1], [1], [0], [0], [1], [1], [1], [0]]
 * hmm.with({ nStates: 2 }).fit({ x }).params.kind // => 'categorical'
 * ```
 */
export const hmm: IterativeEstimator<HmmHyper, learning.BaumWelchState, HmmParams, HmmVerbs> = iterative<
  HmmHyper,
  learning.BaumWelchState,
  HmmParams,
  HmmVerbs
>({
  name: 'sequence/hmm',
  kind: 'clusterer',
  defaults: { nStates: 2, emissions: 'categorical', maxIter: 200, tol: 1e-4, stay: 0.8 },
  hyperSpace: (h) =>
    dict({
      nStates: int(2, 5, { default: h.nStates, label: 'hidden states' }),
      emissions: choice(['categorical', 'gaussian'] as const, { default: h.emissions, label: 'emissions' }),
      tol: real(1e-8, 1, { log: true, default: h.tol, label: 'tolerance' }),
    }),
  algorithm: (data, h, key) => {
    const kind = emissionOf(data.x, h.emissions)
    const sequences = [sequenceOf(data.x, kind)]
    const start = learning.initialHmmParameters(sequences, { states: h.nStates, emission: kind, stay: h.stay }, key)
    return learning.baumWelchAlgorithm(sequences, start, { tol: h.tol })
  },
  maxSteps: (h) => h.maxIter,
  params: (s) => s.params,
  verbs: (p) => ({
    predict: (x) => {
      const path = learning.hmmDecode(p, sequenceOf(x, p.kind)).path
      return fromStorage(Int32Array.from(path), [path.length], 'int32')
    },
    predictProba: (x) => {
      const { posteriors } = learning.hmmDecode(p, sequenceOf(x, p.kind))
      return fromStorage(Float64Array.from(posteriors.flat()), [posteriors.length, p.start.length], 'float64')
    },
    score: (data) => learning.hmmDecode(p, sequenceOf(data.x, p.kind)).logLikelihood,
  }),
  contract: (data) => ({
    'model/posterior': derived<learning.BaumWelchState>({
      description: 'the state posteriors over time and the decoded path under the parameters at this step',
      compute: (s) => readingOf(s.params, data),
    }),
  }),
  records: (s, data) => [readingOf(s.params, data)],
  logs: {
    'solver/logLikelihood': Log(),
    'solver/transition': Log({ detail: 'full' }),
    'solver/emission': Log({ detail: 'full' }),
  },
})
