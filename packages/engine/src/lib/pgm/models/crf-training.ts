/**
 * Training linear-chain CRFs through the solver slot. The model is an objective, the penalised negative conditional
 * log-likelihood of the labelled sequences (`crfSmoothProblem`, with the gradient as expected minus observed feature
 * counts, and unbiased batch estimates for the stochastic solvers); how it is minimised is the slot `crfSolvers`:
 * L-BFGS exact training is the default, every generic solver of `smoothSolvers` is there beside it (the `gradient` mode
 * is stochastic gradient descent with any optimiser, schedule and batch size), and three online algorithms of
 * structured prediction are named modes of the model's own: the structured perceptron (Collins 2002), its averaged
 * form, and the passive-aggressive update with the Hamming loss (Crammer et al. 2006). Each epoch of those decodes the
 * sequences one at a time with the current weights and moves the weights on a mistake; they do not minimise the
 * likelihood but they converge to zero errors on separable data, and the run stops when the reported weights decode every
 * training sequence or an epoch needs no update. Every step puts the penalised objective and the
 * sequence error of the weights on the tape.
 */

import { MdmlError } from '../../base/index.ts'
import {
  analyticObjective,
  familySolvers,
  type SmoothProblem,
  slotAlgorithm,
  type VectorParams,
} from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../random/index.ts'
import { bool, dict, real, type SpaceValue } from '../../space/index.ts'
import { toFlat, type Value } from '../../tensor/index.ts'
import { type Algorithm, type ModeSpec, runAlgorithm, stateFromJson, stateToJson } from '../../trace/index.ts'
import { viterbi } from '../inference/chain.ts'
import {
  type CrfPenalty,
  crfChain,
  crfFeatureCounts,
  crfObjective,
  crfParameters,
  type LabelledSequence,
  type LinearChainCrf,
  withCrfParameters,
} from './crf.ts'

/** What a CRF is trained on: the starting model (labels and vocabularies), the labelled sequences and the penalties. */
export interface CrfTrainingProblem {
  readonly crf: LinearChainCrf
  readonly data: readonly LabelledSequence[]
  readonly penalty?: CrfPenalty
}

/**
 * The CRF training objective as a smooth problem for the solver slot: `crfObjective` (negative conditional
 * log-likelihood plus the L2 penalty, with its analytic gradient from forward-backward) over the flat weight vector.
 * With a batch of sequence indices the likelihood part is rescaled by n / |batch|, an unbiased estimate of the full
 * objective, so the stochastic solvers see the same scale as the exact ones. The L1 penalty of `crfObjective` is not
 * smooth and is ignored here (use `c2`).
 *
 * @param problem - The model, data and penalty.
 * @returns The problem: objective (with batches over the sequences) and the starting weights.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { crfSmoothProblem, linearChainCrf } from 'mdml-engine/pgm'
 * crfSmoothProblem({ crf: linearChainCrf(['A', 'B'], ['f']), data: [{ x: [{ f: 1 }], y: ['A'] }] }).x0.length // => 6
 * ```
 */
export function crfSmoothProblem(problem: CrfTrainingProblem): SmoothProblem {
  const { crf, data } = problem
  const c2 = problem.penalty?.c2 ?? 0
  const n = data.length
  const at = (w: ArrayLike<number>, batch?: readonly number[]): { value: number; gradient: Float64Array } => {
    const model = withCrfParameters(crf, w)
    const rows = batch === undefined ? data : batch.map((i) => data[i] as LabelledSequence)
    const r = crfObjective(model, rows, { c2 })
    if (batch === undefined) return r
    // crfObjective adds the penalty once; rescale only the likelihood part
    const scale = n / Math.max(1, batch.length)
    const pen = crfObjective(model, [], { c2 })
    const gradient = r.gradient.map((g, i) => scale * (g - (pen.gradient[i] ?? 0)) + (pen.gradient[i] ?? 0))
    return { value: scale * (r.value - pen.value) + pen.value, gradient }
  }
  const objectiveOf = (batch?: readonly number[]): ((x: Value) => Value) =>
    analyticObjective({ value: (w) => at(w, batch).value, gradient: (w) => at(w, batch).gradient })
  const full = objectiveOf()
  return {
    objective: ((x: unknown, batch?: readonly number[]) =>
      (batch === undefined ? full : objectiveOf(batch))(x as Value)) as SmoothProblem['objective'],
    x0: Array.from(crfParameters(crf)),
    batches: { n },
  }
}

/** The state of an online CRF trainer: the reported weights, their objective and gradient, and the epoch's mistakes. */
export interface CrfOnlineState {
  /** The weights that are reported (the running average for the averaged perceptron, else the current ones). */
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  readonly step: number
  /** The current weights (equal to `x` except for the averaged perceptron). */
  readonly w: Float64Array
  /** The sum of the weights after every example so far, for averaging. */
  readonly sum: Float64Array
  /** Examples seen. */
  readonly seen: number
  /** Updates made in the last epoch (the perceptron's mistakes). */
  readonly mistakes: number
}

/** The fraction of sequences the weights do not decode exactly right. */
function sequenceError(problem: CrfTrainingProblem, w: ArrayLike<number>): number {
  const model = withCrfParameters(problem.crf, w)
  let wrong = 0
  for (const { x, y } of problem.data) {
    const decoded = viterbi(crfChain(model, x)).path
    const gold = y.map((l) => (typeof l === 'number' ? l : model.labels.indexOf(l)))
    if (decoded.some((k, t) => k !== gold[t])) wrong++
  }
  return problem.data.length === 0 ? 0 : wrong / problem.data.length
}

type OnlineRule = 'perceptron' | 'averaged' | 'passive-aggressive'

function onlineMode(rule: OnlineRule, description: string): ModeSpec<CrfTrainingProblem, CrfOnlineState, VectorParams> {
  return {
    description,
    space: dict({
      learningRate: real(1e-3, 10, { log: true, default: 1, label: 'learning rate' }),
      aggressiveness: real(1e-4, 1e4, { log: true, default: 1, label: 'aggressiveness C' }),
      shuffle: bool({ default: true, label: 'shuffle each epoch' }),
    }),
    make: (problem, knobs): Algorithm<CrfOnlineState> => {
      const eta = (knobs['learningRate'] as number | undefined) ?? 1
      const aggressiveness = (knobs['aggressiveness'] as number | undefined) ?? 1
      const shuffle = (knobs['shuffle'] as boolean | undefined) ?? true
      const smooth = crfSmoothProblem(problem)
      const evaluate = (x: Float64Array): { value: number; grad: Float64Array } => {
        const r = crfObjective(withCrfParameters(problem.crf, x), problem.data, { c2: problem.penalty?.c2 ?? 0 })
        return { value: r.value, grad: r.gradient }
      }
      const P =
        problem.crf.stateWeights.length + problem.crf.transitionWeights.length + problem.crf.bigramWeights.length
      const start = Float64Array.from(smooth.x0 as ArrayLike<number>)
      const report = (
        w: Float64Array,
        sum: Float64Array,
        seen: number,
        mistakes: number,
        step: number,
      ): CrfOnlineState => {
        const x = rule === 'averaged' && seen > 0 ? sum.map((v) => v / seen) : w
        return { x, ...evaluate(x), step, w, sum, seen, mistakes }
      }
      return {
        name: `crf-${rule}`,
        config: { rule, learningRate: eta, aggressiveness, shuffle, sequences: problem.data.length } as never,
        init: () => report(start, new Float64Array(P), 0, -1, 0),
        step: (s, ctx) => {
          const order = shuffle
            ? Array.from(toFlat(permutation(ctx.key, problem.data.length)))
            : Array.from({ length: problem.data.length }, (_, i) => i)
          const w = Float64Array.from(s.w)
          const sum = Float64Array.from(s.sum)
          let seen = s.seen
          let mistakes = 0
          for (const i of order) {
            const { x, y } = problem.data[i] as LabelledSequence
            const model = withCrfParameters(problem.crf, w)
            const chain = crfChain(model, x)
            const truth = crfFeatureCounts(model, x, y)
            // passive-aggressive decodes with the Hamming cost added to every wrong label (cost-augmented inference)
            const gold = y.map((l) => (typeof l === 'number' ? l : model.labels.indexOf(l)))
            const augmented =
              rule === 'passive-aggressive'
                ? {
                    ...chain,
                    logEmission: chain.logEmission.map((row, t) => row.map((v, k) => v + (k === gold[t] ? 0 : 1))),
                  }
                : chain
            const predicted = viterbi(augmented).path
            if (predicted.every((k, t) => k === gold[t])) {
              seen++
              for (let q = 0; q < P; q++) sum[q] = (sum[q] ?? 0) + (w[q] ?? 0)
              continue
            }
            mistakes++
            const guess = crfFeatureCounts(model, x, predicted)
            const diff = truth.map((v, q) => v - (guess[q] ?? 0))
            let step = eta
            if (rule === 'passive-aggressive') {
              const score = (counts: Float64Array): number => counts.reduce((a, v, q) => a + v * (w[q] ?? 0), 0)
              const hamming = predicted.reduce((a, k, t) => a + (k === gold[t] ? 0 : 1), 0)
              const loss = score(guess) - score(truth) + hamming
              const norm = diff.reduce((a, v) => a + v * v, 0)
              step = norm > 0 ? Math.min(aggressiveness, Math.max(loss, 0) / norm) : 0
            }
            for (let q = 0; q < P; q++) w[q] = (w[q] ?? 0) + step * (diff[q] ?? 0)
            seen++
            for (let q = 0; q < P; q++) sum[q] = (sum[q] ?? 0) + (w[q] ?? 0)
          }
          const next = report(w, sum, seen, mistakes, 0)
          return { ...next, step: Math.hypot(...next.x.map((v, q) => v - (s.x[q] ?? 0))) }
        },
        // done once an epoch made no update, or the reported weights decode every training sequence exactly (the
        // passive-aggressive update keeps asking for a margin after the decoding is right; the average of an averaged
        // perceptron may separate before or after the weights themselves stop changing)
        done: (s) =>
          s.mistakes === 0 || (s.mistakes > 0 && sequenceError(problem, s.x) === 0) ? 'converged' : undefined,
        serialise: (s) => stateToJson(s),
        deserialise: (j) => stateFromJson(j) as CrfOnlineState,
      }
    },
    params: (s) => ({ x: Array.from(s.x) }),
  }
}

/**
 * The ways to train a CRF: L-BFGS on the exact objective is the default, beside every generic smooth solver (Newton,
 * BFGS, conjugate gradient, coordinate descent and the stochastic `gradient` mode with its optimiser, schedule and
 * batch), and the online algorithms of structured prediction as modes of the model's own: `perceptron` (Collins),
 * `averaged` (the perceptron reporting the average of its weights over all examples seen) and `passive-aggressive`
 * (PA-I with the Hamming loss).
 */
export const crfSolvers: ReturnType<typeof familySolvers<CrfTrainingProblem>> = familySolvers(
  {
    perceptron: onlineMode(
      'perceptron',
      'structured perceptron: decode each sequence with the current weights and add the difference of the gold and predicted feature counts on a mistake',
    ),
    averaged: onlineMode(
      'averaged',
      'averaged structured perceptron: the same updates, the reported weights the average over every example seen',
    ),
    'passive-aggressive': onlineMode(
      'passive-aggressive',
      'passive-aggressive (PA-I): the smallest change that makes the gold labelling win by its Hamming distance to the cost-augmented prediction, capped by C',
    ),
  },
  crfSmoothProblem,
  {
    label: 'solver',
    description: 'how the conditional log-likelihood is maximised or the structured loss reduced',
    default: 'lbfgs',
  },
)

/** Options of {@link crfTrainAlgorithm} and {@link trainCrf}. */
export interface CrfTrainOptions extends CrfPenalty {
  /** The solver, a value of `crfSolvers.space` (default L-BFGS). */
  readonly solver?: SpaceValue
  /** Stop when the largest gradient entry falls below this (default 1e-6; the online modes stop at zero mistakes). */
  readonly tol?: number
  /** The key for shuffling and batches. */
  readonly key?: Key
}

/** The state of {@link crfTrainAlgorithm}. */
export type CrfTrainState = {
  readonly solver: unknown
  readonly weights: number[]
  /** The penalised negative conditional log-likelihood at the weights. */
  readonly objective: number
  /** The largest gradient entry. */
  readonly gradient: number
  /** The fraction of training sequences the weights do not decode exactly. */
  readonly sequenceError: number
  /** The updates made in the last epoch (online modes; -1 before the first, -2 for the other solvers). */
  readonly mistakes: number
  readonly evaluations: number
  readonly stalled: boolean
}

/**
 * CRF training as a traced {@link Algorithm} over the slot `crfSolvers`: whichever solver is chosen, each step puts the
 * penalised objective, the largest gradient entry and the sequence error on the tape, with the `solver/gradNorm` and
 * `solver/step` of the solver, and the weights.
 *
 * @param problem - The model, data and penalty.
 * @param options - The solver and tolerance.
 * @returns The algorithm.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { crfTrainAlgorithm, linearChainCrf } from 'mdml-engine/pgm'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const data = [{ x: [{ 'w=a': 1 }, { 'w=b': 1 }], y: ['X', 'Y'] }]
 * const crf = linearChainCrf(['X', 'Y'], ['w=a', 'w=b'])
 * runAlgorithm(crfTrainAlgorithm({ crf, data, penalty: { c2: 0.1 } }), { maxSteps: 100 }).state.sequenceError // => 0
 * ```
 */
export function crfTrainAlgorithm(
  problem: CrfTrainingProblem,
  options: CrfTrainOptions = {},
): Algorithm<CrfTrainState> {
  const solver = options.solver ?? { kind: 'lbfgs' }
  const tol = options.tol ?? 1e-6
  const P = problem.crf.stateWeights.length + problem.crf.transitionWeights.length + problem.crf.bigramWeights.length
  if (problem.data.length === 0) throw new MdmlError('value', 'crfTrain: no training sequences')
  return slotAlgorithm<CrfTrainState, CrfTrainingProblem>({
    name: 'crf-training',
    config: { solver, sequences: problem.data.length, c2: problem.penalty?.c2 ?? 0 } as never,
    schema: {
      metrics: {
        objective: { description: 'the penalised negative conditional log-likelihood', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
        sequenceError: {
          description: 'the fraction of training sequences not decoded exactly right',
          direction: 'lower',
        },
        mistakes: {
          description: 'updates made in the last epoch of an online solver (-1 before the first, -2 otherwise)',
        },
      },
      tensors: { weights: { role: 'param', description: 'the CRF weights: state, transition, bigram', shape: [P] } },
    },
    problem,
    table: crfSolvers,
    solver,
    tol,
    key: options.key ?? makeKey(0),
    view: (r) => ({
      weights: Array.from(r.x),
      objective: r.value,
      gradient: r.grad.reduce((a, b) => Math.max(a, Math.abs(b)), 0),
      sequenceError: sequenceError(problem, r.x),
      mistakes: (r as unknown as Partial<CrfOnlineState>).mistakes ?? -2,
      evaluations: 0,
      stalled: false,
    }),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient, sequenceError: s.sequenceError, mistakes: s.mistakes },
      tensors: { weights: { role: 'param', value: s.weights, shape: [P] } },
    }),
  })
}

/**
 * Train a linear-chain CRF with any solver of the slot: the fitted model, the objective, the sequence error and
 * whether the solver stopped on its own criterion.
 *
 * @param crf - The starting model (labels and vocabularies; its weights are the start).
 * @param data - Labelled sequences.
 * @param options - The solver, penalties, tolerance and key.
 * @param maxSteps - The step budget (epochs for the online modes; default 200).
 * @returns The fitted model and its training summary.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { linearChainCrf, trainCrf } from 'mdml-engine/pgm'
 * const data = [{ x: [{ 'w=a': 1 }, { 'w=b': 1 }], y: ['X', 'Y'] }]
 * trainCrf(linearChainCrf(['X', 'Y'], ['w=a', 'w=b']), data, { c2: 0.1 }).sequenceError // => 0
 * ```
 */
export function trainCrf(
  crf: LinearChainCrf,
  data: readonly LabelledSequence[],
  options: CrfTrainOptions = {},
  maxSteps = 200,
): {
  readonly crf: LinearChainCrf
  readonly objective: number
  readonly sequenceError: number
  readonly converged: boolean
  readonly steps: number
} {
  const penalty = { c2: options.c2 ?? 0 }
  const run = runAlgorithm(crfTrainAlgorithm({ crf, data, penalty }, options), {
    maxSteps,
    key: options.key ?? foldIn(makeKey(0), 1),
  })
  return {
    crf: withCrfParameters(crf, run.state.weights),
    objective: run.state.objective,
    sequenceError: run.state.sequenceError,
    converged: run.status === 'converged',
    steps: run.steps,
  }
}
