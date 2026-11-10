/**
 * Training linear-chain CRFs through the solver slot, as a thin layer over the generic structured code. A CRF training
 * problem becomes a `StructuredProblem` (the chain oracle over the dataset compiled to ids once, `crfStructuredProblem`),
 * the objective is the penalised negative conditional log-likelihood (`crfSmoothProblem`, with the gradient as expected
 * minus observed feature counts, and unbiased batch estimates for the stochastic solvers), and how it is minimised is
 * the slot `crfSolvers`, the modes of `structuredSolvers` over the CRF problem: L-BFGS exact training is the default,
 * every generic solver of `smoothSolvers` is there beside it, `owlqn` handles the L1 penalty `c1` with exact zeros, and
 * the online algorithms of structured prediction are named modes: the structured perceptron (Collins 2002), its
 * averaged form, passive-aggressive with the Hamming loss (Crammer et al. 2006) and AROW. Each epoch of the online
 * modes decodes the sequences one at a time with the current weights and moves the weights on a mistake; they do not
 * minimise the likelihood but converge to zero errors on separable data, and the run stops when the reported weights
 * decode every training sequence or an epoch needs no update. Every step puts the penalised objective, the Hamming loss
 * and the sequence error of the weights on the tape.
 */

import { OmaiError } from '../../base/index.ts'
import { familySolvers, type SmoothProblem, slotAlgorithm, type VectorParams } from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import type { SpaceValue } from '../../space/index.ts'
import { type Algorithm, type ModeTable, runAlgorithm } from '../../trace/index.ts'
import { type ChainInput, type ChainLabels, chainOracle } from './chain-oracle.ts'
import { compileChainData } from './compiled.ts'
import { type CrfPenalty, crfParameters, type LabelledSequence, type LinearChainCrf, withCrfParameters } from './crf.ts'
import { structuredSmoothProblem } from './objective.ts'
import type { StructuredProblem } from './oracle.ts'
import {
  overStructuredProblem,
  type StructuredOnlineState,
  structuredModes,
  structuredTrainMetrics,
  structuredTrainView,
} from './solvers.ts'

/** What a CRF is trained on: the starting model (labels and vocabularies), the labelled sequences and the penalties. */
export interface CrfTrainingProblem {
  readonly crf: LinearChainCrf
  readonly data: readonly LabelledSequence[]
  readonly penalty?: CrfPenalty
}

const structured = new WeakMap<CrfTrainingProblem, StructuredProblem<ChainInput, ChainLabels>>()

/**
 * A CRF training problem as a structured problem: the chain oracle over the dataset compiled once (attributes resolved
 * to ids in typed arrays), the penalties and the model's weights as the start. The result is kept for the problem, so
 * building the several algorithms of a solver slot compiles the data once.
 *
 * @param problem - The model, data and penalty.
 * @returns The structured problem the generic solvers train.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { crfStructuredProblem, linearChainCrf } from 'omai-engine/pgm'
 * const p = crfStructuredProblem({ crf: linearChainCrf(['A', 'B'], ['f']), data: [{ x: [{ f: 1 }], y: ['A'] }] })
 * [p.oracle.size, p.data.length] // => [6, 1]
 * ```
 */
export function crfStructuredProblem(problem: CrfTrainingProblem): StructuredProblem<ChainInput, ChainLabels> {
  const known = structured.get(problem)
  if (known !== undefined) return known
  const made: StructuredProblem<ChainInput, ChainLabels> = {
    oracle: chainOracle(problem.crf),
    data: compileChainData(problem.crf, problem.data),
    start: crfParameters(problem.crf),
    ...(problem.penalty === undefined ? {} : { penalty: problem.penalty }),
  }
  structured.set(problem, made)
  return made
}

/**
 * The CRF training objective as a smooth problem for the solver slot: the penalised negative conditional log-likelihood
 * of the labelled sequences (`structuredSmoothProblem` over the chain oracle on the compiled data, equal to
 * `crfObjective`), with the gradient as expected minus observed feature counts, and unbiased batch estimates for the
 * stochastic solvers (the likelihood part rescaled by n / |batch|). The L1 penalty enters by its subgradient (as in
 * `crfObjective`); for exact zeros use the `owlqn` mode of `crfSolvers`.
 *
 * @param problem - The model, data and penalty.
 * @returns The problem: objective (with batches over the sequences) and the starting weights.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { crfSmoothProblem, linearChainCrf } from 'omai-engine/pgm'
 * crfSmoothProblem({ crf: linearChainCrf(['A', 'B'], ['f']), data: [{ x: [{ f: 1 }], y: ['A'] }] }).x0.length // => 6
 * ```
 */
export function crfSmoothProblem(problem: CrfTrainingProblem): SmoothProblem {
  return structuredSmoothProblem(crfStructuredProblem(problem))
}

/** The state of an online CRF trainer: the reported weights, their objective and gradient, and the epoch's mistakes. */
export type CrfOnlineState = StructuredOnlineState

/**
 * The ways to train a CRF: L-BFGS on the exact objective is the default, beside every generic smooth solver (Newton,
 * BFGS, conjugate gradient, coordinate descent and the stochastic `gradient` mode with its optimiser, schedule and
 * batch), `owlqn` for the L1 penalty `c1` (OWL-QN: exact zeros), and the online algorithms of structured prediction as
 * modes: `perceptron` (Collins), `averaged` (the perceptron reporting the average of its weights over all examples
 * seen), `passive-aggressive` (PA-I with the Hamming loss) and `arow`. They are the modes of `structuredSolvers` over
 * the chain oracle.
 *
 * @index table CRF training solvers: L-BFGS, OWL-QN for L1, SGD, perceptron, passive-aggressive, AROW.
 */
export const crfSolvers: ModeTable<CrfTrainingProblem, VectorParams> = familySolvers<CrfTrainingProblem>(
  overStructuredProblem<CrfTrainingProblem>(structuredModes, crfStructuredProblem),
  crfSmoothProblem,
  {
    label: 'solver',
    description: 'how the conditional log-likelihood is maximised or the structured loss reduced',
    default: 'lbfgs',
  },
)

/** Options of {@link crfTrainAlgorithm} and {@link trainCrf}. */
export interface CrfTrainOptions extends CrfPenalty {
  /** The solver, a value of `crfSolvers.space` (default L-BFGS, or OWL-QN when `c1` is given). */
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
  /** The mean Hamming loss of decoding the training sequences. */
  readonly loss: number
  /** The weights away from zero. */
  readonly nonzeros: number
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
 * import { crfTrainAlgorithm, linearChainCrf } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const data = [{ x: [{ 'w=a': 1 }, { 'w=b': 1 }], y: ['X', 'Y'] }]
 * const crf = linearChainCrf(['X', 'Y'], ['w=a', 'w=b'])
 * runAlgorithm(crfTrainAlgorithm({ crf, data, penalty: { c2: 0.1 } }), { maxSteps: 100 }).state.sequenceError // => 0
 * ```
 */
export function crfTrainAlgorithm(
  problem: CrfTrainingProblem,
  options: CrfTrainOptions = {},
): Algorithm<CrfTrainState> {
  if (problem.data.length === 0) throw new OmaiError('value', 'crfTrain: no training sequences')
  const c1 = problem.penalty?.c1 ?? 0
  const solver = options.solver ?? { kind: c1 > 0 ? 'owlqn' : 'lbfgs' }
  const tol = options.tol ?? 1e-6
  const P = crfStructuredProblem(problem).oracle.size
  const metrics = structuredTrainMetrics({
    name: 'sequenceError',
    description: 'the fraction of training sequences not decoded exactly right',
  })
  return slotAlgorithm<CrfTrainState, CrfTrainingProblem>({
    name: 'crf-training',
    config: { solver, sequences: problem.data.length, c2: problem.penalty?.c2 ?? 0, c1 } as never,
    schema: {
      metrics,
      tensors: { weights: { role: 'param', description: 'the CRF weights: state, transition, bigram', shape: [P] } },
    },
    problem,
    table: crfSolvers,
    solver,
    tol,
    key: options.key ?? makeKey(0),
    view: (r) => {
      const { error, ...rest } = structuredTrainView(crfStructuredProblem(problem), r)
      return { ...rest, sequenceError: error, evaluations: 0, stalled: false }
    },
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        gradient: s.gradient,
        loss: s.loss,
        sequenceError: s.sequenceError,
        nonzeros: s.nonzeros,
        mistakes: s.mistakes,
      },
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
 * import { linearChainCrf, trainCrf } from 'omai-engine/pgm'
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
  const penalty = { c2: options.c2 ?? 0, ...(options.c1 === undefined ? {} : { c1: options.c1 }) }
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
