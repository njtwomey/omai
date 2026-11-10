/**
 * The solvers of every structured model, as one slot over a `StructuredProblem` (an oracle, labelled examples and the
 * penalties). The model is an objective, the penalised negative conditional log-likelihood of `structuredObjective`;
 * how it is minimised is `structuredSolvers`: L-BFGS (the default) and every generic smooth solver of `smoothSolvers`
 * beside it, `owlqn` for the L1 penalty (OWL-QN of `optim`: exact zeros at quasi-Newton speed, the L2 part staying in
 * the smooth objective), and the online algorithms of structured prediction as named modes: the structured perceptron
 * (Collins 2002), its averaged form, passive-aggressive updates with the task loss (Crammer et al. 2006) and AROW
 * (Crammer, Kulesza and Dredze 2009). The online modes need only `decode` (with `against` for the cost-augmented ones),
 * `features` and `loss` of the oracle. `structuredTrainAlgorithm` runs any of them as a traced algorithm and
 * `trainStructured` to the end; another family of learners (a structured SVM) joins the table through
 * `structuredSolversWith`.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { ddot, dnrm2 } from '../../numerics/linalg/index.ts'
import {
  familySolvers,
  owlqnMode,
  type SlotState,
  type SolverReached,
  slotAlgorithm,
  type VectorParams,
} from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../../random/index.ts'
import { bool, dict, real, type SpaceValue } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type ModeSpec,
  type ModeTable,
  runAlgorithm,
  stateFromJson,
  stateToJson,
} from '../../trace/index.ts'
import { structuredObjective, structuredSmoothProblem } from './objective.ts'
import type { StructuredProblem } from './oracle.ts'

/** A structured problem with its example and output types erased, what the solver table is a table over. */
export type AnyStructuredProblem = StructuredProblem<unknown, unknown>

/** A mode of a structured solver table: its state is its own, every mode yields the weight vector. */
// biome-ignore lint/suspicious/noExplicitAny: each mode has a state type of its own; the table erases them
export type StructuredMode = ModeSpec<AnyStructuredProblem, any, VectorParams>

/**
 * The task loss of the weights on the training examples: the mean loss of the decoded labelling against the gold one,
 * and the fraction of examples decoded with any loss at all.
 *
 * @param problem - The oracle and the examples.
 * @param w - The weights.
 * @returns The mean task loss and the fraction of examples not decoded exactly right.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf, structuredErrors } from 'omai-engine/pgm'
 * const oracle = chainOracle(linearChainCrf(['A', 'B'], ['f']))
 * const data = [{ x: [{ f: 1 }, { f: 1 }], y: ['A', 'B'] }]
 * structuredErrors({ oracle, data }, [1, -1, 0, 0, 0, 0]) // => { loss: 1, error: 1 }
 * ```
 */
export function structuredErrors<X, Y>(
  problem: StructuredProblem<X, Y>,
  w: ArrayLike<number>,
): { readonly loss: number; readonly error: number } {
  const { oracle, data } = problem
  if (data.length === 0) return { loss: 0, error: 0 }
  let loss = 0
  let wrong = 0
  for (const { x, y } of data) {
    const l = oracle.loss(y, oracle.decode(w, x).y)
    loss += l
    if (l !== 0) wrong++
  }
  return { loss: loss / data.length, error: wrong / data.length }
}

/**
 * The state of an online structured trainer: the reported weights, their penalised objective and gradient, and the
 * epoch's mistakes.
 */
export interface StructuredOnlineState {
  /** The weights that are reported (the running average for the averaged perceptron, else the current ones). */
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  readonly step: number
  /** The current weights (equal to `x` except for the averaged perceptron). */
  readonly w: Float64Array
  /**
   * For the averaged perceptron, the sum over the updates of (the example's number) times the update, from which the
   * average of the weights over all examples is (seen + 1) w minus this, over seen (so averaging costs nothing per
   * example); zeros for the other rules.
   */
  readonly weighted: Float64Array
  /** AROW's diagonal covariance of the weights (empty for the other rules). */
  readonly variance: Float64Array
  /** Examples seen. */
  readonly seen: number
  /** Updates made in the last epoch (the perceptron's mistakes). */
  readonly mistakes: number
}

type OnlineRule = 'perceptron' | 'averaged' | 'passive-aggressive' | 'arow'

function onlineMode(
  rule: OnlineRule,
  description: string,
): ModeSpec<AnyStructuredProblem, StructuredOnlineState, VectorParams> {
  return {
    description,
    space: dict(
      rule === 'arow'
        ? {
            variance: real(1e-4, 1e4, { log: true, default: 1, label: 'initial variance of each weight' }),
            gamma: real(1e-4, 1e4, { log: true, default: 1, label: 'regularisation r' }),
            shuffle: bool({ default: true, label: 'shuffle each epoch' }),
          }
        : {
            learningRate: real(1e-3, 10, { log: true, default: 1, label: 'learning rate' }),
            aggressiveness: real(1e-4, 1e4, { log: true, default: 1, label: 'aggressiveness C' }),
            shuffle: bool({ default: true, label: 'shuffle each epoch' }),
          },
    ),
    make: (problem, knobs): Algorithm<StructuredOnlineState> => {
      const { oracle, data } = problem
      const number = (name: string, fallback: number): number => (knobs[name] as number | undefined) ?? fallback
      const eta = number('learningRate', 1)
      const aggressiveness = number('aggressiveness', 1)
      const variance0 = number('variance', 1)
      const gamma = number('gamma', 1)
      const shuffle = (knobs['shuffle'] as boolean | undefined) ?? true
      const penalty = problem.penalty ?? {}
      const P = oracle.size
      const start = Float64Array.from(problem.start ?? new Float64Array(P))
      const report = (
        w: Float64Array,
        weighted: Float64Array,
        variance: Float64Array,
        seen: number,
        mistakes: number,
        step: number,
      ): StructuredOnlineState => {
        const x = rule === 'averaged' && seen > 0 ? w.map((v, q) => ((seen + 1) * v - (weighted[q] ?? 0)) / seen) : w
        const r = structuredObjective(oracle, data, x, penalty)
        return { x, value: r.value, grad: r.gradient, step, w, weighted, variance, seen, mistakes }
      }
      return {
        name: `structured-${rule}`,
        config: { rule, learningRate: eta, aggressiveness, variance: variance0, gamma, shuffle, examples: data.length },
        init: () =>
          report(
            start,
            new Float64Array(P),
            rule === 'arow' ? new Float64Array(P).fill(variance0) : new Float64Array(0),
            0,
            -1,
            0,
          ),
        step: (s, ctx) => {
          const order = shuffle
            ? Array.from(toFlat(permutation(ctx.key, data.length)))
            : Array.from({ length: data.length }, (_, i) => i)
          const w = Float64Array.from(s.w)
          const weighted = Float64Array.from(s.weighted)
          const variance = Float64Array.from(s.variance)
          let seen = s.seen
          let mistakes = 0
          const costAugmented = rule === 'passive-aggressive' || rule === 'arow'
          for (const i of order) {
            const example = data[i]
            if (example === undefined) continue
            const { x, y } = example
            seen++
            const found = oracle.decode(w, x, costAugmented ? { against: y } : undefined)
            if (oracle.loss(y, found.y) === 0) continue
            mistakes++
            const diff = oracle.features(x, y)
            const guess = oracle.features(x, found.y)
            for (let q = 0; q < P; q++) diff[q] = (diff[q] ?? 0) - (guess[q] ?? 0)
            // the margin violation: the score of the (cost-augmented) prediction over the gold one
            const violation = found.score - ddot(w, oracle.features(x, y))
            const update = (q: number, step: number): void => {
              w[q] = (w[q] ?? 0) + step
              if (rule === 'averaged') weighted[q] = (weighted[q] ?? 0) + seen * step
            }
            if (rule === 'arow') {
              if (violation <= 0) continue
              let confidence = 0
              for (let q = 0; q < P; q++) confidence += (diff[q] ?? 0) ** 2 * (variance[q] ?? 0)
              const beta = 1 / (confidence + gamma)
              for (let q = 0; q < P; q++) {
                const d = diff[q] ?? 0
                if (d === 0) continue
                const sigma = variance[q] ?? 0
                update(q, violation * beta * sigma * d)
                variance[q] = sigma - beta * (sigma * d) ** 2
              }
              continue
            }
            let step = eta
            if (rule === 'passive-aggressive') {
              const norm = ddot(diff, diff)
              step = norm > 0 ? Math.min(aggressiveness, Math.max(violation, 0) / norm) : 0
            }
            for (let q = 0; q < P; q++) if (diff[q] !== 0) update(q, step * (diff[q] ?? 0))
          }
          const next = report(w, weighted, variance, seen, mistakes, 0)
          return { ...next, step: dnrm2(next.x.map((v, q) => v - (s.x[q] ?? 0))) }
        },
        // done once an epoch made no update, or the reported weights decode every training example exactly (the
        // passive-aggressive update keeps asking for a margin after the decoding is right; the average of an averaged
        // perceptron may separate before or after the weights themselves stop changing)
        done: (s) =>
          s.mistakes === 0 || (s.mistakes > 0 && structuredErrors(problem, s.x).error === 0) ? 'converged' : undefined,
        serialise: (s) => stateToJson(s) as Json,
        deserialise: (j) => stateFromJson(j) as StructuredOnlineState,
      }
    },
    params: (s) => ({ x: Array.from(s.x) }),
  }
}

/**
 * The modes of a structured learner that are not generic smooth solvers, by name: `owlqn`, `perceptron`, `averaged`,
 * `passive-aggressive` and `arow`. A model family whose problem is a `StructuredProblem` (or can be mapped to one with
 * {@link overStructuredProblem}) takes these as the extra modes of `familySolvers`.
 */
export const structuredModes: Readonly<Record<string, StructuredMode>> = {
  owlqn: owlqnMode<AnyStructuredProblem>({
    toSmooth: (p) => structuredSmoothProblem({ ...p, penalty: { c2: p.penalty?.c2 ?? 0 } }),
    l1: (p) => p.penalty?.c1 ?? 0,
  }),
  perceptron: onlineMode(
    'perceptron',
    'structured perceptron: decode each example with the current weights and add the difference of the gold and predicted feature counts on a mistake',
  ),
  averaged: onlineMode(
    'averaged',
    'averaged structured perceptron: the same updates, the reported weights the average over every example seen',
  ),
  'passive-aggressive': onlineMode(
    'passive-aggressive',
    'passive-aggressive (PA-I): the smallest change that makes the gold labelling win by its task loss to the cost-augmented prediction, capped by C',
  ),
  arow: onlineMode(
    'arow',
    'AROW (adaptive regularisation of weights): a confidence-weighted update that shrinks the variance of the weights it moves, against the cost-augmented prediction',
  ),
}

/**
 * The modes of a structured solver table over a family's own problem type: each mode builds over `toStructured(p)`.
 *
 * @param modes - Modes over a `StructuredProblem` ({@link structuredModes}, or a family's own).
 * @param toStructured - The family's problem as a structured problem.
 * @returns The modes over the family's problem.
 * @example
 * ```ts
 * import { overStructuredProblem, structuredModes } from 'omai-engine/pgm'
 * type P = { readonly name: string }
 * Object.keys(overStructuredProblem<P>({ perceptron: structuredModes.perceptron! }, () => { throw new Error('unused') })) // => ['perceptron']
 * ```
 */
export function overStructuredProblem<P>(
  modes: Readonly<Record<string, StructuredMode>>,
  toStructured: (problem: P) => AnyStructuredProblem,
  // biome-ignore lint/suspicious/noExplicitAny: each mode has a state type of its own
): Record<string, ModeSpec<P, any, VectorParams>> {
  // biome-ignore lint/suspicious/noExplicitAny: each mode has a state type of its own
  const out: Record<string, ModeSpec<P, any, VectorParams>> = {}
  for (const [name, mode] of Object.entries(modes))
    out[name] = { ...mode, make: (p, k, key) => mode.make(toStructured(p), k, key) }
  return out
}

/**
 * The structured solver table with extra modes of a family beside {@link structuredModes}: the same slot, so a
 * structured SVM or any other learner that needs only the oracle is one more mode.
 *
 * @param extra - More modes over a structured problem, by name.
 * @returns The table: L-BFGS the default, every smooth solver, `owlqn`, the online modes and the extras.
 * @example
 * ```ts
 * import { structuredSolversWith } from 'omai-engine/pgm'
 * structuredSolversWith({}).modeOf({}).name // => 'lbfgs'
 * ```
 */
export function structuredSolversWith(
  extra: Readonly<Record<string, StructuredMode>> = {},
): ModeTable<AnyStructuredProblem, VectorParams> {
  return familySolvers<AnyStructuredProblem>({ ...structuredModes, ...extra }, structuredSmoothProblem, {
    label: 'solver',
    description: 'how the conditional log-likelihood is maximised or the structured loss reduced',
    default: 'lbfgs',
  })
}

/**
 * The ways to train a structured model: L-BFGS on the exact objective is the default, beside every generic smooth solver
 * (Newton, BFGS, conjugate gradient, coordinate descent and the stochastic `gradient` mode), `owlqn` for the L1 penalty
 * c1 (exact zeros), and the online algorithms `perceptron`, `averaged`, `passive-aggressive` and `arow`.
 *
 * @index table Structured-model solvers: L-BFGS, OWL-QN for L1, SGD, perceptron, passive-aggressive, AROW.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { structuredSolvers } from 'omai-engine/pgm'
 * [structuredSolvers.modeOf({}).name, structuredSolvers.names.includes('owlqn'), structuredSolvers.names.includes('arow')] // => ['lbfgs', true, true]
 * ```
 */
export const structuredSolvers: ModeTable<AnyStructuredProblem, VectorParams> = structuredSolversWith()

/** What the traced training of a structured model reports on every step, from the solver's iterate. */
export interface StructuredTrainView {
  readonly weights: number[]
  /** The penalised negative conditional log-likelihood (the L1 term included) at the weights. */
  readonly objective: number
  /** The largest entry of the gradient (the pseudo-gradient for OWL-QN, a subgradient for the online modes). */
  readonly gradient: number
  /** The mean task loss of decoding the training examples. */
  readonly loss: number
  /** The fraction of training examples not decoded exactly right. */
  readonly error: number
  /** The weights away from zero. */
  readonly nonzeros: number
  /** The updates made in the last epoch (online modes; -1 before the first, -2 for the other solvers). */
  readonly mistakes: number
}

/**
 * What a solver's iterate means for the structured model: the weights, the objective and gradient, the task loss and
 * error of decoding the training examples, the number of nonzero weights and the online mistakes. The view and the
 * channels of a model that trains through the slot are built from this.
 *
 * @param problem - The oracle and the examples.
 * @param r - The iterate of any solver of the slot.
 * @returns The view.
 */
export function structuredTrainView(problem: AnyStructuredProblem, r: SolverReached): StructuredTrainView {
  const { loss, error } = structuredErrors(problem, r.x)
  let largest = 0
  for (const g of r.grad) largest = Math.max(largest, Math.abs(g))
  return {
    weights: Array.from(r.x),
    objective: r.value,
    gradient: largest,
    loss,
    error,
    nonzeros: r.x.reduce((k, v) => k + (v === 0 ? 0 : 1), 0),
    mistakes: (r as unknown as Partial<StructuredOnlineState>).mistakes ?? -2,
  }
}

/** The metric channels of {@link structuredTrainView}, with the name and description of the error channel. */
export function structuredTrainMetrics(error: {
  readonly name: string
  readonly description: string
}): Readonly<Record<string, { readonly description: string; readonly direction?: 'lower' | 'higher' }>> {
  return {
    objective: { description: 'the penalised negative conditional log-likelihood', direction: 'lower' },
    gradient: { description: 'the largest gradient entry (pseudo-gradient for OWL-QN)', direction: 'lower' },
    loss: { description: 'the mean task loss of decoding the training examples', direction: 'lower' },
    [error.name]: { description: error.description, direction: 'lower' },
    nonzeros: { description: 'the weights away from zero', direction: 'lower' },
    mistakes: { description: 'updates made in the last epoch of an online solver (-1 before the first, -2 otherwise)' },
  }
}

/** Options of {@link structuredTrainAlgorithm} and {@link trainStructured}. */
export interface StructuredTrainOptions {
  /**
   * The solver, a value of `structuredSolvers.space` (default L-BFGS, or `owlqn` when the problem has an L1 penalty
   * `c1`).
   */
  readonly solver?: SpaceValue
  /** Stop when the largest gradient entry falls below this (default 1e-6; the online modes stop at zero mistakes). */
  readonly tol?: number
  /** The key for shuffling and batches. */
  readonly key?: Key
}

/** The state of {@link structuredTrainAlgorithm}: the solver's, and what {@link structuredTrainView} says of it. */
export interface StructuredTrainState extends StructuredTrainView, SlotState {
  readonly evaluations: number
  readonly stalled: boolean
}

/**
 * Structured training as a traced {@link Algorithm} over the slot `structuredSolvers`: whichever solver is chosen,
 * each step puts the penalised objective, the largest gradient entry, the task loss and error of decoding the training
 * examples and the number of nonzero weights on the tape, with the `solver/gradNorm` and `solver/step` of the solver,
 * and the weights.
 *
 * @param problem - The oracle, labelled examples, penalties and start.
 * @param options - The solver and tolerance.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for no examples.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf, structuredTrainAlgorithm } from 'omai-engine/pgm'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const oracle = chainOracle(linearChainCrf(['X', 'Y'], ['w=a', 'w=b']))
 * const data = [{ x: [{ 'w=a': 1 }, { 'w=b': 1 }], y: ['X', 'Y'] }]
 * runAlgorithm(structuredTrainAlgorithm({ oracle, data, penalty: { c2: 0.1 } }), { maxSteps: 100 }).state.error // => 0
 * ```
 */
export function structuredTrainAlgorithm<X, Y>(
  problem: StructuredProblem<X, Y>,
  options: StructuredTrainOptions = {},
): Algorithm<StructuredTrainState> {
  if (problem.data.length === 0) throw new OmaiError('value', 'structuredTrain: no training examples')
  const c1 = problem.penalty?.c1 ?? 0
  const solver = options.solver ?? { kind: c1 > 0 ? 'owlqn' : 'lbfgs' }
  const P = problem.oracle.size
  return slotAlgorithm<StructuredTrainState, AnyStructuredProblem>({
    name: 'structured-training',
    config: {
      solver,
      examples: problem.data.length,
      c2: problem.penalty?.c2 ?? 0,
      c1,
    } as Json,
    schema: {
      metrics: structuredTrainMetrics({
        name: 'error',
        description: 'the fraction of training examples not decoded exactly right',
      }),
      tensors: { weights: { role: 'param', description: 'the model weights, in the oracle layout', shape: [P] } },
    },
    problem: problem as AnyStructuredProblem,
    table: structuredSolvers,
    solver,
    tol: options.tol ?? 1e-6,
    key: options.key ?? makeKey(0),
    view: (r) => ({ ...structuredTrainView(problem as AnyStructuredProblem, r), evaluations: 0, stalled: false }),
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        gradient: s.gradient,
        loss: s.loss,
        error: s.error,
        nonzeros: s.nonzeros,
        mistakes: s.mistakes,
      },
      tensors: { weights: { role: 'param', value: s.weights, shape: [P] } },
    }),
  })
}

/**
 * Train a structured model with any solver of the slot to the end: the weights, the penalised objective, the task loss
 * and error on the training examples, the nonzero weights and whether the solver stopped on its own criterion.
 *
 * @param problem - The oracle, labelled examples, penalties and start.
 * @param options - The solver, tolerance and key.
 * @param maxSteps - The step budget (epochs for the online modes; default 200).
 * @returns The fitted weights and the training summary.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf, trainStructured } from 'omai-engine/pgm'
 * const oracle = chainOracle(linearChainCrf(['X', 'Y'], ['w=a', 'w=b']))
 * const data = [{ x: [{ 'w=a': 1 }, { 'w=b': 1 }], y: ['X', 'Y'] }]
 * trainStructured({ oracle, data, penalty: { c2: 0.1 } }).error // => 0
 * ```
 */
export function trainStructured<X, Y>(
  problem: StructuredProblem<X, Y>,
  options: StructuredTrainOptions = {},
  maxSteps = 200,
): {
  readonly weights: Float64Array
  readonly objective: number
  readonly loss: number
  readonly error: number
  readonly nonzeros: number
  readonly converged: boolean
  readonly steps: number
} {
  const run = runAlgorithm(structuredTrainAlgorithm(problem, options), {
    maxSteps,
    key: options.key ?? foldIn(makeKey(0), 1),
  })
  const s = run.state
  return {
    weights: Float64Array.from(s.weights),
    objective: s.objective,
    loss: s.loss,
    error: s.error,
    nonzeros: s.nonzeros,
    converged: run.status === 'converged',
    steps: run.steps,
  }
}
