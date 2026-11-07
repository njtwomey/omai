/**
 * The solvers of a smooth objective, as a table of modes shared by every method whose problem is a differentiable
 * function of a parameter vector: gradient descent with its step rules, conjugate gradient, Newton, BFGS, L-BFGS and
 * coordinate descent, each the library's minimiser from `optim` with its own knobs, each yielding the same fitted
 * parameters (the vector). A method whose objective is smooth takes `smoothSolvers` as its solver slot and builds the
 * objective; the modes do the rest, and the controls, grids and tuners see every solver's knobs through the table's
 * space. The `gradient` mode is the stochastic one: an optimiser (SGD with momentum, RMSprop, Adam and the rest), a
 * learning-rate schedule and an optional batch of examples drawn afresh at every step, all of them knobs of the mode.
 */

import type { Json } from '../base/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import type { Key } from '../random/index.ts'
import {
  asModeConfig,
  bool,
  choice,
  completeOption,
  type DictSpace,
  defaults,
  dict,
  int,
  real,
  type SpaceValue,
} from '../space/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  Log,
  type LogSpec,
  type ModeSpec,
  type ModeTable,
  modes,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { boolKnob, choiceKnob, ftolKnob, given, gradTolKnob, numberKnob } from './_knobs.ts'
import {
  type ConjugateGradientBeta,
  conjugateGradientAlgorithm,
  coordinateDescentAlgorithm,
  type GradientDescentOptions,
  gradientDescentAlgorithm,
} from './minimize/gradient.ts'
import { newtonAlgorithm } from './minimize/newton.ts'
import type { Parameters } from './minimize/objective.ts'
import { bfgsAlgorithm, lbfgsAlgorithm } from './minimize/quasi-newton.ts'
import { optimisers } from './optimiser-table.ts'
import { schedules } from './schedule-table.ts'
import { type Batches, type BatchObjective, type GradientState, gradientAlgorithm } from './stochastic.ts'

/**
 * A smooth problem: a differentiable objective of a parameter vector (or tree) and where to start. An objective that
 * is a mean over examples may take the indices of a batch as its second argument and say how many examples there are
 * in `batches`; the `gradient` mode then draws batches. Every other solver, and a problem without `batches`, uses the
 * whole set.
 */
export interface SmoothProblem {
  readonly objective: BatchObjective
  readonly x0: Parameters
  /** The examples the objective sums over, when it can be summed over a part of them. */
  readonly batches?: Batches
}

/** The fitted parameters every smooth solver yields: the vector reached. */
export type VectorParams = {
  readonly x: number[]
}

/** What every vector minimiser's state holds: the iterate. */
interface WithVector {
  readonly x: Float64Array
}

const vector = (s: WithVector): VectorParams => ({ x: Array.from(s.x) })

const gradTol = gradTolKnob

const gradientDescent: ModeSpec<SmoothProblem, WithVector, VectorParams> = {
  description: 'gradient descent: a line search along the negative gradient, a fixed step, or Barzilai-Borwein steps',
  space: dict({
    step: choice(['line-search', 'fixed', 'bb1', 'bb2'], { default: 'line-search', label: 'step rule' }),
    learningRate: real(1e-5, 1, {
      log: true,
      default: 1e-2,
      label: 'learning rate',
      when: { key: 'step', in: ['fixed', 'bb1', 'bb2'] },
    }),
    gradTol,
  }),
  make: (p, k) =>
    gradientDescentAlgorithm(
      p.objective,
      p.x0,
      given({
        step: choiceKnob<NonNullable<GradientDescentOptions['step']>>(k, 'step'),
        learningRate: numberKnob(k, 'learningRate'),
        gradTol: numberKnob(k, 'gradTol'),
      }),
    ),
  params: vector,
}

const conjugateGradient: ModeSpec<SmoothProblem, WithVector, VectorParams> = {
  description: 'nonlinear conjugate gradient with a strong Wolfe line search',
  space: dict({
    beta: choice(['polak-ribiere', 'fletcher-reeves', 'hestenes-stiefel', 'dai-yuan', 'hager-zhang'], {
      default: 'polak-ribiere',
      label: 'direction update',
    }),
    restart: int(0, 100, { default: 0, label: 'restart every (0: never)' }),
    gradTol,
  }),
  make: (p, k) =>
    conjugateGradientAlgorithm(
      p.objective,
      p.x0,
      given({
        beta: choiceKnob<ConjugateGradientBeta>(k, 'beta'),
        restart: (numberKnob(k, 'restart') ?? 0) > 0 ? numberKnob(k, 'restart') : undefined,
        gradTol: numberKnob(k, 'gradTol'),
      }),
    ),
  params: vector,
}

const newton: ModeSpec<SmoothProblem, WithVector, VectorParams> = {
  description: "Newton's method with a modified Hessian and an Armijo line search",
  space: dict({
    modification: choice(['cholesky', 'eigenvalue'], { default: 'cholesky', label: 'Hessian modification' }),
    beta: real(1e-6, 1, { log: true, default: 1e-3, label: 'smallest shift' }),
    gradTol,
  }),
  make: (p, k) =>
    newtonAlgorithm(
      p.objective,
      p.x0,
      given({
        modification: choiceKnob<'cholesky' | 'eigenvalue'>(k, 'modification'),
        beta: numberKnob(k, 'beta'),
        gradTol: numberKnob(k, 'gradTol'),
      }),
    ),
  params: vector,
}

const bfgs: ModeSpec<SmoothProblem, WithVector, VectorParams> = {
  description: 'BFGS with a More-Thuente line search',
  space: dict({ damped: bool({ default: false, label: "Powell's damping" }), gradTol }),
  make: (p, k) =>
    bfgsAlgorithm(p.objective, p.x0, given({ damped: boolKnob(k, 'damped'), gradTol: numberKnob(k, 'gradTol') })),
  params: vector,
}

const lbfgs: ModeSpec<SmoothProblem, WithVector, VectorParams> = {
  description: 'limited-memory BFGS',
  space: dict({
    memory: int(1, 50, { default: 10, label: 'curvature pairs kept' }),
    ftol: ftolKnob,
    gradTol,
  }),
  make: (p, k) =>
    lbfgsAlgorithm(
      p.objective,
      p.x0,
      given({ memory: numberKnob(k, 'memory'), ftol: numberKnob(k, 'ftol'), gradTol: numberKnob(k, 'gradTol') }),
    ),
  params: vector,
}

const coordinateDescent: ModeSpec<SmoothProblem, WithVector, VectorParams> = {
  description: 'coordinate descent: one coordinate at a time, exactly, by a Newton step or by a gradient step',
  space: dict({
    order: choice(['cyclic', 'random', 'greedy'], { default: 'cyclic', label: 'sweep order' }),
    update: choice(['exact', 'newton', 'gradient'], { default: 'exact', label: 'coordinate update' }),
    learningRate: real(1e-5, 1, {
      log: true,
      default: 1e-2,
      label: 'learning rate',
      when: { key: 'update', in: ['gradient'] },
    }),
    gradTol,
  }),
  make: (p, k) =>
    coordinateDescentAlgorithm(
      p.objective,
      p.x0,
      given({
        order: choiceKnob<'cyclic' | 'random' | 'greedy'>(k, 'order'),
        update: choiceKnob<'exact' | 'newton' | 'gradient'>(k, 'update'),
        learningRate: numberKnob(k, 'learningRate'),
        gradTol: numberKnob(k, 'gradTol'),
      }),
    ),
  params: vector,
}

const gradient: ModeSpec<SmoothProblem, GradientState, VectorParams> = {
  description:
    'stochastic gradient: an optimiser (SGD with momentum, RMSprop, Adam and others) at a scheduled learning rate, over the whole set or a batch drawn at every step',
  space: dict({
    optimiser: optimisers.space,
    learningRate: real(1e-5, 10, { log: true, default: 1e-2, label: 'learning rate' }),
    schedule: schedules.space,
    batch: int(0, 1_000_000, { default: 0, label: 'batch size (0: the whole set)' }),
    gradTol,
  }),
  make: (p, k) => {
    const batch = numberKnob(k, 'batch') ?? 0
    return gradientAlgorithm(p.objective, p.x0, {
      optimiser: optimisers.make(completeOption(optimisers, k['optimiser'])),
      schedule: schedules.make(completeOption(schedules, k['schedule'])),
      batch: batch > 0 ? batch : null,
      ...(p.batches === undefined ? {} : { batches: p.batches }),
      ...given({ learningRate: numberKnob(k, 'learningRate'), gradTol: numberKnob(k, 'gradTol') }),
    })
  },
  params: vector,
}

/**
 * A solver table's configuration with the defaults of the chosen solver filled in where the configuration names only
 * some knobs (`{ kind: 'lbfgs', gradTol: 1e-9 }` becomes a complete value of the table's space, which a method's
 * configuration check requires; the `gradient` mode's optimiser and schedule are filled the same way).
 *
 * @param table - A table of modes.
 * @param config - A configuration naming a solver and some of its knobs.
 * @returns The complete configuration.
 * @example
 * ```ts
 * import { smoothSolvers, solverConfigOf } from 'mdml-engine/optim'
 * solverConfigOf(smoothSolvers, { kind: 'lbfgs', gradTol: 1e-9 }) // => { memory: 10, ftol: 2.220446049250313e-9, gradTol: 1e-9, kind: 'lbfgs' }
 * ```
 */
export function solverConfigOf<Problem>(table: ModeTable<Problem, VectorParams>, config: SpaceValue): SpaceValue {
  const name = table.modeOf(config).name
  const given = asModeConfig(config)
  const nested: Record<string, SpaceValue> = {}
  if (name === 'gradient') {
    if ('optimiser' in given) nested['optimiser'] = completeOption(optimisers, given['optimiser'] as SpaceValue)
    if ('schedule' in given) nested['schedule'] = completeOption(schedules, given['schedule'] as SpaceValue)
  }
  return {
    ...(defaults(table.space.options[name] as DictSpace) as object),
    ...given,
    ...nested,
    kind: name,
  } as SpaceValue
}

/**
 * {@link solverConfigOf} for `smoothSolvers`.
 *
 * @param config - A configuration naming a solver and some of its knobs.
 * @returns The complete configuration, the other knobs at their defaults.
 * @example
 * ```ts
 * import { smoothSolverConfig } from 'mdml-engine/optim'
 * smoothSolverConfig({ kind: 'bfgs' }) // => { damped: false, gradTol: 0.00001, kind: 'bfgs' }
 * ```
 */
export const smoothSolverConfig = (config: SpaceValue): SpaceValue => solverConfigOf(smoothSolvers, config)

/**
 * The solvers of a smooth objective as one slot: gradient descent (default), conjugate gradient, Newton, BFGS, L-BFGS,
 * coordinate descent and the stochastic `gradient` mode, each with its own knobs, all yielding the vector reached.
 *
 * @index table Smooth-objective solvers as one slot: gradient descent to L-BFGS.
 * @example
 * ```ts
 * import { smoothSolvers } from 'mdml-engine/optim'
 * smoothSolvers.names // => ['gradientDescent', 'conjugateGradient', 'newton', 'bfgs', 'lbfgs', 'coordinateDescent', 'gradient']
 * smoothSolvers.modeOf({ kind: 'lbfgs' }).name // => 'lbfgs'
 * ```
 */
export const smoothSolvers: ModeTable<SmoothProblem, VectorParams> = modes(
  { gradientDescent, conjugateGradient, newton, bfgs, lbfgs, coordinateDescent, gradient },
  { label: 'solver', description: 'how the objective is minimised' },
)

/** The channels every smooth solver's state yields, whichever solver it is. */
export interface SolverReached {
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  readonly step: number
  /** The learning rate of the last step (the `gradient` mode only). */
  readonly lr?: number
}

/** The paths `loggedSolver` logs on every step, whichever solver. */
export type SolverPath = 'solver/objective' | 'solver/gradNorm' | 'solver/step'

/** The path only the `gradient` mode logs. */
export type GradientPath = 'solver/lr'

/**
 * The paths `loggedSolver` logs, for a method's contract. It is a fragment: spread it into the method's own contract
 * (`defineContract({ ...solverContract, 'model/loss': scalar() })`) and the method's paths include {@link SolverPath}.
 */
export const solverContract: Contract<unknown, SolverPath> = defineContract({
  'solver/objective': scalar({ description: 'the objective at the iterate' }),
  'solver/gradNorm': scalar({
    description: 'the Euclidean norm of the gradient at the iterate (the solvers stop on its infinity norm, `gradTol`)',
  }),
  'solver/step': scalar({ description: 'the length of the last step (or trust radius, as the solver calls it)' }),
})

/** The path only the `gradient` mode logs, for the contract of a method that can take it. */
export const gradientContract: Contract<unknown, GradientPath> = defineContract({
  'solver/lr': scalar({ description: 'the learning rate of the last step, from the optimiser and its schedule' }),
})

/**
 * A solver table's algorithm as a logged algorithm: the table's solver for the configuration, wrapped so each step
 * logs the objective, the gradient's norm and the step length (`solver/objective`, `solver/gradNorm`, `solver/step`),
 * and the learning rate where the solver has one (`solver/lr`), lets the model log what is its own from the iterate
 * (`extra`), and serialises the state for seeking. `loggedSolver` is this for `smoothSolvers`; a family with its own
 * table (`glmSolvers`) logs by the same convention through this.
 *
 * The algorithm's paths are the solver's ({@link SolverPath}, {@link GradientPath}) and the type parameter `K`, the
 * model's own, which `extra` may log; a method's contract must have all of them (the `gradient` mode logs
 * `solver/lr`, so spread {@link gradientContract} too unless the method excludes that mode). `K` is inferred from the
 * method's path type where the algorithm is returned from `make`.
 *
 * @param table - The table of modes, whose states carry `x`, `value`, `grad` and `step`.
 * @param problem - What the modes build over.
 * @param config - A value of the table's space.
 * @param key - The run's key.
 * @param extra - Called each step with the iterate and the state's channels, to log what the model adds.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { loggedSolverOf, smoothSolvers } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * const problem = { objective: (x: Value) => sum(square(x)), x0: [3, 4] }
 * const algorithm = loggedSolverOf(smoothSolvers, problem, { kind: 'gradientDescent', step: 'fixed', learningRate: 0.25 }, key(0))
 * const seen: Record<string, number> = {}
 * algorithm.step(algorithm.init(key(0)), { log: (path: string, v: number) => { seen[path] = v }, key: key(1) } as never)
 * [algorithm.name, seen['solver/objective']] // => ['gradientDescent', 6.25]
 * ```
 */
export function loggedSolverOf<Problem, K extends string = never>(
  table: ModeTable<Problem, VectorParams>,
  problem: Problem,
  config: SpaceValue,
  key: Key,
  extra?: (x: Float64Array, ctx: StepContext<SolverPath | GradientPath | K>, reached: SolverReached) => void,
): Algorithm<unknown, SolverPath | GradientPath | K> {
  const inner = table.make(problem, config, key)
  return {
    name: inner.name,
    config: config as Json,
    init: (k) => inner.init(k),
    step: (state, ctx) => {
      const next = inner.step(state, ctx)
      const r = next as SolverReached
      ctx.log('solver/objective', r.value)
      ctx.log('solver/gradNorm', dnrm2(r.grad))
      ctx.log('solver/step', r.step)
      if (r.lr !== undefined) ctx.log('solver/lr', r.lr)
      extra?.(r.x, ctx, r)
      return next
    },
    ...(inner.done === undefined ? {} : { done: inner.done }),
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j),
  }
}

/**
 * A smooth solver as a logged algorithm: the library's minimiser for the configured solver, wrapped so each step
 * logs the objective, the gradient's norm and the step length (`solver/objective`, `solver/gradNorm`, `solver/step`),
 * and the learning rate where the solver has one (`solver/lr`, the `gradient` mode), lets the method log what is its own from the iterate (`extra`), and serialises its state for seeking.
 *
 * @param problem - The objective and the start.
 * @param config - A value of `smoothSolvers.space` (the tag and the chosen solver's knobs).
 * @param key - The run's key.
 * @param extra - Called each step with the iterate, to log what the method adds.
 * @returns The algorithm, typed by the solver's paths and the method's own (`K`, see {@link loggedSolverOf}).
 * @example
 * ```ts
 * import { loggedSolver } from 'mdml-engine/optim'
 * import { key } from 'mdml-engine/random'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * const problem = { objective: (x: Value) => sum(square(x)), x0: [3, 4] }
 * const algorithm = loggedSolver(problem, { kind: 'gradientDescent', step: 'fixed', learningRate: 0.25 }, key(0))
 * const seen: Record<string, number> = {}
 * algorithm.step(algorithm.init(key(0)), { log: (path: string, v: number) => { seen[path] = v }, key: key(1) } as never)
 * seen['solver/objective'] // => 6.25
 * ```
 */
export function loggedSolver<K extends string = never>(
  problem: SmoothProblem,
  config: SpaceValue,
  key: Key,
  extra?: (x: Float64Array, ctx: StepContext<SolverPath | GradientPath | K>) => void,
): Algorithm<unknown, SolverPath | GradientPath | K> {
  return loggedSolverOf<SmoothProblem, K>(smoothSolvers, problem, config, key, extra)
}

/** What `loggedSolver`'s paths keep by default. */
export const solverDefaults: Readonly<Record<SolverPath, LogSpec>> = {
  'solver/objective': Log(),
  'solver/gradNorm': Log(),
  'solver/step': Log(),
}

/** What the `gradient` mode's learning-rate path keeps by default. */
export const gradientDefaults: Readonly<Record<GradientPath, LogSpec>> = {
  'solver/lr': Log(),
}
