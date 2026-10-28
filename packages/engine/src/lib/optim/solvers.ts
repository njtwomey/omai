/**
 * The solvers of a smooth objective, as a table of modes shared by every method whose problem is a differentiable
 * function of a parameter vector: gradient descent with its step rules, conjugate gradient, Newton, BFGS, L-BFGS and
 * coordinate descent, each the library's minimiser from `optim` with its own knobs, each yielding the same fitted
 * parameters (the vector). A method whose objective is smooth takes `smoothSolvers` as its solver slot and builds the
 * objective; the modes do the rest, and the controls, grids and tuners see every solver's knobs through the table's
 * space.
 */

import type { Json } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import { bool, choice, dict, int, real, type SpaceValue } from '../space/index.ts'
import {
  type Algorithm,
  type Contract,
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
import {
  type ConjugateGradientBeta,
  conjugateGradientAlgorithm,
  coordinateDescentAlgorithm,
  type GradientDescentOptions,
  gradientDescentAlgorithm,
} from './minimize/gradient.ts'
import { newtonAlgorithm } from './minimize/newton.ts'
import type { Objective, Parameters } from './minimize/objective.ts'
import { bfgsAlgorithm, lbfgsAlgorithm } from './minimize/quasi-newton.ts'

/** A smooth problem: a differentiable objective of a parameter vector (or tree) and where to start. */
export interface SmoothProblem {
  readonly objective: Objective
  readonly x0: Parameters
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

/** The gradient tolerance every gradient-based solver stops at (scipy's gtol). */
const gradTol = real(1e-12, 1e-2, { log: true, default: 1e-5, label: 'gradient tolerance' })

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
    gradientDescentAlgorithm(p.objective, p.x0, {
      step: k['step'] as NonNullable<GradientDescentOptions['step']>,
      ...(k['learningRate'] === undefined ? {} : { learningRate: k['learningRate'] as number }),
      gradTol: k['gradTol'] as number,
    }),
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
    conjugateGradientAlgorithm(p.objective, p.x0, {
      beta: k['beta'] as ConjugateGradientBeta,
      ...((k['restart'] as number) > 0 ? { restart: k['restart'] as number } : {}),
      gradTol: k['gradTol'] as number,
    }),
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
    newtonAlgorithm(p.objective, p.x0, {
      modification: k['modification'] as 'cholesky' | 'eigenvalue',
      beta: k['beta'] as number,
      gradTol: k['gradTol'] as number,
    }),
  params: vector,
}

const bfgs: ModeSpec<SmoothProblem, WithVector, VectorParams> = {
  description: 'BFGS with a More-Thuente line search',
  space: dict({ damped: bool({ default: false, label: "Powell's damping" }), gradTol }),
  make: (p, k) => bfgsAlgorithm(p.objective, p.x0, { damped: k['damped'] as boolean, gradTol: k['gradTol'] as number }),
  params: vector,
}

const lbfgs: ModeSpec<SmoothProblem, WithVector, VectorParams> = {
  description: 'limited-memory BFGS',
  space: dict({ memory: int(1, 50, { default: 10, label: 'curvature pairs kept' }), gradTol }),
  make: (p, k) => lbfgsAlgorithm(p.objective, p.x0, { memory: k['memory'] as number, gradTol: k['gradTol'] as number }),
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
    coordinateDescentAlgorithm(p.objective, p.x0, {
      order: k['order'] as 'cyclic' | 'random' | 'greedy',
      update: k['update'] as 'exact' | 'newton' | 'gradient',
      ...(k['learningRate'] === undefined ? {} : { learningRate: k['learningRate'] as number }),
      gradTol: k['gradTol'] as number,
    }),
  params: vector,
}

/**
 * The solvers of a smooth objective as one slot: gradient descent (default), conjugate gradient, Newton, BFGS, L-BFGS
 * and coordinate descent, each with its own knobs, all yielding the vector reached.
 */
export const smoothSolvers: ModeTable<SmoothProblem, VectorParams> = modes(
  { gradientDescent, conjugateGradient, newton, bfgs, lbfgs, coordinateDescent },
  { label: 'solver', description: 'how the objective is minimised' },
)

/** The channels every smooth solver's state yields, whichever solver it is. */
interface Reached {
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  readonly step: number
}

/**
 * A smooth solver as a logged algorithm: the library's minimiser for the configured solver, wrapped so each step
 * logs the objective, the gradient's norm and the step length (`solver/objective`, `solver/gradNorm`, `solver/step`),
 * lets the method log what is its own from the iterate (`extra`), and serialises its state for seeking.
 *
 * @param problem - The objective and the start.
 * @param config - A value of `smoothSolvers.space` (the tag and the chosen solver's knobs).
 * @param key - The run's key.
 * @param extra - Called each step with the iterate, to log what the method adds.
 * @returns The algorithm.
 */
export function loggedSolver(
  problem: SmoothProblem,
  config: SpaceValue,
  key: Key,
  extra?: (x: Float64Array, ctx: StepContext) => void,
): Algorithm<unknown> {
  const inner = smoothSolvers.make(problem, config, key)
  return {
    name: inner.name,
    config: config as Json,
    init: (k) => inner.init(k),
    step: (state, ctx) => {
      const next = inner.step(state, ctx)
      const r = next as Reached
      ctx.log('solver/objective', r.value)
      ctx.log('solver/gradNorm', Math.hypot(...r.grad))
      ctx.log('solver/step', r.step)
      extra?.(r.x, ctx)
      return next
    },
    ...(inner.done === undefined ? {} : { done: inner.done }),
    serialise: (s) => stateToJson(s) as Json,
    deserialise: (j) => stateFromJson(j),
  }
}

/** The paths `loggedSolver` logs, for a method's contract. */
export const solverContract: Contract<unknown> = {
  'solver/objective': scalar({ description: 'the objective at the iterate' }),
  'solver/gradNorm': scalar({ description: 'the norm of the gradient at the iterate' }),
  'solver/step': scalar({ description: 'the length of the last step (or trust radius, as the solver calls it)' }),
}

/** What `loggedSolver`'s paths keep by default. */
export const solverDefaults: Readonly<Record<string, LogSpec>> = {
  'solver/objective': Log(),
  'solver/gradNorm': Log(),
  'solver/step': Log(),
}
