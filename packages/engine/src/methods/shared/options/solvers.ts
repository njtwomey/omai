/**
 * The solvers of a smooth objective, as a table of modes shared by every method whose problem is a differentiable
 * function of a parameter vector: gradient descent with its step rules, conjugate gradient, Newton, BFGS, L-BFGS and
 * coordinate descent, each the library's minimiser from `optim` with its own knobs, each yielding the same fitted
 * parameters (the vector). A method whose objective is smooth takes `smoothSolvers` as its solver slot and builds the
 * objective; the modes do the rest, and the controls, grids and tuners see every solver's knobs through the table's
 * space.
 */

import {
  bfgsAlgorithm,
  type ConjugateGradientBeta,
  conjugateGradientAlgorithm,
  coordinateDescentAlgorithm,
  type GradientDescentOptions,
  gradientDescentAlgorithm,
  lbfgsAlgorithm,
  newtonAlgorithm,
  type Objective,
  type Parameters,
} from '../../../lib/optim/index.ts'
import { bool, choice, dict, int, real } from '../../../lib/space/index.ts'
import { type Mode, type ModeTable, modes } from '../modes.ts'

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

const gradientDescent: Mode<SmoothProblem, WithVector, VectorParams> = {
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

const conjugateGradient: Mode<SmoothProblem, WithVector, VectorParams> = {
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

const newton: Mode<SmoothProblem, WithVector, VectorParams> = {
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

const bfgs: Mode<SmoothProblem, WithVector, VectorParams> = {
  description: 'BFGS with a More-Thuente line search',
  space: dict({ damped: bool({ default: false, label: "Powell's damping" }), gradTol }),
  make: (p, k) => bfgsAlgorithm(p.objective, p.x0, { damped: k['damped'] as boolean, gradTol: k['gradTol'] as number }),
  params: vector,
}

const lbfgs: Mode<SmoothProblem, WithVector, VectorParams> = {
  description: 'limited-memory BFGS',
  space: dict({ memory: int(1, 50, { default: 10, label: 'curvature pairs kept' }), gradTol }),
  make: (p, k) => lbfgsAlgorithm(p.objective, p.x0, { memory: k['memory'] as number, gradTol: k['gradTol'] as number }),
  params: vector,
}

const coordinateDescent: Mode<SmoothProblem, WithVector, VectorParams> = {
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
