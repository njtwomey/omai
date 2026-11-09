// The problems and the solver slots of the linear support vector machines (`linearSvc`, `linearSvr`, liblinear's models).
// A model is an objective over the weights and the scaled intercept `w~ = (w, b / s)` (liblinear appends the constant
// feature `s`, `interceptScaling`, and penalises it with the weights): the L2-penalised squared-hinge (classification)
// and squared epsilon-insensitive (regression) primals are smooth, so every solver of `smoothSolvers` minimises them,
// beside the models' own modes: `dcd`, dual coordinate descent (liblinear's default, for either loss under the L2
// penalty), and `owlqn` for the L1-penalised squared hinge. Not exported from the barrel.

import { OmaiError } from '../base/index.ts'
import {
  analyticObjective,
  familySolvers,
  linearSvmDualCdAlgorithm,
  linearSvrDualCdAlgorithm,
  owlqnAlgorithm,
  type SmoothProblem,
  type VectorParams,
} from '../optim/index.ts'
import { dict, int } from '../space/index.ts'
import type { ModeSpec, ModeTable } from '../trace/index.ts'

/** The losses of the linear classifier: the hinge, and the squared hinge (liblinear's L2 loss). */
export type LinearSvcLoss = 'hinge' | 'squaredHinge'

/** The penalty on the weights of the linear classifier. */
export type LinearPenalty = 'l1' | 'l2'

/** The losses of the linear regressor: epsilon-insensitive, and its square (liblinear's L2 loss). */
export type LinearSvrLoss = 'epsilonInsensitive' | 'squaredEpsilonInsensitive'

/** The rows augmented with the constant intercept feature when it is fitted. */
export interface LinearDesign {
  readonly n: number
  /** Features, without the constant. */
  readonly d: number
  /** Columns of `augmented`: `d`, or `d + 1` with the intercept. */
  readonly width: number
  readonly fitIntercept: boolean
  /** The value of the constant feature. */
  readonly scaling: number
  /** The rows as given `[n, d]`. */
  readonly raw: Float64Array
  /** The rows with the constant appended `[n, width]`. */
  readonly augmented: Float64Array
}

/** The design of rows `raw` (`[n, d]`, row-major) with the constant feature of an intercept. */
export function linearDesign(
  raw: Float64Array,
  n: number,
  d: number,
  fitIntercept: boolean,
  scaling: number,
): LinearDesign {
  const width = d + (fitIntercept ? 1 : 0)
  const augmented = new Float64Array(n * width)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) augmented[i * width + j] = raw[i * d + j] ?? 0
    if (fitIntercept) augmented[i * width + d] = scaling
  }
  return { n, d, width, fitIntercept, scaling, raw, augmented }
}

const dot = (a: Float64Array, row: number, w: ArrayLike<number>, width: number): number => {
  let z = 0
  for (let j = 0; j < width; j++) z += (a[row * width + j] ?? 0) * (w[j] ?? 0)
  return z
}

/** One binary classification problem of the linear SVC. */
export interface LinearSvcProblem extends SmoothProblem {
  readonly design: LinearDesign
  /** Labels in {-1, +1}. */
  readonly y: Float64Array
  readonly C: number
  /** The multiplier of `C` for each row (class and sample weights). */
  readonly weights: Float64Array
  readonly loss: LinearSvcLoss
  readonly penalty: LinearPenalty
  /** The solvers' tolerance. */
  readonly tol: number
  readonly maxIter: number
}

/** One linear regression problem. */
export interface LinearSvrProblem extends SmoothProblem {
  readonly design: LinearDesign
  readonly y: Float64Array
  readonly C: number
  readonly weights: Float64Array
  readonly loss: LinearSvrLoss
  readonly epsilon: number
  readonly tol: number
  readonly maxIter: number
}

/** The primal objective of a classification problem at `w~`: the penalty plus the penalised losses. */
export function svcPrimal(p: LinearSvcProblem, w: ArrayLike<number>): number {
  const { design: d } = p
  let penalty = 0
  for (let j = 0; j < d.width; j++) penalty += p.penalty === 'l1' ? Math.abs(w[j] ?? 0) : 0.5 * (w[j] ?? 0) ** 2
  let loss = 0
  for (let i = 0; i < d.n; i++) {
    const slack = Math.max(0, 1 - (p.y[i] ?? 1) * dot(d.augmented, i, w, d.width))
    loss += p.C * (p.weights[i] ?? 1) * (p.loss === 'hinge' ? slack : slack * slack)
  }
  return penalty + loss
}

/** The primal objective of a regression problem at `w~`. */
export function svrPrimal(p: LinearSvrProblem, w: ArrayLike<number>): number {
  const { design: d } = p
  let penalty = 0
  for (let j = 0; j < d.width; j++) penalty += 0.5 * (w[j] ?? 0) ** 2
  let loss = 0
  for (let i = 0; i < d.n; i++) {
    const excess = Math.max(0, Math.abs(dot(d.augmented, i, w, d.width) - (p.y[i] ?? 0)) - p.epsilon)
    loss += p.C * (p.weights[i] ?? 1) * (p.loss === 'epsilonInsensitive' ? excess : excess * excess)
  }
  return penalty + loss
}

/**
 * The squared-hinge objective as a smooth problem over `w~`, with its analytic gradient and generalised Hessian: the
 * ridge term `1/2 |w|^2` when `ridge` is set (the L2 penalty), the losses alone otherwise (the smooth part of the L1 one).
 */
export function squaredHingeObjective(p: LinearSvcProblem, ridge: boolean): SmoothProblem {
  const { design: d } = p
  const slackOf = (w: Float64Array): Float64Array =>
    Float64Array.from({ length: d.n }, (_, i) => Math.max(0, 1 - (p.y[i] ?? 1) * dot(d.augmented, i, w, d.width)))
  const objective = analyticObjective({
    value: (w) => {
      const e = slackOf(w)
      let v = 0
      if (ridge) for (const c of w) v += 0.5 * c * c
      for (let i = 0; i < d.n; i++) v += p.C * (p.weights[i] ?? 1) * (e[i] ?? 0) ** 2
      return v
    },
    gradient: (w) => {
      const e = slackOf(w)
      const g = ridge ? Float64Array.from(w) : new Float64Array(d.width)
      for (let i = 0; i < d.n; i++) {
        const c = -2 * p.C * (p.weights[i] ?? 1) * (e[i] ?? 0) * (p.y[i] ?? 1)
        if (c !== 0) for (let j = 0; j < d.width; j++) g[j] = (g[j] ?? 0) + c * (d.augmented[i * d.width + j] ?? 0)
      }
      return g
    },
    hessian: (w) => {
      const e = slackOf(w)
      const h = new Float64Array(d.width * d.width)
      if (ridge) for (let j = 0; j < d.width; j++) h[j * d.width + j] = 1
      for (let i = 0; i < d.n; i++) {
        if (!((e[i] ?? 0) > 0)) continue
        const c = 2 * p.C * (p.weights[i] ?? 1)
        for (let a = 0; a < d.width; a++) {
          const xa = c * (d.augmented[i * d.width + a] ?? 0)
          for (let b = 0; b < d.width; b++)
            h[a * d.width + b] = (h[a * d.width + b] ?? 0) + xa * (d.augmented[i * d.width + b] ?? 0)
        }
      }
      return h
    },
  })
  return { objective, x0: new Float64Array(d.width) }
}

/** The squared epsilon-insensitive primal as a smooth problem over `w~` (always L2-penalised). */
export function squaredEpsilonObjective(p: LinearSvrProblem): SmoothProblem {
  const { design: d } = p
  const excess = (w: Float64Array, i: number): number => {
    const r = dot(d.augmented, i, w, d.width) - (p.y[i] ?? 0)
    return Math.sign(r) * Math.max(0, Math.abs(r) - p.epsilon)
  }
  const objective = analyticObjective({
    value: (w) => {
      let v = 0
      for (const c of w) v += 0.5 * c * c
      for (let i = 0; i < d.n; i++) v += p.C * (p.weights[i] ?? 1) * excess(w, i) ** 2
      return v
    },
    gradient: (w) => {
      const g = Float64Array.from(w)
      for (let i = 0; i < d.n; i++) {
        const c = 2 * p.C * (p.weights[i] ?? 1) * excess(w, i)
        if (c !== 0) for (let j = 0; j < d.width; j++) g[j] = (g[j] ?? 0) + c * (d.augmented[i * d.width + j] ?? 0)
      }
      return g
    },
    hessian: (w) => {
      const h = new Float64Array(d.width * d.width)
      for (let j = 0; j < d.width; j++) h[j * d.width + j] = 1
      for (let i = 0; i < d.n; i++) {
        if (excess(w, i) === 0) continue
        const c = 2 * p.C * (p.weights[i] ?? 1)
        for (let a = 0; a < d.width; a++) {
          const xa = c * (d.augmented[i * d.width + a] ?? 0)
          for (let b = 0; b < d.width; b++)
            h[a * d.width + b] = (h[a * d.width + b] ?? 0) + xa * (d.augmented[i * d.width + b] ?? 0)
        }
      }
      return h
    },
  })
  return { objective, x0: new Float64Array(d.width) }
}

const weightsAsC = (p: { readonly weights: Float64Array }): Float64Array => p.weights

const dcdClassifier: ModeSpec<LinearSvcProblem, { readonly w: Float64Array }, VectorParams> = {
  description:
    "dual coordinate descent (Hsieh et al. 2008, liblinear's default): exact in the dual, one sweep of coordinates per step; the L2 penalty, either loss",
  space: dict({}),
  make: (p) =>
    linearSvmDualCdAlgorithm(rowsOf(p.design), p.y, p.C, {
      loss: p.loss === 'hinge' ? 'l1' : 'l2',
      tol: p.tol,
      fitIntercept: p.design.fitIntercept,
      bias: p.design.scaling,
      weights: weightsAsC(p),
    }),
  params: (s) => ({ x: Array.from(s.w) }),
}

const owlqnClassifier: ModeSpec<LinearSvcProblem, { readonly x: Float64Array }, VectorParams> = {
  description:
    'orthant-wise limited-memory quasi-Newton on the L1-penalised squared hinge: exact zeros at quasi-Newton speed',
  space: dict({ memory: int(1, 50, { default: 10, label: 'curvature pairs kept' }) }),
  make: (p, k) => {
    const smooth = squaredHingeObjective(p, false)
    return owlqnAlgorithm(smooth.objective, smooth.x0, {
      l1: 1,
      gradTol: p.tol,
      memory: typeof k['memory'] === 'number' ? k['memory'] : 10,
      ftol: 1e-14,
    })
  },
  params: (s) => ({ x: Array.from(s.x) }),
}

/** The rows of a design as nested arrays, for the dual solver that appends the constant itself. */
function rowsOf(d: LinearDesign): number[][] {
  return Array.from({ length: d.n }, (_, i) => Array.from(d.raw.subarray(i * d.d, (i + 1) * d.d)))
}

/**
 * The solvers of the linear SVC as one slot: `dcd` (default, liblinear's dual coordinate descent, either loss, L2
 * penalty), `owlqn` (the L1 penalty), and every solver of `smoothSolvers` on the squared-hinge primal (`loss:
 * 'squaredHinge'` with the L2 penalty only: the hinge and the L1 norm are not smooth).
 */
export const linearSvcSolvers: ModeTable<LinearSvcProblem, VectorParams> = familySolvers<LinearSvcProblem>(
  { dcd: dcdClassifier, owlqn: owlqnClassifier },
  (p) => squaredHingeObjective(p, true),
  { default: 'dcd', description: 'how the linear support vector classifier is fitted' },
)

const dcdRegressor: ModeSpec<LinearSvrProblem, { readonly w: Float64Array }, VectorParams> = {
  description:
    "dual coordinate descent (Ho and Lin 2012, liblinear's SVR solver): exact in the dual, one sweep of coordinates per step; either loss",
  space: dict({}),
  make: (p) =>
    linearSvrDualCdAlgorithm(rowsOf(p.design), p.y, p.C, {
      loss: p.loss === 'epsilonInsensitive' ? 'l1' : 'l2',
      epsilon: p.epsilon,
      tol: p.tol,
      fitIntercept: p.design.fitIntercept,
      bias: p.design.scaling,
      weights: weightsAsC(p),
    }),
  params: (s) => ({ x: Array.from(s.w) }),
}

/**
 * The solvers of the linear SVR as one slot: `dcd` (default, liblinear's dual coordinate descent, either loss) and every
 * solver of `smoothSolvers` on the squared epsilon-insensitive primal (`loss: 'squaredEpsilonInsensitive'` only).
 */
export const linearSvrSolvers: ModeTable<LinearSvrProblem, VectorParams> = familySolvers<LinearSvrProblem>(
  { dcd: dcdRegressor },
  (p) => squaredEpsilonObjective(p),
  { default: 'dcd', description: 'how the linear support vector regressor is fitted' },
)

/** Whether a solver of {@link linearSvcSolvers} admits a loss and a penalty. */
export function checkLinearSvc(who: string, solver: string, loss: LinearSvcLoss, penalty: LinearPenalty): void {
  if (penalty === 'l1' && loss === 'hinge')
    throw new OmaiError(
      'value',
      `${who}: the L1 penalty is defined for the squared hinge only (liblinear has no L1-regularised hinge)`,
    )
  if (penalty === 'l1' && solver !== 'owlqn')
    throw new OmaiError('value', `${who}: the L1 penalty needs the 'owlqn' solver, not '${solver}'`)
  if (penalty === 'l2' && solver === 'owlqn')
    throw new OmaiError('value', `${who}: the 'owlqn' solver is for the L1 penalty`)
  if (solver !== 'dcd' && solver !== 'owlqn' && loss === 'hinge')
    throw new OmaiError(
      'value',
      `${who}: the hinge loss is not smooth, only the 'dcd' solver reaches it (use loss 'squaredHinge' for ${solver})`,
    )
}

/** Whether a solver of {@link linearSvrSolvers} admits a loss. */
export function checkLinearSvr(who: string, solver: string, loss: LinearSvrLoss): void {
  if (solver !== 'dcd' && loss === 'epsilonInsensitive')
    throw new OmaiError(
      'value',
      `${who}: the epsilon-insensitive loss is not smooth, only the 'dcd' solver reaches it (use loss 'squaredEpsilonInsensitive' for ${solver})`,
    )
}
