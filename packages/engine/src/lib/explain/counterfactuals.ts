/**
 * Counterfactual and contrastive explanations and recourse: the smallest change to a row that flips a model's
 * decision. Wachter's gradient search, diverse (DiCE-style) sets, FACE paths through dense regions of the data,
 * growing spheres (model-agnostic), CEM's pertinent negatives and positives, prototype-guided counterfactuals, linear
 * recourse with costs, actionability constraints (immutable, monotone, bounded, integer features) and the robustness
 * of recourse to model change.
 */

import { grad } from '../autodiff/index.ts'
import { argmaxOf, type Json, OmaiError } from '../base/index.ts'
import { dijkstraAlgorithm, graph, type PathSearchState, type ShortestPathPath } from '../graph/index.ts'
import { keyFor, runOptionsFor } from '../learn/_shared.ts'
import type { FitOptions } from '../learn/estimator.ts'
import { columnOf, matrixView } from '../learn/validation.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  add,
  asTensor,
  div,
  fromStorage,
  log,
  maximum,
  mul,
  relu,
  reshape,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  runAlgorithm,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import { lambdaSchedules } from './counterfactual-options.ts'
import type { DifferentiableModel } from './gradients.ts'

function rowOf(x: TensorLike, who: string): Float64Array {
  const v = Float64Array.from(toFlat(asTensor(x)))
  if (v.length === 0) throw new OmaiError('shape', `${who}: empty point`)
  return v
}

/**
 * Median absolute deviations of the training features (Wachter et al.'s feature scales), 1 where a feature has none.
 *
 * @param training - Rows `[n, d]`.
 * @returns One scale per feature.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * explain.medianAbsoluteDeviations([[0, 1], [1, 1], [4, 1]]) // => [1, 1]
 * ```
 */
export function medianAbsoluteDeviations(training: TensorLike): number[] {
  const m = matrixView(training, 'medianAbsoluteDeviations')
  const median = (v: Float64Array): number => {
    const s = Float64Array.from(v).sort()
    const h = (s.length - 1) / 2
    return ((s[Math.floor(h)] ?? 0) + (s[Math.ceil(h)] ?? 0)) / 2
  }
  return Array.from({ length: m.d }, (_, j) => {
    const c = columnOf(m, j)
    const med = median(c)
    const mad = median(c.map((v) => Math.abs(v - med)))
    return mad > 0 ? mad : 1
  })
}

function scalarModel(model: DifferentiableModel, d: number): (v: Float64Array) => number {
  return (v) => toFlat(model(fromStorage(Float64Array.from(v), [1, d], 'float64')))[0] ?? 0
}

/** What {@link wachterAlgorithm} searches for and how; {@link WachterOptions} adds the tracing options of the function. */
export interface WachterSettings {
  /** The desired model output (for a probability, e.g. 0.5 + margin). */
  readonly target: number
  /** Accept when |f(x') - target| is at most this (default 0.05). */
  readonly tolerance?: number
  /** Feature scales for the weighted L1 distance (default ones; see {@link medianAbsoluteDeviations}). */
  readonly scales?: readonly number[]
  /** The starting weight of the prediction loss (default 0.1), grown each outer round by the `lambdaSchedule`. */
  readonly lambda?: number
  /** The schedule of the weight, a value of `lambdaSchedules.space` (default doubling). */
  readonly lambdaSchedule?: SpaceValue
  /** Outer rounds (default 20) and proximal-gradient iterations per round (default 200). */
  readonly maxRounds?: number
  readonly maxIterations?: number
  /** Features that may not change. */
  readonly immutable?: readonly number[]
}

/** Options of {@link wachterCounterfactual}. */
export interface WachterOptions extends FitOptions, WachterSettings {}

/** A counterfactual explanation. */
export interface Counterfactual {
  /** The counterfactual point. */
  readonly x: readonly number[]
  /** The model's output there. */
  readonly prediction: number
  /** Its weighted L1 distance from the original point. */
  readonly distance: number
  /** Whether the target was reached within tolerance. */
  readonly valid: boolean
}

/** The paths of {@link wachterAlgorithm}. */
export type CounterfactualPath =
  | 'cf/x'
  | 'cf/prediction'
  | 'cf/loss'
  | 'cf/distance'
  | 'cf/validity'
  | 'cf/lambda'
  | 'cf/round'

/**
 * The state of {@link wachterAlgorithm}: the current point, the weight of the prediction loss, where the search is in
 * its round (the proximal-gradient iteration and its backtracked step size) and the model's output at the point.
 */
export interface WachterState {
  /** Lambda rounds finished. */
  readonly round: number
  /** Proximal-gradient iterations done in the current round. */
  readonly iteration: number
  readonly point: Float64Array
  /** The weight of the prediction loss in the current round. */
  readonly lambda: number
  /** The step size of the proximal gradient, adapted by backtracking within the round (restarts at 1). */
  readonly stepSize: number
  /** The model's output at the point. */
  readonly value: number
  /** Whether the state is at the start of a round (the target is tested only there). */
  readonly roundEnd: boolean
  /** The objective lambda (f(x') - target)^2 + distance at the point. */
  readonly loss: number
  /** The weighted L1 distance from the original point. */
  readonly distance: number
}

/** What a stepped Wachter search logs: the point, the model's output, the objective, the distance, validity and lambda. */
export const counterfactualContract: Contract<WachterState, CounterfactualPath> = defineContract({
  'cf/x': tensorSource({ role: 'state', description: 'the counterfactual point after the iteration' }),
  'cf/prediction': scalar({ description: "the model's output at the point" }),
  'cf/loss': scalar({ description: 'lambda (f(x) - target)^2 + the weighted L1 distance', direction: 'lower' }),
  'cf/distance': scalar({ description: 'the weighted L1 distance from the original point', direction: 'lower' }),
  'cf/validity': scalar({ description: '1 when the output is within tolerance of the target, else 0' }),
  'cf/lambda': scalar({ description: 'the weight of the prediction loss in this round' }),
  'cf/round': scalar({ description: 'lambda rounds finished' }),
})

/**
 * Wachter counterfactuals (Wachter, Mittelstadt and Russell 2017) as a stepped algorithm: the closest point, in the
 * MAD-weighted L1 distance, at which the model's output reaches a target, from argmin lambda (f(x') - y')^2 +
 * sum_j |x'_j - x_j| / MAD_j. Each subproblem is solved by proximal gradient descent (the L1 term's proximal map is soft
 * thresholding towards x, which also keeps sparse changes) with a backtracked step; a step is one proximal-gradient
 * iteration, and when the round's subproblem has converged (or its iterations are spent) lambda grows by the
 * `lambdaSchedule` (doubling by default) and the next round starts. The search stops at the end of a round in which the
 * target is met. Immutable features are fixed. The state serialises.
 *
 * @param model - The differentiable model (one output per row; the first column is used).
 * @param x - The original point `[d]`.
 * @param settings - target, tolerance, scales, lambda and its schedule, maxRounds, maxIterations, immutable.
 * @returns The algorithm.
 * @throws OmaiError with code `shape` for an empty point or the wrong number of scales.
 * @concept counterfactual-explanations
 * @example
 * ```ts
 * import { wachterAlgorithm } from 'omai-engine/explain'
 * import { matmul } from 'omai-engine/tensor'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const score = (v) => matmul(v, [[1], [2]])
 * const alg = wachterAlgorithm(score, [0, 0], { target: 2, tolerance: 0.01 })
 * Math.abs(runAlgorithm(alg, { maxSteps: 4000 }).state.value - 2) <= 0.01 // => true
 * ```
 */
export function wachterAlgorithm(
  model: DifferentiableModel,
  x: TensorLike,
  settings: WachterSettings,
): Algorithm<WachterState, CounterfactualPath> {
  const x0 = rowOf(x, 'wachterCounterfactual')
  const d = x0.length
  const scales = settings.scales ?? new Array<number>(d).fill(1)
  if (scales.length !== d) throw new OmaiError('shape', 'wachterCounterfactual: one scale per feature')
  const fixed = new Set(settings.immutable ?? [])
  const tol = settings.tolerance ?? 0.05
  const maxIterations = settings.maxIterations ?? 200
  const maxRounds = settings.maxRounds ?? 20
  const grow = lambdaSchedules.make(settings.lambdaSchedule)
  const f = scalarModel(model, d)
  const gradF = grad((v: Value) => sum(model(v)))
  const distance = (v: Float64Array): number =>
    v.reduce((s, t, j) => s + Math.abs(t - (x0[j] ?? 0)) / (scales[j] ?? 1), 0)
  const lossAt = (v: Float64Array, lambda: number, fv: number): number =>
    lambda * (fv - settings.target) ** 2 + distance(v)
  const measure = (
    round: number,
    iteration: number,
    point: Float64Array,
    lambda: number,
    stepSize: number,
    roundEnd: boolean,
  ): WachterState => {
    const value = f(point)
    return {
      round,
      iteration,
      point,
      lambda,
      stepSize,
      value,
      roundEnd,
      loss: lossAt(point, lambda, value),
      distance: distance(point),
    }
  }
  return {
    name: 'wachter-counterfactual',
    config: {
      features: d,
      target: settings.target,
      tolerance: tol,
      schedule: lambdaSchedules.optionOf(settings.lambdaSchedule).name,
    },
    schema: channels({
      metrics: {
        lambda: { description: 'the weight of the prediction loss' },
        gap: { description: "|f(x') - target|", direction: 'lower' },
      },
    }),
    init: () => measure(0, 0, Float64Array.from(x0), settings.lambda ?? 0.1, 1, true),
    step: (s, ctx) => {
      const lambda = s.lambda
      const objective = (v: Float64Array): number => lossAt(v, lambda, f(v))
      let p = s.point
      let step = s.stepSize
      let moved = 0
      if (s.iteration < maxIterations) {
        const fv = f(p)
        const g = toFlat(gradF(fromStorage(Float64Array.from(p), [1, d], 'float64')) as Tensor).map(
          (t) => 2 * lambda * (fv - settings.target) * t,
        )
        const current = objective(p)
        let next = p
        for (let bt = 0; bt < 30; bt++) {
          next = Float64Array.from(p, (v, j) => {
            if (fixed.has(j)) return x0[j] ?? 0
            const z = v - step * (g[j] ?? 0)
            const shrink = step / (scales[j] ?? 1)
            const delta = z - (x0[j] ?? 0)
            return (x0[j] ?? 0) + Math.sign(delta) * Math.max(Math.abs(delta) - shrink, 0)
          })
          if (objective(next) <= current + 1e-15) break
          step /= 2
        }
        moved = next.reduce((m, v, j) => Math.max(m, Math.abs(v - (p[j] ?? 0))), 0)
        p = next
        step *= 1.5
      }
      const iteration = s.iteration + 1
      const out =
        moved < 1e-10 || iteration >= maxIterations
          ? measure(s.round + 1, 0, p, grow(lambda, s.round), 1, true)
          : measure(s.round, iteration, p, lambda, step, false)
      ctx.log('cf/x', fromStorage(Float64Array.from(out.point), [d], 'float64'))
      ctx.log('cf/prediction', out.value)
      ctx.log('cf/loss', out.loss)
      ctx.log('cf/distance', out.distance)
      ctx.log('cf/validity', Math.abs(out.value - settings.target) <= tol ? 1 : 0)
      ctx.log('cf/lambda', lambda)
      ctx.log('cf/round', out.round)
      return out
    },
    done: (s) =>
      s.roundEnd && Math.abs(s.value - settings.target) <= tol
        ? 'converged'
        : s.round >= maxRounds
          ? 'completed'
          : undefined,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as WachterState,
    observe: (s) => ({ metrics: { lambda: s.lambda, gap: Math.abs(s.value - settings.target) } }),
  }
}

/**
 * Wachter counterfactuals (Wachter, Mittelstadt and Russell 2017): the closest point, in the MAD-weighted L1
 * distance, at which the model's output reaches a target, from argmin lambda (f(x') - y')^2 + sum_j |x'_j - x_j| / MAD_j
 * with lambda doubled until the target is met. Each subproblem is solved by proximal gradient descent (the L1 term's
 * proximal map is soft thresholding towards x, which also keeps sparse changes); immutable features are fixed. This is
 * {@link wachterAlgorithm} run to the end, a traced algorithm with one step per proximal-gradient iteration.
 *
 * @param model - The differentiable model (one output per row; the first column is used).
 * @param x - The original point `[d]`.
 * @param options - target, tolerance, scales, lambda and its schedule, maxRounds, maxIterations, immutable and tracing.
 * @returns The counterfactual.
 * @concept counterfactual-explanations
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { matmul } from 'omai-engine/tensor'
 * const score = (v) => matmul(v, [[1], [2]])
 * const cf = explain.wachterCounterfactual(score, [0, 0], { target: 2, tolerance: 0.01 })
 * cf.valid // => true
 * Math.abs(cf.x[0]) < 0.01 // => true
 * ```
 */
export function wachterCounterfactual(
  model: DifferentiableModel,
  x: TensorLike,
  options: WachterOptions,
): Counterfactual {
  const settings = options
  const algorithm = wachterAlgorithm(model, x, settings)
  const { state } = runAlgorithm(algorithm, {
    ...runOptionsFor(options, 'wachterCounterfactual'),
    maxSteps: (settings.maxRounds ?? 20) * Math.max(1, settings.maxIterations ?? 200),
  })
  const tol = settings.tolerance ?? 0.05
  return {
    x: Array.from(state.point),
    prediction: state.value,
    distance: state.distance,
    valid: Math.abs(state.value - settings.target) <= tol,
  }
}

/** Options of {@link diverseCounterfactuals}. */
export interface DiverseOptions extends FitOptions {
  /** The number of counterfactuals (default 4). */
  readonly count?: number
  /** The desired class: outputs above 0.5 (1) or below (0) for a probability model (default 1). */
  readonly desired?: 0 | 1
  /** Weights of proximity (default 0.5) and diversity (default 1). */
  readonly proximityWeight?: number
  readonly diversityWeight?: number
  /** Feature scales for distances (default ones). */
  readonly scales?: readonly number[]
  /** Gradient iterations (default 500) and the learning rate (default 0.05). */
  readonly maxIterations?: number
  readonly learningRate?: number
  /** The initial spread around x (default 0.5). */
  readonly spread?: number
}

/** The result of {@link diverseCounterfactuals}. */
export interface DiverseCounterfactuals {
  readonly points: readonly (readonly number[])[]
  readonly predictions: readonly number[]
  /** Which reach the desired class. */
  readonly valid: readonly boolean[]
  /** det(K) of the diversity kernel K_ij = 1 / (1 + dist(c_i, c_j)): larger is more diverse. */
  readonly diversity: number
}

/** The paths {@link diverseCounterfactualsAlgorithm} logs. */
export type DicePath = 'dice/x' | 'dice/loss' | 'dice/validity' | 'dice/proximity' | 'dice/diversity' | 'dice/iteration'

/** What a stepped DiCE search logs: the points, the objective, how many are valid, their proximity and diversity. */
export const diceContract: Contract<unknown, DicePath> = defineContract({
  'dice/x': tensorSource({
    role: 'state',
    shape: ['count', 'features'],
    description: 'the counterfactual points after the iteration',
  }),
  'dice/loss': scalar({ description: 'the DiCE objective: hinge + proximity - diversity', direction: 'lower' }),
  'dice/validity': scalar({ description: 'how many of the points reach the desired class', direction: 'higher' }),
  'dice/proximity': scalar({
    description: 'the mean scaled-L1 distance of the points from the original',
    direction: 'lower',
  }),
  'dice/diversity': scalar({ description: 'det(K) of the diversity kernel of the points', direction: 'higher' }),
  'dice/iteration': scalar({ description: 'Adam iterations done' }),
})

/** The state of {@link diverseCounterfactualsAlgorithm}: the points `[k, d]` row-major, Adam's moments and its step count. */
export interface DiceState {
  readonly c: Float64Array
  readonly m: Float64Array
  readonly v: Float64Array
  readonly t: number
}

interface DiceRun {
  readonly algorithm: Algorithm<DiceState, DicePath>
  readonly k: number
  readonly d: number
  readonly finish: (state: DiceState) => DiverseCounterfactuals
}

function diceRun(model: DifferentiableModel, x: TensorLike, options: DiverseOptions): DiceRun {
  const x0 = rowOf(x, 'diverseCounterfactuals')
  const d = x0.length
  const k = options.count ?? 4
  if (!(Number.isInteger(k) && k >= 1)) throw new OmaiError('value', 'diverseCounterfactuals: count must be >= 1')
  const scales = fromStorage(Float64Array.from(options.scales ?? new Array<number>(d).fill(1)), [1, d], 'float64')
  const xt = fromStorage(
    Float64Array.from({ length: k * d }, (_, i) => x0[i % d] ?? 0),
    [k, d],
    'float64',
  )
  const desired = options.desired ?? 1
  const wProx = options.proximityWeight ?? 0.5
  const wDiv = options.diversityWeight ?? 1
  const eps = 1e-6
  const proximity = (c: Value): Value => sum(div(sqrt(add(square(sub(c, xt)), 1e-9)), scales))
  const loss = (c: Value): Value => {
    const p = model(c)
    // Hinge on the logit: max(0, 1 - s z) with z = logit(p), s = +1 for class 1.
    const z = sub(log(add(p, eps)), log(add(sub(1, p), eps)))
    const margin = sub(1, mul(desired === 1 ? 1 : -1, z))
    const hinge = sum(relu(margin))
    let diversity: Value = fromStorage(new Float64Array([0]), [], 'float64')
    if (k > 1) diversity = logDetKernel(c, scales, k)
    return sub(add(mul(hinge, 1 / k), mul(proximity(c), wProx / k)), mul(diversity, wDiv))
  }
  const dl = grad(loss)
  const key = keyFor(options, 'diverseCounterfactuals')
  const lr = options.learningRate ?? 0.05
  const asTensorK = (c: Float64Array): Tensor => fromStorage(Float64Array.from(c), [k, d], 'float64')
  const predictionsOf = (c: Float64Array): number[] => {
    const out = model(asTensorK(c)) as Tensor
    const width = out.shape.length === 2 ? (out.shape[1] ?? 1) : 1
    return toFlat(out).filter((_, i) => i % width === 0)
  }
  const diversityOf = (c: Float64Array): number =>
    k > 1 ? Math.exp(toFlat(logDetKernel(asTensorK(c), scales, k))[0] ?? 0) : 1
  const algorithm: Algorithm<DiceState, DicePath> = {
    name: 'diverse-counterfactuals',
    config: { count: k, features: d },
    schema: channels({ metrics: { loss: { description: 'the DiCE objective', direction: 'lower' } } }),
    init: () => {
      const z = toFlat(normal(foldIn(key, 0), [k, d]))
      return {
        c: Float64Array.from({ length: k * d }, (_, i) => (x0[i % d] ?? 0) + (options.spread ?? 0.5) * (z[i] ?? 0)),
        m: new Float64Array(k * d),
        v: new Float64Array(k * d),
        t: 0,
      }
    },
    step: (s, ctx) => {
      const g = toFlat(dl(asTensorK(s.c)) as Tensor)
      const t = s.t + 1
      const m = Float64Array.from(s.m, (mv, i) => 0.9 * mv + 0.1 * (g[i] ?? 0))
      const v = Float64Array.from(s.v, (vv, i) => 0.999 * vv + 0.001 * (g[i] ?? 0) ** 2)
      const c = Float64Array.from(
        s.c,
        (cv, i) => cv - (lr * ((m[i] ?? 0) / (1 - 0.9 ** t))) / (Math.sqrt((v[i] ?? 0) / (1 - 0.999 ** t)) + 1e-8),
      )
      ctx.log('dice/iteration', t)
      if (ctx.wants('dice/x')) ctx.log('dice/x', fromStorage(Float64Array.from(c), [k, d], 'float64'))
      if (ctx.wants('dice/loss')) ctx.log('dice/loss', toFlat(loss(asTensorK(c)))[0] ?? 0)
      if (ctx.wants('dice/validity'))
        ctx.log('dice/validity', predictionsOf(c).filter((p) => (desired === 1 ? p > 0.5 : p < 0.5)).length)
      if (ctx.wants('dice/proximity')) ctx.log('dice/proximity', (toFlat(proximity(asTensorK(c)))[0] ?? 0) / k)
      if (ctx.wants('dice/diversity')) ctx.log('dice/diversity', diversityOf(c))
      return { c, m, v, t }
    },
    observe: (s) => ({ metrics: { loss: toFlat(loss(asTensorK(s.c)))[0] ?? 0 } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as DiceState,
  }
  const finish = (state: DiceState): DiverseCounterfactuals => {
    const predictions = predictionsOf(state.c)
    return {
      points: Array.from({ length: k }, (_, i) => Array.from(state.c.subarray(i * d, (i + 1) * d))),
      predictions,
      valid: predictions.map((p) => (desired === 1 ? p > 0.5 : p < 0.5)),
      diversity: diversityOf(state.c),
    }
  }
  return { algorithm, k, d, finish }
}

/**
 * Diverse counterfactuals (Mothilal, Sharma and Tan 2020, DiCE) as a stepped algorithm: k points moved jointly by Adam
 * on the hinge loss of reaching the desired class, plus their mean scaled-L1 distance to x, minus the log-determinant of
 * the kernel K_ij = 1 / (1 + dist(c_i, c_j)) (a determinantal point process that spreads them out). One Adam iteration per
 * step; the state (points and moments) serialises.
 *
 * @param model - The differentiable model returning a probability per row.
 * @param x - The original point `[d]`.
 * @param options - count, desired, weights, scales, learningRate, spread and key (`maxIterations` is the caller's step budget).
 * @returns The algorithm.
 * @throws OmaiError with code `shape` for an empty point and `value` for a count below 1.
 * @concept counterfactual-explanations
 * @example
 * ```ts
 * import { diverseCounterfactualsAlgorithm } from 'omai-engine/explain'
 * import { matmul, sigmoid } from 'omai-engine/tensor'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const proba = (v) => sigmoid(matmul(v, [[2], [2]]))
 * const run = runAlgorithm(diverseCounterfactualsAlgorithm(proba, [-1, -1], { count: 3 }), { maxSteps: 30 })
 * run.state.c.length // => 6
 * ```
 */
export function diverseCounterfactualsAlgorithm(
  model: DifferentiableModel,
  x: TensorLike,
  options: DiverseOptions = {},
): Algorithm<DiceState, DicePath> {
  return diceRun(model, x, options).algorithm
}

/**
 * Diverse counterfactuals (Mothilal, Sharma and Tan 2020, DiCE): k points found jointly by gradient descent (Adam) on
 * the hinge loss of reaching the desired class, plus their mean scaled-L1 distance to x, minus the log-determinant of
 * the kernel K_ij = 1 / (1 + dist(c_i, c_j)) (a determinantal point process that spreads them out). This is
 * {@link diverseCounterfactualsAlgorithm} run to the end, a traced algorithm with one step per iteration.
 *
 * @param model - The differentiable model returning a probability per row.
 * @param x - The original point `[d]`.
 * @param options - count, desired, weights, scales, maxIterations, learningRate, spread, key and tracing.
 * @returns The counterfactuals, their predictions and validity, and the diversity.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { matmul, sigmoid } from 'omai-engine/tensor'
 * const proba = (v) => sigmoid(matmul(v, [[2], [2]]))
 * const r = explain.diverseCounterfactuals(proba, [-1, -1], { count: 3 })
 * r.valid // => [true, true, true]
 * ```
 */
export function diverseCounterfactuals(
  model: DifferentiableModel,
  x: TensorLike,
  options: DiverseOptions = {},
): DiverseCounterfactuals {
  const { algorithm, finish } = diceRun(model, x, options)
  const { state } = runAlgorithm(algorithm, {
    ...runOptionsFor(options, 'diverseCounterfactuals'),
    maxSteps: options.maxIterations ?? 500,
  })
  return finish(state)
}

/** log det of K_ij = 1 / (1 + dist(c_i, c_j)) by a Cholesky-free Gaussian elimination on values. */
function logDetKernel(c: Value, scales: Tensor, k: number): Value {
  const rows = Array.from({ length: k }, (_, i) => squeeze(sliceAxis(c, 0, i, i + 1), 0))
  const a: Value[][] = rows.map((ri) =>
    rows.map((rj) => div(1, add(1, sum(div(sqrt(add(square(sub(ri, rj)), 1e-9)), squeeze(scales, 0)))))),
  )
  // Gaussian elimination without pivoting: the kernel is positive definite.
  let logdet: Value = fromStorage(new Float64Array([0]), [], 'float64')
  for (let p = 0; p < k; p++) {
    const rowP = a[p] as Value[]
    const pivot = rowP[p] as Value
    logdet = add(logdet, log(pivot))
    for (let r = p + 1; r < k; r++) {
      const rowR = a[r] as Value[]
      const factor = div(rowR[p] as Value, pivot)
      for (let col = p; col < k; col++) rowR[col] = sub(rowR[col] as Value, mul(factor, rowP[col] as Value))
    }
  }
  return logdet
}

/** Options of {@link faceCounterfactual}. */
export interface FaceOptions {
  /** Neighbours per point in the graph (default 5). */
  readonly kNeighbors?: number
  /** The predicted label (or a predicate on the model's output) the end point must have. */
  readonly accept: (prediction: number) => boolean
  /** Only end at points whose k-NN density estimate is at least this quantile of the data's (default 0.1). */
  readonly densityQuantile?: number
  /** Feature scales for distances (default ones). */
  readonly scales?: readonly number[]
}

/** A FACE path: a feasible sequence of data points from x to a counterfactual. */
export interface FacePath {
  /** Row indices of the data along the path (x itself excluded). */
  readonly path: readonly number[]
  readonly counterfactual: readonly number[]
  /** The path's total density-weighted length. */
  readonly cost: number
}

/** The paths {@link faceCounterfactualAlgorithm} logs. */
export type FacePathName = 'face/settled' | 'face/cost' | 'face/accepted' | 'face/best' | 'face/frontier'

/** What a stepped FACE search logs: the point settled, its path cost, whether it qualifies, and the best so far. */
export const faceContract: Contract<unknown, FacePathName> = defineContract({
  'face/settled': scalar({ description: 'the node settled at this step (0 is x itself, i + 1 is row i of the data)' }),
  'face/cost': scalar({ description: 'the density-weighted path cost to the settled node', direction: 'lower' }),
  'face/accepted': scalar({ description: '1 when the settled point has the wanted outcome and enough density' }),
  'face/best': scalar({ description: 'the node of the cheapest qualifying point so far (-1 before one is found)' }),
  'face/frontier': scalar({ description: 'nodes waiting to be settled' }),
})

/**
 * The state of {@link faceCounterfactualAlgorithm}: the Dijkstra search over the density graph (settled in place, so
 * keep a copy of a state you want to look at later) and the cheapest qualifying node found so far.
 */
export interface FaceState {
  readonly search: PathSearchState
  /** The cheapest qualifying node settled so far (-1: none yet). */
  readonly best: number
}

function faceRun(
  predict: (x: Tensor) => Tensor,
  x: TensorLike,
  data: TensorLike,
  options: FaceOptions,
): {
  readonly algorithm: Algorithm<FaceState, FacePathName>
  readonly finish: (s: FaceState) => FacePath | undefined
  readonly nodes: number
} {
  const x0 = rowOf(x, 'faceCounterfactual')
  const m = matrixView(data, 'faceCounterfactual', { nFeatures: x0.length })
  const d = m.d
  const scales = options.scales ?? new Array<number>(d).fill(1)
  const pts = [
    Array.from(x0),
    ...Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * d, (i + 1) * d))),
  ]
  const n = pts.length
  const kk = Math.min(options.kNeighbors ?? 5, n - 1)
  const dist = (a: readonly number[], b: readonly number[]): number =>
    Math.sqrt(a.reduce((s, v, j) => s + ((v - (b[j] ?? 0)) / (scales[j] ?? 1)) ** 2, 0))
  const knn = pts.map((p, i) =>
    pts
      .map((q, j) => ({ j, dd: dist(p, q) }))
      .filter((e) => e.j !== i)
      .sort((a, b) => a.dd - b.dd || a.j - b.j)
      .slice(0, kk),
  )
  // A k-NN density estimate: inverse of the distance to the k-th neighbour.
  const density = (p: readonly number[]): number => {
    const dd = pts.map((q) => dist(p, q)).sort((a, b) => a - b)
    return 1 / ((dd[kk] ?? dd[dd.length - 1] ?? 1) + 1e-12)
  }
  const edges: [number, number, number][] = []
  knn.forEach((list, i) => {
    for (const { j, dd } of list) {
      const mid = (pts[i] ?? []).map((v, t) => (v + (pts[j]?.[t] ?? 0)) / 2)
      edges.push([i, j, dd / density(mid)])
    }
  })
  const g = graph(edges, { nodes: n, duplicates: 'min' })
  const inner = dijkstraAlgorithm(g, 0)
  const preds = toFlat(predict(fromStorage(m.data, [m.n, d], 'float64')))
  const dens = pts.slice(1).map(density)
  const sorted = [...dens].sort((a, b) => a - b)
  const cut = sorted[Math.floor((options.densityQuantile ?? 0.1) * (sorted.length - 1))] ?? 0
  const qualifies = (i: number): boolean => options.accept(preds[i - 1] ?? Number.NaN) && (dens[i - 1] ?? 0) >= cut
  const algorithm: Algorithm<FaceState, FacePathName> = {
    name: 'face-counterfactual',
    config: { nodes: n, kNeighbors: kk },
    schema: channels({ metrics: { best: { description: 'the cheapest qualifying node so far' } } }),
    init: () => ({ search: inner.init(key0), best: -1 }),
    step: (s, ctx) => {
      // The search settles nodes in order of cost, so the first qualifying node is the cheapest; settling goes on while
      // costs tie, so that the lowest index among equals wins.
      const search = inner.step(s.search, ctx as unknown as StepContext<ShortestPathPath>)
      const u = search.current
      const cost = search.distance[u] ?? Number.POSITIVE_INFINITY
      const ok = u >= 1 && qualifies(u)
      let best = s.best
      if (ok && Number.isFinite(cost)) {
        const bestCost = best < 0 ? Number.POSITIVE_INFINITY : (search.distance[best] ?? Number.POSITIVE_INFINITY)
        if (cost < bestCost || (cost === bestCost && u < best)) best = u
      }
      ctx.log('face/settled', u)
      ctx.log('face/cost', cost)
      ctx.log('face/accepted', ok ? 1 : 0)
      ctx.log('face/best', best)
      ctx.log('face/frontier', search.queue.size)
      return { search, best }
    },
    done: (s) => {
      if (inner.done?.(s.search, 0) !== undefined) return 'converged'
      const u = s.search.current
      if (
        s.best >= 0 &&
        u >= 0 &&
        (s.search.distance[u] ?? 0) > (s.search.distance[s.best] ?? Number.POSITIVE_INFINITY)
      )
        return 'converged'
      return undefined
    },
    observe: (s) => ({ metrics: { best: s.best } }),
    serialise: (s) => ({ search: inner.serialise?.(s.search) ?? null, best: s.best }),
    deserialise: (j) => {
      const o = j as unknown as { readonly search: Json; readonly best: number }
      return { search: inner.deserialise?.(o.search) as PathSearchState, best: o.best }
    },
  }
  const finish = (s: FaceState): FacePath | undefined => {
    if (s.best < 0) return undefined
    const path: number[] = []
    for (let v = s.best, guard = 0; v > 0 && guard <= n; v = s.search.predecessor[v] ?? -1, guard++) path.push(v - 1)
    path.reverse()
    return { path, counterfactual: pts[s.best] ?? [], cost: s.search.distance[s.best] ?? Number.NaN }
  }
  return { algorithm, finish, nodes: n }
}

const key0: Key = keyFor({}, 'faceCounterfactual')

/**
 * FACE (Poyiadzi et al. 2020) as a stepped algorithm: Dijkstra over the density-weighted k-nearest-neighbour graph of x
 * and the data, one node settled per step, stopping once the cheapest dense point the model labels as wanted is
 * known. The state (the search and the best node so far) serialises.
 *
 * @param predict - The model's output for rows `[m, d]` (`[m]`).
 * @param x - The original point `[d]`.
 * @param data - Candidate points `[n, d]` (the training data).
 * @param options - accept, kNeighbors, densityQuantile and scales.
 * @returns The algorithm; it ends `converged` when the search is over (the path is in {@link faceCounterfactual}).
 * @concept counterfactual-explanations
 * @example
 * ```ts
 * import { faceCounterfactualAlgorithm } from 'omai-engine/explain'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { tensor } from 'omai-engine/tensor'
 * const sign = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => (x.data[i] > 2 ? 1 : 0)))
 * const data = [[0], [0.5], [1], [1.5], [2], [2.5], [3]]
 * runAlgorithm(faceCounterfactualAlgorithm(sign, [0.1], data, { accept: (p) => p === 1, kNeighbors: 2 }), { maxSteps: 20 }).state.best // => 6
 * ```
 */
export function faceCounterfactualAlgorithm(
  predict: (x: Tensor) => Tensor,
  x: TensorLike,
  data: TensorLike,
  options: FaceOptions,
): Algorithm<FaceState, FacePathName> {
  return faceRun(predict, x, data, options).algorithm
}

/**
 * FACE: feasible and actionable counterfactual explanations (Poyiadzi et al. 2020): build a k-nearest-neighbour graph
 * of the data whose edge costs are distances divided by the density at their midpoint, so cheap paths stay in
 * populated regions, and return the cheapest path from x to a dense point the model labels as wanted. This is
 * {@link faceCounterfactualAlgorithm} run to the end.
 *
 * @param predict - The model's output for rows `[m, d]` (`[m]`).
 * @param x - The original point `[d]`.
 * @param data - Candidate points `[n, d]` (the training data).
 * @param options - accept, kNeighbors, densityQuantile and scales.
 * @returns The path, or undefined when no acceptable point is reachable.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { tensor } from 'omai-engine/tensor'
 * const sign = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => (x.data[i] > 2 ? 1 : 0)))
 * const data = [[0], [0.5], [1], [1.5], [2], [2.5], [3]]
 * explain.faceCounterfactual(sign, [0.1], data, { accept: (p) => p === 1, kNeighbors: 2 }).counterfactual // => [2.5]
 * ```
 */
export function faceCounterfactual(
  predict: (x: Tensor) => Tensor,
  x: TensorLike,
  data: TensorLike,
  options: FaceOptions,
): FacePath | undefined {
  const { algorithm, finish, nodes } = faceRun(predict, x, data, options)
  return finish(runAlgorithm(algorithm, { maxSteps: nodes + 1 }).state)
}

/** Options of {@link linearRecourse}. */
export interface RecourseOptions {
  /** The cost per unit change of each feature (default ones). */
  readonly costs?: readonly number[]
  /** Bounds on each feature's change, `[lower, upper]` (default unbounded). */
  readonly bounds?: readonly (readonly [number, number])[]
  /** Features that may not change. */
  readonly immutable?: readonly number[]
  /** The score to reach, w . x + b >= margin (default 0). */
  readonly margin?: number
}

/** A recourse action. */
export interface Recourse {
  /** The change to each feature. */
  readonly action: readonly number[]
  /** Its total cost, sum of cost_j |a_j|. */
  readonly cost: number
  /** Whether the margin is reachable within the bounds. */
  readonly feasible: boolean
}

/**
 * Actionable recourse for a linear classifier (Ustun, Spangher and Liu 2019, with linear costs): the cheapest change
 * a with w . (x + a) + b >= margin, each a_j within its bounds and immutable features fixed. With linear costs this is a
 * fractional knapsack, solved exactly by spending on features in order of |w_j| / cost_j.
 *
 * @param weights - The classifier's coefficients `[d]`.
 * @param intercept - Its intercept.
 * @param x - The point `[d]`.
 * @param options - costs, bounds, immutable and margin.
 * @returns The action, its cost and feasibility.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * // Score 2 x0 + x1 - 4 at (0, 0); raising x0 is twice as effective per unit as x1.
 * explain.linearRecourse([2, 1], -4, [0, 0], { bounds: [[0, 1], [0, 5]] }) // => { action: [1, 2], cost: 3, feasible: true }
 * ```
 */
export function linearRecourse(
  weights: readonly number[],
  intercept: number,
  x: TensorLike,
  options: RecourseOptions = {},
): Recourse {
  const x0 = rowOf(x, 'linearRecourse')
  const d = x0.length
  if (weights.length !== d) throw new OmaiError('shape', 'linearRecourse: one weight per feature')
  const costs = options.costs ?? new Array<number>(d).fill(1)
  if (costs.some((c) => !(c > 0))) throw new OmaiError('value', 'linearRecourse: costs must be positive')
  const fixed = new Set(options.immutable ?? [])
  const score = weights.reduce((s, w, j) => s + w * (x0[j] ?? 0), intercept)
  let need = (options.margin ?? 0) - score
  const action = new Array<number>(d).fill(0)
  if (need <= 0) return { action, cost: 0, feasible: true }
  const order = weights
    .map((w, j) => ({ j, w, eff: Math.abs(w) / (costs[j] ?? 1) }))
    .filter((e) => e.w !== 0 && !fixed.has(e.j))
    .sort((a, b) => b.eff - a.eff || a.j - b.j)
  let cost = 0
  for (const { j, w } of order) {
    if (need <= 1e-15) break
    const [lo, hi] = options.bounds?.[j] ?? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]
    const room = w > 0 ? hi : -lo
    if (!(room > 0)) continue
    const step = Math.min(room, need / Math.abs(w))
    action[j] = w > 0 ? step : -step
    need -= step * Math.abs(w)
    cost += step * (costs[j] ?? 1)
  }
  return { action, cost, feasible: need <= 1e-12 }
}

/** What a counterfactual may change (actionability): fixed features, allowed directions and bounds. */
export interface ActionConstraints {
  /** Features that may not change (age, ethnicity). */
  readonly immutable?: readonly number[]
  /** Features that may only go up (`increase`) or down (`decrease`), feature index to direction. */
  readonly monotone?: Readonly<Record<number, 'increase' | 'decrease'>>
  /** Bounds on the counterfactual's values, `[low, high]` per feature (a feature's plausible range). */
  readonly bounds?: readonly (readonly [number, number])[]
  /** Features that must stay whole numbers (counts). */
  readonly integer?: readonly number[]
}

/** A violated constraint. */
export interface ActionViolation {
  readonly feature: number
  readonly reason: 'immutable' | 'increase only' | 'decrease only' | 'below bound' | 'above bound' | 'not integer'
}

/**
 * The constraints a counterfactual breaks: changes to immutable features, moves against a monotone direction, values
 * outside bounds or non-integers where integers are required.
 *
 * @param x - The original point `[d]`.
 * @param counterfactual - The proposed point `[d]`.
 * @param constraints - The actionability constraints.
 * @returns Every violation (empty when the counterfactual is actionable).
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * explain.actionViolations([30, 2], [29, 3], { monotone: { 0: 'increase' } }) // => [{ feature: 0, reason: 'increase only' }]
 * ```
 */
export function actionViolations(
  x: TensorLike,
  counterfactual: TensorLike,
  constraints: ActionConstraints,
): ActionViolation[] {
  const a = rowOf(x, 'actionViolations')
  const b = rowOf(counterfactual, 'actionViolations')
  if (a.length !== b.length) throw new OmaiError('shape', 'actionViolations: the points differ in features')
  const out: ActionViolation[] = []
  const fixed = new Set(constraints.immutable ?? [])
  const ints = new Set(constraints.integer ?? [])
  for (let j = 0; j < a.length; j++) {
    const delta = (b[j] ?? 0) - (a[j] ?? 0)
    const tol = 1e-9 * (1 + Math.abs(a[j] ?? 0))
    if (fixed.has(j) && Math.abs(delta) > tol) out.push({ feature: j, reason: 'immutable' })
    const dir = constraints.monotone?.[j]
    if (dir === 'increase' && delta < -tol) out.push({ feature: j, reason: 'increase only' })
    if (dir === 'decrease' && delta > tol) out.push({ feature: j, reason: 'decrease only' })
    const bound = constraints.bounds?.[j]
    if (bound !== undefined && (b[j] ?? 0) < bound[0] - tol) out.push({ feature: j, reason: 'below bound' })
    if (bound !== undefined && (b[j] ?? 0) > bound[1] + tol) out.push({ feature: j, reason: 'above bound' })
    if (ints.has(j) && Math.abs((b[j] ?? 0) - Math.round(b[j] ?? 0)) > tol)
      out.push({ feature: j, reason: 'not integer' })
  }
  return out
}

/**
 * Project a proposed counterfactual onto the actionable set: immutable features back to the original, monotone ones
 * clipped to their allowed side, values clipped to their bounds and rounded where integers are required.
 *
 * @param x - The original point `[d]`.
 * @param counterfactual - The proposed point `[d]`.
 * @param constraints - The actionability constraints.
 * @returns The nearest actionable point (feature by feature).
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * explain.projectActions([30, 2, 1], [29, 3.6, 5], { monotone: { 0: 'increase' }, integer: [1], immutable: [2] }) // => [30, 4, 1]
 * ```
 */
export function projectActions(x: TensorLike, counterfactual: TensorLike, constraints: ActionConstraints): number[] {
  const a = rowOf(x, 'projectActions')
  const b = Array.from(rowOf(counterfactual, 'projectActions'))
  const fixed = new Set(constraints.immutable ?? [])
  const ints = new Set(constraints.integer ?? [])
  return b.map((v, j) => {
    const orig = a[j] ?? 0
    if (fixed.has(j)) return orig
    let out = v
    const dir = constraints.monotone?.[j]
    if (dir === 'increase') out = Math.max(out, orig)
    if (dir === 'decrease') out = Math.min(out, orig)
    const bound = constraints.bounds?.[j]
    if (bound !== undefined) out = Math.min(bound[1], Math.max(bound[0], out))
    if (ints.has(j)) out = Math.round(out)
    return out
  })
}

/** Options of {@link growingSpheres}. */
export interface GrowingSpheresOptions {
  /** Whether a model output is the desired outcome. */
  readonly accept: (prediction: number) => boolean
  /** The first sphere's radius (default 0.1, in units of the scales). */
  readonly radius?: number
  /** Points sampled per layer (default 200). */
  readonly nSamples?: number
  /** Most layers tried (default 50). */
  readonly maxLayers?: number
  /** Feature scales (default ones). */
  readonly scales?: readonly number[]
  readonly constraints?: ActionConstraints
  readonly key?: Key
}

/** The paths {@link growingSpheresAlgorithm} logs. */
export type GrowingSpheresPath = 'gs/radius' | 'gs/layer' | 'gs/enemies' | 'gs/distance' | 'gs/restored'

/** What a stepped growing-spheres search logs: the radius, the layer, enemies found, the closest one and features restored. */
export const growingSpheresContract: Contract<unknown, GrowingSpheresPath> = defineContract({
  'gs/radius': scalar({ description: 'the radius of the ball (shrinking) or the width of the layers (growing)' }),
  'gs/layer': scalar({ description: 'the layer sampled at this step (0 while the ball still shrinks)' }),
  'gs/enemies': scalar({ description: 'samples in this ball or layer with the desired outcome' }),
  'gs/distance': scalar({
    description: 'the scaled distance of the closest enemy found so far (-1 before one is found)',
    direction: 'lower',
  }),
  'gs/restored': scalar({ description: 'features moved back to the original value while sparsifying' }),
})

/** The phases of {@link growingSpheresAlgorithm}. */
export type GrowingSpheresPhase = 'shrink' | 'grow' | 'sparsify' | 'done'

/**
 * The state of {@link growingSpheresAlgorithm}: the phase, the sampling counter `t` (which fixes the key of each
 * layer), the radius and inner edge `lo` of the current layer, layers tried, the closest enemy found, and the
 * sparsification (the point being thinned, its prediction, the features still to try and how many were restored).
 */
export interface GrowingSpheresState {
  readonly phase: GrowingSpheresPhase
  readonly t: number
  readonly radius: number
  readonly lo: number
  readonly layers: number
  readonly found: { readonly p: readonly number[]; readonly f: number } | null
  readonly point: readonly number[]
  readonly value: number
  readonly order: readonly number[]
  readonly next: number
  readonly restored: number
}

function growingSpheresRun(
  predict: (x: Tensor) => Tensor,
  x: TensorLike,
  options: GrowingSpheresOptions,
): {
  readonly algorithm: Algorithm<GrowingSpheresState, GrowingSpheresPath>
  readonly finish: (s: GrowingSpheresState) => Counterfactual | undefined
  readonly maxSteps: number
} {
  const x0 = rowOf(x, 'growingSpheres')
  const d = x0.length
  const scales = options.scales ?? new Array<number>(d).fill(1)
  const ns = options.nSamples ?? 200
  const key = options.key ?? keyFor({}, 'growingSpheres')
  const constraints = options.constraints ?? {}
  const maxLayers = options.maxLayers ?? 50
  const evalRows = (rows: readonly (readonly number[])[]): number[] =>
    rows.length === 0 ? [] : toFlat(predict(fromStorage(Float64Array.from(rows.flat()), [rows.length, d], 'float64')))
  const dist = (p: readonly number[]): number =>
    Math.sqrt(p.reduce((s, v, j) => s + ((v - (x0[j] ?? 0)) / (scales[j] ?? 1)) ** 2, 0))
  // Uniform samples in the shell lo <= |z| < hi (scaled), constrained.
  const layer = (lo: number, hi: number, t: number): number[][] => {
    const z = toFlat(normal(foldIn(key, 2 * t), [ns, d]))
    const u = toFlat(uniform(foldIn(key, 2 * t + 1), [ns]))
    return Array.from({ length: ns }, (_, i) => {
      const dir = Array.from({ length: d }, (_, j) => z[i * d + j] ?? 0)
      const norm = Math.sqrt(dir.reduce((s, v) => s + v * v, 0)) || 1
      const r = (lo ** d + (u[i] ?? 0) * (hi ** d - lo ** d)) ** (1 / d)
      const p = dir.map((v, j) => (x0[j] ?? 0) + (v / norm) * r * (scales[j] ?? 1))
      return projectActions(x0, p, constraints)
    })
  }
  const empty: GrowingSpheresState = {
    phase: 'shrink',
    t: 0,
    radius: options.radius ?? 0.1,
    lo: 0,
    layers: 0,
    found: null,
    point: [],
    value: Number.NaN,
    order: [],
    next: 0,
    restored: 0,
  }
  const algorithm: Algorithm<GrowingSpheresState, GrowingSpheresPath> = {
    name: 'growing-spheres',
    config: { features: d, nSamples: ns, maxLayers },
    schema: channels({ metrics: { radius: { description: 'the radius' } } }),
    init: () => empty,
    step: (s, ctx) => {
      if (s.phase === 'shrink') {
        // Shrink the first ball until it holds no enemy (at most 30 halvings).
        const pts = layer(0, s.radius, s.t)
        const enemies = evalRows(pts).filter(options.accept).length
        ctx.log('gs/radius', s.radius)
        ctx.log('gs/layer', 0)
        ctx.log('gs/enemies', enemies)
        if (enemies === 0) return { ...s, phase: 'grow', lo: s.radius }
        const radius = s.radius / 2
        const t = s.t + 1
        return t >= 30 ? { ...s, phase: 'grow', t, radius, lo: radius } : { ...s, t, radius }
      }
      if (s.phase === 'grow') {
        const hi = s.lo + s.radius
        const t = s.t + s.layers
        const pts = layer(s.lo, hi, 100 + t)
        const f = evalRows(pts)
        let best = Number.POSITIVE_INFINITY
        let found: GrowingSpheresState['found'] = null
        let enemies = 0
        for (const [i, p] of pts.entries()) {
          if (!options.accept(f[i] ?? Number.NaN)) continue
          enemies += 1
          const dd = dist(p)
          if (dd < best) {
            best = dd
            found = { p, f: f[i] ?? 0 }
          }
        }
        ctx.log('gs/radius', s.radius)
        ctx.log('gs/layer', s.layers + 1)
        ctx.log('gs/enemies', enemies)
        ctx.log('gs/distance', found === null ? -1 : best)
        const layers = s.layers + 1
        if (found === null) return { ...s, lo: hi, layers, phase: layers >= maxLayers ? 'done' : 'grow' }
        // Sparsify: move features back to x's, smallest change first, while the point stays an enemy.
        const order = found.p
          .map((v, j) => ({ j, c: Math.abs(v - (x0[j] ?? 0)) / (scales[j] ?? 1) }))
          .filter((e) => e.c > 0)
          .sort((a, b) => a.c - b.c || a.j - b.j)
          .map((e) => e.j)
        return {
          ...s,
          lo: hi,
          layers,
          found,
          point: [...found.p],
          value: found.f,
          order,
          next: 0,
          phase: order.length === 0 ? 'done' : 'sparsify',
        }
      }
      // sparsify: one feature per step
      const j = s.order[s.next] ?? 0
      const trial = [...s.point]
      trial[j] = x0[j] ?? 0
      const f = evalRows([trial])[0] ?? Number.NaN
      const keep = options.accept(f)
      const restored = s.restored + (keep ? 1 : 0)
      ctx.log('gs/restored', restored)
      ctx.log('gs/distance', dist(keep ? trial : s.point))
      const next = s.next + 1
      return {
        ...s,
        point: keep ? trial : s.point,
        value: keep ? f : s.value,
        next,
        restored,
        phase: next >= s.order.length ? 'done' : 'sparsify',
      }
    },
    done: (s) => (s.phase === 'done' ? (s.found === null ? 'completed' : 'converged') : undefined),
    observe: (s) => ({ metrics: { radius: s.radius } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GrowingSpheresState,
  }
  const finish = (s: GrowingSpheresState): Counterfactual | undefined => {
    if (s.found === null) return undefined
    const l1 = s.point.reduce((a, v, j) => a + Math.abs(v - (x0[j] ?? 0)) / (scales[j] ?? 1), 0)
    return { x: [...s.point], prediction: s.value, distance: l1, valid: options.accept(s.value) }
  }
  return { algorithm, finish, maxSteps: 30 + maxLayers + d + 2 }
}

/**
 * Growing spheres (Laugel et al. 2018) as a stepped algorithm. While the first ball still holds an enemy (a sample
 * with the desired outcome) each step samples a ball of half the radius; then each step samples one spherical layer of
 * growing outer radius until one holds an enemy, and takes the closest; then each step moves one feature back to the
 * original value, smallest change first, if the point stays an enemy. The state serialises; every layer's samples come
 * from a key derived from the options' key and the layer, so a resumed run draws the same points.
 *
 * @param predict - The model.
 * @param x - The original point `[d]`.
 * @param options - accept, radius, nSamples, maxLayers, scales, constraints and key.
 * @returns The algorithm; it ends `converged` with a counterfactual in its state and `completed` without one.
 * @concept counterfactual-explanations
 * @example
 * ```ts
 * import { growingSpheresAlgorithm } from 'omai-engine/explain'
 * import { runAlgorithm } from 'omai-engine/trace'
 * import { tensor } from 'omai-engine/tensor'
 * const f = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i]))
 * runAlgorithm(growingSpheresAlgorithm(f, [0, 0], { accept: (p) => p > 1 }), { maxSteps: 200 }).status // => 'converged'
 * ```
 */
export function growingSpheresAlgorithm(
  predict: (x: Tensor) => Tensor,
  x: TensorLike,
  options: GrowingSpheresOptions,
): Algorithm<GrowingSpheresState, GrowingSpheresPath> {
  return growingSpheresRun(predict, x, options).algorithm
}

/**
 * Growing spheres (Laugel et al. 2018): a model-agnostic counterfactual search. Sample uniformly in a ball of
 * decreasing radius around x until no sample is an "enemy" (has the desired outcome), then in spherical layers of
 * growing radius until one is; take the closest enemy, and make it sparse by moving its features back to x's, smallest
 * change first, while it stays an enemy. Distances are Euclidean in units of the scales; actionability constraints
 * are applied to every sample. This is {@link growingSpheresAlgorithm} run to the end.
 *
 * @param predict - The model.
 * @param x - The original point `[d]`.
 * @param options - accept, radius, nSamples, maxLayers, scales, constraints and key.
 * @returns The counterfactual, or undefined when no enemy is found.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { tensor } from 'omai-engine/tensor'
 * const f = (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[2 * i]))
 * const cf = explain.growingSpheres(f, [0, 0], { accept: (p) => p > 1 })
 * cf.valid && cf.x[1] === 0 // => true
 * ```
 */
export function growingSpheres(
  predict: (x: Tensor) => Tensor,
  x: TensorLike,
  options: GrowingSpheresOptions,
): Counterfactual | undefined {
  const { algorithm, finish, maxSteps } = growingSpheresRun(predict, x, options)
  return finish(runAlgorithm(algorithm, { maxSteps }).state)
}

/** Options shared by the contrastive explanations (CEM) and prototype-guided counterfactuals. */
export interface ContrastiveOptions {
  /** The confidence margin kappa the class gap must exceed (default 0). */
  readonly kappa?: number
  /** The L1 weight beta (default 0.1). */
  readonly beta?: number
  /** The starting weight c of the class loss (default 10), doubled while no solution is valid. */
  readonly c?: number
  /** Rounds of c (default 6) and FISTA iterations per round (default 300) at this step size (default 0.01). */
  readonly rounds?: number
  readonly iterations?: number
  readonly learningRate?: number
  /** The feature range, `[low, high]` per feature (default unbounded). */
  readonly bounds?: readonly (readonly [number, number])[]
  /** The class explained (default the predicted class). */
  readonly classIndex?: number
}

/** A contrastive explanation: the perturbation found and the class it gives. */
export interface ContrastiveResult {
  /** The point (x + delta for a pertinent negative, delta for a pertinent positive). */
  readonly x: readonly number[]
  /** The perturbation delta. */
  readonly delta: readonly number[]
  /** The class predicted at the point, and the class explained. */
  readonly predictedClass: number
  readonly classIndex: number
  /** Whether the point meets the condition (another class for a PN; the same class for a PP). */
  readonly valid: boolean
  /** The class scores at the point. */
  readonly scores: readonly number[]
}

function classScores(model: DifferentiableModel, v: readonly number[]): number[] {
  return toFlat(model(fromStorage(Float64Array.from(v), [1, v.length], 'float64')) as Tensor)
}

/** The paths of the elastic-net search shared by CEM and prototype-guided counterfactuals. */
export type ElasticSearchPath =
  | 'search/delta'
  | 'search/distance'
  | 'search/c'
  | 'search/round'
  | 'search/valid'
  | 'search/best'

/** What a stepped elastic-net search logs: the perturbation, its size, the class-loss weight and the best valid size. */
export const elasticSearchContract: Contract<unknown, ElasticSearchPath> = defineContract({
  'search/delta': tensorSource({
    role: 'state',
    shape: ['features'],
    description: 'the perturbation after the FISTA iteration',
  }),
  'search/distance': scalar({ description: 'the L2 norm of the perturbation', direction: 'lower' }),
  'search/c': scalar({ description: 'the weight of the class loss in this round' }),
  'search/round': scalar({ description: 'the round of c (from 0)' }),
  'search/valid': scalar({ description: '1 when the perturbation changes the class as wanted' }),
  'search/best': scalar({
    description: 'the elastic-net size of the best valid perturbation so far (-1 before one)',
    direction: 'lower',
  }),
})

/**
 * The state of the elastic-net search: the round (a weight `c` of the class loss), the FISTA iteration within it, the
 * iterate `delta`, the momentum point `y` and sequence `tk`, whether the round has found a valid point, and the best
 * valid perturbation by elastic-net size.
 */
export interface ElasticSearchState {
  readonly round: number
  readonly iteration: number
  readonly c: number
  readonly delta: readonly number[]
  readonly y: readonly number[]
  readonly tk: number
  readonly found: boolean
  readonly best: readonly number[] | null
  /** The elastic-net size of `best` (-1 when there is none). */
  readonly bestSize: number
}

/**
 * The elastic-net FISTA search shared by CEM and prototype-guided counterfactuals as a stepped algorithm: minimise
 * c loss(point(delta)) + beta |delta|_1 + |delta|_2^2 (+ an extra smooth term) over a box, by projected FISTA with
 * soft thresholding, one iteration per step; after a round of iterations c doubles when no valid point was found and
 * halves otherwise, and the best valid point by elastic-net size is kept.
 */
function elasticSearchAlgorithm(
  name: string,
  objective: (delta: Value, c: number) => Value,
  valid: (delta: readonly number[]) => boolean,
  d: number,
  box: (j: number) => [number, number],
  options: ContrastiveOptions,
): Algorithm<ElasticSearchState, ElasticSearchPath> {
  const beta = options.beta ?? 0.1
  const lr = options.learningRate ?? 0.01
  const rounds = options.rounds ?? 6
  const iterations = options.iterations ?? 300
  const gradients = new Map<number, (v: Value) => Value | Tensor>()
  const gradAt = (c: number): ((v: Value) => Value | Tensor) => {
    let g = gradients.get(c)
    if (g === undefined) {
      g = grad((v: Value) => objective(v, c)) as (v: Value) => Value | Tensor
      gradients.set(c, g)
    }
    return g
  }
  const fresh = (round: number, c: number, s?: ElasticSearchState): ElasticSearchState => ({
    round,
    iteration: 0,
    c,
    delta: new Array<number>(d).fill(0),
    y: new Array<number>(d).fill(0),
    tk: 1,
    found: false,
    best: s?.best ?? null,
    bestSize: s?.bestSize ?? -1,
  })
  return {
    name,
    config: { features: d, beta, learningRate: lr, rounds, iterations },
    schema: channels({ metrics: { distance: { description: 'the size of the perturbation', direction: 'lower' } } }),
    init: () => fresh(0, options.c ?? 10),
    step: (s, ctx) => {
      const gv = toFlat(gradAt(s.c)(fromStorage(Float64Array.from(s.y), [d], 'float64')) as Tensor)
      const next = Float64Array.from(s.y, (v, j) => {
        const z = v - lr * (gv[j] ?? 0)
        const shrunk = Math.sign(z) * Math.max(0, Math.abs(z) - lr * beta)
        const [lo, hi] = box(j)
        return Math.min(hi, Math.max(lo, shrunk))
      })
      const t1 = (1 + Math.sqrt(1 + 4 * s.tk * s.tk)) / 2
      const y = Float64Array.from(next, (v, j) => v + ((s.tk - 1) / t1) * (v - (s.delta[j] ?? 0)))
      for (let j = 0; j < d; j++) {
        const [lo, hi] = box(j)
        y[j] = Math.min(hi, Math.max(lo, y[j] ?? 0))
      }
      const dv = Array.from(next)
      const ok = valid(dv)
      let best = s.best
      let bestSize = s.bestSize
      if (ok) {
        const size = dv.reduce((a, v) => a + beta * Math.abs(v) + v * v, 0)
        if (best === null || size < bestSize) {
          best = dv
          bestSize = size
        }
      }
      ctx.log('search/delta', fromStorage(Float64Array.from(next), [d], 'float64'))
      ctx.log('search/distance', Math.sqrt(dv.reduce((a, v) => a + v * v, 0)))
      ctx.log('search/c', s.c)
      ctx.log('search/round', s.round)
      ctx.log('search/valid', ok ? 1 : 0)
      ctx.log('search/best', bestSize)
      const found = s.found || ok
      const advanced: ElasticSearchState = {
        round: s.round,
        iteration: s.iteration + 1,
        c: s.c,
        delta: dv,
        y: Array.from(y),
        tk: t1,
        found,
        best,
        bestSize,
      }
      // Doubling c pushes harder towards validity; once valid, halving searches for a smaller perturbation.
      return advanced.iteration >= iterations ? fresh(s.round + 1, found ? s.c / 2 : s.c * 2, advanced) : advanced
    },
    done: (s) => (s.round >= rounds ? 'completed' : undefined),
    observe: (s) => ({ metrics: { distance: Math.sqrt(s.delta.reduce((a, v) => a + v * v, 0)) } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ElasticSearchState,
  }
}

function elasticSearch(
  name: string,
  objective: (delta: Value, c: number) => Value,
  valid: (delta: readonly number[]) => boolean,
  d: number,
  box: (j: number) => [number, number],
  options: ContrastiveOptions,
): number[] | undefined {
  const rounds = options.rounds ?? 6
  const iterations = options.iterations ?? 300
  const alg = elasticSearchAlgorithm(name, objective, valid, d, box, options)
  const end = runAlgorithm(alg, { maxSteps: rounds * iterations }).state
  return end.best === null ? undefined : [...end.best]
}

function classGapLoss(scores: Value, target: number, k: number, sign: 1 | -1, kappa: number): Value {
  // sign 1: the target class must lose (max other - f_t >= kappa); sign -1: it must win.
  const col = (j: number): Value => squeeze(sliceAxis(scores, -1, j, j + 1))
  let other: Value | undefined
  for (let j = 0; j < k; j++) {
    if (j === target) continue
    other = other === undefined ? col(j) : maximum(other, col(j))
  }
  const ft = col(target)
  const gap = sign === 1 ? sub(ft, other ?? ft) : sub(other ?? ft, ft)
  return relu(add(gap, kappa))
}

/**
 * A pertinent negative (Dhurandhar et al. 2018, CEM; alibi's CEM in PN mode): the smallest perturbation delta whose
 * addition changes the predicted class, what must be absent from x for its class to hold, found by projected FISTA on
 * c max(f_t(x + delta) - max_{i != t} f_i(x + delta), -kappa) + beta |delta|_1 + |delta|_2^2 within the feature range.
 *
 * @param model - A differentiable model returning class scores `[n, k]` (probabilities or logits).
 * @param x - The original point `[d]`.
 * @param options - kappa, beta, c, rounds, iterations, learningRate, bounds and classIndex.
 * @returns The pertinent negative, or undefined when none is found.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { matmul } from 'omai-engine/tensor'
 * // Two classes by the sign of x1 - x2: raising x2 above x1 flips it.
 * const scores = (v) => matmul(v, [[1, -1], [-1, 1]])
 * const pn = explain.pertinentNegative(scores, [1, 0.5], { beta: 0.01 })
 * pn.predictedClass // => 1
 * ```
 */
export function pertinentNegative(
  model: DifferentiableModel,
  x: TensorLike,
  options: ContrastiveOptions = {},
): ContrastiveResult | undefined {
  const x0 = rowOf(x, 'pertinentNegative')
  const d = x0.length
  const s0 = classScores(model, Array.from(x0))
  const k = s0.length
  const t = options.classIndex ?? argmaxOf(s0)
  const kappa = options.kappa ?? 0
  const xt = fromStorage(Float64Array.from(x0), [d], 'float64')
  const box = (j: number): [number, number] => {
    const b = options.bounds?.[j]
    return b === undefined
      ? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]
      : [b[0] - (x0[j] ?? 0), b[1] - (x0[j] ?? 0)]
  }
  const objective = (delta: Value, c: number): Value => {
    const scores = model(reshape(add(xt, delta), [1, d]))
    return add(mul(classGapLoss(scores, t, k, 1, kappa), c), sum(square(delta)))
  }
  const valid = (delta: readonly number[]): boolean =>
    argmaxOf(
      classScores(
        model,
        delta.map((v, j) => (x0[j] ?? 0) + v),
      ),
    ) !== t
  const delta = elasticSearch('pertinent-negative', objective, valid, d, box, options)
  if (delta === undefined) return undefined
  const point = delta.map((v, j) => (x0[j] ?? 0) + v)
  const scores = classScores(model, point)
  return { x: point, delta, predictedClass: argmaxOf(scores), classIndex: t, valid: true, scores }
}

/**
 * A pertinent positive (Dhurandhar et al. 2018, CEM; alibi's CEM in PP mode): the smallest part delta of x (between
 * the baseline 0 and x, feature by feature) that on its own keeps x's class, what must be present, found by projected
 * FISTA on c max(max_{i != t} f_i(delta) - f_t(delta), -kappa) + beta |delta|_1 + |delta|_2^2.
 *
 * @param model - A differentiable model returning class scores `[n, k]`.
 * @param x - The original point `[d]` (features measured from an absent value of 0).
 * @param options - kappa, beta, c, rounds, iterations, learningRate and classIndex.
 * @returns The pertinent positive, or undefined when none is found.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { add, matmul } from 'omai-engine/tensor'
 * // Class 0 needs x1 > 0.5; x2 is irrelevant, so the pertinent positive keeps part of x1 only.
 * const scores = (v) => add(matmul(v, [[1, 0], [0, 0]]), [[0, 0.5]])
 * const pp = explain.pertinentPositive(scores, [2, 3])
 * [pp.predictedClass, pp.delta[1] < 1e-6] // => [0, true]
 * ```
 */
export function pertinentPositive(
  model: DifferentiableModel,
  x: TensorLike,
  options: ContrastiveOptions = {},
): ContrastiveResult | undefined {
  const x0 = rowOf(x, 'pertinentPositive')
  const d = x0.length
  const s0 = classScores(model, Array.from(x0))
  const k = s0.length
  const t = options.classIndex ?? argmaxOf(s0)
  const kappa = options.kappa ?? 0
  const box = (j: number): [number, number] => {
    const v = x0[j] ?? 0
    return v >= 0 ? [0, v] : [v, 0]
  }
  const objective = (delta: Value, c: number): Value => {
    const scores = model(reshape(delta, [1, d]))
    return add(mul(classGapLoss(scores, t, k, -1, kappa), c), sum(square(delta)))
  }
  const valid = (delta: readonly number[]): boolean => argmaxOf(classScores(model, delta)) === t
  const delta = elasticSearch('pertinent-positive', objective, valid, d, box, options)
  if (delta === undefined) return undefined
  const scores = classScores(model, delta)
  return { x: delta, delta, predictedClass: argmaxOf(scores), classIndex: t, valid: true, scores }
}

/** Options of {@link prototypeCounterfactual}. */
export interface PrototypeOptions extends ContrastiveOptions {
  /** The weight theta of the prototype term (default 1). */
  readonly theta?: number
  /** Neighbours of x in each class averaged into its prototype (default 5). */
  readonly kPrototype?: number
  /** The target class (default the class whose prototype is nearest, other than x's). */
  readonly targetClass?: number
}

function prototypeRun(
  model: DifferentiableModel,
  x: TensorLike,
  train: TensorLike,
  labels: readonly number[],
  options: PrototypeOptions,
): {
  readonly algorithm: Algorithm<ElasticSearchState, ElasticSearchPath>
  readonly finish: (s: ElasticSearchState) => PrototypeResult | undefined
  readonly maxSteps: number
} {
  const x0 = rowOf(x, 'prototypeCounterfactual')
  const d = x0.length
  const m = matrixView(train, 'prototypeCounterfactual', { nFeatures: d })
  if (labels.length !== m.n) throw new OmaiError('shape', 'prototypeCounterfactual: one label per training row')
  const s0 = classScores(model, Array.from(x0))
  const t = options.classIndex ?? argmaxOf(s0)
  const kp = options.kPrototype ?? 5
  const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * d, (i + 1) * d)))
  const dist2 = (a: readonly number[]): number => a.reduce((s, v, j) => s + (v - (x0[j] ?? 0)) ** 2, 0)
  const prototypes = new Map<number, number[]>()
  for (const c of new Set(labels)) {
    const near = rows
      .map((r, i) => ({ r, i }))
      .filter((e) => labels[e.i] === c)
      .sort((a, b) => dist2(a.r) - dist2(b.r) || a.i - b.i)
      .slice(0, kp)
    prototypes.set(
      c,
      Array.from({ length: d }, (_, j) => near.reduce((s, e) => s + (e.r[j] ?? 0), 0) / Math.max(1, near.length)),
    )
  }
  let target = options.targetClass
  if (target === undefined) {
    let best = Number.POSITIVE_INFINITY
    for (const [c, p] of prototypes) {
      if (c === t) continue
      const v = dist2(p)
      if (v < best) {
        best = v
        target = c
      }
    }
  }
  const proto = target === undefined ? undefined : prototypes.get(target)
  if (target === undefined || proto === undefined)
    throw new OmaiError('value', 'prototypeCounterfactual: no other class to move to')
  const theta = options.theta ?? 1
  const xt = fromStorage(Float64Array.from(x0), [d], 'float64')
  const pt = fromStorage(Float64Array.from(proto), [d], 'float64')
  const k = s0.length
  const box = (j: number): [number, number] => {
    const b = options.bounds?.[j]
    return b === undefined
      ? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]
      : [b[0] - (x0[j] ?? 0), b[1] - (x0[j] ?? 0)]
  }
  const goal = target
  const objective = (delta: Value, c: number): Value => {
    const point = add(xt, delta)
    const scores = model(reshape(point, [1, d]))
    // The target class must win by kappa.
    const loss = classGapLoss(scores, goal, k, -1, options.kappa ?? 0)
    return add(add(mul(loss, c), sum(square(delta))), mul(sum(square(sub(point, pt))), theta))
  }
  const valid = (delta: readonly number[]): boolean =>
    argmaxOf(
      classScores(
        model,
        delta.map((v, j) => (x0[j] ?? 0) + v),
      ),
    ) === goal
  const algorithm = elasticSearchAlgorithm('prototype-counterfactual', objective, valid, d, box, options)
  const finish = (s: ElasticSearchState): PrototypeResult | undefined => {
    const delta = s.best === null ? undefined : [...s.best]
    if (delta === undefined) return undefined
    const point = delta.map((v, j) => (x0[j] ?? 0) + v)
    const scores = classScores(model, point)
    return {
      x: point,
      delta,
      predictedClass: argmaxOf(scores),
      classIndex: t,
      valid: true,
      scores,
      prototype: proto,
      targetClass: goal,
    }
  }
  return { algorithm, finish, maxSteps: (options.rounds ?? 6) * (options.iterations ?? 300) }
}

/** A prototype-guided counterfactual with the prototype it was pulled towards and the class it aims for. */
export type PrototypeResult = ContrastiveResult & {
  readonly prototype: readonly number[]
  readonly targetClass: number
}

/**
 * Prototype-guided counterfactuals as a stepped algorithm (the elastic-net FISTA search of {@link prototypeCounterfactual}),
 * one projected FISTA iteration per step. The class prototypes and the target class are fixed when the algorithm is built;
 * the state (round, iterate, momentum and the best valid perturbation so far) serialises. The counterfactual is
 * `x + state.best`.
 *
 * @param model - A differentiable model returning class scores `[n, k]`.
 * @param x - The original point `[d]`.
 * @param train - Training rows `[n, d]`.
 * @param labels - Their class indices (0..k-1).
 * @param options - theta, kPrototype, targetClass and the {@link ContrastiveOptions}.
 * @returns The algorithm, `rounds * iterations` steps.
 * @throws OmaiError with code `shape` for labels of the wrong length and `value` when there is no other class.
 * @concept counterfactual-explanations
 * @example
 * ```ts
 * import { prototypeCounterfactualAlgorithm } from 'omai-engine/explain'
 * import { matmul } from 'omai-engine/tensor'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const scores = (v) => matmul(v, [[1, -1], [-1, 1]])
 * const train = [[2, 0], [3, 1], [0, 2], [1, 3]]
 * const alg = prototypeCounterfactualAlgorithm(scores, [2, 1], train, [0, 0, 1, 1], { kPrototype: 2, rounds: 2, iterations: 50 })
 * runAlgorithm(alg, { maxSteps: 100 }).steps // => 100
 * ```
 */
export function prototypeCounterfactualAlgorithm(
  model: DifferentiableModel,
  x: TensorLike,
  train: TensorLike,
  labels: readonly number[],
  options: PrototypeOptions = {},
): Algorithm<ElasticSearchState, ElasticSearchPath> {
  return prototypeRun(model, x, train, labels, options).algorithm
}

/**
 * Prototype-guided counterfactuals (Van Looveren and Klaise 2021; alibi's CounterfactualProto without an encoder):
 * the {@link prototypeCounterfactualAlgorithm} run to the end.
 *
 * @param model - A differentiable model returning class scores `[n, k]`.
 * @param x - The original point `[d]`.
 * @param train - Training rows `[n, d]`.
 * @param labels - Their class indices (0..k-1).
 * @param options - theta, kPrototype, targetClass and the {@link ContrastiveOptions}.
 * @returns The counterfactual with its prototype and target class, or undefined when none is found.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { matmul } from 'omai-engine/tensor'
 * const scores = (v) => matmul(v, [[1, -1], [-1, 1]])
 * const train = [[2, 0], [3, 1], [0, 2], [1, 3]]
 * const r = explain.prototypeCounterfactual(scores, [2, 1], train, [0, 0, 1, 1], { kPrototype: 2 })
 * [r.targetClass, r.predictedClass] // => [1, 1]
 * ```
 */
export function prototypeCounterfactual(
  model: DifferentiableModel,
  x: TensorLike,
  train: TensorLike,
  labels: readonly number[],
  options: PrototypeOptions = {},
): PrototypeResult | undefined {
  const { algorithm, finish, maxSteps } = prototypeRun(model, x, train, labels, options)
  return finish(runAlgorithm(algorithm, { maxSteps }).state)
}

/**
 * The robustness of recourse to model change (Upadhyay, Joshi and Lakkaraju 2021; Rawal et al. 2020): the share of
 * alternative models (refits on resampled data, later versions) under which a counterfactual still gets the desired
 * outcome. Recourse that holds only for the model it was computed against is fragile.
 *
 * @param predictors - The alternative models.
 * @param counterfactual - The counterfactual point `[d]`.
 * @param accept - Whether a model output is the desired outcome.
 * @returns The share of models that accept it, and each model's verdict.
 * @example
 * ```ts
 * import * as explain from 'omai-engine/explain'
 * import { tensor } from 'omai-engine/tensor'
 * const shifted = (c) => (x) => tensor(Array.from({ length: x.shape[0] }, (_, i) => x.data[i] - c))
 * explain.recourseRobustness([shifted(0), shifted(0.5), shifted(2)], [1], (p) => p > 0).share // => 0.6666666666666666
 * ```
 */
export function recourseRobustness(
  predictors: readonly ((x: Tensor) => Tensor)[],
  counterfactual: TensorLike,
  accept: (prediction: number) => boolean,
): { readonly share: number; readonly accepted: readonly boolean[] } {
  const p = rowOf(counterfactual, 'recourseRobustness')
  if (predictors.length === 0) throw new OmaiError('value', 'recourseRobustness: no models')
  const accepted = predictors.map((f) =>
    accept(toFlat(f(fromStorage(Float64Array.from(p), [1, p.length], 'float64')))[0] ?? Number.NaN),
  )
  return { share: accepted.filter(Boolean).length / accepted.length, accepted }
}
