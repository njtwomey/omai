/**
 * Variance-reduced stochastic methods for finite sums such as empirical risks: SAG, SAGA (with a proximal step for
 * nonsmooth penalties, as scikit-learn's solvers) and SVRG, which correct each stochastic gradient with stored or
 * snapshot gradients so they converge as fast as full-gradient methods at the cost of one example per step.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { daxpy, idamax } from '../numerics/linalg/index.ts'
import { randint } from '../random/index.ts'
import type { Value } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { type FlatObjective, flatObjective, type MinimizeResult, type Parameters } from './minimize/objective.ts'
import type { ParamTree } from './transform.ts'

/** A finite sum f(x) = (1/n) sum_i f_i(x): the empirical risk of a model on n examples. */
export interface FiniteSum {
  /** The number of terms. */
  readonly n: number
  /** The i-th term f_i, a scalar function of the parameters (one example's loss plus its share of a smooth penalty). */
  readonly term: (params: ParamTree, i: number) => Value
}

/** A proximal operator: argmin_z h(z) + |z - x|^2 / (2 step), for a non-smooth penalty h such as an L1 norm. */
export type ProximalStep = (x: Float64Array, step: number) => Float64Array

/** Options of the variance-reduced methods. */
export interface VarianceReducedOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The step size; by default derived from `lipschitz` (1/L for SAG, 1/(3L) for SAGA, 1/(10L) for SVRG). */
  readonly learningRate?: number
  /** The largest Lipschitz constant of the term gradients. */
  readonly lipschitz?: number
  /** A proximal step for a non-smooth penalty (SAGA and SVRG). */
  readonly prox?: ProximalStep
  /** Stop when max |x - x_prev| / max |x| over an epoch falls to this (scikit-learn's tol, default 1e-6). */
  readonly tol?: number
  /** The epoch cap (default 1000). */
  readonly maxEpochs?: number
}

/** The state of a variance-reduced method: the iterate, a gradient table or snapshot, and per-epoch progress. */
export interface VarianceReducedState {
  readonly x: Float64Array
  readonly epoch: number
  /** max |x - x_prev| / max |x| over the last epoch. */
  readonly change: number
  /** Term-gradient evaluations so far. */
  readonly evaluations: number
  readonly internal: Readonly<Record<string, unknown>>
}

const SCHEMA = channels({
  metrics: {
    loss: { description: 'the finite-sum objective (smooth part)', direction: 'lower' },
    change: { description: 'relative change of the iterate over the epoch' },
    evaluations: { description: 'term-gradient evaluations so far' },
  },
  tensors: { x: { role: 'param', description: 'iterate' } },
})

interface Setup {
  readonly objective: FlatObjective
  readonly grad: (x: Float64Array, i: number) => Float64Array
  readonly value: (x: Float64Array) => number
  readonly step: number
  readonly tol: number
  readonly prox: ProximalStep
}

function setup(
  problem: FiniteSum,
  x0: Parameters,
  options: VarianceReducedOptions,
  factor: number,
  name: string,
): Setup {
  const step = options.learningRate ?? (options.lipschitz === undefined ? Number.NaN : 1 / (factor * options.lipschitz))
  if (!(step > 0)) throw new MdmlError('value', `${name} needs a positive learningRate or lipschitz constant`)
  const terms = new Map<number, FlatObjective>()
  const objective = flatObjective((p) => problem.term(p, 0), x0)
  const termOf = (i: number): FlatObjective => {
    let t = terms.get(i)
    if (t === undefined) {
      t = flatObjective((p) => problem.term(p, i), x0)
      terms.set(i, t)
    }
    return t
  }
  return {
    objective,
    grad: (x, i) => termOf(i).valueAndGrad(x).grad,
    value: (x) => {
      let s = 0
      for (let i = 0; i < problem.n; i++) s += termOf(i).value(x)
      return s / problem.n
    },
    step,
    tol: options.tol ?? 1e-6,
    prox: options.prox ?? ((x) => x),
  }
}

function relativeChange(x: Float64Array, prev: Float64Array): number {
  const d = daxpy(-1, prev, x)
  const top = Math.abs(d[idamax(d)] ?? 0)
  const scale = Math.abs(x[idamax(x)] ?? 0)
  return scale === 0 ? top : top / scale
}

function algorithm(
  name: string,
  s: Setup,
  options: VarianceReducedOptions,
  init: () => VarianceReducedState,
  epoch: (state: VarianceReducedState, indices: Int32Array) => VarianceReducedState,
  draws: number,
  n: number,
): Algorithm<VarianceReducedState> {
  return {
    name,
    config: { learningRate: s.step, tol: s.tol, proximal: options.prox !== undefined } as Json,
    schema: SCHEMA,
    init,
    step: (state, ctx) => epoch(state, randint(ctx.key, [draws], 0, n).data as Int32Array),
    done: (state) =>
      !state.x.every(Number.isFinite) ? 'diverged' : state.epoch > 0 && state.change <= s.tol ? 'converged' : undefined,
    observe: (state) => ({
      metrics: { loss: s.value(state.x), change: state.change, evaluations: state.evaluations },
      tensors: { x: { role: 'param', value: state.x } },
    }),
  }
}

function finish(
  alg: Algorithm<VarianceReducedState>,
  s: Setup,
  options: VarianceReducedOptions,
  n: number,
): MinimizeResult {
  const { learningRate: _l, lipschitz: _c, prox: _p, tol: _t, maxEpochs, ...run } = options
  const result = runAlgorithm(alg, { ...run, maxSteps: maxEpochs ?? 1000 })
  const x = result.state.x
  const full = new Float64Array(x.length)
  for (let i = 0; i < n; i++) {
    const g = s.grad(x, i)
    for (let k = 0; k < x.length; k++) full[k] = (full[k] ?? 0) + (g[k] ?? 0) / n
  }
  return {
    x: s.objective.unravel(x),
    fun: s.value(x),
    grad: s.objective.unravel(full),
    gradNorm: Math.abs(full[idamax(full)] ?? 0),
    steps: result.steps,
    nfev: n,
    ngev: result.state.evaluations + n,
    nhev: 0,
    status: result.status,
    converged: result.status === 'converged',
    message:
      result.status === 'converged'
        ? 'relative change of the iterate below tol'
        : result.status === 'completed'
          ? 'maximum number of epochs reached'
          : `stopped: ${result.status}`,
  }
}

/**
 * Stochastic average gradient (Schmidt, Le Roux and Bach 2017) as a traced algorithm: each step is an epoch of n
 * updates x <- x - (eta / m) d, where d is the sum of the most recent gradient of every term seen so far (m of them)
 * and one term's entry is refreshed per update. Linear convergence for strongly convex sums at the cost of SGD.
 *
 * @param problem - The finite sum.
 * @param x0 - The starting parameters.
 * @param options - Step size (or Lipschitz constant), tolerance and run options.
 * @returns The algorithm.
 * @concept variance-reduction
 * @example
 * ```ts
 * import { sagAlgorithm } from 'mdml-engine/optim'
 * import { square, sub, type Value } from 'mdml-engine/tensor'
 * sagAlgorithm({ n: 2, term: (x, i) => square(sub(x as Value, i)) }, [0], { lipschitz: 2 }).name // => 'sag'
 * ```
 */
export function sagAlgorithm(
  problem: FiniteSum,
  x0: Parameters,
  options: VarianceReducedOptions = {},
): Algorithm<VarianceReducedState> {
  const s = setup(problem, x0, options, 1, 'sag')
  const n = problem.n
  const d0 = s.objective.n
  return algorithm(
    'sag',
    s,
    options,
    () => ({
      x: s.objective.x0,
      epoch: 0,
      change: Number.POSITIVE_INFINITY,
      evaluations: 0,
      internal: { table: new Float64Array(n * d0), sum: new Float64Array(d0), seen: new Uint8Array(n), m: 0 },
    }),
    (state, indices) => {
      const {
        table: t0,
        sum: s0,
        seen: seen0,
        m: m0,
      } = state.internal as {
        table: Float64Array
        sum: Float64Array
        seen: Uint8Array
        m: number
      }
      const table = Float64Array.from(t0)
      const sum = Float64Array.from(s0)
      const seen = Uint8Array.from(seen0)
      let m = m0
      let x = state.x
      for (const i of indices) {
        const g = s.grad(x, i)
        const row = table.subarray(i * d0, (i + 1) * d0)
        for (let k = 0; k < d0; k++) sum[k] = (sum[k] ?? 0) - (row[k] ?? 0) + (g[k] ?? 0)
        row.set(g)
        if (seen[i] === 0) {
          seen[i] = 1
          m += 1
        }
        x = daxpy(-s.step / m, sum, x)
      }
      return {
        x,
        epoch: state.epoch + 1,
        change: relativeChange(x, state.x),
        evaluations: state.evaluations + indices.length,
        internal: { table, sum, seen, m },
      }
    },
    n,
    n,
  )
}

/**
 * Minimise a finite sum by SAG (scikit-learn's solver='sag').
 *
 * @param problem - The finite sum.
 * @param x0 - The starting parameters.
 * @param options - Step size (or Lipschitz constant), tolerance, epoch cap and run options.
 * @returns The result.
 * @concept variance-reduction
 * @example
 * ```ts
 * import { sag } from 'mdml-engine/optim'
 * import { square, sub, toArray, type Value } from 'mdml-engine/tensor'
 * const r = sag({ n: 3, term: (x, i) => square(sub(x as Value, i)) }, [0], { lipschitz: 2, tol: 1e-12 })
 * Math.round((toArray(r.x as Value) as number[])[0] * 1e8) / 1e8 // => 1
 * ```
 */
export function sag(problem: FiniteSum, x0: Parameters, options: VarianceReducedOptions = {}): MinimizeResult {
  const alg = sagAlgorithm(problem, x0, options)
  return finish(alg, setup(problem, x0, options, 1, 'sag'), options, problem.n)
}

/**
 * SAGA (Defazio, Bach and Lacoste-Julien 2014) as a traced algorithm: unbiased variance-reduced steps
 * x <- prox(x - eta (g_j(x) - y_j + mean(y)), eta) with a table y of the latest term gradients, initialised at x0.
 * Supports non-smooth penalties through the proximal step.
 *
 * @param problem - The finite sum (its smooth part).
 * @param x0 - The starting parameters.
 * @param options - Step size (or Lipschitz constant, default step 1/(3L)), proximal step, tolerance and run options.
 * @returns The algorithm.
 * @concept variance-reduction
 * @example
 * ```ts
 * import { sagaAlgorithm } from 'mdml-engine/optim'
 * import { square, sub, type Value } from 'mdml-engine/tensor'
 * sagaAlgorithm({ n: 2, term: (x, i) => square(sub(x as Value, i)) }, [0], { lipschitz: 2 }).name // => 'saga'
 * ```
 */
export function sagaAlgorithm(
  problem: FiniteSum,
  x0: Parameters,
  options: VarianceReducedOptions = {},
): Algorithm<VarianceReducedState> {
  const s = setup(problem, x0, options, 3, 'saga')
  const n = problem.n
  const d0 = s.objective.n
  return algorithm(
    'saga',
    s,
    options,
    () => {
      const x = s.objective.x0
      const table = new Float64Array(n * d0)
      const mean = new Float64Array(d0)
      for (let i = 0; i < n; i++) {
        const g = s.grad(x, i)
        table.set(g, i * d0)
        for (let k = 0; k < d0; k++) mean[k] = (mean[k] ?? 0) + (g[k] ?? 0) / n
      }
      return { x, epoch: 0, change: Number.POSITIVE_INFINITY, evaluations: n, internal: { table, mean } }
    },
    (state, indices) => {
      const { table: t0, mean: m0 } = state.internal as { table: Float64Array; mean: Float64Array }
      const table = Float64Array.from(t0)
      const mean = Float64Array.from(m0)
      let x = state.x
      for (const j of indices) {
        const g = s.grad(x, j)
        const row = table.subarray(j * d0, (j + 1) * d0)
        const direction = new Float64Array(d0)
        for (let k = 0; k < d0; k++) direction[k] = (g[k] ?? 0) - (row[k] ?? 0) + (mean[k] ?? 0)
        x = s.prox(daxpy(-s.step, direction, x), s.step)
        for (let k = 0; k < d0; k++) mean[k] = (mean[k] ?? 0) + ((g[k] ?? 0) - (row[k] ?? 0)) / n
        row.set(g)
      }
      return {
        x,
        epoch: state.epoch + 1,
        change: relativeChange(x, state.x),
        evaluations: state.evaluations + indices.length,
        internal: { table, mean },
      }
    },
    n,
    n,
  )
}

/**
 * Minimise a finite sum (plus an optional non-smooth penalty through `prox`) by SAGA (scikit-learn's solver='saga').
 *
 * @param problem - The finite sum.
 * @param x0 - The starting parameters.
 * @param options - Step size (or Lipschitz constant), proximal step, tolerance, epoch cap and run options.
 * @returns The result.
 * @concept variance-reduction
 * @example
 * ```ts
 * import { saga } from 'mdml-engine/optim'
 * import { square, sub, toArray, type Value } from 'mdml-engine/tensor'
 * const r = saga({ n: 3, term: (x, i) => square(sub(x as Value, i)) }, [0], { lipschitz: 2, tol: 1e-12 })
 * Math.round((toArray(r.x as Value) as number[])[0] * 1e8) / 1e8 // => 1
 * ```
 */
export function saga(problem: FiniteSum, x0: Parameters, options: VarianceReducedOptions = {}): MinimizeResult {
  return finish(sagaAlgorithm(problem, x0, options), setup(problem, x0, options, 3, 'saga'), options, problem.n)
}

/** Options of {@link svrg}: the variance-reduced options plus the inner-loop length. */
export interface SvrgOptions extends VarianceReducedOptions {
  /** Inner updates per snapshot (default 2n, Johnson and Zhang's choice for convex problems). */
  readonly innerSteps?: number
}

/**
 * Stochastic variance-reduced gradient (Johnson and Zhang 2013; proximal form of Xiao and Zhang 2014) as a traced
 * algorithm: each step takes a snapshot x~ with its full gradient mu, then makes m updates
 * x <- prox(x - eta (g_i(x) - g_i(x~) + mu), eta), keeping no gradient table.
 *
 * @param problem - The finite sum.
 * @param x0 - The starting parameters.
 * @param options - Step size (or Lipschitz constant, default 1/(10L)), inner steps, proximal step, tolerance.
 * @returns The algorithm.
 * @concept variance-reduction
 * @example
 * ```ts
 * import { svrgAlgorithm } from 'mdml-engine/optim'
 * import { square, sub, type Value } from 'mdml-engine/tensor'
 * svrgAlgorithm({ n: 2, term: (x, i) => square(sub(x as Value, i)) }, [0], { lipschitz: 2 }).name // => 'svrg'
 * ```
 */
export function svrgAlgorithm(
  problem: FiniteSum,
  x0: Parameters,
  options: SvrgOptions = {},
): Algorithm<VarianceReducedState> {
  const s = setup(problem, x0, options, 10, 'svrg')
  const n = problem.n
  const d0 = s.objective.n
  const inner = options.innerSteps ?? 2 * n
  return algorithm(
    'svrg',
    s,
    options,
    () => ({ x: s.objective.x0, epoch: 0, change: Number.POSITIVE_INFINITY, evaluations: 0, internal: {} }),
    (state, indices) => {
      const snapshot = state.x
      const grads = Array.from({ length: n }, (_, i) => s.grad(snapshot, i))
      const mu = new Float64Array(d0)
      for (const g of grads) for (let k = 0; k < d0; k++) mu[k] = (mu[k] ?? 0) + (g[k] ?? 0) / n
      let x = snapshot
      for (const i of indices) {
        const g = s.grad(x, i)
        const old = grads[i] as Float64Array
        const direction = new Float64Array(d0)
        for (let k = 0; k < d0; k++) direction[k] = (g[k] ?? 0) - (old[k] ?? 0) + (mu[k] ?? 0)
        x = s.prox(daxpy(-s.step, direction, x), s.step)
      }
      return {
        x,
        epoch: state.epoch + 1,
        change: relativeChange(x, snapshot),
        evaluations: state.evaluations + n + indices.length,
        internal: { snapshotGradNorm: Math.abs(mu[idamax(mu)] ?? 0) },
      }
    },
    inner,
    n,
  )
}

/**
 * Minimise a finite sum by SVRG (with a proximal step for non-smooth penalties).
 *
 * @param problem - The finite sum.
 * @param x0 - The starting parameters.
 * @param options - Step size (or Lipschitz constant), inner steps, proximal step, tolerance, epoch cap and run options.
 * @returns The result.
 * @concept variance-reduction
 * @example
 * ```ts
 * import { svrg } from 'mdml-engine/optim'
 * import { square, sub, toArray, type Value } from 'mdml-engine/tensor'
 * const r = svrg({ n: 3, term: (x, i) => square(sub(x as Value, i)) }, [0], { learningRate: 0.2, tol: 1e-12 })
 * Math.round((toArray(r.x as Value) as number[])[0] * 1e8) / 1e8 // => 1
 * ```
 */
export function svrg(problem: FiniteSum, x0: Parameters, options: SvrgOptions = {}): MinimizeResult {
  const { innerSteps: _i, ...rest } = options
  return finish(svrgAlgorithm(problem, x0, options), setup(problem, x0, options, 10, 'svrg'), rest, problem.n)
}
