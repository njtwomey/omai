// The hyperparameter search shared by the Gaussian-process estimators: a solver of the `boundedSolvers` slot (L-BFGS-B
// by default) on a differentiable objective of a flat parameter vector, one iteration per step, restarted from further
// points in turn, keeping the best run. Not exported from the barrel.

import { customVjp } from '../autodiff/index.ts'
import { hyperparameters, type Kernel } from '../kernels/index.ts'
import { type BoundedProblem, boundedSolvers, infNorm, solverConfigOf } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { fromStorage, mul, toFlat, type Value } from '../tensor/index.ts'
import type { Algorithm, StepContext } from '../trace/index.ts'

/** The box of the free hyperparameters in theta (log) coordinates: positive ones within `bounds`, real ones free. */
export function thetaBox(kernel: Kernel, bounds: readonly [number, number]): { lower: number[]; upper: number[] } {
  const lower: number[] = []
  const upper: number[] = []
  for (const hp of hyperparameters(kernel)) {
    for (let i = 0; i < hp.size; i++) {
      lower.push(hp.constraint === 'positive' ? Math.log(bounds[0]) : -Infinity)
      upper.push(hp.constraint === 'positive' ? Math.log(bounds[1]) : Infinity)
    }
  }
  return { lower, upper }
}

/** What a search reads of the state of the solver in progress, whichever solver of the slot it is. */
export interface SearchInner {
  /** The iterate, in the objective's own coordinates. */
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  /** L-BFGS-B's infinity norm of the projected gradient. */
  readonly projectedGradNorm?: number
  /** L-BFGS-B's iteration count. */
  readonly iterations?: number
}

/** The default solver of the Gaussian-process searches: L-BFGS-B at scipy's memory and tolerances. */
export const defaultSearchSolver: SpaceValue = solverConfigOf(boundedSolvers, { kind: 'lbfgsb' })

/** The norm of the gradient a search reports: the projected one where the solver has it, else the largest entry. */
export const gradNormOf = (inner: SearchInner): number => inner.projectedGradNorm ?? infNorm(inner.grad)

/** The state of a search: the run in progress, the starts of all runs, and the best finished run. */
export interface SearchState {
  readonly run: number
  readonly starts: number[][]
  readonly inner: SearchInner | null
  /** Steps taken in the run in progress (for solvers that do not count iterations themselves). */
  readonly steps?: number
  readonly best: { readonly x: number[]; readonly value: number } | null
  readonly finished: boolean
}

/** What a search needs. */
export interface SearchSpec {
  /** The objective to minimise, a differentiable function of the parameter vector. */
  readonly objective: (x: Value) => Value
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  /** The starting points of the runs, drawn from the key (the first is the user's own). */
  readonly starts: (key: Key) => number[][]
  /** Whether to optimise at all (otherwise the first start stands). */
  readonly optimise: boolean
  /** The gradient tolerance of the solver (it overrides the solver's own). */
  readonly gradTol: number
  /** The solver, a value of `boundedSolvers.space` (default L-BFGS-B). */
  readonly solver?: SpaceValue
  /** The most iterations of one run. */
  readonly maxIter: number
}

/** A search as functions of its state. */
export interface Search {
  init(key: Key): SearchState
  step(s: SearchState, ctx: StepContext): SearchState
  /** The best parameters so far, finished or in progress, with their objective value. */
  current(s: SearchState): { readonly x: number[]; readonly value: number }
}

/** The best parameters of a state, finished or in progress, with their objective value. */
export function currentOf(s: SearchState): { readonly x: number[]; readonly value: number } {
  const running = s.inner === null ? null : { x: Array.from(s.inner.x), value: s.inner.value }
  if (s.best === null) return running ?? { x: [], value: Number.NaN }
  if (running === null || s.best.value <= running.value) return s.best
  return running
}

/** A stepped multi-start search with a solver of the bounded slot (L-BFGS-B unless `spec.solver` names another). */
export function lbfgsbSearch(spec: SearchSpec): Search {
  const named = (spec.solver ?? { kind: 'lbfgsb' }) as Readonly<Record<string, SpaceValue>>
  const solver = solverConfigOf(boundedSolvers, { ...named, gradTol: spec.gradTol })
  const algorithms = new Map<number, Algorithm<unknown>>()
  const algorithmFor = (run: number, start: readonly number[], key: Key): Algorithm<unknown> => {
    let alg = algorithms.get(run)
    if (alg === undefined) {
      const problem: BoundedProblem = { objective: spec.objective, x0: start, lower: spec.lower, upper: spec.upper }
      alg = boundedSolvers.make(problem, solver, key)
      algorithms.set(run, alg)
    }
    return alg
  }
  const valueAt = (x: readonly number[]): number =>
    toFlat(spec.objective(fromStorage(Float64Array.from(x), [x.length], 'float64')))[0] ?? Number.NaN
  return {
    current: currentOf,
    init: (key) => {
      const starts = spec.starts(key)
      const first = starts[0] as number[]
      if (!spec.optimise || first.length === 0)
        return { run: 0, starts: [first], inner: null, best: { x: first, value: valueAt(first) }, finished: false }
      return {
        run: 0,
        starts,
        inner: algorithmFor(0, first, key).init(key) as SearchInner,
        steps: 0,
        best: null,
        finished: false,
      }
    },
    step: (s, ctx) => {
      if (s.inner === null) return { ...s, finished: true }
      const alg = algorithmFor(s.run, s.starts[s.run] as number[], ctx.key)
      const inner = alg.step(s.inner, ctx) as SearchInner
      const steps = (s.steps ?? 0) + 1
      const stopped = alg.done?.(inner, 0) !== undefined || (inner.iterations ?? steps) >= spec.maxIter
      if (!stopped) return { ...s, inner, steps }
      const ended = { x: Array.from(inner.x), value: inner.value }
      const best = s.best === null || ended.value < s.best.value ? ended : s.best
      if (s.run + 1 < s.starts.length) {
        const next = algorithmFor(s.run + 1, s.starts[s.run + 1] as number[], ctx.key).init(ctx.key) as SearchInner
        return { ...s, run: s.run + 1, inner: next, steps: 0, best }
      }
      return { ...s, inner, steps, best, finished: true }
    },
  }
}

/**
 * A function to maximise, with its analytic gradient, as the differentiable objective of a minimiser: the negation, whose
 * derivative rule is the gradient. The function is evaluated once per point however often the value and the gradient
 * are asked for, and a value that is not finite becomes a large finite penalty.
 */
export function negatedObjective(
  name: string,
  evaluate: (x: Float64Array) => { readonly value: number; readonly gradient: Float64Array },
): (x: Value) => Value {
  let last:
    | { readonly key: string; readonly result: { readonly value: number; readonly gradient: Float64Array } }
    | undefined
  const at = (x: Value): { readonly value: number; readonly gradient: Float64Array } => {
    const values = Float64Array.from(toFlat(x))
    const key = values.join(',')
    if (last?.key !== key) last = { key, result: evaluate(values) }
    return last.result
  }
  const fn = customVjp({
    name,
    forward: (x) => {
      const value = -at(x).value
      return fromStorage(Float64Array.of(Number.isFinite(value) ? value : 1e25), [], 'float64')
    },
    vjp: (g, [x]) => {
      const gradient = at(x as Value).gradient
      return [
        mul(
          g,
          fromStorage(
            Float64Array.from(gradient, (v) => -v),
            [gradient.length],
            'float64',
          ),
        ),
      ]
    },
  })
  return (x) => fn(x)
}
