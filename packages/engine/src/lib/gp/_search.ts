// The hyperparameter search shared by the Gaussian-process estimators: L-BFGS-B on a differentiable objective of a flat
// parameter vector, one iteration per step, restarted from further points in turn, keeping the best run. Not exported
// from the barrel.

import { customVjp } from '../autodiff/index.ts'
import { type LbfgsbState, lbfgsbAlgorithm } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { fromStorage, mul, toFlat, type Value } from '../tensor/index.ts'
import type { Algorithm, StepContext } from '../trace/index.ts'

/** The state of a search: the run in progress, the starts of all runs, and the best finished run. */
export interface SearchState {
  readonly run: number
  readonly starts: number[][]
  readonly inner: LbfgsbState | null
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
  readonly gradTol: number
  /** The most L-BFGS-B iterations of one run. */
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

/** A stepped multi-start L-BFGS-B search. */
export function lbfgsbSearch(spec: SearchSpec): Search {
  const algorithms = new Map<number, Algorithm<LbfgsbState>>()
  const algorithmFor = (run: number, start: readonly number[]): Algorithm<LbfgsbState> => {
    let alg = algorithms.get(run)
    if (alg === undefined) {
      alg = lbfgsbAlgorithm(spec.objective, start, {
        bounds: { lower: spec.lower, upper: spec.upper },
        gradTol: spec.gradTol,
      })
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
      return { run: 0, starts, inner: algorithmFor(0, first).init(key), best: null, finished: false }
    },
    step: (s, ctx) => {
      if (s.inner === null) return { ...s, finished: true }
      const alg = algorithmFor(s.run, s.starts[s.run] as number[])
      const inner = alg.step(s.inner, ctx)
      const stopped = alg.done?.(inner, 0) !== undefined || inner.iterations >= spec.maxIter
      if (!stopped) return { ...s, inner }
      const ended = { x: Array.from(inner.x), value: inner.value }
      const best = s.best === null || ended.value < s.best.value ? ended : s.best
      if (s.run + 1 < s.starts.length) {
        const next = algorithmFor(s.run + 1, s.starts[s.run + 1] as number[]).init(ctx.key)
        return { ...s, run: s.run + 1, inner: next, best }
      }
      return { ...s, inner, best, finished: true }
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
