/**
 * Maximum likelihood for any structured model: the penalised negative conditional log-likelihood of labelled examples
 * under a `StructuredOracle`, with the gradient as expected minus observed (clamped) feature counts, and that objective
 * as a `SmoothProblem` for the solver slot (exact, or unbiased batch estimates for the stochastic solvers). It is the
 * one place where "a model is an objective" is spelled for structured prediction; the CRFs, the semi-Markov and
 * graphical ones differ only in their oracle.
 */

import { dasum, ddot } from '../../numerics/linalg/index.ts'
import { analyticObjective, type SmoothProblem } from '../../optim/index.ts'
import type { Value } from '../../tensor/index.ts'
import type { StructuredExample, StructuredOracle, StructuredPenalty, StructuredProblem } from './oracle.ts'

const asFloat = (w: ArrayLike<number>): Float64Array => (w instanceof Float64Array ? w : Float64Array.from(w))

/**
 * Add the penalties c2 ||w||^2 + c1 ||w||_1 of a structured objective: to the value (returned) and, in place, their
 * gradient 2 c2 w + c1 sign(w) (the L1 term by its subgradient, zero at zero) to `gradient`. The one version of the
 * penalty shared by the CRF objectives (chain, general and pseudo-likelihood); `structuredObjective` adds the same
 * terms after rescaling a batch.
 *
 * @param w - The weights.
 * @param value - The unpenalised objective.
 * @param gradient - The unpenalised gradient, updated in place.
 * @param penalty - The L2 and L1 coefficients (default none).
 * @returns The penalised objective.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { addStructuredPenalty } from 'omai-engine/pgm'
 * const g = new Float64Array([1, 1])
 * [addStructuredPenalty([2, -1], 5, g, { c2: 1, c1: 0.5 }), Array.from(g)] // => [11.5, [5.5, -1.5]]
 * ```
 */
export function addStructuredPenalty(
  w: ArrayLike<number>,
  value: number,
  gradient: Float64Array,
  penalty: StructuredPenalty = {},
): number {
  const c2 = penalty.c2 ?? 0
  const c1 = penalty.c1 ?? 0
  let v = value
  for (let i = 0; i < w.length; i++) {
    const wi = w[i] ?? 0
    v += c2 * wi * wi + c1 * Math.abs(wi)
    gradient[i] = (gradient[i] ?? 0) + 2 * c2 * wi + c1 * Math.sign(wi)
  }
  return v
}

/**
 * The negative conditional log-likelihood of labelled examples under an oracle, plus the penalties c2 ||w||^2 and
 * c1 ||w||_1, and its gradient: for each example the log-partition function minus the clamped one (the score of the
 * labelling, when the oracle has no `clamped`), whose gradient is the expected minus the clamped feature counts. The
 * L1 term is not smooth and enters by its subgradient c1 sign(w), as in CRFsuite's objective at a point; for exact
 * zeros minimise with `structuredSolvers`' `owlqn` mode, which is given the penalty with `c1` removed. With a batch of example indices the likelihood is rescaled by n / |batch|, an
 * unbiased estimate of the full objective.
 *
 * @param oracle - The structured model.
 * @param data - Labelled examples.
 * @param w - The weights (the point of evaluation).
 * @param penalty - The L2 and L1 coefficients.
 * @param batch - Indices of the examples to sum over (default all).
 * @returns The objective, its gradient and the (rescaled) negative log-likelihood alone.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf, structuredObjective } from 'omai-engine/pgm'
 * const oracle = chainOracle(linearChainCrf(['A', 'B'], ['f']))
 * const r = structuredObjective(oracle, [{ x: [{ f: 1 }], y: ['A'] }], new Float64Array(6))
 * [r.value === Math.LN2, Array.from(r.gradient)] // => [true, [-0.5, 0.5, 0, 0, 0, 0]]
 * ```
 */
export function structuredObjective<X, Y>(
  oracle: StructuredOracle<X, Y>,
  data: readonly StructuredExample<X, Y>[],
  w: ArrayLike<number>,
  penalty: StructuredPenalty = {},
  batch?: readonly number[],
): { readonly value: number; readonly gradient: Float64Array; readonly loss: number } {
  const wf = asFloat(w)
  const gradient = new Float64Array(oracle.size)
  let nll = 0
  const visit = (example: StructuredExample<X, Y> | undefined): void => {
    if (example === undefined) return
    const { x, y } = example
    if (oracle.gradientInto !== undefined) {
      nll += oracle.gradientInto(wf, x, y, gradient)
      return
    }
    const free = oracle.expectations(wf, x)
    const fixed = oracle.clamped?.(wf, x, y)
    const counts = fixed?.counts ?? oracle.features(x, y)
    nll += free.logPartition - (fixed?.logPartition ?? ddot(wf, counts))
    for (let q = 0; q < gradient.length; q++)
      gradient[q] = (gradient[q] ?? 0) + (free.counts[q] ?? 0) - (counts[q] ?? 0)
  }
  if (batch === undefined) for (const example of data) visit(example)
  else for (const i of batch) visit(data[i])
  const scale = batch === undefined ? 1 : data.length / Math.max(1, batch.length)
  const c2 = penalty.c2 ?? 0
  const c1 = penalty.c1 ?? 0
  for (let q = 0; q < gradient.length; q++)
    gradient[q] = scale * (gradient[q] ?? 0) + 2 * c2 * (wf[q] ?? 0) + c1 * Math.sign(wf[q] ?? 0)
  return { value: scale * nll + c2 * ddot(wf, wf) + c1 * dasum(wf), gradient, loss: scale * nll }
}

/** One evaluation shared by the value and the gradient of the same point (the objective asks for both). */
function memoised<T>(at: (w: Float64Array) => T): (w: Float64Array) => T {
  let last: { readonly w: Float64Array; readonly result: T } | undefined
  return (w) => {
    if (last !== undefined && last.w.length === w.length && last.w.every((v, i) => v === w[i])) return last.result
    const result = at(w)
    last = { w: Float64Array.from(w), result }
    return result
  }
}

/**
 * A structured model's training objective as a smooth problem for the solver slot: `structuredObjective` (the penalised
 * negative conditional log-likelihood with its analytic gradient) over the flat weight vector, with unbiased batch
 * estimates for the stochastic solvers. The value and the gradient of a point come from one pass over the data. The L1
 * penalty c1 enters by its subgradient, which the gradient-based solvers can only approximate: use `owlqn` for exact
 * zeros.
 *
 * @param problem - The oracle, data, penalties and start.
 * @returns The problem: objective (with batches over the examples) and the starting weights.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { chainOracle, linearChainCrf, structuredSmoothProblem } from 'omai-engine/pgm'
 * const oracle = chainOracle(linearChainCrf(['A', 'B'], ['f']))
 * structuredSmoothProblem({ oracle, data: [{ x: [{ f: 1 }], y: ['A'] }] }).x0.length // => 6
 * ```
 */
export function structuredSmoothProblem<X, Y>(problem: StructuredProblem<X, Y>): SmoothProblem {
  const { oracle, data } = problem
  const penalty = problem.penalty ?? {}
  const n = data.length
  const objectiveOf = (batch?: readonly number[]): ((x: Value) => Value) => {
    const at = memoised((w: Float64Array) => structuredObjective(oracle, data, w, penalty, batch))
    return analyticObjective({ value: (w) => at(w).value, gradient: (w) => at(w).gradient })
  }
  const full = objectiveOf()
  return {
    objective: ((x: unknown, batch?: readonly number[]) =>
      (batch === undefined ? full : objectiveOf(batch))(x as Value)) as SmoothProblem['objective'],
    x0: Array.from(problem.start ?? new Float64Array(oracle.size)),
    batches: { n },
  }
}
