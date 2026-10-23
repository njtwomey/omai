import { maxOf } from '../../base/index.ts'
import {
  arange,
  argmax,
  broadcastShapes,
  broadcastTo,
  cast,
  clip,
  cumsum,
  equal,
  exp,
  expandDims,
  floor,
  greaterEqual,
  less,
  log,
  log1p,
  logicalAnd,
  moveaxis,
  mul,
  neg,
  primalOf,
  promoteFloat,
  reshape,
  squeeze,
  sub,
  sum,
  type TensorLike,
  takeAlongAxis,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { validateParams } from '../distribution.ts'

/** Whether entries are non-negative integers (the support of count distributions). */
export const isCount = (x: Value): Value => logicalAnd(greaterEqual(x, 0), equal(floor(x), x))

/** The largest entry of a parameter's concrete value (term counts of series and sums are data-dependent choices). */
export function largest(v: Value): number {
  return maxOf(toFlat(primalOf(v)))
}

/** The values 0, ..., kmax on a new leading axis, shaped `[kmax + 1, 1, ..., 1]` to broadcast against a batch. */
export function countAxis(kmax: number, batchRank: number): Value {
  return reshape(arange(0, kmax + 1), [kmax + 1, ...new Array<number>(batchRank).fill(1)])
}

/** -sum_k p_k log p_k over k = 0 .. kmax, with k on a new leading axis so the parameters broadcast against it. */
export function entropyBySum(logPmf: (k: Value) => Value, kmax: number, batchRank: number): Value {
  const lp = logPmf(countAxis(kmax, batchRank))
  const finite = equal(lp, lp)
  const safe = where(logicalAnd(finite, greaterEqual(lp, -1e300)), lp, 0)
  return neg(sum(where(equal(safe, 0), 0, mul(exp(safe), safe)), 0))
}

/**
 * P(X <= x) for a count distribution with support in 0 .. kmax, by cumulative sums of its pmf over that range,
 * gathered at floor(x) (0 below the support, 1 above kmax). Differentiable in the parameters.
 */
export function cdfBySum(logPmf: (k: Value) => Value, x: Value, kmax: number, batchShape: readonly number[]): Value {
  const table = cumsum(exp(logPmf(countAxis(kmax, batchShape.length))), 0)
  // Move the count axis last, broadcast to the points, and gather.
  const shape = broadcastShapes(x.shape, batchShape)
  const lastAxis = broadcastTo(moveaxis(table, 0, -1), [...shape, kmax + 1])
  const k = floor(x)
  const index = cast(expandDims(broadcastTo(clip(k, 0, kmax), shape), -1), 'int32')
  const gathered = squeeze(takeAlongAxis(lastAxis, index, -1), -1)
  return where(less(k, 0), 0, where(greaterEqual(k, kmax), 1, gathered))
}

/** The most probable count in 0 .. kmax (the first, on ties), from a table of the log pmf. */
export function modeBySearch(logPmf: (k: Value) => Value, kmax: number, batchRank: number): Value {
  return cast(argmax(logPmf(countAxis(kmax, batchRank)), 0), 'float64')
}

/** The logits of `{ probs }` or `{ logits }` parameters of a binary outcome (probabilities validated). */
export function binaryLogits(
  params: { readonly probs: TensorLike } | { readonly logits: TensorLike },
  family: string,
): Value {
  if ('logits' in params) return promoteFloat(params.logits)[0] as Value
  const [p] = promoteFloat(params.probs) as [Value]
  validateParams(family, { probs: [p, 'unitInterval'] })
  return sub(log(p), log1p(neg(p)))
}
