import { maxOf, WhiteboxError } from '../../base/index.ts'
import {
  equal,
  exp,
  floor,
  greaterEqual,
  isTensor,
  logicalAnd,
  mul,
  neg,
  promoteFloat,
  reshape,
  sum,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'

/** Whether entries are non-negative integers (the support of count distributions). */
export const isCount = (x: Value): Value => logicalAnd(greaterEqual(x, 0), equal(floor(x), x))

/** The largest entry of a concrete parameter (entropies of count distributions sum their pmf up to a bound from it). */
export function largest(v: Value, name: string): number {
  if (!isTensor(v)) throw new WhiteboxError('not-differentiable', `${name}: entropy needs concrete parameters`)
  return maxOf(toFlat(v))
}

/** -sum_k p_k log p_k over k = 0 .. kmax, with k on a new leading axis so the parameters broadcast against it. */
export function entropyBySum(logPmf: (k: Value) => Value, kmax: number, batchRank: number): Value {
  const k = promoteFloat(Array.from({ length: kmax + 1 }, (_, i) => i))[0] as Value
  const lp = logPmf(reshape(k, [kmax + 1, ...new Array<number>(batchRank).fill(1)]))
  return neg(sum(where(equal(lp, Number.NEGATIVE_INFINITY), 0, mul(exp(lp), lp)), 0))
}
