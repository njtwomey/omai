// Private one-dimensional maximisation for profile likelihoods and cross-validation criteria: a grid scan to locate the
// best bracket, then Brent's root finder on the derivative (from reverse-mode differentiation). The optimisers proper
// live in optim; this only roots a smooth derivative.

import { grad } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import { asTensor, item, type Tensor, type Value } from '../tensor/index.ts'

/** The derivative of a scalar tensor function of one parameter, as a function of numbers. */
export function slopeOf(f: (t: Value) => Value): (t: number) => number {
  const g = grad(f)
  return (t) => item(g(asTensor([t])) as Tensor)
}

/**
 * The maximiser of a smooth function on [lo, hi]: the best of `grid` + 1 equally spaced points, refined to a root of
 * the derivative in the neighbouring cells. Returns an end point when the maximum is on the boundary.
 */
export function maximiseSmooth(
  f: (t: number) => number,
  df: (t: number) => number,
  lo: number,
  hi: number,
  grid = 64,
): number {
  if (!(hi > lo)) throw new OmaiError('value', 'maximise: needs lo < hi')
  let best = lo
  let bestValue = Number.NEGATIVE_INFINITY
  for (let i = 0; i <= grid; i++) {
    const t = lo + ((hi - lo) * i) / grid
    const v = f(t)
    if (v > bestValue) {
      bestValue = v
      best = t
    }
  }
  const step = (hi - lo) / grid
  const a = Math.max(lo, best - step)
  const b = Math.min(hi, best + step)
  const da = df(a)
  const db = df(b)
  if (Math.sign(da) === Math.sign(db) || da === 0 || db === 0) {
    if (da === 0) return a
    if (db === 0) return b
    return best
  }
  return brentq(df, a, b, { xtol: 1e-14, rtol: 1e-14, maxIterations: 200 }).root
}
