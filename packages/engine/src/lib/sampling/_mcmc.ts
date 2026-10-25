/** Shared pieces of the Markov kernels: accept/reject with a key, starting states, and Welford accumulators. */
import { MdmlError } from '../base/index.ts'
import { type Key, uniform } from '../random/index.ts'
import type { MCMCState } from './chains.ts'
import type { LogDensity } from './target.ts'

/** Accept with probability `alpha` using the key's uniform draw (alpha <= 0 or NaN never accepts). */
export function acceptWith(key: Key, alpha: number): boolean {
  if (!(alpha > 0)) return false
  if (alpha >= 1) return true
  return (uniform(key, [1]).data[0] ?? 1) < alpha
}

/** Step sizes must be positive (infinite ones are allowed: they diverge and adaptation shrinks them). */
export function checkStepSize(stepSize: number, name: string): void {
  if (!(stepSize > 0)) throw new MdmlError('value', `${name}: the step size must be positive`)
}

/** One uniform draw in [0, 1). */
export function uniformOne(key: Key): number {
  return uniform(key, [1]).data[0] ?? 0
}

/** A starting state of a gradient-free kernel. */
export function initState(target: LogDensity, position: ArrayLike<number>, name: string): MCMCState {
  if (position.length !== target.dim) {
    throw new MdmlError('shape', `${name}: the position has ${position.length} values, need ${target.dim}`)
  }
  const p = Float64Array.from(position)
  return { position: p, logDensity: target.logDensity(p) }
}

/** Running mean and sum of outer products of deviations (dense, row-major), one point at a time. */
export interface Welford {
  readonly count: number
  readonly mean: Float64Array
  readonly m2: Float64Array
}

/** An empty accumulator; `diagonal` keeps only the variances (m2 of length dim). */
export function welfordInit(dim: number, diagonal = false): Welford {
  return { count: 0, mean: new Float64Array(dim), m2: new Float64Array(diagonal ? dim : dim * dim) }
}

/** Add one point (Welford's update, dense or diagonal by the length of m2). */
export function welfordUpdate(w: Welford, x: ArrayLike<number>): Welford {
  const dim = w.mean.length
  const n = w.count + 1
  const delta = new Float64Array(dim)
  const mean = new Float64Array(dim)
  for (let i = 0; i < dim; i++) {
    delta[i] = (x[i] ?? 0) - (w.mean[i] ?? 0)
    mean[i] = (w.mean[i] ?? 0) + (delta[i] ?? 0) / n
  }
  const m2 = Float64Array.from(w.m2)
  if (m2.length === dim) {
    for (let i = 0; i < dim; i++) m2[i] = (m2[i] ?? 0) + ((x[i] ?? 0) - (mean[i] ?? 0)) * (delta[i] ?? 0)
  } else {
    for (let i = 0; i < dim; i++) {
      const after = (x[i] ?? 0) - (mean[i] ?? 0)
      for (let j = 0; j < dim; j++) m2[i * dim + j] = (m2[i * dim + j] ?? 0) + after * (delta[j] ?? 0)
    }
  }
  return { count: n, mean, m2 }
}

/** The sample covariance (ddof 1) of an accumulator, dense or diagonal like its m2. */
export function welfordCovariance(w: Welford): Float64Array {
  const d = Math.max(1, w.count - 1)
  return w.m2.map((v) => v / d)
}
