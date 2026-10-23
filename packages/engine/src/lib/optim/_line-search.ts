// Shared machinery of the line searches (not exported from the barrel).
import { WhiteboxError } from '../base/index.ts'
import type { LineFunction, LinePoint, LineSearchResult, LineStart, LineTrial } from './line-search.ts'

/** Records the evaluations of a line function. */
/** A line function wrapped to count and record its evaluations. */
export interface Probe {
  readonly trials: LineTrial[]
  at(alpha: number): LinePoint
  valueAt(alpha: number): number
  point(alpha: number): LinePoint | undefined
  finish(alpha: number | undefined, value: number, slope: number, message: string): LineSearchResult
}

export function probe(phi: LineFunction): Probe {
  const trials: LineTrial[] = []
  let evaluations = 0
  let gradEvaluations = 0
  const last = new Map<number, LinePoint>()
  return {
    trials,
    at: (alpha: number): LinePoint => {
      const p = phi.at(alpha)
      evaluations += 1
      gradEvaluations += 1
      trials.push({ alpha, value: p.value, slope: p.slope })
      last.clear()
      last.set(alpha, p)
      return p
    },
    valueAt: (alpha: number): number => {
      if (phi.valueAt === undefined) {
        const p = phi.at(alpha)
        evaluations += 1
        gradEvaluations += 1
        trials.push({ alpha, value: p.value, slope: p.slope })
        last.clear()
        last.set(alpha, p)
        return p.value
      }
      const v = phi.valueAt(alpha)
      evaluations += 1
      trials.push({ alpha, value: v, slope: Number.NaN })
      return v
    },
    point: (alpha: number): LinePoint | undefined => last.get(alpha),
    finish: (alpha: number | undefined, value: number, slope: number, message: string): LineSearchResult => {
      const ok = alpha !== undefined && Number.isFinite(alpha)
      const p = ok ? last.get(alpha) : undefined
      return {
        alpha: ok ? alpha : Number.NaN,
        value,
        slope,
        ...(p?.grad === undefined ? {} : { grad: p.grad }),
        success: ok,
        evaluations,
        gradEvaluations,
        trials,
        message,
      }
    },
  }
}

/** scipy's first trial step: min(1, 2.02 (f0 - f_prev) / phi'(0)), or 1. */
export function firstStep(start: LineStart): number {
  if (start.alpha !== undefined) return start.alpha
  if (start.previousValue !== undefined && start.slope !== 0) {
    const a = Math.min(1, (1.01 * 2 * (start.value - start.previousValue)) / start.slope)
    return a < 0 ? 1 : a
  }
  return 1
}

export function checkConstants(c1: number, c2: number): void {
  if (!(0 < c1 && c1 < c2 && c2 < 1))
    throw new WhiteboxError('value', `line search needs 0 < c1 < c2 < 1, got ${c1}, ${c2}`)
}
