// Machinery shared by the steppers: evaluation counting, scipy's initial-step heuristic and stepper bookkeeping.
import { WhiteboxError } from '../../base/index.ts'
import type { Value } from '../../tensor/index.ts'
import type { SolveCounts, StepAttempt, StepperState, StepperStatus, VectorField } from './problem.ts'
import { combine, numbersOf, rms, type Tol, tolAt } from './_vec.ts'

/** Mutable evaluation counts private to one integration. */
export interface Counter {
  nfev: number
  njev: number
  nlu: number
}

/** A fresh counter. */
export function counter(): Counter {
  return { nfev: 0, njev: 0, nlu: 0 }
}

/** A snapshot of a counter. */
export function snapshot(c: Counter): SolveCounts {
  return { nfev: c.nfev, njev: c.njev, nlu: c.nlu }
}

/** The field with every evaluation counted (and its output checked for shape). */
export function counted(f: VectorField, c: Counter, shape: readonly number[]): VectorField {
  return (t, y) => {
    c.nfev += 1
    const out = f(t, y)
    if (out.shape.length !== shape.length || out.shape.some((n, i) => n !== shape[i])) {
      throw new WhiteboxError('shape', `the vector field returned shape [${out.shape}] for a state of shape [${shape}]`)
    }
    return out
  }
}

/** The integration direction scipy uses: the sign of t_bound - t0, and +1 for an empty interval. */
export function directionOf(t0: number, tBound: number): 1 | -1 {
  return tBound >= t0 ? 1 : -1
}

/**
 * scipy's `select_initial_step` (Hairer, Norsett and Wanner, Solving ODEs I, II.4): a first step from the sizes of y,
 * f and a finite-difference estimate of the second derivative.
 */
export function selectInitialStep(
  f: VectorField,
  t0: number,
  y0: Value,
  tBound: number,
  maxStep: number,
  f0: Value,
  direction: number,
  order: number,
  rtol: Tol,
  atol: Tol,
): number {
  const y = numbersOf(y0)
  if (y.length === 0) return Number.POSITIVE_INFINITY
  const interval = Math.abs(tBound - t0)
  if (interval === 0) return 0
  const fy = numbersOf(f0)
  const scale = Float64Array.from(y, (v, i) => tolAt(atol, i) + Math.abs(v) * tolAt(rtol, i))
  const d0 = rms(Float64Array.from(y, (v, i) => v / (scale[i] ?? 1)))
  const d1 = rms(Float64Array.from(fy, (v, i) => v / (scale[i] ?? 1)))
  let h0 = d0 < 1e-5 || d1 < 1e-5 ? 1e-6 : (0.01 * d0) / d1
  h0 = Math.min(h0, interval)
  const y1 = combine(y0, [[h0 * direction, f0]])
  const f1 = numbersOf(f(t0 + h0 * direction, y1))
  const d2 = rms(Float64Array.from(f1, (v, i) => (v - (fy[i] ?? 0)) / (scale[i] ?? 1))) / h0
  const h1 = d1 <= 1e-15 && d2 <= 1e-15 ? Math.max(1e-6, h0 * 1e-3) : (0.01 / Math.max(d1, d2)) ** (1 / (order + 1))
  return Math.min(100 * h0, h1, interval, maxStep)
}

/** The fields every stepper state starts with. */
export function initialState(
  t0: number,
  y0: Value,
  tBound: number,
  hAbs: number,
  c: Counter,
  order: number,
): StepperState {
  const direction = directionOf(t0, tBound)
  return {
    t: t0,
    y: y0,
    tOld: t0,
    yOld: y0,
    tBound,
    direction,
    hAbs,
    status: y0.shape.reduce((a, b) => a * b, 1) === 0 || t0 === tBound ? 'running' : 'running',
    message: '',
    attempts: [],
    counts: snapshot(c),
    order,
  }
}

/** The status after a successful step to t (scipy: finished once t reaches the bound). */
export function statusAfter(t: number, tBound: number, direction: number): StepperStatus {
  return direction * (t - tBound) >= 0 ? 'finished' : 'running'
}

/** Whether the stepper is at its bound already (an empty interval): the step just finishes. */
export function atBound(s: StepperState): boolean {
  return s.t === s.tBound || numbersOf(s.y).length === 0
}

/** Clip a proposed step magnitude to [minStep, maxStep] the way scipy does before attempting it. */
export function clipStep(hAbs: number, minStep: number, maxStep: number): number {
  if (hAbs > maxStep) return maxStep
  if (hAbs < minStep) return minStep
  return hAbs
}

/** Step towards the bound without overshooting: the actual t_new and signed h. */
export function towards(t: number, hAbs: number, direction: number, tBound: number): { tNew: number; h: number } {
  let tNew = t + hAbs * direction
  if (direction * (tNew - tBound) > 0) tNew = tBound
  return { tNew, h: tNew - t }
}

/** Record an attempt. */
export function attempt(h: number, error: number, accepted: boolean): StepAttempt {
  return { h, error, accepted }
}

/** Validate tolerances as scipy does (rtol floored at 100 eps, atol non-negative). */
export function validTolerances(rtol: Tol, atol: Tol): { rtol: Tol; atol: Tol } {
  const floor = 100 * Number.EPSILON
  const r = typeof rtol === 'number' ? Math.max(rtol, floor) : rtol.map((v) => Math.max(v, floor))
  const bad = typeof atol === 'number' ? atol < 0 : atol.some((v) => v < 0)
  if (bad) throw new WhiteboxError('value', 'the absolute tolerance must be non-negative')
  return { rtol: r, atol }
}

/** Check a positive maximum step. */
export function validMaxStep(maxStep: number | undefined): number {
  const m = maxStep ?? Number.POSITIVE_INFINITY
  if (!(m > 0)) throw new WhiteboxError('value', 'maxStep must be positive')
  return m
}

/** Check a first step against the interval. */
export function validFirstStep(first: number, t0: number, tBound: number): number {
  if (!(first > 0)) throw new WhiteboxError('value', 'firstStep must be positive')
  if (first > Math.abs(tBound - t0)) throw new WhiteboxError('value', 'firstStep exceeds the interval')
  return first
}
