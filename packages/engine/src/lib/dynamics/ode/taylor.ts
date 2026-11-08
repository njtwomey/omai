/**
 * A Taylor-series ODE integrator: at each step the solution's Taylor coefficients up to a chosen order are computed by
 * Taylor-mode automatic differentiation, and the step size is chosen from the decay of the last coefficients.
 */

import { jet } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import type { TensorLike, Value } from '../../tensor/index.ts'
import {
  atBound,
  attempt,
  type Counter,
  checked,
  counter,
  initialState,
  snapshot,
  statusAfter,
  towards,
  validMaxStep,
  validTolerances,
} from './_step.ts'
import { allFinite, combine, floatState, numbersOf, rms, type Term, tolAt } from './_vec.ts'
import type { DenseSegment, OdeVectorField, Stepper, StepperOptions, StepperState } from './problem.ts'

/**
 * The Taylor coefficients y_0 = y, y_1, ..., y_order of the solution of the autonomous ODE y' = f(y) through y, so
 * that y(t + h) = sum_k y_k h^k. Each order comes from one Taylor-mode (jet) pass through f with the coefficients
 * found so far: (k + 1)! y_{k+1} = d^k/dt^k f(y(t)).
 *
 * @param f - The field (autonomous: its value may not depend on time).
 * @param y - The point.
 * @param order - The highest coefficient.
 * @returns `[y_0, ..., y_order]`, each with the state's shape.
 * @throws OmaiError with code `value` for a negative order.
 * @concept taylor-mode-autodiff
 * @example
 * ```ts
 * import { taylorSeriesCoefficients } from 'omai-engine/dynamics/ode'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * // y' = y from 1: the coefficients of e^t.
 * taylorSeriesCoefficients((y) => y, tensor([1]), 3).map((c) => toFlat(c)[0]) // => [1, 1, 0.5, 0.16666666666666666]
 * ```
 */
export function taylorSeriesCoefficients(f: (y: Value) => Value, y: TensorLike, order: number): Value[] {
  if (!Number.isInteger(order) || order < 0) throw new OmaiError('value', 'taylorSeriesCoefficients: order >= 0')
  const y0 = floatState(y)
  const coeffs: Value[] = [y0]
  let fact = 1
  for (let k = 0; k < order; k++) {
    let next: Value
    if (k === 0) next = f(y0)
    else {
      // Derivative series of y(t): d^j y / dt^j = j! y_j for j = 1..k.
      let jf = 1
      const series = coeffs.slice(1, k + 1).map((c, j) => {
        jf *= j + 1
        return combine(null, [[jf, c]])
      })
      const out = jet((v) => f(v), [y0], [series])
      next = out.series[k - 1] as Value
    }
    fact *= k + 1
    coeffs.push(combine(null, [[1 / fact, next]]))
  }
  return coeffs
}

/** Options of {@link taylor}. */
export interface TaylorOptions extends StepperOptions {
  /** The order of the series (default 12). */
  readonly order?: number
}

/** The state of {@link taylor}: the series of the last step. */
export interface TaylorState extends StepperState {
  readonly f: OdeVectorField
  readonly series: readonly Value[]
  readonly h: number
}

/**
 * A Taylor-series integrator: at each step the solution's Taylor coefficients to the given order are computed by
 * Taylor-mode automatic differentiation, the step is chosen from the last two coefficients (Jorba and Zou), and the
 * series itself is the dense output. Very high orders make it efficient at tight tolerances. The field is evaluated
 * with time frozen at the step's start, so it is exact for autonomous fields (append t to the state otherwise).
 *
 * @param options - Tolerances (default rtol 1e-10, atol 1e-12), order (default 12) and step limits.
 * @returns The stepper.
 * @throws OmaiError with code `value` for an order below 2.
 * @concept taylor-series-method
 * @reference Jorba and Zou (2005), A software package for the numerical integration of ODEs by means of high-order
 * Taylor methods.
 * @example
 * ```ts
 * import { solveIvp, taylor } from 'omai-engine/dynamics/ode'
 * import { neg, toFlat } from 'omai-engine/tensor'
 * const y = toFlat(solveIvp((_t, y) => neg(y), [0, 1], [1], { method: taylor(), tEval: [1] }).y)[0] ?? 0
 * Math.abs(y - Math.exp(-1)) < 1e-10 // => true
 * ```
 */
export function taylor(options: TaylorOptions = {}): Stepper<TaylorState> {
  const order = options.order ?? 12
  if (!Number.isInteger(order) || order < 2) throw new OmaiError('value', 'taylor: the order must be at least 2')
  const { rtol, atol } = validTolerances(options.rtol ?? 1e-10, options.atol ?? 1e-12)
  const maxStep = validMaxStep(options.maxStep)
  const stepFor = (series: readonly Value[], y: Value): number => {
    const yn = numbersOf(y)
    const scaled = (v: Value): number => {
      const x = numbersOf(v)
      return rms(Float64Array.from(x, (u, i) => u / (tolAt(atol, i) + tolAt(rtol, i) * Math.abs(yn[i] ?? 0))))
    }
    const a = scaled(series[order - 1] as Value)
    const b = scaled(series[order] as Value)
    const ha = a === 0 ? Number.POSITIVE_INFINITY : (1 / a) ** (1 / (order - 1))
    const hb = b === 0 ? Number.POSITIVE_INFINITY : (1 / b) ** (1 / order)
    return Math.min(maxStep, 0.9 * Math.min(ha, hb))
  }
  return {
    name: 'taylor',
    order,
    stiff: false,
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      return { ...initialState(t0, y0, tBound, 0, c, order), f, series: [], h: 0, counts: snapshot(c) }
    },
    step: (s) => {
      if (s.status !== 'running') throw new OmaiError('value', 'taylor: the integration has ended')
      if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
      const c: Counter = { ...s.counts }
      const series = taylorSeriesCoefficients((y) => s.f(s.t, y), s.y, order)
      c.nfev += order
      const proposal =
        options.firstStep !== undefined && s.series.length === 0 ? options.firstStep : stepFor(series, s.y)
      if (!(proposal > 0) || !Number.isFinite(proposal)) {
        const free = Number.isFinite(proposal) ? proposal : Math.abs(s.tBound - s.t)
        if (!(free > 0)) {
          return { ...s, status: 'failed', message: 'taylor: the series gives no usable step', counts: snapshot(c) }
        }
      }
      const hAbs = Number.isFinite(proposal) ? proposal : Math.abs(s.tBound - s.t)
      const { tNew, h } = towards(s.t, hAbs, s.direction, s.tBound)
      const terms: Term[] = series.slice(1).map((v, k) => [h ** (k + 1), v] as const)
      const yNew = combine(series[0] as Value, terms)
      const ok = allFinite(numbersOf(yNew))
      return {
        ...s,
        tOld: s.t,
        yOld: s.y,
        t: tNew,
        y: yNew,
        series,
        h,
        hAbs,
        attempts: [attempt(h, 1, ok)],
        status: ok ? statusAfter(tNew, s.tBound, s.direction) : 'failed',
        message: ok ? '' : `taylor: the solution became non-finite at t = ${tNew}`,
        counts: snapshot(c),
      }
    },
    dense: (s): DenseSegment => ({
      t0: s.tOld,
      t1: s.t,
      at: (t) => {
        const dt = t - s.tOld
        return combine(
          s.series[0] as Value,
          s.series.slice(1).map((v, k) => [dt ** (k + 1), v] as const),
        )
      },
    }),
  }
}
