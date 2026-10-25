/**
 * Fixed-step explicit Runge-Kutta methods described by their Butcher tableaux: a checked tableau builder, the classical
 * tableaux (Euler, Heun, midpoint, Ralston, Kutta's third order, RK4 and the 3/8 rule), the stage and step functions
 * shared by the other solvers, and a ready stepper for each method.
 */

import { WhiteboxError } from '../../base/index.ts'
import type { Value } from '../../tensor/index.ts'
import { attempt, checked, counter, initialState, snapshot, statusAfter, tally, towards } from './_step.ts'
import { allFinite, combine, numbersOf, type Term } from './_vec.ts'
import { hermiteSegment, type Stepper, type StepperState, type VectorField } from './problem.ts'

/**
 * An explicit Runge-Kutta method as its Butcher tableau: stage times `c`, the strictly lower-triangular coupling `a`
 * (row i holds a_i1 .. a_i,i-1) and weights `b`, with its classical order.
 */
export interface ButcherTableau {
  readonly name: string
  readonly a: readonly (readonly number[])[]
  readonly b: readonly number[]
  readonly c: readonly number[]
  readonly order: number
}

/**
 * Build and check a Butcher tableau for an explicit method: `a` strictly lower triangular, one weight and time per
 * stage, and the row-sum condition c_i = sum_j a_ij.
 *
 * @param name - A name for traces.
 * @param a - The coupling rows (row i has i entries; longer rows must be zero beyond the diagonal).
 * @param b - The weights.
 * @param c - The stage times.
 * @param order - The classical order.
 * @returns The tableau.
 * @throws WhiteboxError with code `value` when the tableau is not explicit or inconsistent.
 * @concept runge-kutta
 * @example
 * ```ts
 * import { butcherTableau } from 'whitebox-engine/dynamics/ode'
 * butcherTableau('midpoint', [[], [0.5]], [0, 1], [0, 0.5], 2).b // => [0, 1]
 * ```
 */
export function butcherTableau(
  name: string,
  a: readonly (readonly number[])[],
  b: readonly number[],
  c: readonly number[],
  order: number,
): ButcherTableau {
  const s = b.length
  if (c.length !== s || a.length !== s) {
    throw new WhiteboxError('value', `butcherTableau: ${name} needs ${s} stage times and coupling rows`)
  }
  const rows = a.map((row, i) => {
    if (row.slice(i).some((v) => v !== 0)) {
      throw new WhiteboxError('value', `butcherTableau: ${name} is not explicit (row ${i} reaches the diagonal)`)
    }
    return Array.from({ length: i }, (_, j) => row[j] ?? 0)
  })
  rows.forEach((row, i) => {
    const sum = row.reduce((x, y) => x + y, 0)
    if (Math.abs(sum - (c[i] ?? 0)) > 1e-12 * Math.max(1, Math.abs(sum))) {
      throw new WhiteboxError('value', `butcherTableau: ${name} row ${i} sums to ${sum}, not c = ${c[i]}`)
    }
  })
  return { name, a: rows, b: [...b], c: [...c], order }
}

/** Forward Euler (order 1). */
export const EULER: ButcherTableau = butcherTableau('euler', [[]], [1], [0], 1)
/** Heun's method, the explicit trapezoid (order 2). */
export const HEUN: ButcherTableau = butcherTableau('heun', [[], [1]], [0.5, 0.5], [0, 1], 2)
/** The explicit midpoint method (order 2). */
export const MIDPOINT: ButcherTableau = butcherTableau('midpoint', [[], [0.5]], [0, 1], [0, 0.5], 2)
/** Ralston's second-order method (minimum error bound). */
export const RALSTON: ButcherTableau = butcherTableau('ralston', [[], [2 / 3]], [0.25, 0.75], [0, 2 / 3], 2)
/** Kutta's third-order method. */
export const KUTTA3: ButcherTableau = butcherTableau(
  'kutta3',
  [[], [0.5], [-1, 2]],
  [1 / 6, 2 / 3, 1 / 6],
  [0, 0.5, 1],
  3,
)
/** The classical fourth-order Runge-Kutta method. */
export const RK4: ButcherTableau = butcherTableau(
  'rk4',
  [[], [0.5], [0, 0.5], [0, 0, 1]],
  [1 / 6, 1 / 3, 1 / 3, 1 / 6],
  [0, 0.5, 0.5, 1],
  4,
)
/** Kutta's 3/8 rule (order 4). */
export const RK38: ButcherTableau = butcherTableau(
  'rk38',
  [[], [1 / 3], [-1 / 3, 1], [1, -1, 1]],
  [1 / 8, 3 / 8, 3 / 8, 1 / 8],
  [0, 1 / 3, 2 / 3, 1],
  4,
)

/**
 * The stages of one explicit Runge-Kutta step: k_i = f(t + c_i h, y + h sum_j a_ij k_j), with k_1 = f0 when given
 * (first-same-as-last reuse).
 *
 * @param f - The vector field.
 * @param t - Start time.
 * @param y - Start state.
 * @param h - Signed step.
 * @param tableau - The method.
 * @param f0 - f(t, y), if already known.
 * @returns The stage derivatives.
 * @example
 * ```ts
 * import { rkStages, EULER } from 'whitebox-engine/dynamics/ode'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * rkStages((_t, y) => y, 0, tensor([2]), 0.1, EULER).map((k) => toArray(k)) // => [[2]]
 * ```
 */
export function rkStages(
  f: VectorField,
  t: number,
  y: Value,
  h: number,
  tableau: Pick<ButcherTableau, 'a' | 'c'>,
  f0?: Value,
): Value[] {
  const ks: Value[] = []
  for (let i = 0; i < tableau.c.length; i++) {
    if (i === 0 && f0 !== undefined) {
      ks.push(f0)
      continue
    }
    const row = tableau.a[i] ?? []
    const terms: Term[] = row.map((aij, j) => [h * aij, ks[j] as Value] as const)
    ks.push(f(t + (tableau.c[i] ?? 0) * h, i === 0 ? y : combine(y, terms)))
  }
  return ks
}

/**
 * One step of an explicit Runge-Kutta method: y + h sum_i b_i k_i.
 *
 * @param f - The vector field.
 * @param t - Start time.
 * @param y - Start state.
 * @param h - Signed step.
 * @param tableau - The method.
 * @param f0 - f(t, y), if already known.
 * @returns The new state.
 * @concept runge-kutta
 * @example
 * ```ts
 * import { rkStep, RK4 } from 'whitebox-engine/dynamics/ode'
 * import { item, tensor } from 'whitebox-engine/tensor'
 * // dy/dt = 1 integrates exactly.
 * Math.abs(item(rkStep((_t, y) => tensor([1]), 0, tensor([0]), 0.5, RK4)) - 0.5) < 1e-15 // => true
 * ```
 */
export function rkStep(f: VectorField, t: number, y: Value, h: number, tableau: ButcherTableau, f0?: Value): Value {
  const ks = rkStages(f, t, y, h, tableau, f0)
  return combine(
    y,
    tableau.b.map((bi, i) => [h * bi, ks[i] as Value] as const),
  )
}

/** Options of the fixed-step steppers. */
export interface FixedStepOptions {
  /** The step size (positive; the last step is shortened to land on the bound). */
  readonly dt: number
}

/** The state of a fixed-step stepper: the derivative at the current point is kept for the next step and for dense output. */
export interface FixedStepState extends StepperState {
  readonly f: VectorField
  readonly fy: Value
  readonly fOld: Value
  readonly dt: number
}

/**
 * A fixed-step stepper for any explicit Runge-Kutta tableau. Steps are `dt` (the last one shortened to the bound);
 * dense output is the cubic Hermite interpolant of each step. A non-finite state fails the integration.
 *
 * @param tableau - The method.
 * @param options - The step size.
 * @returns The stepper.
 * @throws WhiteboxError with code `value` when `dt` is not positive.
 * @concept runge-kutta
 * @example
 * ```ts
 * import { explicitRk, RK4 } from 'whitebox-engine/dynamics/ode'
 * import { tensor } from 'whitebox-engine/tensor'
 * const s = explicitRk(RK4, { dt: 0.1 })
 * s.step(s.init((_t, y) => y, 0, tensor([1]), 1)).t // => 0.1
 * ```
 */
export function explicitRk(tableau: ButcherTableau, options: FixedStepOptions): Stepper<FixedStepState> {
  const dt = options.dt
  if (!(dt > 0) || !Number.isFinite(dt)) throw new WhiteboxError('value', `${tableau.name}: dt must be positive`)
  return {
    name: tableau.name,
    order: tableau.order,
    stiff: false,
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      const fy = tally(f, c)(t0, y0)
      return { ...initialState(t0, y0, tBound, dt, c, tableau.order), f, fy, fOld: fy, dt, counts: snapshot(c) }
    },
    step: (s) => {
      if (s.status !== 'running') throw new WhiteboxError('value', `${tableau.name}: the integration has ended`)
      if (s.t === s.tBound) return { ...s, tOld: s.t, status: 'finished' }
      const c = { ...s.counts }
      const f = tally(s.f, c)
      const { tNew, h } = towards(s.t, dt, s.direction, s.tBound)
      const yNew = rkStep(f, s.t, s.y, h, tableau, s.fy)
      const ok = allFinite(numbersOf(yNew))
      const fNew = ok ? f(tNew, yNew) : yNew
      return {
        ...s,
        tOld: s.t,
        yOld: s.y,
        fOld: s.fy,
        t: tNew,
        y: yNew,
        fy: fNew,
        attempts: [attempt(h, 0, ok)],
        status: ok ? statusAfter(tNew, s.tBound, s.direction) : 'failed',
        message: ok ? '' : `${tableau.name}: the solution became non-finite at t = ${tNew}`,
        counts: { ...c },
      }
    },
    dense: (s) => hermiteSegment(s.tOld, s.yOld, s.fOld, s.t, s.y, s.fy),
  }
}

/**
 * The forward Euler stepper.
 *
 * @param options - The step size.
 * @returns The stepper.
 * @example
 * ```ts
 * import { euler } from 'whitebox-engine/dynamics/ode'
 * euler({ dt: 0.1 }).order // => 1
 * ```
 */
export function euler(options: FixedStepOptions): Stepper<FixedStepState> {
  return explicitRk(EULER, options)
}

/**
 * Heun's (explicit trapezoid) stepper.
 *
 * @param options - The step size.
 * @returns The stepper.
 * @example
 * ```ts
 * import { heun } from 'whitebox-engine/dynamics/ode'
 * heun({ dt: 0.1 }).order // => 2
 * ```
 */
export function heun(options: FixedStepOptions): Stepper<FixedStepState> {
  return explicitRk(HEUN, options)
}

/**
 * The explicit midpoint stepper.
 *
 * @param options - The step size.
 * @returns The stepper.
 * @example
 * ```ts
 * import { midpoint } from 'whitebox-engine/dynamics/ode'
 * midpoint({ dt: 0.1 }).name // => 'midpoint'
 * ```
 */
export function midpoint(options: FixedStepOptions): Stepper<FixedStepState> {
  return explicitRk(MIDPOINT, options)
}

/**
 * Ralston's second-order stepper.
 *
 * @param options - The step size.
 * @returns The stepper.
 * @example
 * ```ts
 * import { ralston } from 'whitebox-engine/dynamics/ode'
 * ralston({ dt: 0.1 }).name // => 'ralston'
 * ```
 */
export function ralston(options: FixedStepOptions): Stepper<FixedStepState> {
  return explicitRk(RALSTON, options)
}

/**
 * Kutta's third-order stepper.
 *
 * @param options - The step size.
 * @returns The stepper.
 * @example
 * ```ts
 * import { kutta3 } from 'whitebox-engine/dynamics/ode'
 * kutta3({ dt: 0.1 }).order // => 3
 * ```
 */
export function kutta3(options: FixedStepOptions): Stepper<FixedStepState> {
  return explicitRk(KUTTA3, options)
}

/**
 * The classical fourth-order Runge-Kutta stepper.
 *
 * @param options - The step size.
 * @returns The stepper.
 * @example
 * ```ts
 * import { rk4 } from 'whitebox-engine/dynamics/ode'
 * rk4({ dt: 0.1 }).order // => 4
 * ```
 */
export function rk4(options: FixedStepOptions): Stepper<FixedStepState> {
  return explicitRk(RK4, options)
}

/**
 * The 3/8-rule fourth-order stepper.
 *
 * @param options - The step size.
 * @returns The stepper.
 * @example
 * ```ts
 * import { rk38 } from 'whitebox-engine/dynamics/ode'
 * rk38({ dt: 0.1 }).name // => 'rk38'
 * ```
 */
export function rk38(options: FixedStepOptions): Stepper<FixedStepState> {
  return explicitRk(RK38, options)
}
