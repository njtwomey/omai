/**
 * Adaptive explicit Runge-Kutta solvers: embedded pairs whose two solutions of different order give a local error
 * estimate that sets the next step size. The tableaux (Heun-Euler, Bogacki-Shampine, Fehlberg, Cash-Karp,
 * Dormand-Prince, Tsitouras, DOP853), integral and PI step controllers, and a generic stepper that follows scipy's
 * `RungeKutta`, with `rk23`, `rk45` and `dop853` matching scipy's methods.
 */

import { WhiteboxError } from '../../base/index.ts'
import type { Value } from '../../tensor/index.ts'
import * as dop from './_dop853.ts'
import {
  atBound,
  attempt,
  type Counter,
  checked,
  clipStep,
  counter,
  initialState,
  selectInitialStep,
  snapshot,
  statusAfter,
  tally,
  towards,
  validFirstStep,
  validMaxStep,
  validTolerances,
} from './_step.ts'
import { combine, errorScale, minStep, numbersOf, rms, type Term } from './_vec.ts'
import { type ButcherTableau, butcherTableau, rkStages } from './explicit.ts'
import {
  type DenseSegment,
  hermiteSegment,
  type StepAttempt,
  type Stepper,
  type StepperOptions,
  type StepperState,
  type VectorField,
} from './problem.ts'

/**
 * An embedded Runge-Kutta pair in scipy's form: the propagated method's tableau, error weights `e` over the stages and
 * the derivative at the new point (so `e` has one more entry than `b`; the error estimate is h sum e_i k_i), the order
 * of the error estimate, and an optional dense-output matrix `p` (`[stages + 1, q]`, scipy's `P`).
 */
export interface EmbeddedTableau extends ButcherTableau {
  readonly e: readonly number[]
  readonly errorOrder: number
  readonly p?: readonly (readonly number[])[]
}

function embedded(
  base: ButcherTableau,
  e: readonly number[],
  errorOrder: number,
  p?: readonly (readonly number[])[],
): EmbeddedTableau {
  if (e.length !== base.b.length + 1) {
    throw new WhiteboxError('value', `${base.name}: the error weights need one entry per stage plus one`)
  }
  return { ...base, e: [...e], errorOrder, ...(p === undefined ? {} : { p }) }
}

/**
 * An embedded pair from its two weight vectors: `b` is propagated, `bHat` gives the error estimate b - bHat.
 *
 * @param tableau - The propagated method.
 * @param bHat - The embedded weights (one per stage; a further entry weights the derivative at the new point).
 * @param errorOrder - The order of the embedded estimate (the error is O(h^(errorOrder + 1))).
 * @returns The pair.
 * @throws WhiteboxError with code `value` when the lengths do not match.
 * @example
 * ```ts
 * import { embeddedPair, HEUN } from 'whitebox-engine/dynamics/ode'
 * embeddedPair(HEUN, [1, 0], 1).e // => [-0.5, 0.5, 0]
 * ```
 */
export function embeddedPair(tableau: ButcherTableau, bHat: readonly number[], errorOrder: number): EmbeddedTableau {
  const s = tableau.b.length
  if (bHat.length !== s && bHat.length !== s + 1) {
    throw new WhiteboxError('value', `embeddedPair: ${tableau.name} needs ${s} embedded weights`)
  }
  const e = Array.from({ length: s + 1 }, (_, i) => (tableau.b[i] ?? 0) - (bHat[i] ?? 0))
  return embedded(tableau, e, errorOrder)
}

/** Heun-Euler (2/1): Heun's method with Euler as the error estimate. */
export const HEUN_EULER: EmbeddedTableau = embedded(
  butcherTableau('heun-euler', [[], [1]], [0.5, 0.5], [0, 1], 2),
  [-0.5, 0.5, 0],
  1,
)

/** Bogacki-Shampine (3/2), scipy's RK23, with its cubic dense output. */
export const BOGACKI_SHAMPINE: EmbeddedTableau = embedded(
  butcherTableau('rk23', [[], [1 / 2], [0, 3 / 4]], [2 / 9, 1 / 3, 4 / 9], [0, 1 / 2, 3 / 4], 3),
  [5 / 72, -1 / 12, -1 / 9, 1 / 8],
  2,
  [
    [1, -4 / 3, 5 / 9],
    [0, 1, -2 / 3],
    [0, 4 / 3, -8 / 9],
    [0, -1, 1],
  ],
)

const FEHLBERG_A = [
  [],
  [1 / 4],
  [3 / 32, 9 / 32],
  [1932 / 2197, -7200 / 2197, 7296 / 2197],
  [439 / 216, -8, 3680 / 513, -845 / 4104],
  [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
]
const FEHLBERG_B4 = [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0]
const FEHLBERG_B5 = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]

/** Runge-Kutta-Fehlberg (4/5): the fourth-order solution is propagated and the fifth-order one estimates its error. */
export const FEHLBERG: EmbeddedTableau = embeddedPair(
  butcherTableau('fehlberg45', FEHLBERG_A, FEHLBERG_B4, [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2], 4),
  FEHLBERG_B5,
  4,
)

/** Cash-Karp (5/4): the fifth-order solution is propagated (local extrapolation). */
export const CASH_KARP: EmbeddedTableau = embeddedPair(
  butcherTableau(
    'cash-karp',
    [
      [],
      [1 / 5],
      [3 / 40, 9 / 40],
      [3 / 10, -9 / 10, 6 / 5],
      [-11 / 54, 5 / 2, -70 / 27, 35 / 27],
      [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
    ],
    [37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771],
    [0, 1 / 5, 3 / 10, 3 / 5, 1, 7 / 8],
    5,
  ),
  [2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4],
  4,
)

/** Dormand-Prince (5/4), scipy's RK45, with its quartic dense output (Shampine's optimal c6). */
export const DORMAND_PRINCE: EmbeddedTableau = embedded(
  butcherTableau(
    'rk45',
    [
      [],
      [1 / 5],
      [3 / 40, 9 / 40],
      [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    ],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1],
    5,
  ),
  [-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40],
  4,
  [
    [1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
    [0, 0, 0, 0],
    [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
    [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
    [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
    [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
    [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
  ],
)

/** Tsitouras (5/4) (Tsitouras 2011, the pair behind Tsit5), first same as last; dense output is cubic Hermite. */
export const TSITOURAS: EmbeddedTableau = embedded(
  butcherTableau(
    'tsit5',
    [
      [],
      [0.161],
      [-0.008480655492356989, 0.335480655492357],
      [2.897153057105493, -6.359448489975075, 4.3622954328695815],
      [5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525],
      [5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383],
    ],
    [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774],
    [0, 0.161, 0.327, 0.9, 0.9800255409045097, 1],
    5,
  ),
  [
    -0.001780011052225777,
    -0.0008164344596567469,
    0.007880878010261995,
    -0.1447110071732629,
    0.5823571654525552,
    -0.45808210592918697,
    1 / 66,
  ],
  4,
)

/** The DOP853 coefficients (Hairer, Norsett and Wanner) as used by scipy: 12 stages, order 8, 5- and 3-order estimates. */
export const DOP853_TABLEAU: EmbeddedTableau = embedded(
  butcherTableau('dop853', dop.A.slice(0, dop.N_STAGES), dop.B, dop.C.slice(0, dop.N_STAGES), 8),
  dop.E5,
  7,
)

/**
 * How the next step size follows from the error: `integral` is scipy's controller (factor safety err^(-1/(q+1)));
 * `pi` adds the previous error (Gustafsson's PI controller, factor safety err^(-beta1) prev^(beta2), defaults
 * beta1 = 0.7/(q+1), beta2 = 0.4/(q+1)), which damps step-size oscillation; an object gives the three exponents of a
 * PID controller, factor safety err_n^(-beta1) err_{n-1}^(beta2) err_{n-2}^(-beta3) (diffrax's PIDController with
 * beta1 = (p + i + d)/k, beta2 = (p + 2d)/k, beta3 = d/k).
 */
export type StepController =
  | 'integral'
  | 'pi'
  | { readonly beta1: number; readonly beta2: number; readonly beta3?: number }

/** Options of the adaptive Runge-Kutta steppers. */
export interface AdaptiveOptions extends StepperOptions {
  readonly controller?: StepController
  /** Safety factor (default 0.9). */
  readonly safety?: number
  /** Smallest step reduction on rejection (default 0.2). */
  readonly minFactor?: number
  /** Largest step growth (default 10). */
  readonly maxFactor?: number
  /** Attempts allowed per step before failing (default 50). */
  readonly maxAttempts?: number
}

/** The state of an adaptive Runge-Kutta stepper: the stages of the last step and the controller's memory. */
export interface AdaptiveState extends StepperState {
  readonly f: VectorField
  readonly fy: Value
  readonly k: readonly Value[]
  /** The signed size of the last accepted step. */
  readonly h: number
  /** The error norm of the last accepted step (for the PI controller). */
  readonly errPrev: number
  /** The error norm of the step before (for the PID controller). */
  readonly errPrev2: number
}

function controllerBetas(controller: StepController, q: number): { beta1: number; beta2: number; beta3: number } {
  if (controller === 'integral') return { beta1: 1 / (q + 1), beta2: 0, beta3: 0 }
  if (controller === 'pi') return { beta1: 0.7 / (q + 1), beta2: 0.4 / (q + 1), beta3: 0 }
  return { beta1: controller.beta1, beta2: controller.beta2, beta3: controller.beta3 ?? 0 }
}

function dop853ErrorNorm(k: readonly Value[], h: number, scale: Float64Array): number {
  const ks = k.map((v) => numbersOf(v))
  const n = scale.length
  let e5 = 0
  let e3 = 0
  for (let i = 0; i < n; i++) {
    let a = 0
    let b = 0
    for (let j = 0; j < ks.length; j++) {
      a += (dop.E5[j] ?? 0) * (ks[j]?.[i] ?? 0)
      b += (dop.E3[j] ?? 0) * (ks[j]?.[i] ?? 0)
    }
    e5 += (a / (scale[i] ?? 1)) ** 2
    e3 += (b / (scale[i] ?? 1)) ** 2
  }
  if (e5 === 0 && e3 === 0) return 0
  return (Math.abs(h) * e5) / Math.sqrt((e5 + 0.01 * e3) * n)
}

function weightedNorm(e: readonly number[], k: readonly Value[], h: number, scale: Float64Array): number {
  const ks = k.map((v) => numbersOf(v))
  const err = new Float64Array(scale.length)
  for (let i = 0; i < err.length; i++) {
    let a = 0
    for (let j = 0; j < ks.length; j++) a += (e[j] ?? 0) * (ks[j]?.[i] ?? 0)
    err[i] = (h * a) / (scale[i] ?? 1)
  }
  return rms(err)
}

/**
 * An adaptive stepper for an embedded Runge-Kutta pair, following scipy's `RungeKutta` exactly with the integral
 * controller (initial step by `select_initial_step`, RMS error norm against atol + rtol max(|y|, |y_new|), factors
 * clipped to [0.2, 10], no growth right after a rejection). Every attempt is kept in the state, so rejected steps
 * appear in traces. A step fails when the step size underflows or after `maxAttempts` rejections.
 *
 * @param tableau - The pair.
 * @param options - Tolerances (default rtol 1e-3, atol 1e-6), step limits and controller.
 * @returns The stepper.
 * @concept adaptive-step-size
 * @example
 * ```ts
 * import { adaptiveRk, DORMAND_PRINCE } from 'whitebox-engine/dynamics/ode'
 * import { tensor } from 'whitebox-engine/tensor'
 * const s = adaptiveRk(DORMAND_PRINCE)
 * s.step(s.init((_t, y) => y, 0, tensor([1]), 1)).attempts.length // => 1
 * ```
 */
export function adaptiveRk(tableau: EmbeddedTableau, options: AdaptiveOptions = {}): Stepper<AdaptiveState> {
  const { rtol, atol } = validTolerances(options.rtol ?? 1e-3, options.atol ?? 1e-6)
  const maxStep = validMaxStep(options.maxStep)
  const safety = options.safety ?? 0.9
  const minFactor = options.minFactor ?? 0.2
  const maxFactor = options.maxFactor ?? 10
  const maxAttempts = options.maxAttempts ?? 50
  const { beta1, beta2, beta3 } = controllerBetas(options.controller ?? 'integral', tableau.errorOrder)
  const isDop = tableau === DOP853_TABLEAU
  const norm = (k: readonly Value[], h: number, scale: Float64Array): number =>
    isDop ? dop853ErrorNorm(k, h, scale) : weightedNorm(tableau.e, k, h, scale)
  return {
    name: tableau.name,
    order: tableau.order,
    stiff: false,
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      const fy = tally(f, c)(t0, y0)
      const base = initialState(t0, y0, tBound, 0, c, tableau.order)
      const hAbs =
        options.firstStep === undefined
          ? selectInitialStep(tally(f, c), t0, y0, tBound, maxStep, fy, base.direction, tableau.errorOrder, rtol, atol)
          : validFirstStep(options.firstStep, t0, tBound)
      return { ...base, hAbs, f, fy, k: [], h: 0, errPrev: 1, errPrev2: 1, counts: snapshot(c) }
    },
    step: (s) => {
      if (s.status !== 'running') throw new WhiteboxError('value', `${tableau.name}: the integration has ended`)
      if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
      const c: Counter = { ...s.counts }
      const f = tally(s.f, c)
      const hmin = minStep(s.t, s.direction)
      let hAbs = clipStep(s.hAbs, hmin, maxStep)
      const attempts: StepAttempt[] = []
      let rejected = false
      for (let tries = 0; tries < maxAttempts; tries++) {
        if (hAbs < hmin) {
          return failed(s, c, attempts, `${tableau.name}: required step size is less than spacing between numbers`)
        }
        const { tNew, h } = towards(s.t, hAbs, s.direction, s.tBound)
        hAbs = Math.abs(h)
        const ks = rkStages(f, s.t, s.y, h, tableau, s.fy)
        const yNew = combine(
          s.y,
          tableau.b.map((bi, i) => [h * bi, ks[i] as Value] as const),
        )
        const fNew = f(tNew, yNew)
        ks.push(fNew)
        const scale = errorScale(numbersOf(s.y), numbersOf(yNew), rtol, atol)
        const err = norm(ks, h, scale)
        if (err < 1) {
          let factor: number
          if (err === 0) factor = maxFactor
          else {
            const memory = (beta2 === 0 ? 1 : s.errPrev ** beta2) * (beta3 === 0 ? 1 : s.errPrev2 ** -beta3)
            factor = Math.min(maxFactor, safety * err ** -beta1 * memory)
          }
          if (rejected) factor = Math.min(1, factor)
          if (beta2 !== 0 || beta3 !== 0) factor = Math.max(minFactor, factor)
          attempts.push(attempt(h, err, true))
          return {
            ...s,
            tOld: s.t,
            yOld: s.y,
            t: tNew,
            y: yNew,
            fy: fNew,
            k: ks,
            h,
            hAbs: hAbs * factor,
            errPrev: Math.max(err, 1e-4),
            errPrev2: s.errPrev,
            attempts,
            status: statusAfter(tNew, s.tBound, s.direction),
            counts: snapshot(c),
          }
        }
        attempts.push(attempt(h, err, false))
        const shrink = Number.isFinite(err) ? Math.max(minFactor, safety * err ** -beta1) : minFactor
        hAbs *= shrink
        rejected = true
      }
      return failed(s, c, attempts, `${tableau.name}: ${maxAttempts} consecutive step rejections`)
    },
    dense: (s) => (isDop ? dop853Dense(s) : rkDense(tableau, s)),
  }
}

function failed(s: AdaptiveState, c: Counter, attempts: StepAttempt[], message: string): AdaptiveState {
  return { ...s, status: 'failed', message, attempts, counts: snapshot(c) }
}

function rkDense(tableau: EmbeddedTableau, s: AdaptiveState): DenseSegment {
  const p = tableau.p
  const fOld = s.k[0]
  if (p === undefined || fOld === undefined) return hermiteSegment(s.tOld, s.yOld, fOld ?? s.fy, s.t, s.y, s.fy)
  const h = s.h
  const q = p[0]?.length ?? 0
  return {
    t0: s.tOld,
    t1: s.t,
    at: (t) => {
      const x = h === 0 ? 0 : (t - s.tOld) / h
      const powers: number[] = []
      let xp = 1
      for (let j = 0; j < q; j++) {
        xp *= x
        powers.push(xp)
      }
      const terms: Term[] = s.k.map((ki, i) => {
        const row = p[i] ?? []
        let w = 0
        for (let j = 0; j < q; j++) w += (row[j] ?? 0) * (powers[j] ?? 0)
        return [h * w, ki] as const
      })
      return combine(s.yOld, terms)
    },
  }
}

function dop853Dense(s: AdaptiveState): DenseSegment {
  const h = s.h
  const ks: Value[] = [...s.k]
  for (let st = dop.N_STAGES + 1; st < dop.N_STAGES_EXTENDED; st++) {
    const row = dop.A[st] ?? []
    const terms: Term[] = row.map((a, j) => [h * a, ks[j] as Value] as const)
    ks.push(s.f(s.tOld + (dop.C[st] ?? 0) * h, combine(s.yOld, terms)))
  }
  const fOld = ks[0] as Value
  const dy = combine(s.y, [[-1, s.yOld]])
  const F: Value[] = [
    dy,
    combine(null, [
      [h, fOld],
      [-1, dy],
    ]),
    combine(null, [
      [2, dy],
      [-h, s.fy],
      [-h, fOld],
    ]),
    ...dop.D.map((row) =>
      combine(
        null,
        row.map((d, j) => [h * d, ks[j] as Value] as const),
      ),
    ),
  ]
  return {
    t0: s.tOld,
    t1: s.t,
    at: (t) => {
      const x = h === 0 ? 0 : (t - s.tOld) / h
      // y = (((F6 x + F5)(1 - x) + F4) x + ...) as scipy evaluates it, written as weights on each F.
      const w = new Float64Array(F.length)
      for (let i = 0; i < F.length; i++) {
        const k = F.length - 1 - i
        w[k] = (w[k] ?? 0) + 1
        const m = i % 2 === 0 ? x : 1 - x
        for (let j = 0; j < w.length; j++) w[j] = (w[j] ?? 0) * m
      }
      return combine(
        s.yOld,
        F.map((fk, k) => [w[k] ?? 0, fk] as const),
      )
    },
  }
}

/**
 * scipy's RK23 (Bogacki-Shampine 3/2).
 *
 * @param options - Tolerances, step limits and controller.
 * @returns The stepper.
 * @example
 * ```ts
 * import { rk23 } from 'whitebox-engine/dynamics/ode'
 * rk23().order // => 3
 * ```
 */
export function rk23(options: AdaptiveOptions = {}): Stepper<AdaptiveState> {
  return adaptiveRk(BOGACKI_SHAMPINE, options)
}

/**
 * scipy's RK45 (Dormand-Prince 5/4).
 *
 * @param options - Tolerances, step limits and controller.
 * @returns The stepper.
 * @example
 * ```ts
 * import { rk45 } from 'whitebox-engine/dynamics/ode'
 * rk45().order // => 5
 * ```
 */
export function rk45(options: AdaptiveOptions = {}): Stepper<AdaptiveState> {
  return adaptiveRk(DORMAND_PRINCE, options)
}

/**
 * scipy's DOP853 (eighth order with seventh-order dense output).
 *
 * @param options - Tolerances, step limits and controller.
 * @returns The stepper.
 * @example
 * ```ts
 * import { dop853 } from 'whitebox-engine/dynamics/ode'
 * dop853().order // => 8
 * ```
 */
export function dop853(options: AdaptiveOptions = {}): Stepper<AdaptiveState> {
  return adaptiveRk(DOP853_TABLEAU, options)
}

/**
 * The Heun-Euler (2/1) pair.
 *
 * @param options - Tolerances, step limits and controller.
 * @returns The stepper.
 * @example
 * ```ts
 * import { heunEuler } from 'whitebox-engine/dynamics/ode'
 * heunEuler().order // => 2
 * ```
 */
export function heunEuler(options: AdaptiveOptions = {}): Stepper<AdaptiveState> {
  return adaptiveRk(HEUN_EULER, options)
}

/**
 * Runge-Kutta-Fehlberg 4(5).
 *
 * @param options - Tolerances, step limits and controller.
 * @returns The stepper.
 * @example
 * ```ts
 * import { fehlberg45 } from 'whitebox-engine/dynamics/ode'
 * fehlberg45().order // => 4
 * ```
 */
export function fehlberg45(options: AdaptiveOptions = {}): Stepper<AdaptiveState> {
  return adaptiveRk(FEHLBERG, options)
}

/**
 * The Cash-Karp 5(4) pair.
 *
 * @param options - Tolerances, step limits and controller.
 * @returns The stepper.
 * @example
 * ```ts
 * import { cashKarp } from 'whitebox-engine/dynamics/ode'
 * cashKarp().name // => 'cash-karp'
 * ```
 */
export function cashKarp(options: AdaptiveOptions = {}): Stepper<AdaptiveState> {
  return adaptiveRk(CASH_KARP, options)
}

/**
 * Tsitouras' 5(4) pair (Tsit5).
 *
 * @param options - Tolerances, step limits and controller.
 * @returns The stepper.
 * @example
 * ```ts
 * import { tsit5 } from 'whitebox-engine/dynamics/ode'
 * tsit5().name // => 'tsit5'
 * ```
 */
export function tsit5(options: AdaptiveOptions = {}): Stepper<AdaptiveState> {
  return adaptiveRk(TSITOURAS, options)
}
