/**
 * Linear multistep methods, which reuse derivatives from past steps instead of computing new stages: fixed-step
 * Adams-Bashforth, Adams-Moulton and the PECE predictor-corrector of orders 1 to 5, and the variable-step,
 * variable-order Adams method in Nordsieck form that forms the nonstiff half of LSODE and LSODA.
 */

import { MdmlError } from '../../base/index.ts'
import type { Value } from '../../tensor/index.ts'
import { implicitCorrection, isTracedState, type JacobianOption, jacobianAt, luApply, realLu } from './_newton.ts'
import {
  atBound,
  attempt,
  type Counter,
  checked,
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
import { allFinite, combine, errorScale, fromNumbers, minStep, numbersOf, rms, scaledRms, type Term } from './_vec.ts'
import { RK4, rkStep } from './explicit.ts'
import {
  type DenseSegment,
  hermiteSegment,
  type StepAttempt,
  type Stepper,
  type StepperOptions,
  type StepperState,
  type VectorField,
} from './problem.ts'

/** Adams-Bashforth weights of orders 1 to 5, on f_n, f_{n-1}, ... */
export const ADAMS_BASHFORTH: readonly (readonly number[])[] = [
  [1],
  [3 / 2, -1 / 2],
  [23 / 12, -16 / 12, 5 / 12],
  [55 / 24, -59 / 24, 37 / 24, -9 / 24],
  [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720],
]

/** Adams-Moulton weights of orders 1 to 5, on f_{n+1}, f_n, f_{n-1}, ... */
export const ADAMS_MOULTON: readonly (readonly number[])[] = [
  [1],
  [1 / 2, 1 / 2],
  [5 / 12, 8 / 12, -1 / 12],
  [9 / 24, 19 / 24, -5 / 24, 1 / 24],
  [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720],
]

/** Options of the fixed-step multistep methods. */
export interface MultistepOptions {
  readonly dt: number
  /** The order, 1 to 5 (default 4). */
  readonly order?: 1 | 2 | 3 | 4 | 5
  readonly jac?: JacobianOption
  /** For predictor-corrector: corrector evaluations per step (1 is PECE, 2 is PECECE; default 1). */
  readonly corrections?: number
}

/** The state of a fixed-step multistep method: past derivatives, most recent first. */
export interface MultistepState extends StepperState {
  readonly f: VectorField
  readonly fy: Value
  readonly fOld: Value
  readonly history: readonly Value[]
  /** Milne's estimate of the local error of the last predictor-corrector step (0 otherwise). */
  readonly milne: number
}

type MultistepKind = 'bashforth' | 'moulton' | 'pece'

function multistep(kind: MultistepKind, options: MultistepOptions): Stepper<MultistepState> {
  const dt = options.dt
  const order = options.order ?? 4
  if (!(dt > 0) || !Number.isFinite(dt)) throw new MdmlError('value', 'multistep: dt must be positive')
  if (!Number.isInteger(order) || order < 1 || order > 5) throw new MdmlError('value', 'multistep: order is 1 to 5')
  const jac = options.jac ?? 'autodiff'
  const corrections = options.corrections ?? 1
  const name =
    kind === 'bashforth'
      ? `adams-bashforth-${order}`
      : kind === 'moulton'
        ? `adams-moulton-${order}`
        : `abm-pece-${order}`
  const ab = ADAMS_BASHFORTH[order - 1] ?? [1]
  const am = ADAMS_MOULTON[order - 1] ?? [1]
  // Milne's device for an order-p pair: error ~ C*/(C - C*) (corrected - predicted).
  const abError = [1 / 2, 5 / 12, 3 / 8, 251 / 720, 95 / 288][order - 1] ?? 0
  const amError = [-1 / 2, -1 / 12, -1 / 24, -19 / 720, -3 / 160][order - 1] ?? 0
  const milneFactor = Math.abs(amError / (abError - amError))
  return {
    name,
    order,
    stiff: kind === 'moulton',
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      const fy = tally(f, c)(t0, y0)
      return {
        ...initialState(t0, y0, tBound, dt, c, order),
        f,
        fy,
        fOld: fy,
        history: [fy],
        milne: 0,
        counts: snapshot(c),
      }
    },
    step: (s) => {
      if (s.status !== 'running') throw new MdmlError('value', `${name}: the integration has ended`)
      if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
      const c: Counter = { ...s.counts }
      const f = tally(s.f, c)
      const { tNew, h } = towards(s.t, dt, s.direction, s.tBound)
      // Too little history, or a shortened last step: start (or finish) with RK4.
      const needed = kind === 'moulton' ? order - 1 : order
      const uniform = Math.abs(Math.abs(h) - dt) <= 1e-12 * Math.max(1, dt)
      let yNew: Value
      let milne = 0
      if (s.history.length < needed || !uniform) {
        yNew = rkStep(f, s.t, s.y, h, RK4, s.fy)
      } else if (kind === 'bashforth') {
        yNew = combine(
          s.y,
          ab.map((w, j) => [h * w, s.history[j] as Value] as const),
        )
      } else if (kind === 'pece') {
        const yPred = combine(
          s.y,
          ab.map((w, j) => [h * w, s.history[j] as Value] as const),
        )
        yNew = yPred
        for (let k = 0; k < corrections; k++) {
          const fp = f(tNew, yNew)
          yNew = combine(s.y, [
            [h * (am[0] ?? 0), fp],
            ...am.slice(1).map((w, j) => [h * w, s.history[j] as Value] as const),
          ])
        }
        const diff = Float64Array.from(numbersOf(yNew), (v, i) => v - (numbersOf(yPred)[i] ?? 0))
        milne = milneFactor * rms(diff)
      } else {
        const base = combine(
          s.y,
          am.slice(1).map((w, j) => [h * w, s.history[j] as Value] as const),
        )
        const ha = h * (am[0] ?? 1)
        const z = newtonSolve(f, s.f, tNew, base, ha, s.fy, jac, c)
        if (z === undefined) {
          return {
            ...s,
            status: 'failed',
            message: `${name}: Newton's method did not converge at t = ${tNew}`,
            attempts: [attempt(h, Number.POSITIVE_INFINITY, false)],
            counts: snapshot(c),
          }
        }
        yNew = isTracedState(base, s.fy)
          ? implicitCorrection(
              z,
              base.shape,
              (v) =>
                combine(v, [
                  [-1, base],
                  [-ha, f(tNew, v)],
                ]),
              () => {
                const n = z.length
                const jz = jacobianAt(s.f, tNew, fromNumbers(z, base.shape), jac)
                return Float64Array.from(jz, (v, k) => (k % (n + 1) === 0 ? 1 : 0) - ha * v)
              },
            )
          : fromNumbers(z, base.shape)
      }
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
        history: [fNew, ...s.history].slice(0, 5),
        milne,
        attempts: [attempt(h, milne, ok)],
        status: ok ? statusAfter(tNew, s.tBound, s.direction) : 'failed',
        message: ok ? '' : `${name}: the solution became non-finite at t = ${tNew}`,
        counts: snapshot(c),
      }
    },
    dense: (s) => hermiteSegment(s.tOld, s.yOld, s.fOld, s.t, s.y, s.fy),
  }
}

function newtonSolve(
  f: VectorField,
  raw: VectorField,
  t: number,
  base: Value,
  ha: number,
  slope: Value,
  jac: JacobianOption,
  c: Counter,
): Float64Array | undefined {
  const b = numbersOf(base)
  const n = b.length
  const g = numbersOf(slope)
  const z = Float64Array.from(b, (v, i) => v + ha * (g[i] ?? 0))
  for (let refresh = 0; refresh < 2; refresh++) {
    const J = jacobianAt(raw, t, fromNumbers(z, base.shape), jac)
    c.njev += 1
    const lu = realLu(
      Float64Array.from(J, (v, k) => (k % (n + 1) === 0 ? 1 : 0) - ha * v),
      n,
    )
    c.nlu += 1
    for (let it = 0; it < 10; it++) {
      const fz = numbersOf(f(t, fromNumbers(z, base.shape)))
      if (!allFinite(fz)) return undefined
      const dz = luApply(
        lu,
        Float64Array.from(z, (v, i) => -(v - (b[i] ?? 0) - ha * (fz[i] ?? 0))),
      )
      for (let i = 0; i < n; i++) z[i] = (z[i] ?? 0) + (dz[i] ?? 0)
      if (rms(dz) / (1 + rms(z)) <= 1e-12) return z
    }
  }
  return undefined
}

/**
 * The explicit Adams-Bashforth method of order 1 to 5 with a fixed step (started by RK4).
 *
 * @param options - Step size and order (default 4).
 * @returns The stepper.
 * @throws MdmlError with code `value` for a bad step or order.
 * @concept linear-multistep
 * @example
 * ```ts
 * import { adamsBashforth } from 'mdml-engine/dynamics/ode'
 * adamsBashforth({ dt: 0.1, order: 3 }).name // => 'adams-bashforth-3'
 * ```
 */
export function adamsBashforth(options: MultistepOptions): Stepper<MultistepState> {
  return multistep('bashforth', options)
}

/**
 * The implicit Adams-Moulton method of order 1 to 5 with a fixed step, each step solved by Newton's method (started by
 * RK4; differentiable through an implicit-function correction when traced).
 *
 * @param options - Step size, order (default 4) and Jacobian.
 * @returns The stepper.
 * @concept linear-multistep
 * @example
 * ```ts
 * import { adamsMoulton } from 'mdml-engine/dynamics/ode'
 * adamsMoulton({ dt: 0.1, order: 2 }).name // => 'adams-moulton-2'
 * ```
 */
export function adamsMoulton(options: MultistepOptions): Stepper<MultistepState> {
  return multistep('moulton', options)
}

/**
 * The Adams-Bashforth-Moulton predictor-corrector of order 1 to 5 (PECE, or more corrections), with Milne's estimate of
 * the local error in each step's attempt record.
 *
 * @param options - Step size, order (default 4) and corrections per step.
 * @returns The stepper.
 * @concept predictor-corrector
 * @example
 * ```ts
 * import { pece } from 'mdml-engine/dynamics/ode'
 * pece({ dt: 0.1 }).name // => 'abm-pece-4'
 * ```
 */
export function pece(options: MultistepOptions): Stepper<MultistepState> {
  return multistep('pece', options)
}

/** The Adams-Moulton error coefficients gamma*_q (q = 0..13): Sum_{j<=k} gamma*_j / (k + 1 - j) = 0, gamma*_0 = 1. */
export const ADAMS_ERROR_COEFFICIENTS: readonly number[] = (() => {
  const g = [1]
  for (let k = 1; k <= 13; k++) {
    let s = 0
    for (let j = 0; j < k; j++) s += (g[j] ?? 0) / (k + 1 - j)
    g.push(-s)
  }
  return g
})()

/**
 * The Nordsieck corrector vector l of the order-q Adams-Moulton method: the coefficients of
 * Lambda(x) = (1/(q-1)!) integral_{-1}^{x} prod_{i=1}^{q-1} (u + i) du, so l_1 = 1 and l_0 is the method's implicit
 * weight.
 *
 * @param q - The order (1 to 12).
 * @returns `[l_0, ..., l_q]`.
 * @throws MdmlError with code `value` for an order outside 1..12.
 * @concept nordsieck
 * @example
 * ```ts
 * import { nordsieckAdamsL } from 'mdml-engine/dynamics/ode'
 * nordsieckAdamsL(2) // => [0.5, 1, 0.5]
 * ```
 */
export function nordsieckAdamsL(q: number): number[] {
  if (!Number.isInteger(q) || q < 1 || q > 12) throw new MdmlError('value', 'nordsieckAdamsL: order is 1 to 12')
  // prod_{i=1}^{q-1} (u + i) as coefficients in u.
  let poly = [1]
  for (let i = 1; i < q; i++) {
    const next = new Array<number>(poly.length + 1).fill(0)
    poly.forEach((c, k) => {
      next[k] = (next[k] ?? 0) + c * i
      next[k + 1] = (next[k + 1] ?? 0) + c
    })
    poly = next
  }
  let fact = 1
  for (let i = 2; i < q; i++) fact *= i
  // Integrate from -1 to x: sum c_k (x^{k+1} - (-1)^{k+1}) / (k+1).
  const out = new Array<number>(q + 1).fill(0)
  poly.forEach((c, k) => {
    out[k + 1] = (out[k + 1] ?? 0) + c / (k + 1) / fact
    out[0] = (out[0] ?? 0) - (c * (-1) ** (k + 1)) / (k + 1) / fact
  })
  return out
}

/** The state of {@link adams}: the Nordsieck array [y, h y', h^2 y''/2, ...] at the current point. */
export interface NordsieckState extends StepperState {
  readonly f: VectorField
  readonly fy: Value
  readonly z: readonly Value[]
  /** The signed step the Nordsieck array is scaled to. */
  readonly h: number
  /** The corrections of the last two steps (for raising the order). */
  readonly e: Value | undefined
  readonly ePrev: Value | undefined
  readonly nEqualSteps: number
  /** The largest Lipschitz estimate seen by the corrector in the last step (for stiffness detection). */
  readonly lipschitz: number
  /** The previous Nordsieck array and step, for dense output of the last step. */
  readonly zOld: readonly Value[]
  readonly hOld: number
}

/** Options of {@link adams}. */
export interface AdamsOptions extends StepperOptions {
  /** The highest order (1 to 12, default 12). */
  readonly maxOrder?: number
}

function pascalPredict(z: readonly Value[]): Value[] {
  const q = z.length - 1
  const out: Value[] = []
  for (let j = 0; j <= q; j++) {
    const terms: Term[] = []
    let binom = 1
    for (let k = j; k <= q; k++) {
      terms.push([binom, z[k] as Value])
      binom = (binom * (k + 1)) / (k + 1 - j)
    }
    out.push(combine(null, terms))
  }
  return out
}

function rescale(z: readonly Value[], r: number): Value[] {
  let rj = 1
  return z.map((v, j) => {
    const w = j === 0 ? v : combine(null, [[rj, v]])
    rj *= r
    return w
  })
}

/**
 * Variable-step, variable-order Adams-Moulton (orders 1 to 12) in Nordsieck form, the nonstiff half of LSODE and
 * LSODA: predict by the Pascal matrix, correct by functional iteration (at most three iterations), estimate the local
 * error from the correction, change the step by rescaling the Nordsieck array, and try orders q - 1, q and q + 1 after
 * q + 1 steps at one order. The corrector's Lipschitz estimate is kept in the state for stiffness detection.
 *
 * @param options - Tolerances (default rtol 1e-3, atol 1e-6), step limits and highest order.
 * @returns The stepper.
 * @concept nordsieck
 * @example
 * ```ts
 * import { adams } from 'mdml-engine/dynamics/ode'
 * adams().order // => 12
 * ```
 */
export function adams(options: AdamsOptions = {}): Stepper<NordsieckState> {
  const { rtol, atol } = validTolerances(options.rtol ?? 1e-3, options.atol ?? 1e-6)
  const maxStep = validMaxStep(options.maxStep)
  const maxOrder = Math.min(12, Math.max(1, options.maxOrder ?? 12))
  const errConst = (q: number): number => {
    let fact = 1
    for (let i = 2; i <= q; i++) fact *= i
    return Math.abs(ADAMS_ERROR_COEFFICIENTS[q] ?? 0) * fact
  }
  return {
    name: 'adams',
    order: maxOrder,
    stiff: false,
    init: (field, t0, y0, tBound) => {
      const c = counter()
      const f = checked(field, y0.shape)
      const fy = tally(f, c)(t0, y0)
      const base = initialState(t0, y0, tBound, 0, c, 1)
      const hAbs =
        options.firstStep === undefined
          ? selectInitialStep(tally(f, c), t0, y0, tBound, maxStep, fy, base.direction, 1, rtol, atol)
          : validFirstStep(options.firstStep, t0, tBound)
      const h = hAbs * base.direction
      const z = [y0, combine(null, [[h, fy]])]
      return {
        ...base,
        hAbs,
        f,
        fy,
        z,
        h,
        e: undefined,
        ePrev: undefined,
        nEqualSteps: 0,
        lipschitz: 0,
        zOld: z,
        hOld: h,
        counts: snapshot(c),
      }
    },
    step: (s) => {
      if (s.status !== 'running') throw new MdmlError('value', 'adams: the integration has ended')
      if (atBound(s)) return { ...s, tOld: s.t, t: s.tBound, status: 'finished', attempts: [] }
      const c: Counter = { ...s.counts }
      const f = tally(s.f, c)
      const hmin = minStep(s.t, s.direction)
      const q = s.order
      let hAbs = Math.min(Math.max(s.hAbs, hmin), maxStep)
      let z = rescale(s.z, (hAbs * s.direction) / s.h)
      let nEqual = hAbs === Math.abs(s.h) ? s.nEqualSteps : 0
      const attempts: StepAttempt[] = []
      let lipschitz = 0
      for (let tries = 0; tries < 60; tries++) {
        if (hAbs < hmin) {
          return { ...s, status: 'failed', message: 'adams: step size underflow', attempts, counts: snapshot(c) }
        }
        const { tNew, h } = towards(s.t, hAbs, s.direction, s.tBound)
        if (Math.abs(h) !== hAbs) {
          z = rescale(z, Math.abs(h) / hAbs)
          hAbs = Math.abs(h)
          nEqual = 0
        }
        const l = nordsieckAdamsL(q)
        const pred = pascalPredict(z)
        const y0p = pred[0] as Value
        const y0n = numbersOf(y0p)
        const scale = errorScale(y0n, y0n, rtol, atol)
        // Functional iteration for e = h f(t_new, y_pred + l0 e) - (h y')_pred.
        let e = combine(null, [[0, y0p]])
        let yIter = y0p
        let converged = false
        let fPrev: Float64Array | undefined
        let yPrev: Float64Array | undefined
        for (let m = 0; m < 3; m++) {
          const fv = f(tNew, yIter)
          const fn = numbersOf(fv)
          if (!allFinite(fn)) break
          const eNew = combine(null, [
            [h, fv],
            [-1, pred[1] as Value],
          ])
          const de = Float64Array.from(numbersOf(eNew), (v, i) => v - (numbersOf(e)[i] ?? 0))
          const yn = numbersOf(yIter)
          if (fPrev !== undefined && yPrev !== undefined) {
            const dy = rms(Float64Array.from(yn, (v, i) => v - (yPrev?.[i] ?? 0)))
            const df = rms(Float64Array.from(fn, (v, i) => v - (fPrev?.[i] ?? 0)))
            if (dy > 0) lipschitz = Math.max(lipschitz, df / dy)
          }
          fPrev = fn
          yPrev = yn
          e = eNew
          yIter = combine(y0p, [[l[0] ?? 1, e]])
          const del = scaledRms(de, scale) * (l[0] ?? 1)
          if (m > 0 && del <= 0.1) {
            converged = true
            break
          }
          if (m === 0 && del === 0) {
            converged = true
            break
          }
        }
        if (!converged) {
          attempts.push(attempt(h, Number.POSITIVE_INFINITY, false))
          z = rescale(z, 0.25)
          hAbs *= 0.25
          nEqual = 0
          continue
        }
        const en = numbersOf(e)
        const yNew = yIter
        const errScale = errorScale(numbersOf(s.y), numbersOf(yNew), rtol, atol)
        const err = errConst(q) * scaledRms(en, errScale)
        if (!(err <= 1)) {
          attempts.push(attempt(h, err, false))
          const factor = Number.isFinite(err) ? Math.max(0.2, 0.9 * err ** (-1 / (q + 1))) : 0.2
          z = rescale(z, factor)
          hAbs *= factor
          nEqual = 0
          continue
        }
        attempts.push(attempt(h, err, true))
        const zNew = pred.map((p, j) => combine(p, [[l[j] ?? 0, e]]))
        nEqual += 1
        // Next step size and order.
        let factor = err === 0 ? 10 : Math.min(10, 0.9 * err ** (-1 / (q + 1)))
        let newQ = q
        let zNext = zNew
        if (nEqual >= q + 1) {
          const down =
            q > 1 ? errConst(q - 1) * q * scaledRms(numbersOf(zNew[q] as Value), errScale) : Number.POSITIVE_INFINITY
          let up = Number.POSITIVE_INFINITY
          if (q < maxOrder && s.e !== undefined && s.order === q) {
            const ep = numbersOf(s.e)
            const diff = Float64Array.from(en, (v, i) => v - (ep[i] ?? 0))
            up = (errConst(q + 1) / (q + 1)) * scaledRms(diff, errScale)
          }
          const fDown = down === 0 ? 10 : (0.9 / 1.3) * down ** (-1 / q)
          const fSame = factor / 1.2
          const fUp = up === 0 ? 10 : (0.9 / 1.4) * up ** (-1 / (q + 2))
          if (fUp > fSame && fUp >= fDown && q < maxOrder && Number.isFinite(up)) {
            newQ = q + 1
            zNext = [...zNew, combine(null, [[1 / (q + 1), e]])]
            factor = Math.min(10, fUp)
          } else if (fDown > fSame && q > 1) {
            newQ = q - 1
            zNext = zNew.slice(0, q)
            factor = Math.min(10, fDown)
          } else factor = Math.min(10, Math.max(fSame, 1))
          nEqual = 0
        } else factor = 1
        const fNew = f(tNew, yNew)
        return {
          ...s,
          tOld: s.t,
          yOld: s.y,
          t: tNew,
          y: yNew,
          fy: fNew,
          zOld: zNew,
          hOld: h,
          z: zNext,
          h,
          hAbs: Math.min(hAbs * factor, maxStep),
          order: newQ,
          e: newQ === q ? e : undefined,
          ePrev: s.e,
          nEqualSteps: nEqual,
          lipschitz,
          attempts,
          status: statusAfter(tNew, s.tBound, s.direction),
          counts: snapshot(c),
        }
      }
      return { ...s, status: 'failed', message: 'adams: too many step rejections', attempts, counts: snapshot(c) }
    },
    dense: (s) => nordsieckDense(s),
  }
}

/** The Nordsieck polynomial of the last step, evaluated backwards from its end. */
function nordsieckDense(s: NordsieckState): DenseSegment {
  const z = s.zOld
  const h = s.hOld
  return {
    t0: s.tOld,
    t1: s.t,
    at: (t) => {
      const x = h === 0 ? 0 : (t - s.t) / h
      let xp = 1
      const terms: Term[] = z.map((v) => {
        const term = [xp, v] as const
        xp *= x
        return term
      })
      return combine(null, terms)
    },
  }
}
