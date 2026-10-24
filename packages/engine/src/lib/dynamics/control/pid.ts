import { clamp, WhiteboxError } from '../../base/index.ts'
import { c2dMatrices, isDiscrete, type Lti, toSs } from '../../systems/index.ts'
import { primalOf, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'

/** Gains and implementation details of a PID controller. */
export interface PidOptions {
  readonly kp: number
  readonly ki?: number
  readonly kd?: number
  /** The sample time. */
  readonly dt: number
  /** The derivative filter time constant Tf (default kd / (10 kp), N = 10; 0 for none). */
  readonly tf?: number
  /** Output limits. */
  readonly outputMin?: number
  readonly outputMax?: number
  /** Anti-windup: stop integrating while saturated (`clamping`, the default) or feed the saturation back (`back-calculation`). */
  readonly antiWindup?: 'clamping' | 'back-calculation' | 'none'
  /** Back-calculation gain (default 1 / Ti, or 1). */
  readonly kb?: number
  /** Setpoint weights of the proportional and derivative terms (default 1 and 0: derivative on measurement). */
  readonly b?: number
  readonly c?: number
}

/** The memory of a PID controller between samples. */
export interface PidState {
  readonly integral: number
  readonly derivative: number
  readonly previousError: number
  readonly started: boolean
}

/** A discrete PID controller: `step` maps (state, setpoint, measurement) to the output and the next state. */
export interface PidController {
  readonly options: PidOptions
  init(): PidState
  step(state: PidState, setpoint: number, measurement: number): { readonly u: number; readonly state: PidState }
}

/**
 * A discrete PID controller in parallel form with setpoint weighting, a first-order filter on the derivative
 * (backward Euler: D_k = Tf/(Tf + dt) D_{k-1} + kd/(Tf + dt) (e_d,k - e_d,k-1)), output limits and anti-windup by
 * conditional integration or back-calculation.
 *
 * @param options - Gains, sample time, filter, limits, anti-windup and setpoint weights.
 * @returns The controller.
 * @throws WhiteboxError with code `value` for a non-positive sample time.
 * @concept pid-control
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * const pid = control.pid({ kp: 2, ki: 1, dt: 0.1 })
 * pid.step(pid.init(), 1, 0).u // => 2.1
 * ```
 */
export function pid(options: PidOptions): PidController {
  if (!(options.dt > 0)) throw new WhiteboxError('value', 'pid: the sample time must be positive')
  const { kp, dt } = options
  const ki = options.ki ?? 0
  const kd = options.kd ?? 0
  const tf = options.tf ?? (kd > 0 && kp !== 0 ? kd / (10 * Math.abs(kp)) : 0)
  const lo = options.outputMin ?? Number.NEGATIVE_INFINITY
  const hi = options.outputMax ?? Number.POSITIVE_INFINITY
  const aw = options.antiWindup ?? 'clamping'
  const kb = options.kb ?? (ki !== 0 && kp !== 0 ? ki / kp : 1)
  const b = options.b ?? 1
  const c = options.c ?? 0
  return {
    options,
    init: () => ({ integral: 0, derivative: 0, previousError: 0, started: false }),
    step: (s, setpoint, y) => {
      const e = setpoint - y
      const ed = c * setpoint - y
      const p = kp * (b * setpoint - y)
      const de = s.started ? ed - s.previousError : 0
      const d = kd === 0 ? 0 : (tf / (tf + dt)) * s.derivative + (kd / (tf + dt)) * de
      const integralCandidate = s.integral + ki * dt * e
      const raw = p + integralCandidate + d
      const u = clamp(raw, lo, hi)
      let integral = integralCandidate
      if (aw === 'clamping' && u !== raw && Math.sign(e) === Math.sign(raw - u)) integral = s.integral
      else if (aw === 'back-calculation') integral = integralCandidate + kb * dt * (u - raw)
      return { u, state: { integral, derivative: d, previousError: ed, started: true } }
    },
  }
}

/** The closed-loop response of {@link simulatePid}. */
export interface PidResponse {
  readonly t: number[]
  readonly y: number[]
  readonly u: number[]
  readonly setpoint: number[]
  /** Integral of absolute error. */
  readonly iae: number
  /** Overshoot as a fraction of the final setpoint. */
  readonly overshoot: number
}

/**
 * Simulate a PID loop around a single-input single-output plant (exactly discretised with a zero-order hold at the
 * controller's sample time) as a traced algorithm with the output, input and error as metrics.
 *
 * @param plant - The plant (any LTI representation; continuous or discrete with the same sample time).
 * @param controller - The controller.
 * @param options - Duration, setpoint (constant or a function of time), output disturbance and tracing.
 * @returns The response with IAE and overshoot.
 * @concept pid-control
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * import { tf } from 'whitebox-engine/systems'
 * const r = control.simulatePid(tf([1], [1, 1]), control.pid({ kp: 2, ki: 2, dt: 0.01 }), { duration: 10 })
 * Math.abs((r.y.at(-1) ?? 0) - 1) < 1e-3 // => true
 * ```
 */
export function simulatePid(
  plant: Lti,
  controller: PidController,
  options: {
    readonly duration: number
    readonly setpoint?: number | ((t: number) => number)
    readonly disturbance?: (t: number) => number
  } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>>,
): PidResponse {
  const dt = controller.options.dt
  const sys = toSs(plant)
  const discrete = isDiscrete(sys)
  if (discrete && Math.abs(sys.dt - dt) > 1e-12) throw new WhiteboxError('value', 'simulatePid: sample times differ')
  const m = discrete ? { a: sys.a, b: sys.b, c: sys.c, d: sys.d } : c2dMatrices(sys.a, sys.b, sys.c, sys.d, dt)
  const A = toFlat(primalOf(m.a))
  const B = toFlat(primalOf(m.b))
  const C = toFlat(primalOf(m.c))
  const D = toFlat(primalOf(m.d))[0] ?? 0
  const n = Math.round(Math.sqrt(A.length))
  const steps = Math.round(options.duration / dt)
  const sp = (t: number): number =>
    options.setpoint === undefined ? 1 : typeof options.setpoint === 'number' ? options.setpoint : options.setpoint(t)
  interface S {
    readonly x: number[]
    readonly ctrl: PidState
    readonly t: number[]
    readonly y: number[]
    readonly u: number[]
    readonly r: number[]
  }
  const output = (x: readonly number[], u: number): number =>
    C.reduce((acc, ci, i) => acc + ci * (x[i] ?? 0), 0) + D * u
  const algorithm: Algorithm<S> = {
    name: 'pid-loop',
    config: { kp: controller.options.kp, ki: controller.options.ki ?? 0, kd: controller.options.kd ?? 0, dt },
    schema: channels({
      metrics: {
        y: { description: 'plant output' },
        u: { description: 'control input' },
        error: { description: 'setpoint minus output' },
      },
    }),
    init: () => ({ x: new Array<number>(n).fill(0), ctrl: controller.init(), t: [], y: [], u: [], r: [] }),
    step: (s, ctx) => {
      const t = ctx.t * dt
      const r = sp(t)
      const yMeas = output(s.x, 0) + (options.disturbance?.(t) ?? 0)
      const { u, state } = controller.step(s.ctrl, r, yMeas)
      const x = Array.from({ length: n }, (_, i) => {
        let v = (B[i] ?? 0) * u
        for (let j = 0; j < n; j++) v += (A[i * n + j] ?? 0) * (s.x[j] ?? 0)
        return v
      })
      s.t.push(t)
      s.y.push(yMeas)
      s.u.push(u)
      s.r.push(r)
      return { ...s, x, ctrl: state }
    },
    observe: (s) => ({
      metrics: { y: s.y.at(-1) ?? 0, u: s.u.at(-1) ?? 0, error: (s.r.at(-1) ?? 0) - (s.y.at(-1) ?? 0) },
    }),
  }
  const st = runAlgorithm(algorithm, { maxSteps: steps, ...options }).state
  const final = st.r.at(-1) ?? 1
  let peak = Number.NEGATIVE_INFINITY
  for (const v of st.y) peak = Math.max(peak, v)
  const iae = st.y.reduce((acc, v, i) => acc + Math.abs((st.r[i] ?? 0) - v) * dt, 0)
  return {
    t: st.t,
    y: st.y,
    u: st.u,
    setpoint: st.r,
    iae,
    overshoot: final === 0 ? 0 : Math.max(0, (peak - final) / Math.abs(final)),
  }
}

/** PID gains in parallel form u = kp e + ki int e + kd de/dt. */
export interface PidGains {
  readonly kp: number
  readonly ki: number
  readonly kd: number
}

/**
 * Ziegler-Nichols tuning from the ultimate gain Ku and period Tu (the closed-loop oscillation method): P kp = 0.5 Ku;
 * PI kp = 0.45 Ku, Ti = Tu/1.2; PID kp = 0.6 Ku, Ti = Tu/2, Td = Tu/8.
 *
 * @param ku - The ultimate gain.
 * @param tu - The ultimate period.
 * @param type - `P`, `PI` or `PID` (default).
 * @returns The gains.
 * @concept pid-tuning
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * control.zieglerNichols(10, 2) // => { kp: 6, ki: 6, kd: 1.5 }
 * ```
 */
export function zieglerNichols(ku: number, tu: number, type: 'P' | 'PI' | 'PID' = 'PID'): PidGains {
  if (type === 'P') return { kp: 0.5 * ku, ki: 0, kd: 0 }
  if (type === 'PI') {
    const kp = 0.45 * ku
    return { kp, ki: kp / (tu / 1.2), kd: 0 }
  }
  const kp = 0.6 * ku
  return { kp, ki: kp / (tu / 2), kd: kp * (tu / 8) }
}

/**
 * Cohen-Coon tuning for a first-order-plus-dead-time plant K e^{-theta s} / (tau s + 1).
 *
 * @param k - The process gain K.
 * @param tau - The time constant.
 * @param theta - The dead time.
 * @param type - `P`, `PI` or `PID` (default).
 * @returns The gains.
 * @throws WhiteboxError with code `value` for non-positive tau or theta.
 * @concept pid-tuning
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * Math.round(control.cohenCoon(1, 10, 2, 'P').kp * 1e6) / 1e6 // => 5.333333
 * ```
 */
export function cohenCoon(k: number, tau: number, theta: number, type: 'P' | 'PI' | 'PID' = 'PID'): PidGains {
  if (!(tau > 0) || !(theta > 0)) throw new WhiteboxError('value', 'cohenCoon: tau and theta must be positive')
  const r = theta / tau
  if (type === 'P') return { kp: (1 / (k * r)) * (1 + r / 3), ki: 0, kd: 0 }
  if (type === 'PI') {
    const kp = (1 / (k * r)) * (0.9 + r / 12)
    const ti = (theta * (30 + 3 * r)) / (9 + 20 * r)
    return { kp, ki: kp / ti, kd: 0 }
  }
  const kp = (1 / (k * r)) * (4 / 3 + r / 4)
  const ti = (theta * (32 + 6 * r)) / (13 + 8 * r)
  const td = (4 * theta) / (11 + 2 * r)
  return { kp, ki: kp / ti, kd: kp * td }
}

/** The result of {@link relayAutotune}. */
export interface RelayAutotuneResult {
  /** The ultimate gain 4 d / (pi a). */
  readonly ku: number
  /** The ultimate period (the oscillation period). */
  readonly tu: number
  /** The oscillation amplitude a. */
  readonly amplitude: number
  readonly gains: PidGains
  readonly oscillating: boolean
}

/**
 * Relay auto-tuning (Astrom and Hagglund 1984): put a relay of amplitude d (with optional hysteresis) in feedback with
 * the plant, wait for the limit cycle, read its amplitude a and period Tu, estimate the ultimate gain by describing
 * functions as Ku = 4 d / (pi a), and tune with Ziegler-Nichols.
 *
 * @param plant - The plant.
 * @param options - Relay amplitude (default 1), hysteresis (default 0), sample time (default 0.01), duration (default
 * 100) and the tuning type.
 * @returns Ku, Tu, the amplitude and the gains.
 * @concept relay-autotuning
 * @example
 * ```ts
 * import { control } from 'whitebox-engine/dynamics'
 * import { tf } from 'whitebox-engine/systems'
 * // 1 / (s + 1)^3 has Ku = 8 and Tu = 2 pi / sqrt(3) = 3.628.
 * const r = control.relayAutotune(tf([1], [1, 3, 3, 1]), { dt: 0.005, duration: 60 })
 * Math.abs(r.tu - 3.628) < 0.1 && Math.abs(r.ku - 8) < 0.8 // => true
 * ```
 */
export function relayAutotune(
  plant: Lti,
  options: {
    readonly amplitude?: number
    readonly hysteresis?: number
    readonly dt?: number
    readonly duration?: number
    readonly type?: 'P' | 'PI' | 'PID'
  } = {},
): RelayAutotuneResult {
  const d = options.amplitude ?? 1
  const eps = options.hysteresis ?? 0
  const dt = options.dt ?? 0.01
  const relay: PidController = {
    options: { kp: 1, dt },
    init: () => ({ integral: d, derivative: 0, previousError: 0, started: false }),
    step: (s, setpoint, y) => {
      const e = setpoint - y
      // integral holds the relay's output (its memory for the hysteresis).
      const u = e > eps ? d : e < -eps ? -d : s.integral
      return { u, state: { ...s, integral: u } }
    },
  }
  const r = simulatePid(plant, relay, { duration: options.duration ?? 100, setpoint: 0, disturbance: () => 0 })
  // Use the second half (the established limit cycle): zero crossings of the output give the period.
  const start = Math.floor(r.y.length / 2)
  const ys = r.y.slice(start)
  const ts = r.t.slice(start)
  const ups: number[] = []
  for (let i = 1; i < ys.length; i++) {
    const a = ys[i - 1] ?? 0
    const b = ys[i] ?? 0
    if (a < 0 && b >= 0) ups.push((ts[i - 1] ?? 0) + (dt * -a) / (b - a))
  }
  let top = Number.NEGATIVE_INFINITY
  let bottom = Number.POSITIVE_INFINITY
  for (const v of ys) {
    top = Math.max(top, v)
    bottom = Math.min(bottom, v)
  }
  const amplitude = (top - bottom) / 2
  const oscillating = ups.length >= 2 && amplitude > 0
  const tu = oscillating ? ((ups.at(-1) ?? 0) - (ups[0] ?? 0)) / (ups.length - 1) : Number.NaN
  const ku = oscillating ? (4 * d) / (Math.PI * amplitude) : Number.NaN
  return { ku, tu, amplitude, gains: zieglerNichols(ku, tu, options.type ?? 'PID'), oscillating }
}
