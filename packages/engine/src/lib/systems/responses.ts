/**
 * Time-domain responses of linear systems, scipy.signal style: `lsim` (and `dlsim`) simulation of the response to any
 * input by exact discretisation, also as a traced algorithm, the step and impulse responses, the free response from an
 * initial state, default time grids from the slowest pole, and the measures read off a step response (rise time,
 * overshoot, settling time).
 */

import { type Json, linspaceNumbers, maxOf, minOf, OmaiError } from '../base/index.ts'
import { interp } from '../numerics/interpolate/index.ts'
import { eigvals, expm } from '../numerics/linalg/index.ts'
import {
  add,
  concat,
  eye,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../trace/index.ts'
import { float, numbers } from './_shared.ts'
import { dcgain } from './frequency.ts'
import { isDiscrete, type Lti, type StateSpace, toSs } from './representations.ts'

// Time responses follow scipy.signal (lsim, step, impulse; dlsim, dstep, dimpulse) and python-control's
// initial_response. Outputs are `[nt]` for single-output systems and `[nt, outputs]` otherwise; states are
// `[nt, states]`. Everything is differentiable in the system matrices, the input and the initial state.

/** A simulated response. */
export interface TimeResponse {
  /** Sample times `[nt]`. */
  readonly t: Tensor
  /** Outputs: `[nt]` for one output, `[nt, outputs]` otherwise. */
  readonly y: Value
  /** States `[nt, states]`. */
  readonly x: Value
}

/** Options for {@link lsim} and {@link lsimAlgorithm}. */
export interface LsimOptions {
  /** Initial state (zeros by default). */
  readonly x0?: TensorLike
  /** Continuous time: interpolate the input linearly between samples (first-order hold, default) or hold it. */
  readonly interp?: boolean
}

/** Trace options for the response functions: a tracer, a policy, a run id or a parent run. */
export type TraceOptions = Partial<Omit<RunOptions, 'maxSteps' | 'start'>>

/** The state of {@link lsimAlgorithm}: the sample index, the state and the outputs so far. */
export interface LsimState {
  readonly k: number
  readonly x: Value
  readonly outputs: readonly Value[]
  readonly states: readonly Value[]
}

const SCHEMA = channels({
  metrics: {
    t: { description: 'the sample time' },
    y0: { description: 'the first output (y1, y2, ... the next ones, up to eight)' },
  },
  tensors: {
    x: { role: 'state', description: 'the state at this sample' },
    y: { role: 'output', description: 'the output at this sample' },
  },
})

function outputMetrics(t: number, y: Value | undefined): Record<string, number> {
  const out: Record<string, number> = { t }
  if (y !== undefined) for (const [i, v] of numbers(y).slice(0, 8).entries()) out[`y${i}`] = v
  return out
}

/** x[k+1] = Phi x[k] + G0 u[k] + G1 u[k+1], y[k] = C x[k] + D u[k]. */
interface Recursion {
  readonly phi: Value
  readonly g0: Value
  readonly g1: Value | null
  readonly c: Value
  readonly d: Value
}

function recursion(s: StateSpace, h: number, hold: 'zoh' | 'foh'): Recursion {
  if (isDiscrete(s)) return { phi: s.a, g0: s.b, g1: null, c: s.c, d: s.d }
  const n = s.a.shape[0] ?? 0
  const m = s.b.shape[1] ?? 0
  if (hold === 'zoh') {
    const em = expm(concat([concat([mul(s.a, h), mul(s.b, h)], 1), zeros([m, n + m])], 0))
    const top = sliceAxis(em, 0, 0, n)
    return { phi: sliceAxis(top, 1, 0, n), g0: sliceAxis(top, 1, n, n + m), g1: null, c: s.c, d: s.d }
  }
  const em = expm(
    concat(
      [
        concat([mul(s.a, h), mul(s.b, h), zeros([n, m])], 1),
        concat([zeros([m, n + m]), eye(m)], 1),
        zeros([m, n + 2 * m]),
      ],
      0,
    ),
  )
  const top = sliceAxis(em, 0, 0, n)
  const f1 = sliceAxis(top, 1, n, n + m)
  const f2 = sliceAxis(top, 1, n + m, n + 2 * m)
  return { phi: sliceAxis(top, 1, 0, n), g0: add(f1, mul(f2, -1)), g1: f2, c: s.c, d: s.d }
}

const column = (v: Value): Value => reshape(v, [-1, 1])
const flat = (v: Value): Value => reshape(v, [-1])

/** Inputs as `[nt, m]` (a 1-D input for one input channel, `null` or 0 for none). */
function inputMatrix(u: TensorLike | null, nt: number, m: number): Value | null {
  if (u === null) return null
  const uv = float(u)
  if (uv.shape.length === 0) {
    if (numbers(uv)[0] === 0) return null
    throw new OmaiError('shape', 'lsim: a scalar input must be 0')
  }
  const um = uv.shape.length === 1 ? reshape(uv, [-1, 1]) : uv
  if ((um.shape[0] ?? 0) !== nt) throw new OmaiError('shape', 'lsim: the input needs one row per time')
  if ((um.shape[1] ?? 0) !== m) throw new OmaiError('shape', `lsim: the system has ${m} inputs`)
  return um
}

function timeGrid(s: StateSpace, u: Value | null, t: TensorLike | undefined): { times: number[]; u: Value | null } {
  if (!isDiscrete(s)) {
    if (t === undefined) throw new OmaiError('value', 'lsim: continuous-time simulation needs the times')
    return { times: numbers(float(t)), u }
  }
  // dlsim: samples every dt up to the last time, the input interpolated linearly onto them.
  if (t === undefined) {
    const nt = u === null ? 0 : (u.shape[0] ?? 0)
    return { times: Array.from({ length: nt }, (_, k) => k * s.dt), u }
  }
  const given = numbers(float(t))
  const stop = given[given.length - 1] ?? 0
  const count = Math.floor(stop / s.dt) + 1
  const times = linspaceNumbers(0, stop, count)
  if (u === null) return { times, u }
  const cols = Array.from({ length: u.shape[1] ?? 0 }, (_, j) => interp(times, given, flat(sliceAxis(u, 1, j, j + 1))))
  return { times, u: stack(cols, 1) }
}

/**
 * The simulation recursion of {@link lsim} as a traced algorithm: one step per sample, the state and output observed
 * as tensors. Continuous systems are discretised exactly for the sample spacing (zero- or first-order hold, by the
 * matrix exponential of an augmented matrix, as scipy).
 *
 * @param sys - The system.
 * @param u - Inputs `[nt]` or `[nt, inputs]`, or `null` for none.
 * @param t - Sample times (equally spaced, starting at 0 or later); optional in discrete time.
 * @param options - Initial state and hold.
 * @returns The algorithm (run it for nt - 1 steps) and the sample times.
 * @concept time-response
 * @example
 * ```ts
 * import { lsimAlgorithm, tf } from 'omai-engine/systems'
 * lsimAlgorithm(tf([1], [1, 1]), [1, 1, 1], [0, 1, 2]).algorithm.name // => 'lsim'
 * ```
 */
export function lsimAlgorithm(
  sys: Lti,
  u: TensorLike | null,
  t?: TensorLike,
  options: LsimOptions = {},
): { readonly algorithm: Algorithm<LsimState>; readonly t: Tensor } {
  const s = toSs(sys)
  const n = s.a.shape[0] ?? 0
  const m = s.b.shape[1] ?? 0
  const given = t === undefined ? undefined : numbers(float(t))
  const raw = inputMatrix(u, given?.length ?? (u === null ? 0 : (float(u).shape[0] ?? 0)), m)
  const { times, u: um } = timeGrid(s, raw, t)
  const nt = times.length
  if (nt === 0) throw new OmaiError('value', 'lsim: no samples')
  const h = nt > 1 ? (times[1] ?? 0) - (times[0] ?? 0) : 1
  if (!isDiscrete(s)) {
    for (let k = 1; k < nt; k++) {
      const step = (times[k] ?? 0) - (times[k - 1] ?? 0)
      if (Math.abs(step - h) > 1e-8 + 1e-5 * Math.abs(h))
        throw new OmaiError('value', 'lsim: times must be equally spaced')
    }
    if ((times[0] ?? 0) < 0) throw new OmaiError('value', 'lsim: the initial time must be nonnegative')
  }
  const r = recursion(s, h, options.interp === false ? 'zoh' : 'foh')
  let x0: Value = options.x0 === undefined ? zeros([n, 1]) : column(float(options.x0))
  const t0 = times[0] ?? 0
  if (!isDiscrete(s) && t0 > 0) x0 = matmul(expm(mul(s.a, t0)), x0)
  const uAt = (k: number): Value | null => (um === null ? null : column(sliceAxis(um, 0, k, k + 1)))
  const output = (x: Value, k: number): Value => {
    const uk = uAt(k)
    return flat(uk === null ? matmul(r.c, x) : add(matmul(r.c, x), matmul(r.d, uk)))
  }
  const algorithm: Algorithm<LsimState> = {
    name: 'lsim',
    config: { states: n, inputs: m, samples: nt, dt: isDiscrete(s) ? s.dt : h, discrete: isDiscrete(s) } as Json,
    schema: SCHEMA,
    init: () => ({ k: 0, x: x0, outputs: [output(x0, 0)], states: [flat(x0)] }),
    step: (st) => {
      const k = st.k
      let next = matmul(r.phi, st.x)
      const u0 = uAt(k)
      if (u0 !== null) next = add(next, matmul(r.g0, u0))
      const u1 = uAt(k + 1)
      if (u1 !== null && r.g1 !== null) next = add(next, matmul(r.g1, u1))
      return { k: k + 1, x: next, outputs: [...st.outputs, output(next, k + 1)], states: [...st.states, flat(next)] }
    },
    done: (st) => (st.k >= nt - 1 ? 'completed' : undefined),
    observe: (st) => ({
      metrics: outputMetrics(times[st.k] ?? 0, st.outputs[st.outputs.length - 1]),
      tensors: {
        x: { role: 'state', value: st.states[st.states.length - 1] ?? zeros([n]) },
        y: { role: 'output', value: st.outputs[st.outputs.length - 1] ?? zeros([0]) },
      },
    }),
  }
  return { algorithm, t: tensor(times) as Tensor }
}

function finish(result: LsimState, t: Tensor, p: number): TimeResponse {
  const y = stack([...result.outputs], 0)
  return { t, y: p === 1 ? reshape(y, [-1]) : y, x: stack([...result.states], 0) }
}

/**
 * Simulate a system's response to an input (scipy.signal.lsim for continuous systems, dlsim for discrete ones): exact
 * discretisation with the input held (zero-order) or interpolated linearly (first-order) between samples.
 * Differentiable in the system matrices, the input and the initial state; traced as {@link lsimAlgorithm}.
 *
 * @param sys - The system.
 * @param u - Inputs `[nt]` or `[nt, inputs]`, or `null` for none.
 * @param t - Sample times; in discrete time optional (then one sample per input row, dt apart).
 * @param options - Initial state, hold, and trace options.
 * @returns Times, outputs and states.
 * @throws OmaiError with code `value` for unequal time steps or a negative start.
 * @concept time-response
 * @example
 * ```ts
 * import { lsim, tf } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(lsim(tf([1], [1, -0.5], { dt: 1 }), [1, 0, 0]).y) // => [0, 1, 0.5]
 * ```
 */
export function lsim(
  sys: Lti,
  u: TensorLike | null,
  t?: TensorLike,
  options: LsimOptions & TraceOptions = {},
): TimeResponse {
  const { x0, interp: hold, ...run } = options
  const { algorithm, t: times } = lsimAlgorithm(sys, u, t, {
    ...(x0 === undefined ? {} : { x0 }),
    ...(hold === undefined ? {} : { interp: hold }),
  })
  const nt = times.shape[0] ?? 0
  const result = runAlgorithm(algorithm, { ...run, maxSteps: nt })
  return finish(result.state, times, toSs(sys).c.shape[0] ?? 0)
}

/** Options for {@link step}, {@link impulse} and {@link initial}. */
export interface ResponseOptions extends TraceOptions {
  /** Sample times (default: scipy's choice from the slowest pole, or n samples in discrete time). */
  readonly t?: TensorLike
  /** Number of samples when the times are not given (default 100). */
  readonly n?: number
  /** Initial state (zeros by default). */
  readonly x0?: TensorLike
  /** The input channel of a multi-input system (default 0). */
  readonly input?: number
}

/**
 * Sample times for a continuous response (scipy's `_default_response_times`): n points over seven time constants of
 * the slowest pole.
 *
 * @param a - The state matrix.
 * @param n - Number of samples (default 100).
 * @returns The times.
 * @example
 * ```ts
 * import { defaultResponseTimes } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(defaultResponseTimes([[-7]], 3)) // => [0, 0.5, 1]
 * ```
 */
export function defaultResponseTimes(a: TensorLike, n = 100): Tensor {
  const { re } = eigvals(primalOf(float(a)))
  let r = minOf(toFlat(re).map(Math.abs))
  if (r === 0) r = 1
  return tensor(linspaceNumbers(0, 7 / r, n)) as Tensor
}

function responseTimes(s: StateSpace, options: ResponseOptions): Tensor {
  if (options.t !== undefined) return float(options.t) as Tensor
  const n = options.n ?? 100
  // scipy's dstep and dimpulse: linspace(0, n dt, n, endpoint=False).
  const spacing = (n * s.dt) / n
  if (isDiscrete(s)) return tensor(Array.from({ length: n }, (_, k) => k * spacing)) as Tensor
  return defaultResponseTimes(s.a, n)
}

function channel(s: StateSpace, input: number): number {
  const m = s.b.shape[1] ?? 0
  if (input < 0 || input >= m) throw new OmaiError('value', `the system has no input ${input}`)
  return input
}

/**
 * The step response (scipy.signal.step and dstep): the output for a unit step on one input, with the input held
 * between samples.
 *
 * @param sys - The system.
 * @param options - Times or sample count, initial state, input channel, trace options.
 * @returns Times, outputs and states.
 * @concept time-response
 * @index function Step response of a linear system.
 * @example
 * ```ts
 * import { step, tf } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(step(tf([1], [1, -0.5], { dt: 1 }), { n: 3 }).y) // => [0, 1, 1.5]
 * ```
 */
export function step(sys: Lti, options: ResponseOptions = {}): TimeResponse {
  const s = toSs(sys)
  const { t: _t, n: _n, input, ...rest } = options
  const t = responseTimes(s, options)
  const nt = t.shape[0] ?? 0
  const m = s.b.shape[1] ?? 0
  const i = channel(s, input ?? 0)
  const u = Array.from({ length: nt }, () => Array.from({ length: m }, (_, j) => (j === i ? 1 : 0)))
  return lsim(s, tensor(u), t, { ...rest, interp: false })
}

/**
 * The impulse response (scipy.signal.impulse and dimpulse). In continuous time the impulse moves the state to
 * x0 + B e_i and the response is C e^{At} (x0 + B e_i) (D, which multiplies a Dirac delta, is left out, as scipy); in
 * discrete time it is the response to a unit pulse at sample 0.
 *
 * @param sys - The system.
 * @param options - Times or sample count, initial state, input channel, trace options.
 * @returns Times, outputs and states.
 * @concept time-response
 * @example
 * ```ts
 * import { impulse, tf } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(impulse(tf([1], [1, -0.5], { dt: 1 }), { n: 3 }).y) // => [0, 1, 0.5]
 * ```
 */
export function impulse(sys: Lti, options: ResponseOptions = {}): TimeResponse {
  const s = toSs(sys)
  const { t: _t, n: _n, input, x0, ...rest } = options
  const t = responseTimes(s, options)
  const i = channel(s, input ?? 0)
  const bi = flat(sliceAxis(s.b, 1, i, i + 1))
  if (!isDiscrete(s)) {
    const start = x0 === undefined ? bi : add(bi, flat(float(x0)))
    return lsim(s, null, t, { ...rest, x0: start, interp: false })
  }
  const nt = t.shape[0] ?? 0
  const m = s.b.shape[1] ?? 0
  const u = Array.from({ length: nt }, (_, k) => Array.from({ length: m }, (_, j) => (k === 0 && j === i ? 1 : 0)))
  return lsim(s, tensor(u), t, { ...rest, ...(x0 === undefined ? {} : { x0 }) })
}

/**
 * The free response from an initial state with no input (python-control initial_response): y = C e^{At} x0, or
 * C A^k x0 in discrete time.
 *
 * @param sys - The system.
 * @param x0 - The initial state.
 * @param options - Times or sample count, trace options.
 * @returns Times, outputs and states.
 * @concept time-response
 * @example
 * ```ts
 * import { initial, ss } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(initial(ss([[0.5]], [[1]], [[2]], [[0]], { dt: 1 }), [1], { n: 3 }).y) // => [2, 1, 0.5]
 * ```
 */
export function initial(sys: Lti, x0: TensorLike, options: Omit<ResponseOptions, 'x0' | 'input'> = {}): TimeResponse {
  const s = toSs(sys)
  const { t: _t, n: _n, ...rest } = options
  const t = responseTimes(s, options)
  return lsim(s, null, t, { ...rest, x0, interp: false })
}

/** What {@link stepMeasures} reads off a step response. */
export interface StepMeasures {
  /** Time to go from 10% to 90% of the final value (the limits are options). */
  readonly riseTime: number
  /** The time after which the response stays within the threshold (2%) of the final value; NaN if it never settles. */
  readonly settlingTime: number
  /** The least value after the rise. */
  readonly settlingMin: number
  /** The greatest value after the rise. */
  readonly settlingMax: number
  /** The percentage by which the response exceeds the final value. */
  readonly overshoot: number
  /** The percentage by which it dips the wrong way before rising (a non-minimum-phase zero's undershoot). */
  readonly undershoot: number
  /** The largest magnitude of the response. */
  readonly peak: number
  /** The time of the peak. */
  readonly peakTime: number
  /** The final value the response is measured against. */
  readonly steadyStateValue: number
}

/** Options of {@link stepMeasures} and {@link stepInfo}. */
export interface StepMeasureOptions {
  /** The final value (the last sample for data, the DC gain for a system). */
  readonly final?: number
  /** Settling band as a fraction of the final value (default 0.02). */
  readonly settlingThreshold?: number
  /** The fractions of the final value that bound the rise (default `[0.1, 0.9]`). */
  readonly riseLimits?: readonly [number, number]
}

/**
 * The step-response measures of sampled data (python-control's `step_info`): rise time between two fractions of the final
 * value, settling time to a band, the overshoot and undershoot as percentages, the peak with its time, and the range after
 * the rise. A response that never reaches a rise limit, or whose final value is not finite, gives NaN measures.
 *
 * @param t - Sample times, increasing.
 * @param y - The response, one value per time.
 * @param options - The final value, the settling band and the rise limits.
 * @returns The measures.
 * @throws OmaiError with code `shape` when `t` and `y` differ in length or are empty.
 * @concept time-response
 * @example
 * ```ts
 * import { stepMeasures } from 'omai-engine/systems'
 * const t = [0, 1, 2, 3, 4]
 * const m = stepMeasures(t, [0, 0.5, 1.2, 1.05, 1], {})
 * [m.overshoot.toFixed(1), m.peakTime, m.riseTime] // => ['20.0', 2, 1]
 * ```
 */
export function stepMeasures(
  t: ArrayLike<number>,
  y: ArrayLike<number>,
  options: StepMeasureOptions = {},
): StepMeasures {
  const n = t.length
  if (n === 0 || y.length !== n) throw new OmaiError('shape', 'stepMeasures: give one response value per time')
  const final = options.final ?? (y[n - 1] as number)
  const band = options.settlingThreshold ?? 0.02
  const [lowFraction, highFraction] = options.riseLimits ?? [0.1, 0.9]
  const nan: StepMeasures = {
    riseTime: Number.NaN,
    settlingTime: Number.NaN,
    settlingMin: Number.NaN,
    settlingMax: Number.NaN,
    overshoot: Number.NaN,
    undershoot: Number.NaN,
    peak: Number.POSITIVE_INFINITY,
    peakTime: Number.POSITIVE_INFINITY,
    steadyStateValue: Number.NaN,
  }
  if (!Number.isFinite(final)) return nan
  const sign = Math.sign(final)
  const yv = Array.from(y)
  const first = (limit: number): number => yv.findIndex((v) => sign * (v - limit * final) >= 0)
  const lower = first(lowFraction)
  const upper = first(highFraction)
  if (lower < 0 || upper < 0) return { ...nan, steadyStateValue: final }
  let settled = 0
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs((yv[i] as number) / final - 1) >= band) {
      settled = i + 1
      break
    }
  }
  const after = yv.slice(upper)
  const top = maxOf(yv.map((v) => sign * v))
  const overshoot =
    Math.abs(top) - Math.abs(final) > 0 ? Math.abs((100 * (Math.abs(top) - Math.abs(final))) / final) : 0
  const lowest = yv.reduce((best, v, i) => (sign * v < sign * (yv[best] as number) ? i : best), 0)
  const dip = yv[lowest] as number
  const undershoot = sign * dip < 0 ? (-100 * dip) / final : 0
  const peakIndex = yv.reduce((best, v, i) => (Math.abs(v) > Math.abs(yv[best] as number) ? i : best), 0)
  return {
    riseTime: (t[upper] as number) - (t[lower] as number),
    settlingTime: settled < n ? (t[settled] as number) : Number.NaN,
    settlingMin: Math.min(minOf(after), final),
    settlingMax: Math.max(maxOf(after), final),
    overshoot,
    undershoot,
    peak: Math.abs(yv[peakIndex] as number),
    peakTime: t[peakIndex] as number,
    steadyStateValue: final,
  }
}

/**
 * The step-response measures of a single-input single-output system: its step response, measured against its DC gain
 * (python-control's `step_info`).
 *
 * @param sys - The system (SISO, stable for the DC gain to be the final value).
 * @param options - Times (as {@link step}), the final value, the settling band and the rise limits.
 * @returns The measures.
 * @throws OmaiError with code `shape` for a system with several inputs or outputs.
 * @concept time-response
 * @example
 * ```ts
 * import { stepInfo, tf } from 'omai-engine/systems'
 * // A first-order lag 1 / (s + 1): rise time ln 9, no overshoot, 2% settling at ln 50.
 * const m = stepInfo(tf([1], [1, 1]), { t: Array.from({ length: 2001 }, (_, i) => i * 0.005) })
 * [m.riseTime.toFixed(2), m.overshoot, m.settlingTime.toFixed(2)] // => ['2.20', 0, '3.92']
 * ```
 */
export function stepInfo(sys: Lti, options: StepMeasureOptions & Pick<ResponseOptions, 't' | 'n'> = {}): StepMeasures {
  const s = toSs(sys)
  if ((s.b.shape[1] ?? 0) !== 1 || (s.c.shape[0] ?? 0) !== 1) {
    throw new OmaiError('shape', 'stepInfo: a single-input single-output system')
  }
  const { t, n, ...rest } = options
  const r = step(sys, { ...(t === undefined ? {} : { t }), ...(n === undefined ? {} : { n }) })
  const final = options.final ?? (toFlat(dcgain(sys))[0] as number)
  return stepMeasures(toFlat(r.t), toFlat(r.y), { ...rest, final })
}
