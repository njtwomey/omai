import { type Json, linspaceNumbers, minOf, WhiteboxError } from '../base/index.ts'
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
    throw new WhiteboxError('shape', 'lsim: a scalar input must be 0')
  }
  const um = uv.shape.length === 1 ? reshape(uv, [-1, 1]) : uv
  if ((um.shape[0] ?? 0) !== nt) throw new WhiteboxError('shape', 'lsim: the input needs one row per time')
  if ((um.shape[1] ?? 0) !== m) throw new WhiteboxError('shape', `lsim: the system has ${m} inputs`)
  return um
}

function timeGrid(s: StateSpace, u: Value | null, t: TensorLike | undefined): { times: number[]; u: Value | null } {
  if (!isDiscrete(s)) {
    if (t === undefined) throw new WhiteboxError('value', 'lsim: continuous-time simulation needs the times')
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
 * import { lsimAlgorithm, tf } from 'whitebox-engine/systems'
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
  if (nt === 0) throw new WhiteboxError('value', 'lsim: no samples')
  const h = nt > 1 ? (times[1] ?? 0) - (times[0] ?? 0) : 1
  if (!isDiscrete(s)) {
    for (let k = 1; k < nt; k++) {
      const step = (times[k] ?? 0) - (times[k - 1] ?? 0)
      if (Math.abs(step - h) > 1e-8 + 1e-5 * Math.abs(h))
        throw new WhiteboxError('value', 'lsim: times must be equally spaced')
    }
    if ((times[0] ?? 0) < 0) throw new WhiteboxError('value', 'lsim: the initial time must be nonnegative')
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
 * @throws WhiteboxError with code `value` for unequal time steps or a negative start.
 * @concept time-response
 * @example
 * ```ts
 * import { lsim, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { defaultResponseTimes } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
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
  if (input < 0 || input >= m) throw new WhiteboxError('value', `the system has no input ${input}`)
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
 * @example
 * ```ts
 * import { step, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { impulse, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { initial, ss } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(initial(ss([[0.5]], [[1]], [[2]], [[0]], { dt: 1 }), [1], { n: 3 }).y) // => [2, 1, 0.5]
 * ```
 */
export function initial(sys: Lti, x0: TensorLike, options: Omit<ResponseOptions, 'x0' | 'input'> = {}): TimeResponse {
  const s = toSs(sys)
  const { t: _t, n: _n, ...rest } = options
  const t = responseTimes(s, options)
  return lsim(s, null, t, { ...rest, x0, interp: false })
}
