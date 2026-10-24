import { WhiteboxError } from '../base/index.ts'
import type { ComplexNumber } from '../numerics/differentiate/index.ts'
import type { ComplexArray } from '../numerics/polynomial/index.ts'
import { matmul, reshape, sliceAxis, type TensorLike, tensor, transpose, type Value } from '../tensor/index.ts'
import { asComplexArray, float, fromList, numbers, polyOf, rootsOf, toList, trimLeading } from './_shared.ts'
import { controllabilityStaircase } from './_staircase.ts'
import { abcdNormalize, normalize, ss2tf, ss2zpk, tf2ss, tf2zpk, zpk2ss, zpk2tf } from './conversions.ts'
import { type SosPairing, sos2tf, sos2zpk, zpk2sos } from './sections.ts'

// Systems carry a sample time `dt`: 0 for continuous time (s), positive for discrete time (z), as python-control.

/** A state-space system x' = A x + B u, y = C x + D u (or x[k+1] = A x[k] + B u[k] in discrete time). */
export interface StateSpace {
  readonly kind: 'ss'
  readonly a: Value
  readonly b: Value
  readonly c: Value
  readonly d: Value
  readonly dt: number
}

/** A transfer function num(s) / den(s), highest degree first; `num` is `[k]`, or `[outputs, k]` for one input. */
export interface TransferFunction {
  readonly kind: 'tf'
  readonly num: Value
  readonly den: Value
  readonly dt: number
}

/** A transfer function as zeros, poles and gain: k prod (s - z) / prod (s - p). */
export interface ZerosPolesGain {
  readonly kind: 'zpk'
  readonly zeros: ComplexArray
  readonly poles: ComplexArray
  readonly gain: number
  readonly dt: number
}

/** A cascade of second-order sections, rows [b0, b1, b2, a0, a1, a2]. */
export interface SecondOrderSections {
  readonly kind: 'sos'
  readonly sections: Value
  readonly dt: number
}

/** Any linear time-invariant system. */
export type Lti = StateSpace | TransferFunction | ZerosPolesGain | SecondOrderSections

/** Options shared by the constructors. */
export interface LtiOptions {
  /** Sample time: 0 (default) for continuous time, positive for discrete time. */
  readonly dt?: number
}

function sampleTime(options: LtiOptions): number {
  const dt = options.dt ?? 0
  if (!(dt >= 0) || !Number.isFinite(dt)) throw new WhiteboxError('value', `the sample time must be >= 0, got ${dt}`)
  return dt
}

/**
 * A state-space system (scipy.signal.StateSpace, python-control ss), differentiable in its matrices.
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, m]`.
 * @param c - C `[p, n]`.
 * @param d - D `[p, m]` (zeros when omitted).
 * @param options - The sample time.
 * @returns The system.
 * @throws WhiteboxError with code `shape` when the matrices do not fit.
 * @concept state-space
 * @example
 * ```ts
 * import { ss } from 'whitebox-engine/systems'
 * ss([[-1]], [[1]], [[1]]).d.shape // => [1, 1]
 * ```
 */
export function ss(a: TensorLike, b: TensorLike, c: TensorLike, d?: TensorLike, options: LtiOptions = {}): StateSpace {
  return { kind: 'ss', ...abcdNormalize(a, b, c, d), dt: sampleTime(options) }
}

/**
 * A transfer function (scipy.signal.TransferFunction, python-control tf), normalised as scipy (monic denominator),
 * differentiable in its coefficients.
 *
 * @param num - Numerator, highest degree first (`[k]`, or `[outputs, k]`).
 * @param den - Denominator, highest degree first.
 * @param options - The sample time.
 * @returns The system.
 * @concept transfer-functions
 * @example
 * ```ts
 * import { tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(tf([2], [2, 4]).den) // => [1, 2]
 * ```
 */
export function tf(num: TensorLike, den: TensorLike, options: LtiOptions = {}): TransferFunction {
  return { kind: 'tf', ...normalize(num, den), dt: sampleTime(options) }
}

/**
 * A transfer function by its zeros, poles and gain (scipy.signal.ZerosPolesGain, python-control zpk).
 *
 * @param zeros - Zeros, real or as parallel parts (conjugate pairs).
 * @param poles - Poles, real or as parallel parts (conjugate pairs).
 * @param gain - The gain.
 * @param options - The sample time.
 * @returns The system.
 * @concept transfer-functions
 * @example
 * ```ts
 * import { zpk } from 'whitebox-engine/systems'
 * zpk([], [-1, -2], 3).poles.re // => [-1, -2]
 * ```
 */
export function zpk(
  zeros: readonly number[] | ComplexArray,
  poles: readonly number[] | ComplexArray,
  gain: number,
  options: LtiOptions = {},
): ZerosPolesGain {
  return {
    kind: 'zpk',
    zeros: asComplexArray(zeros),
    poles: asComplexArray(poles),
    gain,
    dt: sampleTime(options),
  }
}

/**
 * A cascade of second-order sections (the `sos` format of scipy.signal).
 *
 * @param sections - Rows [b0, b1, b2, a0, a1, a2], `[n, 6]`.
 * @param options - The sample time.
 * @returns The system.
 * @throws WhiteboxError with code `shape` unless the sections are `[n, 6]`.
 * @concept second-order-sections
 * @example
 * ```ts
 * import { sos } from 'whitebox-engine/systems'
 * sos([[1, 0, 0, 1, -0.5, 0]], { dt: 1 }).sections.shape // => [1, 6]
 * ```
 */
export function sos(sections: TensorLike, options: LtiOptions = {}): SecondOrderSections {
  const s = float(sections)
  if (s.shape.length !== 2 || s.shape[1] !== 6) throw new WhiteboxError('shape', 'sos: expected [n, 6] sections')
  return { kind: 'sos', sections: s, dt: sampleTime(options) }
}

/**
 * Whether a system is discrete time (a positive sample time).
 *
 * @param sys - The system.
 * @returns `true` for discrete time.
 * @example
 * ```ts
 * import { isDiscrete, tf } from 'whitebox-engine/systems'
 * isDiscrete(tf([1], [1, -0.5], { dt: 0.1 })) // => true
 * ```
 */
export function isDiscrete(sys: Lti): boolean {
  return sys.dt > 0
}

/** The numbers of inputs, outputs and states of a system. */
export interface Dimensions {
  readonly inputs: number
  readonly outputs: number
  readonly states: number
}

/**
 * The numbers of inputs, outputs and states (python-control ninputs, noutputs, nstates; the states of a transfer
 * function are those of its controller canonical realisation).
 *
 * @param sys - The system.
 * @returns The dimensions.
 * @example
 * ```ts
 * import { dimensions, tf } from 'whitebox-engine/systems'
 * dimensions(tf([1], [1, 3, 2])) // => { inputs: 1, outputs: 1, states: 2 }
 * ```
 */
export function dimensions(sys: Lti): Dimensions {
  switch (sys.kind) {
    case 'ss':
      return { inputs: sys.b.shape[1] ?? 0, outputs: sys.c.shape[0] ?? 0, states: sys.a.shape[0] ?? 0 }
    case 'tf':
      return {
        inputs: 1,
        outputs: sys.num.shape.length === 2 ? (sys.num.shape[0] ?? 1) : 1,
        states: (sys.den.shape[0] ?? 1) - 1,
      }
    case 'zpk':
      return { inputs: 1, outputs: 1, states: sys.poles.re.length }
    case 'sos':
      return { inputs: 1, outputs: 1, states: 2 * (sys.sections.shape[0] ?? 0) }
  }
}

/**
 * Any system in state-space form (scipy's `_as_ss`): transfer functions by the controller canonical form.
 *
 * @param sys - The system.
 * @returns The state-space system, same sample time.
 * @concept state-space
 * @example
 * ```ts
 * import { tf, toSs } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(toSs(tf([1], [1, 3, 2])).a) // => [[-3, -2], [1, 0]]
 * ```
 */
export function toSs(sys: Lti): StateSpace {
  switch (sys.kind) {
    case 'ss':
      return sys
    case 'tf':
      return { kind: 'ss', ...tf2ss(sys.num, sys.den), dt: sys.dt }
    case 'zpk':
      return { kind: 'ss', ...zpk2ss(sys.zeros, sys.poles, sys.gain), dt: sys.dt }
    case 'sos': {
      const { num, den } = sos2tf(sys.sections)
      return { kind: 'ss', ...tf2ss(num, den), dt: sys.dt }
    }
  }
}

/** Which input (and output) of a multivariable system a conversion takes. */
export interface ChannelOptions {
  readonly input?: number
  readonly output?: number
}

/**
 * Any system as a transfer function (scipy's `_as_tf`); a state-space system gives the transfer functions from one
 * input to every output.
 *
 * @param sys - The system.
 * @param options - The input of a state-space system (default 0).
 * @returns The transfer function, same sample time.
 * @concept transfer-functions
 * @example
 * ```ts
 * import { ss, toTf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(toTf(ss([[-2]], [[1]], [[3]])).num) // => [0, 3]
 * ```
 */
export function toTf(sys: Lti, options: ChannelOptions = {}): TransferFunction {
  switch (sys.kind) {
    case 'tf':
      return sys
    case 'ss': {
      const { num, den } = ss2tf(sys.a, sys.b, sys.c, sys.d, options.input ?? 0)
      return { kind: 'tf', num: num.shape[0] === 1 ? reshape(num, [num.shape[1] ?? 0]) : num, den, dt: sys.dt }
    }
    case 'zpk':
      return { kind: 'tf', ...zpk2tf(sys.zeros, sys.poles, sys.gain), dt: sys.dt }
    case 'sos':
      return { kind: 'tf', ...sos2tf(sys.sections), dt: sys.dt }
  }
}

/**
 * Any single-output channel as zeros, poles and gain (scipy's `_as_zpk`).
 *
 * @param sys - The system.
 * @param options - The input and output of a multivariable system (default 0 and 0).
 * @returns The zero-pole-gain system, same sample time.
 * @example
 * ```ts
 * import { tf, toZpk } from 'whitebox-engine/systems'
 * toZpk(tf([3], [1, 1])).gain // => 3
 * ```
 */
export function toZpk(sys: Lti, options: ChannelOptions = {}): ZerosPolesGain {
  switch (sys.kind) {
    case 'zpk':
      return sys
    case 'sos':
      return { kind: 'zpk', ...sos2zpk(sys.sections, { pad: false }), dt: sys.dt }
    case 'ss': {
      const out = options.output ?? 0
      const c = sliceAxis(sys.c, 0, out, out + 1)
      const d = sliceAxis(sys.d, 0, out, out + 1)
      return { kind: 'zpk', ...ss2zpk(sys.a, sys.b, c, d, options.input ?? 0), dt: sys.dt }
    }
    case 'tf': {
      const out = options.output ?? 0
      const num = sys.num.shape.length === 2 ? reshape(sliceAxis(sys.num, 0, out, out + 1), [-1]) : sys.num
      return { kind: 'zpk', ...tf2zpk(num, sys.den), dt: sys.dt }
    }
  }
}

/**
 * Any single-input single-output system as second-order sections (scipy.signal.zpk2sos of its zeros, poles and
 * gain), with the same response: analog systems, and discrete ones with fewer zeros than poles, pair minimally;
 * others pair 'nearest'.
 *
 * @param sys - The system.
 * @param options - The pairing rule.
 * @returns The sections, same sample time.
 * @concept second-order-sections
 * @example
 * ```ts
 * import { toSos, zpk } from 'whitebox-engine/systems'
 * toSos(zpk([], [0.5, 0.25], 1, { dt: 1 })).sections.shape // => [1, 6]
 * ```
 */
export function toSos(sys: Lti, options: { readonly pairing?: SosPairing } = {}): SecondOrderSections {
  if (sys.kind === 'sos') return sys
  const z = toZpk(sys)
  const analog = !isDiscrete(sys)
  // Padding to equal counts would add zeros at the origin (a time advance), so unequal counts pair minimally.
  const equal = z.zeros.re.length === z.poles.re.length
  const pairing = options.pairing ?? (analog || !equal ? 'minimal' : 'nearest')
  const sections = zpk2sos(z.zeros, z.poles, z.gain, { analog, pairing })
  return { kind: 'sos', sections, dt: sys.dt }
}

/** Options for {@link minreal}. */
export interface MinrealOptions {
  /** Tolerance: for pole-zero cancellation, or the rank tolerance of the staircase reductions. */
  readonly tol?: number
}

function cancel(zeros: ComplexArray, poles: ComplexArray, tol: number | undefined): [ComplexNumber[], ComplexNumber[]] {
  const kept: ComplexNumber[] = []
  const ps = toList(poles)
  const sqrtEps = Math.sqrt(Number.EPSILON)
  for (const z of toList(zeros)) {
    const t = tol ?? 1000 * Math.max(Number.EPSILON, Math.hypot(z.re, z.im) * sqrtEps)
    const i = ps.findIndex((p) => Math.hypot(z.re - p.re, z.im - p.im) < t)
    if (i >= 0) ps.splice(i, 1)
    else kept.push(z)
  }
  return [kept, ps]
}

function minrealSs(sys: StateSpace, tol: number | undefined): StateSpace {
  const n = sys.a.shape[0] ?? 0
  if (n === 0) return sys
  const reduce = (a: Value, b: Value, c: Value, dual: boolean): [Value, Value, Value] => {
    const { t, rank } = dual
      ? controllabilityStaircase(transpose(a), transpose(c), tol)
      : controllabilityStaircase(a, b, tol)
    const k = a.shape[0] ?? 0
    const basis = sliceAxis(t, 1, 0, rank)
    if (rank === k) return [a, b, c]
    const bt = transpose(basis)
    return [matmul(bt, matmul(a, basis)), matmul(bt, b), matmul(c, basis)]
  }
  const [a1, b1, c1] = reduce(sys.a, sys.b, sys.c, false)
  if ((a1.shape[0] ?? 0) === 0) return sys
  const [a2, b2, c2] = reduce(a1, b1, c1, true)
  return { kind: 'ss', a: a2, b: b2, c: c2, d: sys.d, dt: sys.dt }
}

/**
 * A minimal realisation (python-control minreal): a state-space system loses its uncontrollable then unobservable
 * states (orthogonal staircase reductions, as SLICOT TB01PD); a transfer function or zero-pole-gain system loses
 * cancelling pole-zero pairs (closer than `tol`, default 1000 max(eps, |z| sqrt(eps))).
 *
 * @param sys - The system.
 * @param options - The tolerance.
 * @returns A system of the same kind with no redundant states, poles or zeros.
 * @concept minimal-realisation
 * @example
 * ```ts
 * import { minreal, tf } from 'whitebox-engine/systems'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(minreal(tf([1, 1], [1, 3, 2])).den) // => [1, 2]
 * ```
 */
export function minreal<T extends Lti>(sys: T, options: MinrealOptions = {}): T {
  return minrealOf(sys, options) as T
}

function minrealOf(sys: Lti, options: MinrealOptions): Lti {
  switch (sys.kind) {
    case 'ss':
      return minrealSs(sys, options.tol)
    case 'zpk': {
      const [z, p] = cancel(sys.zeros, sys.poles, options.tol)
      return { ...sys, zeros: fromList(z), poles: fromList(p) }
    }
    case 'sos':
      return toSos(minrealOf(toZpk(sys), options))
    case 'tf': {
      if (sys.num.shape.length === 2) throw new WhiteboxError('shape', 'minreal: one numerator row is supported')
      const b = trimLeading(numbers(sys.num))
      const a = numbers(sys.den)
      const gain = (b[0] ?? 0) / (a[0] ?? 1)
      if (gain === 0) return { ...sys, num: tensor([0]), den: tensor([1]) }
      const [z, p] = cancel(rootsOf(b), rootsOf(a), options.tol)
      return { ...sys, num: tensor(polyOf(fromList(z)).map((v) => v * gain)), den: tensor(polyOf(fromList(p))) }
    }
  }
}
