/**
 * Turning a continuous-time system into a discrete-time one for a given sample time (scipy.signal.cont2discrete,
 * python-control sample_system): zero- and first-order hold, the bilinear (Tustin) and other generalised bilinear
 * transforms, impulse invariance and pole-zero matching, plus a differentiable discretisation of state-space matrices.
 */

import { OmaiError } from '../base/index.ts'
import { complexOps as C, type ComplexNumber } from '../numerics/differentiate/index.ts'
import { expm, solve } from '../numerics/linalg/index.ts'
import { add, concat, eye, matmul, mul, sliceAxis, sub, type TensorLike, transpose, zeros } from '../tensor/index.ts'
import { fromList, numbers, toList } from './_shared.ts'
import type { StateSpaceMatrices } from './conversions.ts'
import { dcgain } from './frequency.ts'
import {
  isDiscrete,
  type Lti,
  type StateSpace,
  ss,
  toSos,
  toSs,
  toTf,
  toZpk,
  type ZerosPolesGain,
} from './representations.ts'

/** A continuous-to-discrete conversion method (scipy.signal.cont2discrete names, plus python-control's 'matched'). */
export type DiscretisationMethod =
  | 'zoh'
  | 'foh'
  | 'gbt'
  | 'bilinear'
  | 'tustin'
  | 'euler'
  | 'forward_diff'
  | 'backward_diff'
  | 'impulse'
  | 'matched'

/** Options for {@link cont2discrete}. */
export interface Cont2DiscreteOptions {
  /** The method (default 'zoh'). */
  readonly method?: DiscretisationMethod
  /** The weight of the generalised bilinear transform, in [0, 1] (needed for 'gbt'). */
  readonly alpha?: number
  /** For the bilinear (Tustin) method: the frequency (rad per time unit) where the responses match exactly. */
  readonly prewarpFrequency?: number
}

function gbt(s: StateSpace, dt: number, alpha: number): StateSpace {
  if (!(alpha >= 0 && alpha <= 1)) throw new OmaiError('value', 'cont2discrete: alpha must be in [0, 1]')
  const n = s.a.shape[0] ?? 0
  const identity = eye(n)
  const ima = sub(identity, mul(s.a, alpha * dt))
  const ad = solve(ima, add(identity, mul(s.a, (1 - alpha) * dt)))
  const bd = solve(ima, mul(s.b, dt))
  const cd = transpose(solve(transpose(ima), transpose(s.c)))
  const dd = add(s.d, mul(matmul(s.c, bd), alpha))
  return { kind: 'ss', a: ad, b: bd, c: cd, d: dd, dt }
}

function holdOrImpulse(s: StateSpace, dt: number, method: 'zoh' | 'foh' | 'impulse'): StateSpace {
  const n = s.a.shape[0] ?? 0
  const m = s.b.shape[1] ?? 0
  if (method === 'zoh') {
    const top = sliceAxis(expm(mul(concat([concat([s.a, s.b], 1), zeros([m, n + m])], 0), dt)), 0, 0, n)
    return { kind: 'ss', a: sliceAxis(top, 1, 0, n), b: sliceAxis(top, 1, n, n + m), c: s.c, d: s.d, dt }
  }
  if (method === 'foh') {
    const em = concat(
      [
        concat([mul(s.a, dt), mul(s.b, dt), zeros([n, m])], 1),
        concat([zeros([m, n + m]), eye(m)], 1),
        zeros([m, n + 2 * m]),
      ],
      0,
    )
    const top = sliceAxis(expm(em), 0, 0, n)
    const ms11 = sliceAxis(top, 1, 0, n)
    const ms12 = sliceAxis(top, 1, n, n + m)
    const ms13 = sliceAxis(top, 1, n + m, n + 2 * m)
    return {
      kind: 'ss',
      a: ms11,
      b: add(sub(ms12, ms13), matmul(ms11, ms13)),
      c: s.c,
      d: add(s.d, matmul(s.c, ms13)),
      dt,
    }
  }
  if (numbers(s.d).some((v) => Math.abs(v) > 1e-8)) {
    throw new OmaiError('value', 'cont2discrete: the impulse method needs a strictly proper system (D = 0)')
  }
  const ad = expm(mul(s.a, dt))
  const bd = mul(matmul(ad, s.b), dt)
  return { kind: 'ss', a: ad, b: bd, c: s.c, d: mul(matmul(s.c, s.b), dt), dt }
}

function matched(sys: Lti, dt: number): ZerosPolesGain {
  const z = toZpk(sys)
  const map = (q: ComplexNumber): ComplexNumber => C.exp(C.mul(q, dt))
  const zz = toList(z.zeros).map(map)
  const zp = toList(z.poles).map(map)
  let pre = C.of(1)
  for (const q of zz) pre = C.mul(pre, C.sub(C.of(1), q))
  for (const q of zp) pre = C.div(pre, C.sub(C.of(1), q))
  const k = numbers(dcgain(toTf(sys))).at(0) ?? Number.NaN
  const gain = k / pre.re
  if (!Number.isFinite(gain)) {
    throw new OmaiError('value', 'cont2discrete: the matched method needs a finite, nonzero dc gain')
  }
  return { kind: 'zpk', zeros: fromList(zz), poles: fromList(zp), gain, dt }
}

/** A discretised state-space system in the kind of the original. */
function asKindOf(original: Lti, d: Lti): Lti {
  switch (original.kind) {
    case 'ss':
      return toSs(d)
    case 'tf':
      return toTf(d)
    case 'zpk':
      return toZpk(d)
    case 'sos':
      return toSos(d)
  }
}

/**
 * Discretise a continuous-time system (scipy.signal.cont2discrete, python-control sample_system): zero-order hold
 * (exact for piecewise-constant inputs), first-order hold (exact for piecewise-linear inputs), the generalised
 * bilinear transform (alpha = 0 forward Euler, 1/2 Tustin, 1 backward Euler), Tustin with frequency prewarping,
 * impulse invariance, and the matched pole-zero mapping z = e^{sT} (python-control's, single-input single-output,
 * gain matched at dc). The result has the kind of the input (transfer functions through state space, as scipy) and is
 * differentiable in the system for every method but 'matched'.
 *
 * @param sys - A continuous-time system.
 * @param dt - The sample time.
 * @param options - Method, gbt weight and prewarp frequency.
 * @returns The discrete-time system.
 * @throws OmaiError with code `value` for a discrete input, a bad sample time or alpha, a non-strictly-proper
 * system with 'impulse', or an infinite dc gain with 'matched'.
 * @concept discretisation
 * @example
 * ```ts
 * import { cont2discrete, ss } from 'omai-engine/systems'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(cont2discrete(ss([[-1]], [[1]], [[1]]), 0.5).a) - Math.exp(-0.5)) < 1e-15 // => true
 * ```
 */
export function cont2discrete<T extends Lti>(sys: T, dt: number, options: Cont2DiscreteOptions = {}): T {
  return discretised(sys, dt, options) as T
}

function discretised(sys: Lti, dt: number, options: Cont2DiscreteOptions): Lti {
  if (isDiscrete(sys)) throw new OmaiError('value', 'cont2discrete: the system is already discrete')
  if (!(dt > 0) || !Number.isFinite(dt)) throw new OmaiError('value', 'cont2discrete: dt must be positive')
  const method = options.method ?? 'zoh'
  if (method === 'matched') return asKindOf(sys, matched(sys, dt))
  const s = toSs(sys)
  let d: StateSpace
  switch (method) {
    case 'gbt':
      if (options.alpha === undefined) throw new OmaiError('value', "cont2discrete: 'gbt' needs alpha")
      d = gbt(s, dt, options.alpha)
      break
    case 'bilinear':
    case 'tustin': {
      const w = options.prewarpFrequency
      // Prewarping: transform with the sample time that maps w exactly, then keep the true sample time.
      const warp = w === undefined || w === 0 ? dt : (2 * Math.tan((w * dt) / 2)) / w
      d = { ...gbt(s, warp, 0.5), dt }
      break
    }
    case 'euler':
    case 'forward_diff':
      d = gbt(s, dt, 0)
      break
    case 'backward_diff':
      d = gbt(s, dt, 1)
      break
    case 'zoh':
    case 'foh':
    case 'impulse':
      d = holdOrImpulse(s, dt, method)
      break
    default:
      throw new OmaiError('value', `cont2discrete: unknown method '${method as string}'`)
  }
  return asKindOf(sys, d)
}

/**
 * The discrete matrices of a continuous state-space realisation (scipy.signal.cont2discrete on an (A, B, C, D)
 * tuple), differentiable in the matrices.
 *
 * @param a - A.
 * @param b - B.
 * @param c - C.
 * @param d - D.
 * @param dt - The sample time.
 * @param options - Method, gbt weight and prewarp frequency ('matched' is not available on matrices).
 * @returns Ad, Bd, Cd and Dd.
 * @example
 * ```ts
 * import { c2dMatrices } from 'omai-engine/systems'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(c2dMatrices([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], [[0]], 0.5).b) // => [[0.125], [0.5]]
 * ```
 */
export function c2dMatrices(
  a: TensorLike,
  b: TensorLike,
  c: TensorLike,
  d: TensorLike | undefined,
  dt: number,
  options: Omit<Cont2DiscreteOptions, 'method'> & { readonly method?: Exclude<DiscretisationMethod, 'matched'> } = {},
): StateSpaceMatrices {
  const r = toSs(cont2discrete(ss(a, b, c, d), dt, options))
  return { a: r.a, b: r.b, c: r.c, d: r.d }
}
